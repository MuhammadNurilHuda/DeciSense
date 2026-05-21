from __future__ import annotations

import json

import pandas as pd
import pytest

from ds_engine.profiling.target_profile import (
    TargetProfileError,
    TargetProfileResult,
    create_target_profile,
)


def _issue_codes(result: TargetProfileResult) -> set[str]:
    return {issue.code for issue in result.issues}


def test_create_target_profile_summarizes_binary_classification_target() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40, 50],
            "target": [1, 1, 1, 0, 0],
        }
    )

    result = create_target_profile(
        dataframe,
        target_column="target",
        task_type="classification",
    )

    assert result.target_column == "target"
    assert result.task_type == "classification"
    assert result.row_count == 5
    assert result.non_null_count == 5
    assert result.missing_count == 0
    assert result.unique_count == 2
    assert result.majority_class == 1
    assert result.majority_class_ratio == 0.6
    assert result.minority_class == 0
    assert result.minority_class_ratio == 0.4
    assert result.class_imbalance_ratio == 1.5
    assert result.is_usable_for_modeling is True
    assert result.issues == []


def test_create_target_profile_flags_single_class_classification_target() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [10, 20, 30],
            "target": [1, 1, 1],
        }
    )

    result = create_target_profile(
        dataframe,
        target_column="target",
        task_type="classification",
    )

    assert result.unique_count == 1
    assert "single_class_target" in _issue_codes(result)
    assert result.has_critical_issues is True
    assert result.is_usable_for_modeling is False


def test_create_target_profile_flags_missing_and_imbalanced_classification_target() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "feature": list(range(11)),
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, None],
        }
    )

    result = create_target_profile(
        dataframe,
        target_column="target",
        task_type="classification",
        classification_imbalance_ratio_threshold=5.0,
        classification_minority_ratio_threshold=0.15,
    )

    assert result.missing_count == 1
    assert result.missing_ratio == 0.090909
    assert "target_missing_values" in _issue_codes(result)
    assert "class_imbalance" in _issue_codes(result)
    assert result.class_imbalance_ratio == 9.0


def test_create_target_profile_summarizes_regression_target() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = create_target_profile(
        dataframe,
        target_column="target",
        task_type="regression",
    )

    assert result.task_type == "regression"
    assert result.unique_count == 4
    assert result.numeric_summary is not None
    assert result.numeric_summary.count == 4
    assert result.numeric_summary.mean == 25.0
    assert result.numeric_summary.median == 25.0
    assert result.numeric_summary.minimum == 10.0
    assert result.numeric_summary.maximum == 40.0
    assert result.class_distribution == []
    assert result.is_usable_for_modeling is True


def test_create_target_profile_flags_invalid_regression_target() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": ["10", "bad-value", "30"],
        }
    )

    result = create_target_profile(
        dataframe,
        target_column="target",
        task_type="regression",
    )

    assert "non_numeric_regression_target" in _issue_codes(result)
    assert result.has_critical_issues is True
    assert result.is_usable_for_modeling is False


def test_create_target_profile_raises_when_target_column_is_missing() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "label": [0, 1, 0],
        }
    )

    with pytest.raises(TargetProfileError, match="Target column not found"):
        create_target_profile(
            dataframe,
            target_column="target",
            task_type="classification",
        )


def test_target_profile_to_dict_is_json_serializable() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
        }
    )

    result = create_target_profile(
        dataframe,
        target_column="target",
        task_type="classification",
    )
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["class_distribution"][0]["label"] == "2024-01-01T00:00:00"
