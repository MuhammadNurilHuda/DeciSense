from __future__ import annotations

import pandas as pd

from ds_engine.intake.validate_tabular import (
    TabularValidationResult,
    validate_tabular_dataset,
)


def test_validate_tabular_dataset_returns_valid_result_for_normal_dataframe() -> None:
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "income": [5000, 7000, 9000],
        }
    )

    result = validate_tabular_dataset(df)

    assert isinstance(result, TabularValidationResult)
    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []
    assert result.row_count == 3
    assert result.column_count == 2


def test_validate_tabular_dataset_rejects_empty_dataframe() -> None:
    df = pd.DataFrame()

    result = validate_tabular_dataset(df)

    assert result.is_valid is False
    assert "Dataset is empty." in result.errors


def test_validate_tabular_dataset_rejects_dataframe_with_too_few_rows() -> None:
    df = pd.DataFrame({"feature": [1], "target": [0]})

    result = validate_tabular_dataset(df, min_rows=2)

    assert result.is_valid is False
    assert "Dataset must contain at least 2 row(s), but found 1." in result.errors


def test_validate_tabular_dataset_rejects_dataframe_with_only_missing_values() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [None, None],
            "feature_b": [None, None],
        }
    )

    result = validate_tabular_dataset(df)

    assert result.is_valid is False
    assert "All dataset columns contain only missing values." in result.errors


def test_validate_tabular_dataset_warns_for_fully_missing_column() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [None, None, None],
        }
    )

    result = validate_tabular_dataset(df)

    assert result.is_valid is True
    assert result.errors == []
    assert any(
        "Some columns contain only missing values" in warning
        for warning in result.warnings
    )


def test_validate_tabular_dataset_warns_for_row_with_only_missing_values() -> None:
    df = pd.DataFrame(
        {
            "feature_a": [1, None, 3],
            "feature_b": [10, None, 30],
        }
    )

    result = validate_tabular_dataset(df)

    assert result.is_valid is True
    assert "Some rows contain only missing values." in result.warnings
