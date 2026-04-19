from __future__ import annotations

import pandas as pd

from ds_engine.intake.infer_task import (
    TaskInferenceResult,
    _infer_task_type_from_target,
    infer_task_from_dataframe,
)


def test_infer_task_from_dataframe_returns_not_found_when_no_target_column_matches() -> None:
    df = pd.DataFrame(
        {
            "age": [20, 21, 22],
            "income": [1000, 2000, 3000],
        }
    )

    result = infer_task_from_dataframe(df)

    assert isinstance(result, TaskInferenceResult)
    assert result.candidate_target is None
    assert result.task_type is None
    assert result.status == "not_found"


def test_infer_task_from_dataframe_returns_ambiguous_when_multiple_target_candidates_exist() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1, 0],
            "label": [1, 0, 1],
            "feature": [10, 20, 30],
        }
    )

    result = infer_task_from_dataframe(df)

    assert result.candidate_target is None
    assert result.task_type is None
    assert result.status == "ambiguous"


def test_infer_task_from_dataframe_infers_classification_for_string_target() -> None:
    df = pd.DataFrame(
        {
            "feature": [1.2, 3.4, 5.6],
            "target": ["yes", "no", "yes"],
        }
    )

    result = infer_task_from_dataframe(df)

    assert result.candidate_target == "target"
    assert result.task_type == "classification"
    assert result.status == "ok"


def test_infer_task_from_dataframe_infers_classification_for_small_integer_label_set() -> None:
    df = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )

    result = infer_task_from_dataframe(df)

    assert result.candidate_target == "target"
    assert result.task_type == "classification"
    assert result.status == "ok"


def test_infer_task_from_dataframe_infers_regression_for_numeric_target_with_many_unique_values() -> None:
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [10.5, 20.1, 30.2, 40.8, 50.3],
        }
    )

    result = infer_task_from_dataframe(df)

    assert result.candidate_target == "target"
    assert result.task_type == "regression"
    assert result.status == "ok"


def test_infer_task_from_dataframe_respects_custom_target_candidates() -> None:
    df = pd.DataFrame(
        {
            "sales": [100, 120, 140],
            "feature_a": [1, 2, 3],
        }
    )

    result = infer_task_from_dataframe(
        df,
        target_candidates=("sales",),
    )

    assert result.candidate_target == "sales"
    assert result.task_type == "regression"
    assert result.status == "ok"

def test_infer_task_from_dataframe_infers_classification_for_multiclass_integer_labels() -> None:
    df = pd.DataFrame(
        {
            "feature": [10, 11, 12, 13, 14, 15],
            "target": [0, 1, 2, 0, 1, 2],
        }
    )

    result = infer_task_from_dataframe(df)

    assert result.candidate_target == "target"
    assert result.task_type == "classification"
    assert result.status == "ok"

def test_infer_task_type_from_target_returns_classification_for_boolean_target() -> None:
    target = pd.Series([True, False, True])

    result = _infer_task_type_from_target(
        target,
        classification_unique_value_threshold=20,
    )

    assert result == "classification"

def test_infer_task_from_dataframe_returns_ambiguous_for_all_missing_target_column() -> None:
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [None, None, None],
        }
    )

    result = infer_task_from_dataframe(df)

    assert result.candidate_target == "target"
    assert result.task_type is None
    assert result.status == "ambiguous"

    