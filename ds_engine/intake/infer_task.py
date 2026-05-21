from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)

TaskType = Literal["classification", "regression"]
InferenceStatus = Literal["ok", "ambiguous", "not_found"]

DEFAULT_TARGET_CANDIDATES: tuple[str, ...] = (
    "target",
    "label",
    "y",
    "class",
    "outcome",
)


@dataclass(frozen=True)
class TaskInferenceResult:
    """Structured result for target-column and task-type inference."""

    candidate_target: str | None
    task_type: TaskType | None
    status: InferenceStatus
    reasoning: list[str]


def infer_task_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    target_candidates: tuple[str, ...] = DEFAULT_TARGET_CANDIDATES,
    classification_unique_value_threshold: int = 20,
) -> TaskInferenceResult:
    """
    Returns
    -------
    TaskInferenceResult
        Structured inference result
    """

    normalized_candidates = {name.strip().lower(): name for name in dataframe.columns}

    matched_targets = [
        normalized_candidates[candidate.lower()]
        for candidate in target_candidates
        if candidate.lower() in normalized_candidates
    ]

    if len(matched_targets) == 0:
        return TaskInferenceResult(
            candidate_target=None,
            task_type=None,
            status="not_found",
            reasoning=["No target column matched the configured candidate names."],
        )

    if len(matched_targets) > 1:
        return TaskInferenceResult(
            candidate_target=None,
            task_type=None,
            status="ambiguous",
            reasoning=[f"Multiple target-like columns were found: {matched_targets}."],
        )

    target_column = matched_targets[0]
    target_series = dataframe[target_column]
    if target_series.dropna().empty:
        return TaskInferenceResult(
            candidate_target=target_column,
            task_type=None,
            status="ambiguous",
            reasoning=[
                f"Matched target column '{target_column}'.",
                "The matched target column contains only missing values.",
            ],
        )
    inferred_task_type = _infer_task_type_from_target(
        target_series,
        classification_unique_value_threshold=classification_unique_value_threshold,
    )

    reasoning = [f"Matched target column '{target_column}'."]
    if inferred_task_type == "classification":
        reasoning.append(
            "Target was inferred as classification based on dtype and/or limited unique values."
        )
    else:
        reasoning.append(
            "Target was inferred as regression because it is numeric with many unique values."
        )
    return TaskInferenceResult(
        candidate_target=target_column,
        task_type=inferred_task_type,
        status="ok",
        reasoning=reasoning,
    )


def _infer_task_type_from_target(
    target_series: pd.Series, *, classification_unique_value_threshold: int
) -> TaskType:
    """
    Infer task type from a target series using simple, stable heuristics.
    """

    non_null_target = target_series.dropna()

    if non_null_target.empty:
        return "regression"
    if is_bool_dtype(target_series):
        return "classification"
    if not is_numeric_dtype(target_series):
        return "classification"
    if _is_label_like_numeric_target(
        non_null_target, max_class_count=classification_unique_value_threshold
    ):
        return "classification"
    return "regression"


def _is_label_like_numeric_target(
    non_null_target: pd.Series,
    *,
    max_class_count: int,
) -> bool:
    """
    Return True when a numeric target looks like encoded class labels
    rather than a regression target.
    """
    unique_values = sorted(pd.unique(non_null_target))
    unique_count = len(unique_values)
    unique_ratio = unique_count / len(non_null_target)

    if unique_count == 0 or unique_count > max_class_count:
        return False

    if any(not float(value).is_integer() for value in unique_values):
        return False

    integer_values = [int(value) for value in unique_values]

    if integer_values == [0, 1] or integer_values == [-1, 1]:
        return True

    if unique_count > 2 and unique_ratio > 0.5:
        return False

    consecutive_from_zero = integer_values == list(range(0, unique_count))
    consecutive_from_one = integer_values == list(range(1, unique_count + 1))

    return consecutive_from_zero or consecutive_from_one
