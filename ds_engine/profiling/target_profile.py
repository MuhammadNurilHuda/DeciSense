"""menganalisis target column setelah infer_task.py berhasil menentukan candidate_target dan task_type"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd

from ds_engine.intake.infer_task import TaskType


TargetIssueSeverity = Literal["info", "warning", "critical"]


class TargetProfileError(Exception):
    """Raised when a target profile cannot be created."""


@dataclass(frozen=True)
class TargetProfileIssue:
    """A target-specific data issue."""

    code: str
    severity: TargetIssueSeverity
    message: str
    metric_value: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "metric_value": self.metric_value,
        }


@dataclass(frozen=True)
class ClassDistributionItem:
    """Distribution summary for one classification target class."""

    label: Any
    count: int
    proportion: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": _to_json_safe_value(self.label),
            "count": self.count,
            "proportion": self.proportion,
        }


@dataclass(frozen=True)
class NumericTargetSummary:
    """Numeric summary for a regression target."""

    count: int
    mean: float | None
    std: float | None
    minimum: float | None
    q25: float | None
    median: float | None
    q75: float | None
    maximum: float | None
    skewness: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "q25": self.q25,
            "median": self.median,
            "q75": self.q75,
            "maximum": self.maximum,
            "skewness": self.skewness,
        }


@dataclass(frozen=True)
class TargetProfileResult:
    """Target-column profile for classification or regression tasks."""

    target_column: str
    task_type: TaskType
    row_count: int
    non_null_count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    class_distribution: list[ClassDistributionItem] = field(default_factory=list)
    majority_class: Any | None = None
    majority_class_ratio: float | None = None
    minority_class: Any | None = None
    minority_class_ratio: float | None = None
    class_imbalance_ratio: float | None = None
    numeric_summary: NumericTargetSummary | None = None
    issues: list[TargetProfileIssue] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == "critical" for issue in self.issues)

    @property
    def is_usable_for_modeling(self) -> bool:
        return not self.has_critical_issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "task_type": self.task_type,
            "row_count": self.row_count,
            "non_null_count": self.non_null_count,
            "missing_count": self.missing_count,
            "missing_ratio": self.missing_ratio,
            "unique_count": self.unique_count,
            "class_distribution": [
                item.to_dict() for item in self.class_distribution
            ],
            "majority_class": _to_json_safe_value(self.majority_class),
            "majority_class_ratio": self.majority_class_ratio,
            "minority_class": _to_json_safe_value(self.minority_class),
            "minority_class_ratio": self.minority_class_ratio,
            "class_imbalance_ratio": self.class_imbalance_ratio,
            "numeric_summary": (
                self.numeric_summary.to_dict()
                if self.numeric_summary is not None
                else None
            ),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def create_target_profile(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    task_type: TaskType,
    target_missing_warning_threshold: float = 0.0,
    classification_imbalance_ratio_threshold: float = 5.0,
    classification_minority_ratio_threshold: float = 0.10,
    classification_high_cardinality_threshold: int = 50,
) -> TargetProfileResult:
    """
    Create a target-specific profile for downstream analysis and model planning.
    """
    if target_column not in dataframe.columns:
        raise TargetProfileError(f"Target column not found: {target_column}")

    target_series = dataframe[target_column]
    row_count = len(dataframe)
    non_null_target = target_series.dropna()

    non_null_count = int(len(non_null_target))
    missing_count = int(row_count - non_null_count)
    missing_ratio = _safe_ratio(missing_count, row_count)
    unique_count = int(non_null_target.nunique(dropna=True))

    issues: list[TargetProfileIssue] = []

    if missing_ratio > target_missing_warning_threshold:
        issues.append(
            TargetProfileIssue(
                code="target_missing_values",
                severity="warning",
                message="The target column contains missing values.",
                metric_value=missing_ratio,
            )
        )

    if non_null_target.empty:
        issues.append(
            TargetProfileIssue(
                code="target_all_missing",
                severity="critical",
                message="The target column contains only missing values.",
            )
        )
        return TargetProfileResult(
            target_column=target_column,
            task_type=task_type,
            row_count=row_count,
            non_null_count=non_null_count,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            unique_count=unique_count,
            issues=issues,
        )

    if task_type == "classification":
        return _create_classification_target_profile(
            target_column=target_column,
            task_type=task_type,
            row_count=row_count,
            non_null_count=non_null_count,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            unique_count=unique_count,
            non_null_target=non_null_target,
            issues=issues,
            imbalance_ratio_threshold=classification_imbalance_ratio_threshold,
            minority_ratio_threshold=classification_minority_ratio_threshold,
            high_cardinality_threshold=classification_high_cardinality_threshold,
        )

    return _create_regression_target_profile(
        target_column=target_column,
        task_type=task_type,
        row_count=row_count,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_ratio=missing_ratio,
        unique_count=unique_count,
        non_null_target=non_null_target,
        issues=issues,
    )


def _create_classification_target_profile(
    *,
    target_column: str,
    task_type: TaskType,
    row_count: int,
    non_null_count: int,
    missing_count: int,
    missing_ratio: float,
    unique_count: int,
    non_null_target: pd.Series,
    issues: list[TargetProfileIssue],
    imbalance_ratio_threshold: float,
    minority_ratio_threshold: float,
    high_cardinality_threshold: int,
) -> TargetProfileResult:
    """Create a profile for a classification target."""
    class_distribution = _build_class_distribution(non_null_target)

    majority_item = class_distribution[0] if class_distribution else None
    minority_item = class_distribution[-1] if class_distribution else None

    majority_class = majority_item.label if majority_item is not None else None
    majority_class_ratio = (
        majority_item.proportion if majority_item is not None else None
    )
    minority_class = minority_item.label if minority_item is not None else None
    minority_class_ratio = (
        minority_item.proportion if minority_item is not None else None
    )

    class_imbalance_ratio = None
    if majority_item is not None and minority_item is not None and len(class_distribution) > 1:
        class_imbalance_ratio = round(
            majority_item.count / max(minority_item.count, 1),
            6,
        )

    if unique_count == 1:
        issues.append(
            TargetProfileIssue(
                code="single_class_target",
                severity="critical",
                message=(
                    "The classification target contains only one class. "
                    "A supervised classification model cannot learn class separation."
                ),
            )
        )

    if unique_count > high_cardinality_threshold:
        issues.append(
            TargetProfileIssue(
                code="high_cardinality_classification_target",
                severity="warning",
                message=(
                    "The classification target has many distinct classes. "
                    "This may require special evaluation and modeling choices."
                ),
                metric_value=unique_count,
            )
        )

    if (
        class_imbalance_ratio is not None
        and minority_class_ratio is not None
        and (
            class_imbalance_ratio >= imbalance_ratio_threshold
            or minority_class_ratio <= minority_ratio_threshold
        )
    ):
        issues.append(
            TargetProfileIssue(
                code="class_imbalance",
                severity="warning",
                message=(
                    "The classification target is imbalanced. "
                    "Use stratified splits and imbalance-aware metrics."
                ),
                metric_value=class_imbalance_ratio,
            )
        )

    return TargetProfileResult(
        target_column=target_column,
        task_type=task_type,
        row_count=row_count,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_ratio=missing_ratio,
        unique_count=unique_count,
        class_distribution=class_distribution,
        majority_class=majority_class,
        majority_class_ratio=majority_class_ratio,
        minority_class=minority_class,
        minority_class_ratio=minority_class_ratio,
        class_imbalance_ratio=class_imbalance_ratio,
        issues=issues,
    )


def _create_regression_target_profile(
    *,
    target_column: str,
    task_type: TaskType,
    row_count: int,
    non_null_count: int,
    missing_count: int,
    missing_ratio: float,
    unique_count: int,
    non_null_target: pd.Series,
    issues: list[TargetProfileIssue],
) -> TargetProfileResult:
    """Create a profile for a regression target."""
    numeric_target = pd.to_numeric(non_null_target, errors="coerce")
    invalid_numeric_count = int(numeric_target.isna().sum())
    numeric_target = numeric_target.dropna()

    if invalid_numeric_count > 0:
        issues.append(
            TargetProfileIssue(
                code="non_numeric_regression_target",
                severity="critical",
                message=(
                    "The regression target contains non-numeric values. "
                    "Regression requires a numeric target."
                ),
                metric_value=invalid_numeric_count,
            )
        )

    if numeric_target.empty:
        issues.append(
            TargetProfileIssue(
                code="empty_numeric_regression_target",
                severity="critical",
                message="No valid numeric values were found in the regression target.",
            )
        )
        numeric_summary = None
    else:
        numeric_summary = _summarize_numeric_target(numeric_target)

        if numeric_target.nunique(dropna=True) == 1:
            issues.append(
                TargetProfileIssue(
                    code="constant_regression_target",
                    severity="critical",
                    message=(
                        "The regression target contains only one unique value. "
                        "A supervised regression model cannot learn meaningful variation."
                    ),
                )
            )

    return TargetProfileResult(
        target_column=target_column,
        task_type=task_type,
        row_count=row_count,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_ratio=missing_ratio,
        unique_count=unique_count,
        numeric_summary=numeric_summary,
        issues=issues,
    )


def _build_class_distribution(non_null_target: pd.Series) -> list[ClassDistributionItem]:
    """Build class distribution sorted by descending class frequency."""
    total_count = len(non_null_target)
    value_counts = non_null_target.value_counts(dropna=True)

    return [
        ClassDistributionItem(
            label=_to_json_safe_value(label),
            count=int(count),
            proportion=_safe_ratio(int(count), total_count),
        )
        for label, count in value_counts.items()
    ]


def _summarize_numeric_target(numeric_target: pd.Series) -> NumericTargetSummary:
    """Create descriptive statistics for a numeric regression target."""
    return NumericTargetSummary(
        count=int(len(numeric_target)),
        mean=_to_optional_float(numeric_target.mean()),
        std=_to_optional_float(numeric_target.std()),
        minimum=_to_optional_float(numeric_target.min()),
        q25=_to_optional_float(numeric_target.quantile(0.25)),
        median=_to_optional_float(numeric_target.median()),
        q75=_to_optional_float(numeric_target.quantile(0.75)),
        maximum=_to_optional_float(numeric_target.max()),
        skewness=_to_optional_float(numeric_target.skew()),
    )


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded ratio while protecting against division by zero."""
    if denominator == 0:
        return 0.0

    return round(float(numerator) / float(denominator), 6)


def _to_optional_float(value: Any) -> float | None:
    """Convert numeric values to rounded floats while preserving null-like values."""
    if pd.isna(value):
        return None

    return round(float(value), 6)


def _to_json_safe_value(value: Any) -> Any:
    """Convert common pandas/numpy scalar values into JSON-friendly values."""
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            extracted_value = value.item()
            if extracted_value is not value:
                return _to_json_safe_value(extracted_value)
        except (AttributeError, ValueError, TypeError):
            pass

    return value