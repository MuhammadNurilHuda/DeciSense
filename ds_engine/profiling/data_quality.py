from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from ds_engine.profiling.schema_profile import (
    SchemaProfileResult,
    create_schema_profile,
)

QualitySeverity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class DataQualityIssue:
    """A single dataset-level quality issue."""

    code: str
    severity: QualitySeverity
    message: str
    columns: list[str] = field(default_factory=list)
    metric_value: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "columns": list(self.columns),
            "metric_value": self.metric_value,
        }


@dataclass(frozen=True)
class DataQualityResult:
    """Dataset-level data quality summary."""

    row_count: int
    column_count: int
    duplicate_row_count: int
    duplicate_row_ratio: float
    missing_cell_count: int
    missing_cell_ratio: float
    rows_with_missing_count: int
    rows_with_missing_ratio: float
    fully_missing_row_count: int
    fully_missing_row_ratio: float
    columns_with_missing_values: list[str]
    fully_missing_columns: list[str]
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    possible_id_columns: list[str]
    issues: list[DataQualityIssue]

    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == "critical" for issue in self.issues)

    @property
    def is_usable_for_analysis(self) -> bool:
        return not self.has_critical_issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_row_ratio": self.duplicate_row_ratio,
            "missing_cell_count": self.missing_cell_count,
            "missing_cell_ratio": self.missing_cell_ratio,
            "rows_with_missing_count": self.rows_with_missing_count,
            "rows_with_missing_ratio": self.rows_with_missing_ratio,
            "fully_missing_row_count": self.fully_missing_row_count,
            "fully_missing_row_ratio": self.fully_missing_row_ratio,
            "columns_with_missing_values": list(self.columns_with_missing_values),
            "fully_missing_columns": list(self.fully_missing_columns),
            "constant_columns": list(self.constant_columns),
            "high_cardinality_columns": list(self.high_cardinality_columns),
            "possible_id_columns": list(self.possible_id_columns),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def create_data_quality_report(
    dataframe: pd.DataFrame,
    *,
    schema_profile: SchemaProfileResult | None = None,
    duplicate_row_warning_threshold: float = 0.01,
    missing_cell_warning_threshold: float = 0.05,
    missing_cell_critical_threshold: float = 0.50,
    possible_id_unique_ratio_threshold: float = 0.95,
) -> DataQualityResult:
    """
    Create dataset-level quality diagnostics for a tabular dataframe.

    This function does not mutate the dataframe. It only summarizes quality
    risks that downstream analysis, model planning, and reporting should know.
    """
    profile = schema_profile or create_schema_profile(dataframe)

    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    total_cell_count = row_count * column_count

    missing_mask = dataframe.isna()
    missing_cell_count = int(missing_mask.sum().sum())
    missing_cell_ratio = _safe_ratio(missing_cell_count, total_cell_count)

    rows_with_missing_count = (
        int(missing_mask.any(axis=1).sum()) if row_count > 0 and column_count > 0 else 0
    )
    rows_with_missing_ratio = _safe_ratio(rows_with_missing_count, row_count)

    fully_missing_row_count = (
        int(missing_mask.all(axis=1).sum()) if row_count > 0 and column_count > 0 else 0
    )
    fully_missing_row_ratio = _safe_ratio(fully_missing_row_count, row_count)

    columns_with_missing_values = (
        [str(column) for column in dataframe.columns[missing_mask.any(axis=0)].tolist()]
        if column_count > 0
        else []
    )

    duplicate_row_count = _count_duplicate_rows(dataframe)
    duplicate_row_ratio = _safe_ratio(duplicate_row_count, row_count)

    possible_id_columns = _detect_possible_id_columns(
        profile,
        unique_ratio_threshold=possible_id_unique_ratio_threshold,
    )

    issues = _build_quality_issues(
        duplicate_row_count=duplicate_row_count,
        duplicate_row_ratio=duplicate_row_ratio,
        duplicate_row_warning_threshold=duplicate_row_warning_threshold,
        missing_cell_ratio=missing_cell_ratio,
        missing_cell_warning_threshold=missing_cell_warning_threshold,
        missing_cell_critical_threshold=missing_cell_critical_threshold,
        fully_missing_row_count=fully_missing_row_count,
        fully_missing_row_ratio=fully_missing_row_ratio,
        columns_with_missing_values=columns_with_missing_values,
        fully_missing_columns=profile.all_missing_columns,
        constant_columns=profile.constant_columns,
        high_cardinality_columns=profile.high_cardinality_columns,
        possible_id_columns=possible_id_columns,
    )

    return DataQualityResult(
        row_count=row_count,
        column_count=column_count,
        duplicate_row_count=duplicate_row_count,
        duplicate_row_ratio=duplicate_row_ratio,
        missing_cell_count=missing_cell_count,
        missing_cell_ratio=missing_cell_ratio,
        rows_with_missing_count=rows_with_missing_count,
        rows_with_missing_ratio=rows_with_missing_ratio,
        fully_missing_row_count=fully_missing_row_count,
        fully_missing_row_ratio=fully_missing_row_ratio,
        columns_with_missing_values=columns_with_missing_values,
        fully_missing_columns=profile.all_missing_columns,
        constant_columns=profile.constant_columns,
        high_cardinality_columns=profile.high_cardinality_columns,
        possible_id_columns=possible_id_columns,
        issues=issues,
    )


def _build_quality_issues(
    *,
    duplicate_row_count: int,
    duplicate_row_ratio: float,
    duplicate_row_warning_threshold: float,
    missing_cell_ratio: float,
    missing_cell_warning_threshold: float,
    missing_cell_critical_threshold: float,
    fully_missing_row_count: int,
    fully_missing_row_ratio: float,
    columns_with_missing_values: list[str],
    fully_missing_columns: list[str],
    constant_columns: list[str],
    high_cardinality_columns: list[str],
    possible_id_columns: list[str],
) -> list[DataQualityIssue]:
    """Build human-readable quality issues from computed metrics."""
    issues: list[DataQualityIssue] = []

    if missing_cell_ratio >= missing_cell_critical_threshold:
        issues.append(
            DataQualityIssue(
                code="critical_missingness",
                severity="critical",
                message=(
                    "A large share of dataset cells are missing. "
                    "This may make analysis and modeling unreliable."
                ),
                columns=columns_with_missing_values,
                metric_value=missing_cell_ratio,
            )
        )
    elif missing_cell_ratio >= missing_cell_warning_threshold:
        issues.append(
            DataQualityIssue(
                code="missing_values",
                severity="warning",
                message=(
                    "The dataset contains missing values that should be handled "
                    "before modeling."
                ),
                columns=columns_with_missing_values,
                metric_value=missing_cell_ratio,
            )
        )

    if (
        duplicate_row_ratio >= duplicate_row_warning_threshold
        and duplicate_row_count > 0
    ):
        issues.append(
            DataQualityIssue(
                code="duplicate_rows",
                severity="warning",
                message=(
                    "The dataset contains duplicate rows. "
                    "These may bias analysis or model evaluation."
                ),
                metric_value=duplicate_row_ratio,
            )
        )

    if fully_missing_row_count > 0:
        issues.append(
            DataQualityIssue(
                code="fully_missing_rows",
                severity="warning",
                message="Some rows contain only missing values.",
                metric_value=fully_missing_row_ratio,
            )
        )

    if fully_missing_columns:
        issues.append(
            DataQualityIssue(
                code="fully_missing_columns",
                severity="warning",
                message="Some columns contain only missing values.",
                columns=fully_missing_columns,
            )
        )

    if constant_columns:
        issues.append(
            DataQualityIssue(
                code="constant_columns",
                severity="warning",
                message=(
                    "Some columns contain only one unique non-null value and "
                    "are unlikely to help modeling."
                ),
                columns=constant_columns,
            )
        )

    if high_cardinality_columns:
        issues.append(
            DataQualityIssue(
                code="high_cardinality_columns",
                severity="warning",
                message=(
                    "Some categorical/text columns have high cardinality and "
                    "may require careful encoding or exclusion."
                ),
                columns=high_cardinality_columns,
            )
        )

    if possible_id_columns:
        issues.append(
            DataQualityIssue(
                code="possible_id_columns",
                severity="warning",
                message=(
                    "Some columns look like identifiers. They may be useful for joins "
                    "or tracking, but should usually be excluded from modeling."
                ),
                columns=possible_id_columns,
            )
        )

    return issues


def _detect_possible_id_columns(
    schema_profile: SchemaProfileResult,
    *,
    unique_ratio_threshold: float,
) -> list[str]:
    """Detect columns that look like row identifiers."""
    possible_id_columns: list[str] = []

    for column in schema_profile.columns:
        normalized_name = column.column_name.strip().lower()
        name_looks_like_id = (
            normalized_name == "id"
            or normalized_name.endswith("_id")
            or normalized_name.endswith("id")
            or "identifier" in normalized_name
        )

        if (
            name_looks_like_id
            and not column.is_all_missing
            and column.unique_ratio >= unique_ratio_threshold
        ):
            possible_id_columns.append(column.column_name)

    return possible_id_columns


def _count_duplicate_rows(dataframe: pd.DataFrame) -> int:
    """Count duplicate rows safely."""
    if dataframe.empty:
        return 0

    try:
        return int(dataframe.duplicated().sum())
    except TypeError:
        return 0


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded ratio while protecting against division by zero."""
    if denominator == 0:
        return 0.0

    return round(float(numerator) / float(denominator), 6)
