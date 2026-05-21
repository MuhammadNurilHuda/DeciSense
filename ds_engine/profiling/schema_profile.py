"""
Perlu di cek ulang, apakah sudah sesuai dengan kebutuhan atau belum.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

InferredColumnType = Literal[
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "empty",
    "unknown",
]


@dataclass(frozen=True)
class ColumnSchemaProfile:
    """Schema-level profile for a single dataframe column."""

    column_name: str
    dtype: str
    inferred_type: InferredColumnType
    non_null_count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    is_all_missing: bool
    is_high_cardinality: bool
    sample_values: list[Any]


@dataclass(frozen=True)
class SchemaProfileResult:
    """Schema-level profile for a dataframe."""

    row_count: int
    column_count: int
    columns: list[ColumnSchemaProfile]

    @property
    def numeric_columns(self) -> list[str]:
        return self._columns_by_type("numeric")

    @property
    def categorical_columns(self) -> list[str]:
        return self._columns_by_type("categorical")

    @property
    def boolean_columns(self) -> list[str]:
        return self._columns_by_type("boolean")

    @property
    def datetime_columns(self) -> list[str]:
        return self._columns_by_type("datetime")

    @property
    def text_columns(self) -> list[str]:
        return self._columns_by_type("text")

    @property
    def empty_columns(self) -> list[str]:
        return self._columns_by_type("empty")

    @property
    def constant_columns(self) -> list[str]:
        return [column.column_name for column in self.columns if column.is_constant]

    @property
    def all_missing_columns(self) -> list[str]:
        return [column.column_name for column in self.columns if column.is_all_missing]

    @property
    def high_cardinality_columns(self) -> list[str]:
        return [
            column.column_name for column in self.columns if column.is_high_cardinality
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert schema profile into a JSON-friendly dictionary."""
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "boolean_columns": self.boolean_columns,
            "datetime_columns": self.datetime_columns,
            "text_columns": self.text_columns,
            "empty_columns": self.empty_columns,
            "constant_columns": self.constant_columns,
            "all_missing_columns": self.all_missing_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
            "columns": [
                {
                    "column_name": column.column_name,
                    "dtype": column.dtype,
                    "inferred_type": column.inferred_type,
                    "non_null_count": column.non_null_count,
                    "missing_count": column.missing_count,
                    "missing_ratio": column.missing_ratio,
                    "unique_count": column.unique_count,
                    "unique_ratio": column.unique_ratio,
                    "is_constant": column.is_constant,
                    "is_all_missing": column.is_all_missing,
                    "is_high_cardinality": column.is_high_cardinality,
                    "sample_values": column.sample_values,
                }
                for column in self.columns
            ],
        }

    def _columns_by_type(self, inferred_type: InferredColumnType) -> list[str]:
        return [
            column.column_name
            for column in self.columns
            if column.inferred_type == inferred_type
        ]


def create_schema_profile(
    dataframe: pd.DataFrame,
    *,
    high_cardinality_min_unique_count: int = 50,
    high_cardinality_unique_ratio_threshold: float = 0.8,
    text_avg_length_threshold: int = 40,
    text_unique_ratio_threshold: float = 0.8,
    sample_value_count: int = 3,
) -> SchemaProfileResult:
    """
    Create a schema-level profile for a tabular dataframe.

    This function focuses on column structure, not full statistical EDA.
    """
    row_count = len(dataframe)
    column_profiles = [
        _profile_column(
            dataframe.iloc[:, column_index],
            column_name=str(column_name),
            row_count=row_count,
            high_cardinality_min_unique_count=high_cardinality_min_unique_count,
            high_cardinality_unique_ratio_threshold=(
                high_cardinality_unique_ratio_threshold
            ),
            text_avg_length_threshold=text_avg_length_threshold,
            text_unique_ratio_threshold=text_unique_ratio_threshold,
            sample_value_count=sample_value_count,
        )
        for column_index, column_name in enumerate(dataframe.columns)
    ]

    return SchemaProfileResult(
        row_count=row_count,
        column_count=len(dataframe.columns),
        columns=column_profiles,
    )


def _profile_column(
    series: pd.Series,
    *,
    column_name: str,
    row_count: int,
    high_cardinality_min_unique_count: int,
    high_cardinality_unique_ratio_threshold: float,
    text_avg_length_threshold: int,
    text_unique_ratio_threshold: float,
    sample_value_count: int,
) -> ColumnSchemaProfile:
    """Create a schema profile for one column."""
    non_null_series = series.dropna()

    non_null_count = int(len(non_null_series))
    missing_count = int(row_count - non_null_count)
    missing_ratio = round(missing_count / row_count, 6) if row_count > 0 else 0.0

    unique_count = int(non_null_series.nunique(dropna=True))
    unique_ratio = (
        round(unique_count / non_null_count, 6) if non_null_count > 0 else 0.0
    )

    inferred_type = _infer_column_type(
        series,
        non_null_series=non_null_series,
        text_avg_length_threshold=text_avg_length_threshold,
        text_unique_ratio_threshold=text_unique_ratio_threshold,
    )

    is_all_missing = non_null_count == 0
    is_constant = unique_count == 1 and non_null_count > 0
    is_high_cardinality = _is_high_cardinality(
        inferred_type=inferred_type,
        unique_count=unique_count,
        non_null_count=non_null_count,
        min_unique_count=high_cardinality_min_unique_count,
        unique_ratio_threshold=high_cardinality_unique_ratio_threshold,
    )

    return ColumnSchemaProfile(
        column_name=column_name,
        dtype=str(series.dtype),
        inferred_type=inferred_type,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_ratio=missing_ratio,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        is_constant=is_constant,
        is_all_missing=is_all_missing,
        is_high_cardinality=is_high_cardinality,
        sample_values=_build_sample_values(non_null_series, sample_value_count),
    )


def _infer_column_type(
    series: pd.Series,
    *,
    non_null_series: pd.Series,
    text_avg_length_threshold: int,
    text_unique_ratio_threshold: float,
) -> InferredColumnType:
    """Infer a practical column type for tabular analysis."""
    if non_null_series.empty:
        return "empty"

    if is_bool_dtype(series) or _contains_only_boolean_values(non_null_series):
        return "boolean"

    if is_datetime64_any_dtype(series):
        return "datetime"

    if is_numeric_dtype(series):
        return "numeric"

    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    if is_string_dtype(series) or series.dtype == "object":
        average_length = non_null_series.astype(str).str.len().mean()
        unique_ratio = non_null_series.nunique(dropna=True) / len(non_null_series)

        if (
            average_length >= text_avg_length_threshold
            and unique_ratio >= text_unique_ratio_threshold
        ):
            return "text"

        return "categorical"

    return "unknown"


def _contains_only_boolean_values(non_null_series: pd.Series) -> bool:
    """Return True when an object-like series contains only boolean values."""
    return bool(non_null_series.map(lambda value: isinstance(value, bool)).all())


def _is_high_cardinality(
    *,
    inferred_type: InferredColumnType,
    unique_count: int,
    non_null_count: int,
    min_unique_count: int,
    unique_ratio_threshold: float,
) -> bool:
    """Return True when a categorical/text column has many distinct values."""
    if inferred_type not in {"categorical", "text"}:
        return False

    if non_null_count == 0:
        return False

    unique_ratio = unique_count / non_null_count

    return unique_count >= min_unique_count and unique_ratio >= unique_ratio_threshold


def _build_sample_values(
    non_null_series: pd.Series,
    sample_value_count: int,
) -> list[Any]:
    """Return JSON-friendly unique sample values from a column."""
    if sample_value_count <= 0 or non_null_series.empty:
        return []

    unique_values = pd.unique(non_null_series)
    return [_to_json_safe_value(value) for value in unique_values[:sample_value_count]]


def _to_json_safe_value(value: Any) -> Any:
    """Convert common pandas/numpy scalar values into JSON-friendly values."""
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
