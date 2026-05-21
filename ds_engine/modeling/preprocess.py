from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ds_engine.intake.infer_task import TaskType
from ds_engine.profiling.data_quality import DataQualityResult
from ds_engine.profiling.schema_profile import (
    SchemaProfileResult,
    create_schema_profile,
)

FeatureExclusionReason = Literal[
    "target_column",
    "all_missing",
    "constant",
    "possible_id",
    "text",
    "datetime",
    "high_cardinality",
    "unknown_type",
]


class PreprocessingError(Exception):
    """Raised when supervised modeling data cannot be prepared."""


@dataclass(frozen=True)
class FeatureExclusion:
    """Reason why a feature was excluded from modeling."""

    column_name: str
    reason: FeatureExclusionReason

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PreparedModelingDataset:
    """Prepared supervised dataset metadata and feature selection result."""

    X: pd.DataFrame
    y: pd.Series
    target_column: str
    task_type: TaskType
    numeric_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]
    excluded_features: list[FeatureExclusion] = field(default_factory=list)
    dropped_target_missing_rows: int = 0

    @property
    def included_features(self) -> list[str]:
        return [
            *self.numeric_features,
            *self.categorical_features,
            *self.boolean_features,
        ]

    @property
    def row_count(self) -> int:
        return len(self.X)

    @property
    def feature_count(self) -> int:
        return len(self.included_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "task_type": self.task_type,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "boolean_features": list(self.boolean_features),
            "included_features": list(self.included_features),
            "excluded_features": [
                exclusion.to_dict() for exclusion in self.excluded_features
            ],
            "dropped_target_missing_rows": self.dropped_target_missing_rows,
        }


@dataclass(frozen=True)
class TrainTestSplitResult:
    """Train/test split output for supervised modeling."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    stratified: bool
    test_size: float | int
    random_state: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_row_count": len(self.X_train),
            "test_row_count": len(self.X_test),
            "feature_count": len(self.X_train.columns),
            "stratified": self.stratified,
            "test_size": self.test_size,
            "random_state": self.random_state,
        }


def prepare_modeling_dataset(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    task_type: TaskType,
    schema_profile: SchemaProfileResult | None = None,
    data_quality_result: DataQualityResult | None = None,
    drop_missing_target_rows: bool = True,
    exclude_possible_id_columns: bool = True,
    exclude_text_columns: bool = True,
    exclude_datetime_columns: bool = True,
    exclude_high_cardinality_columns: bool = True,
    exclude_constant_columns: bool = True,
    exclude_all_missing_columns: bool = True,
) -> PreparedModelingDataset:
    """
    Prepare a dataframe for supervised tabular modeling.

    This function selects safe modeling features, drops rows with missing target
    values when requested, and returns X/y plus feature metadata.
    """
    if target_column not in dataframe.columns:
        raise PreprocessingError(f"Target column not found: {target_column}")

    if dataframe.empty:
        raise PreprocessingError(
            "Cannot prepare modeling data from an empty dataframe."
        )

    working_dataframe = dataframe.copy()
    missing_target_mask = working_dataframe[target_column].isna()
    dropped_target_missing_rows = int(missing_target_mask.sum())

    if dropped_target_missing_rows > 0:
        if not drop_missing_target_rows:
            raise PreprocessingError(
                "Target column contains missing values and drop_missing_target_rows=False."
            )

        working_dataframe = working_dataframe.loc[~missing_target_mask].copy()

    if working_dataframe.empty:
        raise PreprocessingError(
            "No rows remain after dropping rows with missing target values."
        )

    profile = schema_profile or create_schema_profile(working_dataframe)
    possible_id_columns = _get_possible_id_columns(
        schema_profile=profile,
        data_quality_result=data_quality_result,
    )

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    boolean_features: list[str] = []
    exclusions: dict[str, FeatureExclusionReason] = {
        target_column: "target_column",
    }

    for column_profile in profile.columns:
        column_name = column_profile.column_name

        if column_name == target_column:
            continue

        if column_name not in working_dataframe.columns:
            continue

        exclusion_reason = _get_feature_exclusion_reason(
            column_name=column_name,
            inferred_type=column_profile.inferred_type,
            is_all_missing=column_profile.is_all_missing,
            is_constant=column_profile.is_constant,
            is_high_cardinality=column_profile.is_high_cardinality,
            possible_id_columns=possible_id_columns,
            exclude_possible_id_columns=exclude_possible_id_columns,
            exclude_text_columns=exclude_text_columns,
            exclude_datetime_columns=exclude_datetime_columns,
            exclude_high_cardinality_columns=exclude_high_cardinality_columns,
            exclude_constant_columns=exclude_constant_columns,
            exclude_all_missing_columns=exclude_all_missing_columns,
        )

        if exclusion_reason is not None:
            exclusions[column_name] = exclusion_reason
            continue

        if column_profile.inferred_type == "numeric":
            numeric_features.append(column_name)
        elif column_profile.inferred_type == "categorical":
            categorical_features.append(column_name)
        elif column_profile.inferred_type == "boolean":
            boolean_features.append(column_name)
        else:
            exclusions[column_name] = "unknown_type"

    included_features = [
        *numeric_features,
        *categorical_features,
        *boolean_features,
    ]

    if not included_features:
        raise PreprocessingError("No usable modeling features remain after filtering.")

    X = working_dataframe[included_features].copy()
    y = _prepare_target_series(
        working_dataframe[target_column],
        task_type=task_type,
        target_column=target_column,
    )

    return PreparedModelingDataset(
        X=X,
        y=y,
        target_column=target_column,
        task_type=task_type,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
        excluded_features=[
            FeatureExclusion(column_name=column_name, reason=reason)
            for column_name, reason in exclusions.items()
            if column_name != target_column
        ],
        dropped_target_missing_rows=dropped_target_missing_rows,
    )


def build_sklearn_preprocessor(
    prepared_dataset: PreparedModelingDataset,
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Build a sklearn preprocessing transformer for the prepared dataset.

    Numeric features use median imputation and optional scaling.
    Categorical/boolean features use most-frequent imputation and one-hot encoding.
    """
    transformers: list[tuple[str, SklearnPipeline, list[str]]] = []

    if prepared_dataset.numeric_features:
        numeric_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]

        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        transformers.append(
            (
                "numeric",
                SklearnPipeline(numeric_steps),
                prepared_dataset.numeric_features,
            )
        )

    categorical_like_features = [
        *prepared_dataset.categorical_features,
        *prepared_dataset.boolean_features,
    ]

    if categorical_like_features:
        transformers.append(
            (
                "categorical",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _build_one_hot_encoder()),
                    ]
                ),
                categorical_like_features,
            )
        )

    if not transformers:
        raise PreprocessingError("No preprocessing transformers could be created.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def create_train_test_split(
    prepared_dataset: PreparedModelingDataset,
    *,
    test_size: float | int = 0.2,
    random_state: int = 42,
    stratify_classification: bool = True,
) -> TrainTestSplitResult:
    """Create a train/test split for supervised modeling."""
    if prepared_dataset.row_count < 2:
        raise PreprocessingError("At least 2 rows are required to create a split.")

    stratify = None
    stratified = False

    if (
        prepared_dataset.task_type == "classification"
        and stratify_classification
        and _can_stratify_target(prepared_dataset.y, test_size=test_size)
    ):
        stratify = prepared_dataset.y
        stratified = True

    X_train, X_test, y_train, y_test = train_test_split(
        prepared_dataset.X,
        prepared_dataset.y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    return TrainTestSplitResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        stratified=stratified,
        test_size=test_size,
        random_state=random_state,
    )


def _get_feature_exclusion_reason(
    *,
    column_name: str,
    inferred_type: str,
    is_all_missing: bool,
    is_constant: bool,
    is_high_cardinality: bool,
    possible_id_columns: set[str],
    exclude_possible_id_columns: bool,
    exclude_text_columns: bool,
    exclude_datetime_columns: bool,
    exclude_high_cardinality_columns: bool,
    exclude_constant_columns: bool,
    exclude_all_missing_columns: bool,
) -> FeatureExclusionReason | None:
    """Return feature exclusion reason when a column should not be modeled."""
    if exclude_all_missing_columns and is_all_missing:
        return "all_missing"

    if exclude_constant_columns and is_constant:
        return "constant"

    if exclude_possible_id_columns and column_name in possible_id_columns:
        return "possible_id"

    if exclude_text_columns and inferred_type == "text":
        return "text"

    if exclude_datetime_columns and inferred_type == "datetime":
        return "datetime"

    if exclude_high_cardinality_columns and is_high_cardinality:
        return "high_cardinality"

    if inferred_type in {"empty", "unknown"}:
        return "unknown_type"

    return None


def _prepare_target_series(
    target_series: pd.Series,
    *,
    task_type: TaskType,
    target_column: str,
) -> pd.Series:
    """Prepare target values for modeling."""
    prepared_target = target_series.copy()

    if task_type == "regression":
        prepared_target = pd.to_numeric(prepared_target, errors="coerce")
        if prepared_target.isna().any():
            raise PreprocessingError(
                f"Regression target '{target_column}' contains non-numeric values."
            )

    return prepared_target


def _get_possible_id_columns(
    *,
    schema_profile: SchemaProfileResult,
    data_quality_result: DataQualityResult | None,
) -> set[str]:
    """Return possible ID columns from data quality result or schema heuristics."""
    if data_quality_result is not None:
        return set(data_quality_result.possible_id_columns)

    possible_id_columns: set[str] = set()

    for column in schema_profile.columns:
        normalized_name = column.column_name.strip().lower()
        name_looks_like_id = (
            normalized_name == "id"
            or normalized_name.endswith("_id")
            or normalized_name.endswith("id")
            or "identifier" in normalized_name
        )

        if name_looks_like_id and column.unique_ratio >= 0.95:
            possible_id_columns.add(column.column_name)

    return possible_id_columns


def _build_one_hot_encoder() -> OneHotEncoder:
    """Build OneHotEncoder with compatibility across sklearn versions."""
    parameters: dict[str, Any] = {
        "handle_unknown": "ignore",
    }

    signature = inspect.signature(OneHotEncoder)
    if "sparse_output" in signature.parameters:
        parameters["sparse_output"] = False
    else:
        parameters["sparse"] = False

    return OneHotEncoder(**parameters)


def _can_stratify_target(
    target: pd.Series,
    *,
    test_size: float | int,
) -> bool:
    """Return True when stratified train/test split is safe."""
    class_counts = target.value_counts(dropna=False)

    if len(class_counts) < 2:
        return False

    if int(class_counts.min()) < 2:
        return False

    row_count = len(target)

    if isinstance(test_size, float):
        test_row_count = ceil(row_count * test_size)
    else:
        test_row_count = int(test_size)

    train_row_count = row_count - test_row_count
    class_count = len(class_counts)

    return test_row_count >= class_count and train_row_count >= class_count
