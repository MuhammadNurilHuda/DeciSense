from __future__ import annotations

import json

import pandas as pd
import pytest

from ds_engine.modeling.preprocess import (
    PreparedModelingDataset,
    PreprocessingError,
    build_sklearn_preprocessor,
    create_train_test_split,
    prepare_modeling_dataset,
)
from ds_engine.profiling.data_quality import create_data_quality_report
from ds_engine.profiling.schema_profile import create_schema_profile


def _build_schema_and_quality(dataframe: pd.DataFrame):
    schema_profile = create_schema_profile(
        dataframe,
        high_cardinality_min_unique_count=4,
        high_cardinality_unique_ratio_threshold=0.8,
        text_avg_length_threshold=30,
        text_unique_ratio_threshold=0.8,
    )
    data_quality_result = create_data_quality_report(
        dataframe,
        schema_profile=schema_profile,
    )
    return schema_profile, data_quality_result


def test_prepare_modeling_dataset_selects_safe_features_and_excludes_risky_features() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "city": ["A", "B", "A", "C"],
            "is_active": [True, False, True, True],
            "customer_id": ["C001", "C002", "C003", "C004"],
            "feedback": [
                "The onboarding experience was smooth and helpful.",
                "The support process was clear and fast.",
                "The service explanation was detailed and useful.",
                "The documentation was readable and practical.",
            ],
            "constant_feature": [1, 1, 1, 1],
            "empty_feature": [None, None, None, None],
            "target": [0, 1, 0, 1],
        }
    )
    schema_profile, data_quality_result = _build_schema_and_quality(dataframe)

    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
        schema_profile=schema_profile,
        data_quality_result=data_quality_result,
    )

    assert isinstance(prepared, PreparedModelingDataset)
    assert prepared.numeric_features == ["age"]
    assert prepared.categorical_features == ["city"]
    assert prepared.boolean_features == ["is_active"]
    assert prepared.included_features == ["age", "city", "is_active"]
    assert prepared.X.columns.tolist() == ["age", "city", "is_active"]
    assert prepared.y.tolist() == [0, 1, 0, 1]

    exclusions = {item.column_name: item.reason for item in prepared.excluded_features}
    assert exclusions["customer_id"] == "possible_id"
    assert exclusions["feedback"] == "text"
    assert exclusions["constant_feature"] == "constant"
    assert exclusions["empty_feature"] == "all_missing"


def test_prepare_modeling_dataset_drops_rows_with_missing_target() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, None, 1],
        }
    )

    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )

    assert prepared.row_count == 2
    assert prepared.dropped_target_missing_rows == 1
    assert prepared.X["feature"].tolist() == [1, 3]
    assert prepared.y.tolist() == [0, 1]


def test_prepare_modeling_dataset_raises_when_target_column_is_missing() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "label": [0, 1, 0],
        }
    )

    with pytest.raises(PreprocessingError, match="Target column not found"):
        prepare_modeling_dataset(
            dataframe,
            target_column="target",
            task_type="classification",
        )


def test_build_sklearn_preprocessor_can_fit_transform_prepared_features() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, None, 50],
            "city": ["A", "B", "A", None],
            "target": [0, 1, 0, 1],
        }
    )

    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )
    preprocessor = build_sklearn_preprocessor(prepared)

    transformed = preprocessor.fit_transform(prepared.X)

    assert transformed.shape[0] == prepared.row_count
    assert transformed.shape[1] >= 2


def test_create_train_test_split_stratifies_when_classification_target_supports_it() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "feature": list(range(10)),
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )
    split = create_train_test_split(
        prepared,
        test_size=0.4,
        random_state=42,
    )

    assert split.stratified is True
    assert len(split.X_train) == 6
    assert len(split.X_test) == 4
    assert set(split.y_train.unique()) == {0, 1}
    assert set(split.y_test.unique()) == {0, 1}


def test_create_train_test_split_falls_back_when_classification_target_cannot_be_stratified() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 0, 0, 1],
        }
    )

    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )
    split = create_train_test_split(
        prepared,
        test_size=0.5,
        random_state=42,
    )

    assert split.stratified is False
    assert len(split.X_train) == 2
    assert len(split.X_test) == 2


def test_prepared_modeling_dataset_to_dict_is_json_serializable() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [10.0, 20.0, 30.0],
        }
    )

    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="regression",
    )
    payload = prepared.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["target_column"] == "target"
    assert payload["task_type"] == "regression"
    assert payload["feature_count"] == 1
