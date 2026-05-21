from __future__ import annotations

import json

import pandas as pd

from ds_engine.planning.model_recommender import (
    ModelRecommendationResult,
    recommend_model_for_tabular_data,
)
from ds_engine.profiling.data_quality import create_data_quality_report
from ds_engine.profiling.schema_profile import create_schema_profile
from ds_engine.profiling.target_profile import create_target_profile


def _build_inputs(
    dataframe: pd.DataFrame,
    *,
    task_type: str,
):
    schema_profile = create_schema_profile(
        dataframe,
        high_cardinality_min_unique_count=4,
        high_cardinality_unique_ratio_threshold=0.8,
    )
    data_quality_result = create_data_quality_report(
        dataframe,
        schema_profile=schema_profile,
    )
    target_profile_result = create_target_profile(
        dataframe,
        target_column="target",
        task_type=task_type,  # type: ignore[arg-type]
    )
    return schema_profile, data_quality_result, target_profile_result


def _candidate_names(result: ModelRecommendationResult) -> list[str]:
    return [candidate.model_name for candidate in result.candidates]


def test_recommend_model_for_numeric_classification_prefers_hist_gradient_boosting() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "income": [1000, 2000, 3000, 4000, 5000, 6000],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    schema_profile, data_quality_result, target_profile_result = _build_inputs(
        dataframe,
        task_type="classification",
    )

    result = recommend_model_for_tabular_data(
        schema_profile=schema_profile,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
    )

    assert result.status == "ready"
    assert result.recommended_model == "HistGradientBoostingClassifier"
    assert result.is_ready_for_training_approval is True
    assert result.recommended_candidate is not None
    assert result.recommended_candidate.role == "recommended"
    assert "LogisticRegression" in _candidate_names(result)


def test_recommend_model_uses_catboost_when_many_categorical_features_and_optional_dependencies_allowed() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "city": ["A", "B", "C", "D", "E", "F"],
            "merchant_category": [
                "food",
                "retail",
                "food",
                "travel",
                "retail",
                "travel",
            ],
            "customer_id": ["C001", "C002", "C003", "C004", "C005", "C006"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    schema_profile, data_quality_result, target_profile_result = _build_inputs(
        dataframe,
        task_type="classification",
    )

    result = recommend_model_for_tabular_data(
        schema_profile=schema_profile,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
        allow_optional_dependencies=True,
    )

    assert result.status == "ready"
    assert result.recommended_model == "CatBoostClassifier"
    assert result.recommended_candidate is not None
    assert result.recommended_candidate.optional_dependency == "catboost"
    assert result.recommended_candidate.initial_params["loss_function"] == "Logloss"


def test_recommend_model_prefers_random_forest_for_imbalanced_classification() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": list(range(10)),
            "feature_b": [value * 2 for value in range(10)],
            "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )
    schema_profile, data_quality_result, target_profile_result = _build_inputs(
        dataframe,
        task_type="classification",
    )

    result = recommend_model_for_tabular_data(
        schema_profile=schema_profile,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
    )

    assert result.status == "ready"
    assert result.recommended_model == "RandomForestClassifier"
    assert result.recommended_candidate is not None
    assert result.recommended_candidate.initial_params["class_weight"] == "balanced"


def test_recommend_model_for_numeric_regression_prefers_hist_gradient_boosting() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [10, 20, 30, 40, 50, 60],
            "target": [100.0, 120.0, 135.0, 160.0, 190.0, 210.0],
        }
    )
    schema_profile, data_quality_result, target_profile_result = _build_inputs(
        dataframe,
        task_type="regression",
    )

    result = recommend_model_for_tabular_data(
        schema_profile=schema_profile,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
    )

    assert result.status == "ready"
    assert result.recommended_model == "HistGradientBoostingRegressor"
    assert "Ridge" in _candidate_names(result)
    assert result.confidence == "high"


def test_recommend_model_blocks_when_target_has_critical_issue() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "target": [1, 1, 1, 1],
        }
    )
    schema_profile, data_quality_result, target_profile_result = _build_inputs(
        dataframe,
        task_type="classification",
    )

    result = recommend_model_for_tabular_data(
        schema_profile=schema_profile,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
    )

    assert result.status == "blocked"
    assert result.recommended_model is None
    assert result.candidates == []
    assert result.is_ready_for_training_approval is False
    assert any("only one class" in reason for reason in result.blocked_reasons)


def test_model_recommendation_result_to_dict_is_json_serializable() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )
    schema_profile, data_quality_result, target_profile_result = _build_inputs(
        dataframe,
        task_type="classification",
    )

    result = recommend_model_for_tabular_data(
        schema_profile=schema_profile,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
    )
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["status"] == "ready"
    assert payload["recommended_model"] == "HistGradientBoostingClassifier"
