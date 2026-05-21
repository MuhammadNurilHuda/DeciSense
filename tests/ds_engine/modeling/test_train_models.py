from __future__ import annotations

import json

import pandas as pd

from ds_engine.modeling.preprocess import prepare_modeling_dataset
from ds_engine.modeling.train_models import (
    ModelTrainingResult,
    train_model_candidates,
)
from ds_engine.planning.model_recommender import (
    ModelCandidate,
    ModelRecommendationResult,
)


def _ready_recommendation(
    *,
    task_type: str,
    candidates: list[ModelCandidate],
) -> ModelRecommendationResult:
    recommended_model = next(
        candidate.model_name
        for candidate in candidates
        if candidate.role == "recommended"
    )

    return ModelRecommendationResult(
        status="ready",
        task_type=task_type,  # type: ignore[arg-type]
        recommended_model=recommended_model,
        confidence="medium",
        candidates=candidates,
        reasoning=["Test recommendation."],
        warnings=[],
        blocked_reasons=[],
        dataset_notes={},
    )


def test_train_model_candidates_trains_classification_candidates_and_selects_best() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "feature_a": list(range(20)),
            "feature_b": [value % 3 for value in range(20)],
            "target": [0] * 10 + [1] * 10,
        }
    )
    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )
    recommendation = _ready_recommendation(
        task_type="classification",
        candidates=[
            ModelCandidate(
                model_name="LogisticRegression",
                task_type="classification",
                role="recommended",
                initial_params={
                    "max_iter": 1000,
                    "solver": "lbfgs",
                    "class_weight": None,
                    "random_state": 42,
                },
                preprocessing_notes=[],
                reasoning=[],
            ),
            ModelCandidate(
                model_name="RandomForestClassifier",
                task_type="classification",
                role="challenger",
                initial_params={
                    "n_estimators": 50,
                    "random_state": 42,
                    "n_jobs": -1,
                },
                preprocessing_notes=[],
                reasoning=[],
            ),
        ],
    )

    result = train_model_candidates(
        prepared_dataset=prepared,
        model_recommendation_result=recommendation,
        test_size=0.3,
        random_state=42,
    )

    assert isinstance(result, ModelTrainingResult)
    assert result.status == "completed"
    assert len(result.experiments) == 2
    assert all(experiment.status == "success" for experiment in result.experiments)
    assert result.best_experiment is not None
    assert result.best_experiment.primary_metric_name == "f1_macro"
    assert result.best_experiment.primary_metric_value is not None
    assert result.train_row_count == 14
    assert result.test_row_count == 6
    assert result.feature_count == 2
    assert result.stratified_split is True


def test_train_model_candidates_trains_regression_candidate() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": list(range(20)),
            "feature_b": [value * 2 for value in range(20)],
            "target": [float(value * 3 + 5) for value in range(20)],
        }
    )
    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="regression",
    )
    recommendation = _ready_recommendation(
        task_type="regression",
        candidates=[
            ModelCandidate(
                model_name="Ridge",
                task_type="regression",
                role="recommended",
                initial_params={
                    "alpha": 1.0,
                    "random_state": 42,
                },
                preprocessing_notes=[],
                reasoning=[],
            ),
        ],
    )

    result = train_model_candidates(
        prepared_dataset=prepared,
        model_recommendation_result=recommendation,
        test_size=0.25,
        random_state=42,
    )

    assert result.status == "completed"
    assert result.best_experiment is not None
    assert result.best_experiment.model_name == "Ridge"
    assert result.best_experiment.primary_metric_name == "rmse"
    assert result.best_experiment.primary_metric_value is not None
    assert "mae" in result.best_experiment.test_metrics
    assert "r2" in result.best_experiment.test_metrics
    assert result.train_row_count == 15
    assert result.test_row_count == 5


def test_train_model_candidates_returns_failed_when_recommendation_is_blocked() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [1, 1, 1, 1],
        }
    )
    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )
    recommendation = ModelRecommendationResult(
        status="blocked",
        task_type="classification",
        recommended_model=None,
        confidence="low",
        candidates=[],
        reasoning=[],
        warnings=[],
        blocked_reasons=["The classification target contains only one class."],
        dataset_notes={},
    )

    result = train_model_candidates(
        prepared_dataset=prepared,
        model_recommendation_result=recommendation,
    )

    assert result.status == "failed"
    assert result.experiments == []
    assert result.best_experiment is None
    assert "Model recommendation is not ready for training approval." in result.errors
    assert "The classification target contains only one class." in result.errors


def test_train_model_candidates_skips_candidate_with_missing_optional_dependency() -> (
    None
):
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )
    recommendation = _ready_recommendation(
        task_type="classification",
        candidates=[
            ModelCandidate(
                model_name="CatBoostClassifier",
                task_type="classification",
                role="recommended",
                optional_dependency="decisense_missing_dependency",
                initial_params={},
                preprocessing_notes=[],
                reasoning=[],
            ),
        ],
    )

    result = train_model_candidates(
        prepared_dataset=prepared,
        model_recommendation_result=recommendation,
    )

    assert result.status == "failed"
    assert len(result.experiments) == 1
    assert result.experiments[0].status == "skipped"
    assert result.best_experiment is None
    assert "optional dependency" in result.warnings[0]


def test_model_training_result_to_dict_is_json_serializable() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": list(range(12)),
            "target": [0, 1] * 6,
        }
    )
    prepared = prepare_modeling_dataset(
        dataframe,
        target_column="target",
        task_type="classification",
    )
    recommendation = _ready_recommendation(
        task_type="classification",
        candidates=[
            ModelCandidate(
                model_name="LogisticRegression",
                task_type="classification",
                role="recommended",
                initial_params={
                    "max_iter": 1000,
                    "solver": "lbfgs",
                    "random_state": 42,
                },
                preprocessing_notes=[],
                reasoning=[],
            ),
        ],
    )

    result = train_model_candidates(
        prepared_dataset=prepared,
        model_recommendation_result=recommendation,
        test_size=0.25,
        random_state=42,
    )
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["status"] == "completed"
    assert payload["best_experiment"]["model_name"] == "LogisticRegression"
    assert payload["best_experiment"]["primary_metric_name"] == "f1_macro"
