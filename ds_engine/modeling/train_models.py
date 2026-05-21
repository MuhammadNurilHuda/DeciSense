from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from importlib.util import find_spec
from math import sqrt
from typing import Any, Literal

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline

from ds_engine.intake.infer_task import TaskType
from ds_engine.modeling.preprocess import (
    PreparedModelingDataset,
    PreprocessingError,
    build_sklearn_preprocessor,
    create_train_test_split,
)
from ds_engine.planning.model_recommender import (
    CandidateRole,
    ModelCandidate,
    ModelRecommendationResult,
)

ExperimentStatus = Literal["success", "failed", "skipped"]
TrainingRunStatus = Literal["completed", "completed_with_failures", "failed"]
OverfittingRisk = Literal["low", "medium", "high", "unknown"]


class ModelTrainingError(Exception):
    """Raised when a model training operation cannot be completed."""


@dataclass(frozen=True)
class ModelExperimentResult:
    """Result for one trained model candidate."""

    model_name: str
    role: CandidateRole
    status: ExperimentStatus
    task_type: TaskType
    initial_params: dict[str, Any]
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    higher_is_better: bool = True
    generalization_gap: float | None = None
    overfitting_risk: OverfittingRisk = "unknown"
    fitted_pipeline: SklearnPipeline | None = field(
        default=None, repr=False, compare=False
    )
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "role": self.role,
            "status": self.status,
            "task_type": self.task_type,
            "initial_params": dict(self.initial_params),
            "train_metrics": dict(self.train_metrics),
            "test_metrics": dict(self.test_metrics),
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": self.primary_metric_value,
            "higher_is_better": self.higher_is_better,
            "generalization_gap": self.generalization_gap,
            "overfitting_risk": self.overfitting_risk,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ModelTrainingResult:
    """Training result across all selected model candidates."""

    status: TrainingRunStatus
    task_type: TaskType
    experiments: list[ModelExperimentResult]
    best_experiment: ModelExperimentResult | None = None
    train_row_count: int = 0
    test_row_count: int = 0
    feature_count: int = 0
    stratified_split: bool = False
    random_state: int = 42
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def successful_experiments(self) -> list[ModelExperimentResult]:
        return [experiment for experiment in self.experiments if experiment.is_success]

    @property
    def is_success(self) -> bool:
        return self.status in {"completed", "completed_with_failures"} and bool(
            self.successful_experiments
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_type": self.task_type,
            "train_row_count": self.train_row_count,
            "test_row_count": self.test_row_count,
            "feature_count": self.feature_count,
            "stratified_split": self.stratified_split,
            "random_state": self.random_state,
            "best_experiment": (
                self.best_experiment.to_dict()
                if self.best_experiment is not None
                else None
            ),
            "experiments": [experiment.to_dict() for experiment in self.experiments],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def train_model_candidates(
    *,
    prepared_dataset: PreparedModelingDataset,
    model_recommendation_result: ModelRecommendationResult,
    test_size: float | int = 0.2,
    random_state: int = 42,
    max_candidates: int | None = None,
) -> ModelTrainingResult:
    """
    Train recommended model candidates and select the best experiment.

    Candidate-level failures are captured as failed/skipped experiments so one
    broken model does not crash the full training run.
    """
    task_type = prepared_dataset.task_type

    if not model_recommendation_result.is_ready_for_training_approval:
        return ModelTrainingResult(
            status="failed",
            task_type=task_type,
            experiments=[],
            errors=[
                "Model recommendation is not ready for training approval.",
                *model_recommendation_result.blocked_reasons,
            ],
        )

    candidates = model_recommendation_result.candidates
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    if not candidates:
        return ModelTrainingResult(
            status="failed",
            task_type=task_type,
            experiments=[],
            errors=["No model candidates were provided for training."],
        )

    try:
        split = create_train_test_split(
            prepared_dataset,
            test_size=test_size,
            random_state=random_state,
        )
    except PreprocessingError as exc:
        return ModelTrainingResult(
            status="failed",
            task_type=task_type,
            experiments=[],
            errors=[f"Failed to create train/test split: {exc}"],
        )

    experiments = [
        _train_single_candidate(
            candidate=candidate,
            prepared_dataset=prepared_dataset,
            X_train=split.X_train,
            X_test=split.X_test,
            y_train=split.y_train,
            y_test=split.y_test,
        )
        for candidate in candidates
    ]

    successful_experiments = [
        experiment for experiment in experiments if experiment.is_success
    ]
    best_experiment = _select_best_experiment(successful_experiments, task_type)

    warnings = _collect_experiment_messages(experiments, field_name="warnings")
    errors = _collect_experiment_messages(experiments, field_name="errors")

    if not successful_experiments:
        status: TrainingRunStatus = "failed"
        if not errors:
            errors = ["No model candidate completed successfully."]
    elif len(successful_experiments) == len(experiments):
        status = "completed"
    else:
        status = "completed_with_failures"

    return ModelTrainingResult(
        status=status,
        task_type=task_type,
        experiments=experiments,
        best_experiment=best_experiment,
        train_row_count=len(split.X_train),
        test_row_count=len(split.X_test),
        feature_count=prepared_dataset.feature_count,
        stratified_split=split.stratified,
        random_state=random_state,
        warnings=warnings,
        errors=errors,
    )


def _train_single_candidate(
    *,
    candidate: ModelCandidate,
    prepared_dataset: PreparedModelingDataset,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ModelExperimentResult:
    """Train and evaluate one candidate model."""
    if candidate.optional_dependency and not _dependency_is_available(
        candidate.optional_dependency
    ):
        return ModelExperimentResult(
            model_name=candidate.model_name,
            role=candidate.role,
            status="skipped",
            task_type=candidate.task_type,
            initial_params=candidate.initial_params,
            warnings=[
                f"Skipped {candidate.model_name} because optional dependency "
                f"'{candidate.optional_dependency}' is not installed."
            ],
        )

    try:
        estimator = _build_estimator(candidate)
        preprocessor = build_sklearn_preprocessor(prepared_dataset)
        fitted_pipeline = SklearnPipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        fitted_pipeline.fit(X_train, y_train)

        train_metrics = _evaluate_fitted_pipeline(
            fitted_pipeline,
            X_train,
            y_train,
            task_type=candidate.task_type,
        )
        test_metrics = _evaluate_fitted_pipeline(
            fitted_pipeline,
            X_test,
            y_test,
            task_type=candidate.task_type,
        )

        primary_metric_name, higher_is_better = _primary_metric(candidate.task_type)
        primary_metric_value = test_metrics.get(primary_metric_name)
        train_primary_metric_value = train_metrics.get(primary_metric_name)

        generalization_gap = _compute_generalization_gap(
            train_value=train_primary_metric_value,
            test_value=primary_metric_value,
            higher_is_better=higher_is_better,
        )
        overfitting_risk = _estimate_overfitting_risk(
            generalization_gap=generalization_gap,
            higher_is_better=higher_is_better,
        )

        return ModelExperimentResult(
            model_name=candidate.model_name,
            role=candidate.role,
            status="success",
            task_type=candidate.task_type,
            initial_params=candidate.initial_params,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            primary_metric_name=primary_metric_name,
            primary_metric_value=primary_metric_value,
            higher_is_better=higher_is_better,
            generalization_gap=generalization_gap,
            overfitting_risk=overfitting_risk,
            fitted_pipeline=fitted_pipeline,
        )

    except Exception as exc:
        return ModelExperimentResult(
            model_name=candidate.model_name,
            role=candidate.role,
            status="failed",
            task_type=candidate.task_type,
            initial_params=candidate.initial_params,
            errors=[f"Failed to train {candidate.model_name}: {exc}"],
        )


def _build_estimator(candidate: ModelCandidate) -> Any:
    """Instantiate a supported sklearn estimator from a model candidate."""
    registry = {
        "LogisticRegression": LogisticRegression,
        "Ridge": Ridge,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    }

    estimator_class = registry.get(candidate.model_name)
    if estimator_class is None:
        raise ModelTrainingError(
            f"Unsupported model for local sklearn training: {candidate.model_name}"
        )

    valid_params = inspect.signature(estimator_class).parameters
    filtered_params = {
        name: value
        for name, value in candidate.initial_params.items()
        if name in valid_params
    }

    return estimator_class(**filtered_params)


def _evaluate_fitted_pipeline(
    fitted_pipeline: SklearnPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task_type: TaskType,
) -> dict[str, float]:
    """Evaluate a fitted sklearn pipeline."""
    predictions = fitted_pipeline.predict(X)

    if task_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y, predictions)), 6),
            "f1_macro": round(float(f1_score(y, predictions, average="macro")), 6),
        }

        auc_value = _safe_roc_auc(fitted_pipeline, X, y)
        if auc_value is not None:
            metrics["roc_auc"] = auc_value

        return metrics

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = sqrt(mse)
    r2 = r2_score(y, predictions)

    return {
        "mae": round(float(mae), 6),
        "rmse": round(float(rmse), 6),
        "r2": round(float(r2), 6),
    }


def _safe_roc_auc(
    fitted_pipeline: SklearnPipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> float | None:
    """Compute ROC-AUC when supported and safe."""
    if not hasattr(fitted_pipeline, "predict_proba"):
        return None

    try:
        probabilities = fitted_pipeline.predict_proba(X)
        unique_classes = pd.Series(y).nunique(dropna=False)

        if unique_classes == 2:
            return round(float(roc_auc_score(y, probabilities[:, 1])), 6)

        return round(
            float(roc_auc_score(y, probabilities, multi_class="ovr")),
            6,
        )
    except Exception:
        return None


def _primary_metric(task_type: TaskType) -> tuple[str, bool]:
    """Return primary metric name and whether higher is better."""
    if task_type == "classification":
        return "f1_macro", True

    return "rmse", False


def _compute_generalization_gap(
    *,
    train_value: float | None,
    test_value: float | None,
    higher_is_better: bool,
) -> float | None:
    """Compute train-test generalization gap using the primary metric."""
    if train_value is None or test_value is None:
        return None

    if higher_is_better:
        return round(float(train_value - test_value), 6)

    if train_value == 0:
        return round(float(test_value - train_value), 6)

    return round(float((test_value - train_value) / abs(train_value)), 6)


def _estimate_overfitting_risk(
    *,
    generalization_gap: float | None,
    higher_is_better: bool,
) -> OverfittingRisk:
    """Estimate overfitting risk from the primary metric gap."""
    if generalization_gap is None:
        return "unknown"

    if generalization_gap <= 0:
        return "low"

    if higher_is_better:
        if generalization_gap >= 0.15:
            return "high"
        if generalization_gap >= 0.07:
            return "medium"
        return "low"

    if generalization_gap >= 0.30:
        return "high"
    if generalization_gap >= 0.15:
        return "medium"
    return "low"


def _select_best_experiment(
    successful_experiments: list[ModelExperimentResult],
    task_type: TaskType,
) -> ModelExperimentResult | None:
    """Select best successful experiment by primary test metric."""
    if not successful_experiments:
        return None

    _, higher_is_better = _primary_metric(task_type)

    return sorted(
        successful_experiments,
        key=lambda experiment: (
            experiment.primary_metric_value is None,
            experiment.primary_metric_value
            if experiment.primary_metric_value is not None
            else float("-inf")
            if higher_is_better
            else float("inf"),
        ),
        reverse=higher_is_better,
    )[0]


def _dependency_is_available(package_name: str) -> bool:
    """Return True when an optional dependency can be imported."""
    return find_spec(package_name) is not None


def _collect_experiment_messages(
    experiments: list[ModelExperimentResult],
    *,
    field_name: Literal["warnings", "errors"],
) -> list[str]:
    """Collect warning/error messages from experiments."""
    messages: list[str] = []

    for experiment in experiments:
        values = experiment.warnings if field_name == "warnings" else experiment.errors
        messages.extend(values)

    return _deduplicate_strings(messages)


def _deduplicate_strings(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving order."""
    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)

    return unique_values
