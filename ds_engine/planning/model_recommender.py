from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ds_engine.intake.infer_task import TaskType
from ds_engine.profiling.data_quality import DataQualityResult
from ds_engine.profiling.schema_profile import SchemaProfileResult
from ds_engine.profiling.target_profile import TargetProfileResult

CandidateRole = Literal["recommended", "baseline", "challenger"]
RecommendationStatus = Literal["ready", "blocked"]
RecommendationConfidence = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelCandidate:
    """A candidate model plan for tabular ML experimentation."""

    model_name: str
    task_type: TaskType
    role: CandidateRole
    initial_params: dict[str, Any]
    preprocessing_notes: list[str]
    reasoning: list[str]
    concerns: list[str] = field(default_factory=list)
    optional_dependency: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "role": self.role,
            "initial_params": dict(self.initial_params),
            "preprocessing_notes": list(self.preprocessing_notes),
            "reasoning": list(self.reasoning),
            "concerns": list(self.concerns),
            "optional_dependency": self.optional_dependency,
        }


@dataclass(frozen=True)
class ModelRecommendationResult:
    """Structured model recommendation for the analysis-before-training stage."""

    status: RecommendationStatus
    task_type: TaskType
    recommended_model: str | None
    confidence: RecommendationConfidence
    candidates: list[ModelCandidate]
    reasoning: list[str]
    warnings: list[str]
    blocked_reasons: list[str]
    dataset_notes: dict[str, Any]

    @property
    def is_ready_for_training_approval(self) -> bool:
        return self.status == "ready" and self.recommended_model is not None

    @property
    def recommended_candidate(self) -> ModelCandidate | None:
        for candidate in self.candidates:
            if candidate.role == "recommended":
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "task_type": self.task_type,
            "recommended_model": self.recommended_model,
            "confidence": self.confidence,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reasoning": list(self.reasoning),
            "warnings": list(self.warnings),
            "blocked_reasons": list(self.blocked_reasons),
            "dataset_notes": dict(self.dataset_notes),
        }


def recommend_model_for_tabular_data(
    *,
    schema_profile: SchemaProfileResult,
    data_quality_result: DataQualityResult,
    target_profile_result: TargetProfileResult,
    prefer_interpretable: bool = False,
    allow_optional_dependencies: bool = False,
) -> ModelRecommendationResult:
    """
    Recommend the first model setup to try for a validated tabular ML task.

    This does not train a model. It only creates a reasoned starting point for
    the user approval step before experimentation.
    """
    task_type = target_profile_result.task_type
    dataset_notes = _build_dataset_notes(
        schema_profile=schema_profile,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
    )

    blocked_reasons = _build_blocked_reasons(
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
    )
    warnings = _build_recommendation_warnings(
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
        dataset_notes=dataset_notes,
        allow_optional_dependencies=allow_optional_dependencies,
    )

    if blocked_reasons:
        return ModelRecommendationResult(
            status="blocked",
            task_type=task_type,
            recommended_model=None,
            confidence="low",
            candidates=[],
            reasoning=[
                "Model recommendation is blocked because the dataset or target has critical issues."
            ],
            warnings=warnings,
            blocked_reasons=blocked_reasons,
            dataset_notes=dataset_notes,
        )

    recommended_candidate = _choose_recommended_candidate(
        task_type=task_type,
        dataset_notes=dataset_notes,
        target_profile_result=target_profile_result,
        prefer_interpretable=prefer_interpretable,
        allow_optional_dependencies=allow_optional_dependencies,
    )

    baseline_candidate = _build_baseline_candidate(
        task_type=task_type,
        target_profile_result=target_profile_result,
        role="baseline",
    )

    challenger_candidate = _build_challenger_candidate(
        task_type=task_type,
        recommended_model=recommended_candidate.model_name,
        dataset_notes=dataset_notes,
        target_profile_result=target_profile_result,
    )

    candidates = _deduplicate_candidates(
        [
            recommended_candidate,
            baseline_candidate,
            challenger_candidate,
        ]
    )

    confidence = _estimate_recommendation_confidence(
        dataset_notes=dataset_notes,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
        allow_optional_dependencies=allow_optional_dependencies,
    )

    return ModelRecommendationResult(
        status="ready",
        task_type=task_type,
        recommended_model=recommended_candidate.model_name,
        confidence=confidence,
        candidates=candidates,
        reasoning=list(recommended_candidate.reasoning),
        warnings=warnings,
        blocked_reasons=[],
        dataset_notes=dataset_notes,
    )


def _choose_recommended_candidate(
    *,
    task_type: TaskType,
    dataset_notes: dict[str, Any],
    target_profile_result: TargetProfileResult,
    prefer_interpretable: bool,
    allow_optional_dependencies: bool,
) -> ModelCandidate:
    """Choose the strongest first model candidate before actual training."""
    if prefer_interpretable:
        return _build_baseline_candidate(
            task_type=task_type,
            target_profile_result=target_profile_result,
            role="recommended",
        )

    if dataset_notes["has_many_categorical_features"] and allow_optional_dependencies:
        return _build_catboost_candidate(
            task_type=task_type,
            target_profile_result=target_profile_result,
            role="recommended",
        )

    if task_type == "classification" and dataset_notes["has_class_imbalance"]:
        return _build_random_forest_candidate(
            task_type=task_type,
            target_profile_result=target_profile_result,
            role="recommended",
            reason_prefix=(
                "Random Forest is recommended first because the classification "
                "target is imbalanced and the model can use class weighting."
            ),
        )

    if dataset_notes["is_mostly_numeric"]:
        return _build_hist_gradient_boosting_candidate(
            task_type=task_type,
            role="recommended",
            reason_prefix=(
                "Histogram-based Gradient Boosting is recommended first because "
                "the feature set is mostly numeric and likely benefits from "
                "non-linear tabular modeling."
            ),
        )

    return _build_random_forest_candidate(
        task_type=task_type,
        target_profile_result=target_profile_result,
        role="recommended",
        reason_prefix=(
            "Random Forest is recommended first as a robust non-linear baseline "
            "for mixed tabular data."
        ),
    )


def _build_baseline_candidate(
    *,
    task_type: TaskType,
    target_profile_result: TargetProfileResult,
    role: CandidateRole,
) -> ModelCandidate:
    """Build a simple interpretable baseline candidate."""
    if task_type == "classification":
        class_weight = (
            "balanced"
            if _target_has_issue(target_profile_result, "class_imbalance")
            else None
        )
        params = {
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": class_weight,
            "random_state": 42,
        }

        return ModelCandidate(
            model_name="LogisticRegression",
            task_type=task_type,
            role=role,
            initial_params=params,
            preprocessing_notes=[
                "Impute missing numeric values.",
                "One-hot encode categorical features.",
                "Scale numeric features before fitting.",
            ],
            reasoning=[
                "Logistic Regression provides a fast, interpretable classification baseline.",
                "It helps verify whether a simple linear decision boundary is already competitive.",
            ],
            concerns=[
                "May underfit non-linear relationships.",
                "Can struggle with high-cardinality categorical features after one-hot encoding.",
            ],
        )

    return ModelCandidate(
        model_name="Ridge",
        task_type=task_type,
        role=role,
        initial_params={
            "alpha": 1.0,
            "random_state": 42,
        },
        preprocessing_notes=[
            "Impute missing numeric values.",
            "One-hot encode categorical features.",
            "Scale numeric features before fitting.",
        ],
        reasoning=[
            "Ridge Regression provides a fast, interpretable regression baseline.",
            "It is a good first check for linear signal before trying heavier models.",
        ],
        concerns=[
            "May underfit strong non-linear relationships.",
            "Sensitive to unhandled outliers in the regression target.",
        ],
    )


def _build_random_forest_candidate(
    *,
    task_type: TaskType,
    target_profile_result: TargetProfileResult,
    role: CandidateRole,
    reason_prefix: str,
) -> ModelCandidate:
    """Build a Random Forest candidate."""
    if task_type == "classification":
        class_weight = (
            "balanced"
            if _target_has_issue(target_profile_result, "class_imbalance")
            else None
        )
        return ModelCandidate(
            model_name="RandomForestClassifier",
            task_type=task_type,
            role=role,
            initial_params={
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 2,
                "class_weight": class_weight,
                "random_state": 42,
                "n_jobs": -1,
            },
            preprocessing_notes=[
                "Impute missing values before fitting.",
                "Encode categorical features before fitting.",
                "Exclude likely identifier columns from training features.",
            ],
            reasoning=[
                reason_prefix,
                "It is robust to mixed feature scales and provides a strong tabular benchmark.",
            ],
            concerns=[
                "Can overfit noisy data if trees become too deep.",
                "Can be slower and larger than linear baselines.",
            ],
        )

    return ModelCandidate(
        model_name="RandomForestRegressor",
        task_type=task_type,
        role=role,
        initial_params={
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
        preprocessing_notes=[
            "Impute missing values before fitting.",
            "Encode categorical features before fitting.",
            "Exclude likely identifier columns from training features.",
        ],
        reasoning=[
            reason_prefix,
            "It is robust to mixed feature scales and useful for non-linear regression baselines.",
        ],
        concerns=[
            "Can overfit noisy data if trees become too deep.",
            "Predictions may be less smooth than gradient boosting for regression.",
        ],
    )


def _build_hist_gradient_boosting_candidate(
    *,
    task_type: TaskType,
    role: CandidateRole,
    reason_prefix: str,
) -> ModelCandidate:
    """Build a scikit-learn histogram gradient boosting candidate."""
    model_name = (
        "HistGradientBoostingClassifier"
        if task_type == "classification"
        else "HistGradientBoostingRegressor"
    )

    return ModelCandidate(
        model_name=model_name,
        task_type=task_type,
        role=role,
        initial_params={
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.1,
            "random_state": 42,
        },
        preprocessing_notes=[
            "Impute or encode features consistently before fitting.",
            "Encode categorical features before fitting.",
            "Use validation-based early stopping when available.",
        ],
        reasoning=[
            reason_prefix,
            "Gradient boosting is often a strong first choice for structured tabular data.",
        ],
        concerns=[
            "Needs careful validation to detect overfitting.",
            "Categorical features still need a consistent preprocessing strategy.",
        ],
    )


def _build_catboost_candidate(
    *,
    task_type: TaskType,
    target_profile_result: TargetProfileResult,
    role: CandidateRole,
) -> ModelCandidate:
    """Build a CatBoost candidate when optional dependencies are allowed."""
    if task_type == "classification":
        is_multiclass = target_profile_result.unique_count > 2
        return ModelCandidate(
            model_name="CatBoostClassifier",
            task_type=task_type,
            role=role,
            optional_dependency="catboost",
            initial_params={
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "MultiClass" if is_multiclass else "Logloss",
                "eval_metric": "MultiClass" if is_multiclass else "AUC",
                "random_seed": 42,
                "verbose": False,
            },
            preprocessing_notes=[
                "Pass categorical feature indices/names to CatBoost during fitting.",
                "Exclude likely identifier columns from training features.",
                "Use stratified validation for classification.",
            ],
            reasoning=[
                "CatBoost is recommended because the dataset has many categorical features.",
                "It can handle categorical variables more naturally than one-hot-heavy sklearn baselines.",
            ],
            concerns=[
                "Requires the optional catboost package.",
                "Still needs validation checks for overfitting and leakage.",
            ],
        )

    return ModelCandidate(
        model_name="CatBoostRegressor",
        task_type=task_type,
        role=role,
        optional_dependency="catboost",
        initial_params={
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": False,
        },
        preprocessing_notes=[
            "Pass categorical feature indices/names to CatBoost during fitting.",
            "Exclude likely identifier columns from training features.",
            "Use a validation split for early stopping.",
        ],
        reasoning=[
            "CatBoost is recommended because the dataset has many categorical features.",
            "It is a strong first candidate for tabular regression with categorical inputs.",
        ],
        concerns=[
            "Requires the optional catboost package.",
            "Still needs validation checks for overfitting and target outliers.",
        ],
    )


def _build_challenger_candidate(
    *,
    task_type: TaskType,
    recommended_model: str,
    dataset_notes: dict[str, Any],
    target_profile_result: TargetProfileResult,
) -> ModelCandidate:
    """Build one challenger candidate to compare against the recommended model."""
    if recommended_model.startswith("HistGradientBoosting"):
        return _build_random_forest_candidate(
            task_type=task_type,
            target_profile_result=target_profile_result,
            role="challenger",
            reason_prefix="Random Forest is useful as a challenger to gradient boosting.",
        )

    return _build_hist_gradient_boosting_candidate(
        task_type=task_type,
        role="challenger",
        reason_prefix=(
            "Histogram-based Gradient Boosting is useful as a challenger model "
            "for tabular data."
        ),
    )


def _build_dataset_notes(
    *,
    schema_profile: SchemaProfileResult,
    data_quality_result: DataQualityResult,
    target_profile_result: TargetProfileResult,
) -> dict[str, Any]:
    """Summarize dataset traits used for model recommendation."""
    target_column = target_profile_result.target_column
    feature_profiles = [
        column
        for column in schema_profile.columns
        if column.column_name != target_column
    ]
    feature_count = len(feature_profiles)

    numeric_feature_count = sum(
        column.inferred_type == "numeric" for column in feature_profiles
    )
    categorical_feature_count = sum(
        column.inferred_type == "categorical" for column in feature_profiles
    )
    text_feature_count = sum(
        column.inferred_type == "text" for column in feature_profiles
    )
    high_cardinality_feature_count = sum(
        column.is_high_cardinality for column in feature_profiles
    )

    categorical_ratio = _safe_ratio(categorical_feature_count, feature_count)
    numeric_ratio = _safe_ratio(numeric_feature_count, feature_count)

    return {
        "row_count": schema_profile.row_count,
        "column_count": schema_profile.column_count,
        "feature_count": feature_count,
        "numeric_feature_count": numeric_feature_count,
        "categorical_feature_count": categorical_feature_count,
        "text_feature_count": text_feature_count,
        "high_cardinality_feature_count": high_cardinality_feature_count,
        "categorical_feature_ratio": categorical_ratio,
        "numeric_feature_ratio": numeric_ratio,
        "is_small_dataset": schema_profile.row_count < 1_000,
        "is_large_dataset": schema_profile.row_count >= 100_000,
        "is_mostly_numeric": numeric_ratio >= 0.70,
        "has_many_categorical_features": (
            categorical_ratio >= 0.30 or high_cardinality_feature_count > 0
        ),
        "has_text_features": text_feature_count > 0,
        "has_missing_values": data_quality_result.missing_cell_count > 0,
        "missing_cell_ratio": data_quality_result.missing_cell_ratio,
        "has_class_imbalance": _target_has_issue(
            target_profile_result,
            "class_imbalance",
        ),
        "possible_id_columns": list(data_quality_result.possible_id_columns),
        "high_cardinality_columns": list(data_quality_result.high_cardinality_columns),
    }


def _build_blocked_reasons(
    *,
    data_quality_result: DataQualityResult,
    target_profile_result: TargetProfileResult,
) -> list[str]:
    """Return reasons that should block model recommendation/training approval."""
    blocked_reasons: list[str] = []

    for issue in data_quality_result.issues:
        if issue.severity == "critical":
            blocked_reasons.append(issue.message)

    for issue in target_profile_result.issues:
        if issue.severity == "critical":
            blocked_reasons.append(issue.message)

    return blocked_reasons


def _build_recommendation_warnings(
    *,
    data_quality_result: DataQualityResult,
    target_profile_result: TargetProfileResult,
    dataset_notes: dict[str, Any],
    allow_optional_dependencies: bool,
) -> list[str]:
    """Create model-planning warnings from quality and target diagnostics."""
    warnings: list[str] = []

    warnings.extend(
        issue.message
        for issue in data_quality_result.issues
        if issue.severity in {"warning", "critical"}
    )
    warnings.extend(
        issue.message
        for issue in target_profile_result.issues
        if issue.severity in {"warning", "critical"}
    )

    if dataset_notes["has_text_features"]:
        warnings.append(
            "Text-like columns were detected. The first MVP model should exclude them or use simple text preprocessing."
        )

    if (
        dataset_notes["has_many_categorical_features"]
        and not allow_optional_dependencies
    ):
        warnings.append(
            "Many categorical or high-cardinality features were detected. CatBoost may be a stronger optional candidate later."
        )

    return _deduplicate_strings(warnings)


def _estimate_recommendation_confidence(
    *,
    dataset_notes: dict[str, Any],
    data_quality_result: DataQualityResult,
    target_profile_result: TargetProfileResult,
    allow_optional_dependencies: bool,
) -> RecommendationConfidence:
    """Estimate confidence in the pre-training recommendation."""
    if (
        data_quality_result.has_critical_issues
        or target_profile_result.has_critical_issues
    ):
        return "low"

    warning_count = len(data_quality_result.issues) + len(target_profile_result.issues)

    if (
        dataset_notes["has_many_categorical_features"]
        and not allow_optional_dependencies
    ):
        return "medium"

    if warning_count == 0 and dataset_notes["feature_count"] > 0:
        return "high"

    return "medium"


def _target_has_issue(
    target_profile_result: TargetProfileResult,
    issue_code: str,
) -> bool:
    return any(issue.code == issue_code for issue in target_profile_result.issues)


def _deduplicate_candidates(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    """Keep first occurrence of each model name while preserving order."""
    seen: set[str] = set()
    unique_candidates: list[ModelCandidate] = []

    for candidate in candidates:
        if candidate.model_name in seen:
            continue
        seen.add(candidate.model_name)
        unique_candidates.append(candidate)

    return unique_candidates


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


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded ratio while protecting against division by zero."""
    if denominator == 0:
        return 0.0

    return round(float(numerator) / float(denominator), 6)
