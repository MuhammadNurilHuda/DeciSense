from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ds_engine.modeling.preprocess import (
    PreparedModelingDataset,
    PreprocessingError,
    prepare_modeling_dataset,
)
from ds_engine.modeling.train_models import (
    ModelExperimentResult,
    ModelTrainingResult,
    train_model_candidates,
)
from ds_engine.workflows.analysis_workflow import AnalysisWorkflowResult

TrainingWorkflowStatus = Literal[
    "completed",
    "completed_with_failures",
    "failed",
    "not_ready",
]


@dataclass(frozen=True)
class TrainingWorkflowResult:
    """High-level result for the training workflow after user approval."""

    run_id: str
    status: TrainingWorkflowStatus
    prepared_dataset: PreparedModelingDataset | None = None
    model_training_result: ModelTrainingResult | None = None
    artifact_dir: Path | None = None
    artifact_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.status in {"completed", "completed_with_failures"}

    @property
    def best_experiment(self) -> ModelExperimentResult | None:
        if self.model_training_result is None:
            return None
        return self.model_training_result.best_experiment

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "is_success": self.is_success,
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
            "artifact_paths": [str(path) for path in self.artifact_paths],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "prepared_dataset": (
                self.prepared_dataset.to_dict()
                if self.prepared_dataset is not None
                else None
            ),
            "model_training_result": (
                self.model_training_result.to_dict()
                if self.model_training_result is not None
                else None
            ),
        }

    def to_telegram_summary(self) -> str:
        """Return a concise Telegram-friendly training summary."""
        if not self.is_success or self.model_training_result is None:
            return _build_failed_training_summary(self)

        best_experiment = self.model_training_result.best_experiment
        if best_experiment is None:
            return _build_failed_training_summary(self)

        primary_metric = (
            f"{best_experiment.primary_metric_name}="
            f"{best_experiment.primary_metric_value}"
            if best_experiment.primary_metric_name is not None
            else "primary metric unavailable"
        )

        successful_count = len(self.model_training_result.successful_experiments)
        total_count = len(self.model_training_result.experiments)

        return "\n".join(
            [
                "Training completed.",
                "",
                f"Experiments completed: {successful_count}/{total_count}",
                f"Best model: {best_experiment.model_name}",
                f"Best test metric: {primary_metric}",
                f"Overfitting risk: {best_experiment.overfitting_risk}",
                f"Train rows: {self.model_training_result.train_row_count}",
                f"Test rows: {self.model_training_result.test_row_count}",
            ]
        ).strip()


def run_training_workflow(
    analysis_workflow_result: AnalysisWorkflowResult,
    *,
    test_size: float | int = 0.2,
    random_state: int = 42,
    max_candidates: int | None = None,
    persist_artifacts: bool = False,
    runs_root: str | Path = "runs",
) -> TrainingWorkflowResult:
    """
    Run the model-training workflow after user approval.

    This function expects the analysis workflow to already be ready for training
    approval. It prepares modeling data, trains model candidates, and optionally
    persists training artifacts.
    """
    if not analysis_workflow_result.is_ready_for_training_approval:
        return TrainingWorkflowResult(
            run_id=analysis_workflow_result.run_id,
            status="not_ready",
            errors=[
                "Analysis workflow is not ready for training approval.",
                f"Current workflow status: {analysis_workflow_result.status}",
                f"Current report status: {analysis_workflow_result.analysis_report.status}",
            ],
        )

    pipeline_result = analysis_workflow_result.pipeline_result
    missing_components = _find_missing_training_components(analysis_workflow_result)
    if missing_components:
        return TrainingWorkflowResult(
            run_id=analysis_workflow_result.run_id,
            status="failed",
            errors=[
                "Training workflow is missing required analysis components.",
                *missing_components,
            ],
        )

    assert pipeline_result.loaded_dataset is not None
    assert pipeline_result.task_inference_result is not None
    assert pipeline_result.task_inference_result.candidate_target is not None
    assert pipeline_result.task_inference_result.task_type is not None
    assert pipeline_result.model_recommendation_result is not None

    try:
        prepared_dataset = prepare_modeling_dataset(
            pipeline_result.loaded_dataset.dataframe,
            target_column=pipeline_result.task_inference_result.candidate_target,
            task_type=pipeline_result.task_inference_result.task_type,
            schema_profile=pipeline_result.schema_profile_result,
            data_quality_result=pipeline_result.data_quality_result,
        )
    except PreprocessingError as exc:
        return TrainingWorkflowResult(
            run_id=analysis_workflow_result.run_id,
            status="failed",
            errors=[f"Failed to prepare modeling dataset: {exc}"],
        )

    model_training_result = train_model_candidates(
        prepared_dataset=prepared_dataset,
        model_recommendation_result=pipeline_result.model_recommendation_result,
        test_size=test_size,
        random_state=random_state,
        max_candidates=max_candidates,
    )

    workflow_result = TrainingWorkflowResult(
        run_id=analysis_workflow_result.run_id,
        status=_infer_training_workflow_status(model_training_result),
        prepared_dataset=prepared_dataset,
        model_training_result=model_training_result,
        warnings=_deduplicate_strings(
            [
                *analysis_workflow_result.warnings,
                *model_training_result.warnings,
            ]
        ),
        errors=list(model_training_result.errors),
    )

    if not persist_artifacts:
        return workflow_result

    return _persist_training_artifacts_if_possible(
        workflow_result,
        runs_root=runs_root,
    )


def save_training_workflow_artifacts(
    training_workflow_result: TrainingWorkflowResult,
    *,
    runs_root: str | Path = "runs",
) -> tuple[Path, list[Path]]:
    """
    Persist training workflow artifacts under:

    runs/<run_id>/training/
    """
    if training_workflow_result.prepared_dataset is None:
        raise ValueError("Cannot persist training artifacts without prepared dataset.")

    if training_workflow_result.model_training_result is None:
        raise ValueError("Cannot persist training artifacts without training result.")

    training_dir = (
        Path(runs_root).expanduser().resolve()
        / training_workflow_result.run_id
        / "training"
    )
    training_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths: list[Path] = []

    prepared_dataset_path = training_dir / "prepared_modeling_dataset.json"
    _write_json(
        prepared_dataset_path, training_workflow_result.prepared_dataset.to_dict()
    )
    artifact_paths.append(prepared_dataset_path)

    training_result_path = training_dir / "model_training_result.json"
    _write_json(
        training_result_path, training_workflow_result.model_training_result.to_dict()
    )
    artifact_paths.append(training_result_path)

    training_summary_path = training_dir / "training_summary.txt"
    training_summary_path.write_text(
        training_workflow_result.to_telegram_summary(),
        encoding="utf-8",
    )
    artifact_paths.append(training_summary_path)

    manifest_path = training_dir / "training_manifest.json"
    _write_json(
        manifest_path,
        {
            "run_id": training_workflow_result.run_id,
            "workflow_status": training_workflow_result.status,
            "artifact_files": [
                path.relative_to(training_dir).as_posix() for path in artifact_paths
            ],
        },
    )
    artifact_paths.append(manifest_path)

    return training_dir, artifact_paths


def _find_missing_training_components(
    analysis_workflow_result: AnalysisWorkflowResult,
) -> list[str]:
    """Return missing components required for training."""
    pipeline_result = analysis_workflow_result.pipeline_result
    missing_components: list[str] = []

    if pipeline_result.loaded_dataset is None:
        missing_components.append("Missing loaded dataset.")

    if pipeline_result.task_inference_result is None:
        missing_components.append("Missing task inference result.")
    else:
        if pipeline_result.task_inference_result.candidate_target is None:
            missing_components.append("Missing target column.")
        if pipeline_result.task_inference_result.task_type is None:
            missing_components.append("Missing task type.")

    if pipeline_result.schema_profile_result is None:
        missing_components.append("Missing schema profile result.")

    if pipeline_result.data_quality_result is None:
        missing_components.append("Missing data quality result.")

    if pipeline_result.target_profile_result is None:
        missing_components.append("Missing target profile result.")

    if pipeline_result.model_recommendation_result is None:
        missing_components.append("Missing model recommendation result.")

    return missing_components


def _infer_training_workflow_status(
    model_training_result: ModelTrainingResult,
) -> TrainingWorkflowStatus:
    """Map model training status into workflow status."""
    if model_training_result.status == "completed":
        return "completed"

    if model_training_result.status == "completed_with_failures":
        return "completed_with_failures"

    return "failed"


def _persist_training_artifacts_if_possible(
    training_workflow_result: TrainingWorkflowResult,
    *,
    runs_root: str | Path,
) -> TrainingWorkflowResult:
    """Persist training artifacts without turning successful training into failure."""
    try:
        artifact_dir, artifact_paths = save_training_workflow_artifacts(
            training_workflow_result,
            runs_root=runs_root,
        )
        return replace(
            training_workflow_result,
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
        )
    except (OSError, TypeError, ValueError) as exc:
        return replace(
            training_workflow_result,
            warnings=[
                *training_workflow_result.warnings,
                f"Failed to persist training artifacts: {exc}",
            ],
        )


def _build_failed_training_summary(
    training_workflow_result: TrainingWorkflowResult,
) -> str:
    """Build a Telegram-friendly failed training summary."""
    lines = [
        "Training could not be completed.",
        "",
        f"Status: {training_workflow_result.status}",
    ]

    if training_workflow_result.errors:
        lines.extend(["", "Error detail(s):"])
        lines.extend(f"- {error}" for error in training_workflow_result.errors[:5])

    return "\n".join(lines).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON artifact."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


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
