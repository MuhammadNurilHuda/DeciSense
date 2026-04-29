from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any, Literal

import pandas as pd
import json

from ds_engine.intake.infer_task import (
    DEFAULT_TARGET_CANDIDATES,
    TaskInferenceResult,
    infer_task_from_dataframe
)
from ds_engine.intake.load_data import (
    DataLoadError,
    LoadedDataset,
    load_tabular_data
)
from ds_engine.intake.validate_tabular import (
    TabularValidationResult,
    validate_tabular_dataset
)

from ds_engine.profiling.data_quality import (
    DataQualityResult,
    create_data_quality_report,
)

from ds_engine.profiling.schema_profile import (
    SchemaProfileResult,
    create_schema_profile,
)

from ds_engine.profiling.target_profile import (
    TargetProfileError,
    TargetProfileResult,
    create_target_profile,
)
from ds_engine.planning.model_recommender import (
    ModelRecommendationResult,
    recommend_model_for_tabular_data,
)



PipelineStatus = Literal[
    "completed",
    "load_failed",
    "validation_failed",
    "pipeline_error"
]

@dataclass(frozen=True)
class IntakePipelineConfig:
    """ """
    min_rows: int = 2
    min_columns: int = 1
    target_candidates: tuple[str, ...] = DEFAULT_TARGET_CANDIDATES
    classification_unique_value_threshold: int = 20
    persist_artifacts: bool = False
    runs_root: Path = field(default_factory=lambda: Path("runs"))
    include_preview_in_artifacts: bool = True
    preview_row_count: int = 5
    high_cardinality_min_unique_count: int = 50
    high_cardinality_unique_ratio_threshold: float = 0.8
    text_avg_length_threshold: int = 40
    text_unique_ratio_threshold: float = 0.8
    duplicate_row_warning_threshold: float = 0.01
    missing_cell_warning_threshold: float = 0.05
    missing_cell_critical_threshold: float = 0.50
    possible_id_unique_ratio_threshold: float = 0.95
    target_missing_warning_threshold: float = 0.0
    classification_imbalance_ratio_threshold: float = 5.0
    classification_minority_ratio_threshold: float = 0.10
    classification_high_cardinality_threshold: int = 50
    prefer_interpretable_model: bool = False
    allow_optional_model_dependencies: bool = False

@dataclass(frozen=True)
class IntakePipelineResult:
    """ """
    run_id: str
    status: PipelineStatus
    source_file: str
    loaded_dataset: LoadedDataset | None = None
    validation_result: TabularValidationResult | None = None
    task_inference_result: TaskInferenceResult | None = None
    schema_profile_result: SchemaProfileResult | None = None
    data_quality_result: DataQualityResult | None = None
    target_profile_result: TargetProfileResult | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    started_at_utc: str = ""
    completed_at_utc: str = ""
    elapsed_second: float = 0.0
    artifact_path: Path | None = None
    model_recommendation_result: ModelRecommendationResult | None = None

    @property
    def dataframe(self) -> pd.DataFrame | None:
        """ """
        if self.loaded_dataset is None:
            return None
        return self.loaded_dataset.dataframe
        
    @property
    def is_ready_for_downstream_analysis(self) -> bool:
        """ """
        return (
            self.status == 'completed'
            and self.validation_result is not None
            and self.validation_result.is_valid
        )

    @property
    def is_ready_for_model_planning(self) -> bool:
        """Return True when the dataset, target, and quality checks support model planning."""
        quality_is_usable = (
            self.data_quality_result is None
            or self.data_quality_result.is_usable_for_analysis
        )
        target_is_usable = (
            self.target_profile_result is not None
            and self.target_profile_result.is_usable_for_modeling
        )
        return (
            self.is_ready_for_downstream_analysis
            and quality_is_usable
            and target_is_usable
            and self.task_inference_result is not None
            and self.task_inference_result.status == "ok"
        )
    
    @property
    def requires_user_target_input(self) -> bool:
        """ """
        return (
            self.is_ready_for_downstream_analysis
            and self.task_inference_result is not None
            and self.task_inference_result != "ok"
        )
    
    @property
    def is_ready_for_training_approval(self) -> bool:
        """Return True when the pipeline has a model recommendation ready for user approval."""
        return (
            self.is_ready_for_model_planning
            and self.model_recommendation_result is not None
            and self.model_recommendation_result.is_ready_for_training_approval
        )

    def to_dict(
        self,
        *,
        include_preview: bool = False,
        preview_row_count: int = 5) -> dict[str, Any]:
        """ """
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status,
            "source_file": self.source_file,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "elapsed_second": self.elapsed_second,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "loaded_dataset": None,
            "validation_result": None,
            "task_inference_result": None,
            "schema_profile_result": None,
            "data_quality_result": None,
            "target_profile_result": None,
            "model_recommendation_result": None,
        }

        if self.loaded_dataset is not None:
            payload["loaded_dataset"] = _serialize_loaded_dataset(
                self.loaded_dataset,
                include_preview=include_preview,
                preview_row_count=preview_row_count
            )
        
        if self.validation_result is not None:
            payload["validation_result"] = {
                "is_valid": self.validation_result.is_valid,
                "errors": list(self.validation_result.errors),
                "warnings": list(self.validation_result.warnings),
                "row_count": self.validation_result.row_count,
                "colum_count": self.validation_result.column_count
            }

        if self.task_inference_result is not None:
            payload["task_inference_result"] = {
                "candidate_target": self.task_inference_result.candidate_target,
                "task_type": self.task_inference_result.task_type,
                "status": self.task_inference_result.status,
                "reasoning": list(self.task_inference_result.reasoning)
            }

        if self.schema_profile_result is not None:
            payload["schema_profile_result"] = self.schema_profile_result.to_dict()

        if self.data_quality_result is not None:
            payload["data_quality_result"] = self.data_quality_result.to_dict()

        if self.target_profile_result is not None:
            payload["target_profile_result"] = self.target_profile_result.to_dict()

        if self.model_recommendation_result is not None:
            payload["model_recommendation_result"] = (
                self.model_recommendation_result.to_dict()
            )

        return payload

def run_intake_pipeline(
    file_path: str | Path,
    *,
    config: IntakePipelineConfig | None = None,
    run_id: str | None = None) -> IntakePipelineResult:
    """
    Execute the intake pipeline:
    1. load dataset
    2. validate tabular structure
    3. infer target column and task type

    Notes
    -----
    - Validation failure is treated as a hard stop for downstream analysis.
    - Target ambiguity is preserved as a completed intake result, so the caller
      can ask the user for clarification instead of treating it as a crash.
    """

    pipeline_config = config or IntakePipelineConfig()
    actual_run_id = run_id or _generate_run_id()
    started_at = datetime.now(timezone.utc)
    source_file = str(Path(file_path).expanduser())

    try:
        loaded_dataset = load_tabular_data(file_path)
        validation_result = validate_tabular_dataset(
            loaded_dataset.dataframe,
            min_rows=pipeline_config.min_rows,
            min_columns=pipeline_config.min_columns
        )

        if not validation_result.is_valid:
            result = _build_result(
                run_id=actual_run_id,
                status="validation_failed",
                source_file=str(loaded_dataset.file_path),
                started_at=started_at,
                loaded_dataset=loaded_dataset,
                validation_result=validation_result,
                errors=list(validation_result.errors),
                warnings=list(validation_result.warnings)
            )
            return _maybe_persist_result(result, pipeline_config)

        schema_profile_result = create_schema_profile(
            loaded_dataset.dataframe,
            high_cardinality_min_unique_count=(
                pipeline_config.high_cardinality_min_unique_count
            ),
            high_cardinality_unique_ratio_threshold=(
                pipeline_config.high_cardinality_unique_ratio_threshold
            ),
            text_avg_length_threshold=pipeline_config.text_avg_length_threshold,
            text_unique_ratio_threshold=pipeline_config.text_unique_ratio_threshold,
        )

        data_quality_result = create_data_quality_report(
            loaded_dataset.dataframe,
            schema_profile=schema_profile_result,
            duplicate_row_warning_threshold=(
                pipeline_config.duplicate_row_warning_threshold
            ),
            missing_cell_warning_threshold=(
                pipeline_config.missing_cell_warning_threshold
            ),
            missing_cell_critical_threshold=(
                pipeline_config.missing_cell_critical_threshold
            ),
            possible_id_unique_ratio_threshold=(
                pipeline_config.possible_id_unique_ratio_threshold
            ),
        )

        task_inference_result = infer_task_from_dataframe(
            loaded_dataset.dataframe,
            target_candidates=pipeline_config.target_candidates,
            classification_unique_value_threshold=(
                pipeline_config.classification_unique_value_threshold
            ),
        )

        target_profile_result, target_profile_warnings = (
            _create_target_profile_if_available(
                loaded_dataset.dataframe,
                task_inference_result=task_inference_result,
                config=pipeline_config,
            )
        )

        model_recommendation_result, model_recommendation_warnings = (
            _create_model_recommendation_if_available(
                schema_profile_result=schema_profile_result,
                data_quality_result=data_quality_result,
                target_profile_result=target_profile_result,
                config=pipeline_config,
            )
        )

        warnings = list(validation_result.warnings)
        warnings.extend(_format_data_quality_warnings(data_quality_result))
        warnings.extend(target_profile_warnings)
        warnings.extend(model_recommendation_warnings)
        
        result = _build_result(
            run_id=actual_run_id,
            status="completed",
            source_file=str(loaded_dataset.file_path),
            started_at=started_at,
            loaded_dataset=loaded_dataset,
            validation_result=validation_result,
            task_inference_result=task_inference_result,
            schema_profile_result=schema_profile_result,
            data_quality_result=data_quality_result,
            target_profile_result=target_profile_result,
            model_recommendation_result=model_recommendation_result,
            warnings=warnings,
        )
        return _maybe_persist_result(result, pipeline_config)

    except (FileNotFoundError, DataLoadError) as exc:
        result = _build_result(
            run_id=actual_run_id,
            status="load_failed",
            source_file=source_file,
            started_at=started_at,
            errors=[str(exc)]
        )
        return _maybe_persist_result(result, pipeline_config)

    except Exception as exc:
        result = _build_result(
            run_id=actual_run_id,
            status="pipeline_error",
            source_file=source_file,
            started_at=started_at,
            errors=[f"Unexpected intake pipeline error: {exc}"]
         )
        return _maybe_persist_result(result, pipeline_config)

def save_intake_pipeline_artifacts(
    result: IntakePipelineResult,
    *,
    runs_root: str | Path = "runs",
    include_preview: bool = True,
    preview_row_count: int = 5) -> Path:
    """
    Persist an intake pipeline result as JSON under the run directory.

    The output path is:
    runs/<run_id>/intake/intake_result.json
    """
    output_dir = Path(runs_root).expanduser().resolve() / result.run_id / "intake"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "intake_result.json"
    payload = result.to_dict(
        include_preview=include_preview,
        preview_row_count=preview_row_count
    )

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return output_path

def _serialize_loaded_dataset(
    loaded_dataset: LoadedDataset,
    *,
    include_preview: bool,
    preview_row_count: int) -> dict[str, Any]:
    """ """
    payload: dict[str, Any] = {
        "file_path": str(loaded_dataset.file_path),
        "file_name": loaded_dataset.file_name,
        "file_extension": loaded_dataset.file_extension,
        "row_count": loaded_dataset.row_count,
        "column_count": loaded_dataset.column_count,
        "column_names": [str(column) for column in loaded_dataset.dataframe.columns]
    }

    if include_preview:
        payload["preview_rows"] = _build_preview_rows(
            loaded_dataset.dataframe,
            preview_row_count
        )
    return payload

def _build_preview_rows(
    dataframe: pd.DataFrame,
    preview_row_count: int) -> list[dict[str, Any]]:
    """ """
    if preview_row_count <= 0 or dataframe.empty:
        return []

    preview = dataframe.head(preview_row_count).copy()
    preview = preview.where(pd.notna(preview), None)
    return preview.to_dict(orient="records")

def _generate_run_id() -> str:
    """ """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{token_hex(3)}"

def _build_result(
    *,
    run_id: str,
    status: PipelineStatus,
    source_file: str,
    started_at: datetime,
    loaded_dataset: LoadedDataset | None = None,
    validation_result: TabularValidationResult | None = None,
    task_inference_result: TaskInferenceResult | None = None,
    schema_profile_result: SchemaProfileResult | None = None,
    data_quality_result: DataQualityResult | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    target_profile_result: TargetProfileResult | None = None,
    model_recommendation_result: ModelRecommendationResult | None = None) -> IntakePipelineResult:
    """
    
    """
    completed_at = datetime.now(timezone.utc)

    return IntakePipelineResult(
        run_id=run_id,
        status=status,
        source_file=source_file,
        loaded_dataset=loaded_dataset,
        validation_result=validation_result,
        task_inference_result=task_inference_result,
        errors=errors or [],
        warnings=warnings or [],
        started_at_utc=_to_utc_iso(started_at),
        completed_at_utc=_to_utc_iso(completed_at),
        elapsed_second=round((completed_at - started_at).total_seconds(),6),
        schema_profile_result=schema_profile_result,
        data_quality_result=data_quality_result,
        target_profile_result=target_profile_result,
        model_recommendation_result=model_recommendation_result,
    )

def _to_utc_iso(value: datetime) -> str:
    """ """
    return value.astimezone(timezone.utc).isoformat().replace("+00.00", "Z")

def _maybe_persist_result(
    result: IntakePipelineResult,
    config: IntakePipelineConfig) -> IntakePipelineResult:
    """ """
    if not config.persist_artifacts:
        return result
    
    try:
        artifact_path = save_intake_pipeline_artifacts(
            result,
            runs_root=config.runs_root,
            include_preview=config.include_preview_in_artifacts,
            preview_row_count=config.preview_row_count
        )
        return replace(result, artifact_path=artifact_path)
    except (OSError, TypeError, ValueError) as exc:
        warnings = [
            *result.warnings,
            f"Failed to persist intake artifacts: {exc}"
        ]
        return replace(result, warnings=warnings)

def _format_data_quality_warnings(
    data_quality_result: DataQualityResult,
) -> list[str]:
    """Convert data quality issues into pipeline-level warnings."""
    return [
        f"{issue.severity}: {issue.message}"
        for issue in data_quality_result.issues
        if issue.severity in {"warning", "critical"}
    ]

def _create_target_profile_if_available(
    dataframe: pd.DataFrame,
    *,
    task_inference_result: TaskInferenceResult,
    config: IntakePipelineConfig,
) -> tuple[TargetProfileResult | None, list[str]]:
    """
    Create target profile only when target inference is resolved.

    Unresolved target inference is not treated as a pipeline failure because
    the caller may still ask the user to choose the target column.
    """
    if task_inference_result.status != "ok":
        return None, list(task_inference_result.reasoning)

    if (
        task_inference_result.candidate_target is None
        or task_inference_result.task_type is None
    ):
        return None, [
            "Task inference was marked as resolved, but target column or task type is missing."
        ]

    try:
        target_profile_result = create_target_profile(
            dataframe,
            target_column=task_inference_result.candidate_target,
            task_type=task_inference_result.task_type,
            target_missing_warning_threshold=(
                config.target_missing_warning_threshold
            ),
            classification_imbalance_ratio_threshold=(
                config.classification_imbalance_ratio_threshold
            ),
            classification_minority_ratio_threshold=(
                config.classification_minority_ratio_threshold
            ),
            classification_high_cardinality_threshold=(
                config.classification_high_cardinality_threshold
            ),
        )
    except TargetProfileError as exc:
        return None, [f"Failed to create target profile: {exc}"]

    return target_profile_result, _format_target_profile_warnings(
        target_profile_result
    )


def _format_target_profile_warnings(
    target_profile_result: TargetProfileResult,
) -> list[str]:
    """Convert target profile issues into pipeline-level warnings."""
    return [
        f"{issue.severity}: {issue.message}"
        for issue in target_profile_result.issues
        if issue.severity in {"warning", "critical"}
    ]

def _create_model_recommendation_if_available(
    *,
    schema_profile_result: SchemaProfileResult,
    data_quality_result: DataQualityResult,
    target_profile_result: TargetProfileResult | None,
    config: IntakePipelineConfig,
) -> tuple[ModelRecommendationResult | None, list[str]]:
    """
    Create model recommendation when the target profile exists.

    A blocked recommendation is still returned because it contains useful
    reasons explaining why training should not proceed yet.
    """
    if target_profile_result is None:
        return None, []

    try:
        model_recommendation_result = recommend_model_for_tabular_data(
            schema_profile=schema_profile_result,
            data_quality_result=data_quality_result,
            target_profile_result=target_profile_result,
            prefer_interpretable=config.prefer_interpretable_model,
            allow_optional_dependencies=config.allow_optional_model_dependencies,
        )
    except (TypeError, ValueError) as exc:
        return None, [f"Failed to create model recommendation: {exc}"]

    warnings = list(model_recommendation_result.warnings)

    if model_recommendation_result.status == "blocked":
        warnings.extend(model_recommendation_result.blocked_reasons)

    return model_recommendation_result, _deduplicate_strings(warnings)

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

    