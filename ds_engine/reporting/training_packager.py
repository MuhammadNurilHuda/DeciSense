from __future__ import annotations

import json
import re
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib

from ds_engine.modeling.train_models import ModelExperimentResult
from ds_engine.workflows.analysis_workflow import AnalysisWorkflowResult
from ds_engine.workflows.training_workflow import TrainingWorkflowResult

TrainingPackageStatus = Literal["success", "failed"]
TrainingPackageType = Literal["full_training"]


@dataclass(frozen=True)
class TrainingPackageResult:
    """Result of packaging analysis and training artifacts into a tarball."""

    run_id: str
    status: TrainingPackageStatus
    package_type: TrainingPackageType
    package_path: Path | None
    package_dir: Path | None
    included_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        return self.status == "success" and self.package_path is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "package_type": self.package_type,
            "package_path": str(self.package_path) if self.package_path else None,
            "package_dir": str(self.package_dir) if self.package_dir else None,
            "included_files": [str(path) for path in self.included_files],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def package_training_artifacts(
    *,
    analysis_workflow_result: AnalysisWorkflowResult,
    training_workflow_result: TrainingWorkflowResult,
    runs_root: str | Path = "runs",
    package_name: str | None = None,
    include_source_data: bool = False,
    include_fitted_pipelines: bool = True,
) -> TrainingPackageResult:
    """
    Package full training artifacts into a .tar.gz bundle.

    This package represents the "yes, continue training" branch.
    """
    warnings: list[str] = []

    if analysis_workflow_result.run_id != training_workflow_result.run_id:
        return _failed_package_result(
            run_id=analysis_workflow_result.run_id,
            warnings=warnings,
            errors=[
                "Run ID mismatch between analysis workflow result and training workflow result."
            ],
        )

    if not training_workflow_result.is_success:
        return _failed_package_result(
            run_id=analysis_workflow_result.run_id,
            warnings=warnings,
            errors=[
                "Training package cannot be created because training did not complete successfully.",
                *training_workflow_result.errors,
            ],
        )

    if training_workflow_result.prepared_dataset is None:
        return _failed_package_result(
            run_id=analysis_workflow_result.run_id,
            warnings=warnings,
            errors=[
                "Training package cannot be created without prepared dataset metadata."
            ],
        )

    if training_workflow_result.model_training_result is None:
        return _failed_package_result(
            run_id=analysis_workflow_result.run_id,
            warnings=warnings,
            errors=[
                "Training package cannot be created without model training result."
            ],
        )

    try:
        run_root = (
            Path(runs_root).expanduser().resolve() / analysis_workflow_result.run_id
        )
        package_dir = run_root / "package" / "full_training"
        bundle_dir = run_root / "bundles"

        analysis_dir = package_dir / "analysis"
        training_dir = package_dir / "training"
        models_dir = training_dir / "models"

        analysis_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        included_files: list[Path] = []

        analysis_report_json = analysis_dir / "analysis_report.json"
        _write_json(
            analysis_report_json, analysis_workflow_result.analysis_report.to_dict()
        )
        included_files.append(analysis_report_json)

        analysis_report_md = analysis_dir / "analysis_report.md"
        _write_text(
            analysis_report_md, analysis_workflow_result.analysis_report.to_markdown()
        )
        included_files.append(analysis_report_md)

        pipeline_result_json = analysis_dir / "pipeline_result.json"
        _write_json(
            pipeline_result_json,
            analysis_workflow_result.pipeline_result.to_dict(include_preview=True),
        )
        included_files.append(pipeline_result_json)

        analysis_workflow_json = analysis_dir / "analysis_workflow_result.json"
        _write_json(analysis_workflow_json, analysis_workflow_result.to_dict())
        included_files.append(analysis_workflow_json)

        prepared_dataset_json = training_dir / "prepared_modeling_dataset.json"
        _write_json(
            prepared_dataset_json,
            training_workflow_result.prepared_dataset.to_dict(),
        )
        included_files.append(prepared_dataset_json)

        model_training_json = training_dir / "model_training_result.json"
        _write_json(
            model_training_json,
            training_workflow_result.model_training_result.to_dict(),
        )
        included_files.append(model_training_json)

        training_workflow_json = training_dir / "training_workflow_result.json"
        _write_json(training_workflow_json, training_workflow_result.to_dict())
        included_files.append(training_workflow_json)

        training_summary_txt = training_dir / "training_summary.txt"
        _write_text(
            training_summary_txt, training_workflow_result.to_telegram_summary()
        )
        included_files.append(training_summary_txt)

        if include_fitted_pipelines:
            model_files, model_warnings = _save_fitted_pipelines(
                training_workflow_result.model_training_result.experiments,
                models_dir=models_dir,
            )
            included_files.extend(model_files)
            warnings.extend(model_warnings)

        if include_source_data:
            source_file, source_warning = _copy_source_data_if_available(
                analysis_workflow_result,
                package_dir=package_dir,
            )
            if source_file is not None:
                included_files.append(source_file)
            if source_warning is not None:
                warnings.append(source_warning)

        manifest_path = package_dir / "package_manifest.json"
        _write_json(
            manifest_path,
            {
                "run_id": analysis_workflow_result.run_id,
                "package_type": "full_training",
                "analysis_report_status": analysis_workflow_result.analysis_report.status,
                "analysis_workflow_status": analysis_workflow_result.status,
                "training_workflow_status": training_workflow_result.status,
                "training_run_status": (
                    training_workflow_result.model_training_result.status
                ),
                "best_model": (
                    training_workflow_result.best_experiment.model_name
                    if training_workflow_result.best_experiment is not None
                    else None
                ),
                "include_source_data": include_source_data,
                "include_fitted_pipelines": include_fitted_pipelines,
                "included_files": [
                    _relative_path(path, root=package_dir) for path in included_files
                ],
                "warnings": warnings,
            },
        )
        included_files.append(manifest_path)

        tarball_path = bundle_dir / (
            package_name or f"{analysis_workflow_result.run_id}_full_training.tar.gz"
        )
        _create_tarball(
            tarball_path,
            included_files=included_files,
            archive_root=package_dir,
        )

        return TrainingPackageResult(
            run_id=analysis_workflow_result.run_id,
            status="success",
            package_type="full_training",
            package_path=tarball_path,
            package_dir=package_dir,
            included_files=included_files,
            warnings=warnings,
            errors=[],
        )

    except (OSError, TypeError, ValueError, tarfile.TarError) as exc:
        return _failed_package_result(
            run_id=analysis_workflow_result.run_id,
            warnings=warnings,
            errors=[f"Failed to package training artifacts: {exc}"],
        )


def _save_fitted_pipelines(
    experiments: list[ModelExperimentResult],
    *,
    models_dir: Path,
) -> tuple[list[Path], list[str]]:
    """Persist fitted sklearn pipelines for successful experiments."""
    model_files: list[Path] = []
    warnings: list[str] = []

    successful_experiments = [
        experiment for experiment in experiments if experiment.is_success
    ]

    for index, experiment in enumerate(successful_experiments, start=1):
        if experiment.fitted_pipeline is None:
            warnings.append(
                f"Fitted pipeline is missing for {experiment.model_name}; model file was not saved."
            )
            continue

        model_file = models_dir / (
            f"{index:02d}_{_safe_filename(experiment.role)}_"
            f"{_safe_filename(experiment.model_name)}.joblib"
        )
        joblib.dump(experiment.fitted_pipeline, model_file)
        model_files.append(model_file)

    if successful_experiments and not model_files:
        warnings.append("No fitted model pipelines were saved.")

    return model_files, warnings


def _copy_source_data_if_available(
    analysis_workflow_result: AnalysisWorkflowResult,
    *,
    package_dir: Path,
) -> tuple[Path | None, str | None]:
    """Copy source data into package when explicitly requested."""
    loaded_dataset = analysis_workflow_result.pipeline_result.loaded_dataset

    if loaded_dataset is None:
        return None, "Source data was requested but no loaded dataset is available."

    source_path = loaded_dataset.file_path
    if not source_path.exists():
        return None, f"Source data was requested but file was not found: {source_path}"

    source_dir = package_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    destination_path = source_dir / source_path.name
    shutil.copy2(source_path, destination_path)

    return destination_path, None


def _create_tarball(
    tarball_path: Path,
    *,
    included_files: list[Path],
    archive_root: Path,
) -> None:
    """Create a .tar.gz archive from selected package files."""
    with tarfile.open(tarball_path, mode="w:gz") as archive:
        for file_path in included_files:
            if not file_path.exists() or not file_path.is_file():
                continue

            archive.add(
                file_path,
                arcname=_relative_path(file_path, root=archive_root),
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON artifact."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    """Write text artifact."""
    path.write_text(content, encoding="utf-8")


def _relative_path(path: Path, *, root: Path) -> str:
    """Return stable relative path for manifests and tar archive names."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _safe_filename(value: str) -> str:
    """Convert arbitrary labels into safe file-name segments."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "artifact"


def _failed_package_result(
    *,
    run_id: str,
    warnings: list[str],
    errors: list[str],
) -> TrainingPackageResult:
    """Build a failed package result."""
    return TrainingPackageResult(
        run_id=run_id,
        status="failed",
        package_type="full_training",
        package_path=None,
        package_dir=None,
        included_files=[],
        warnings=warnings,
        errors=errors,
    )
