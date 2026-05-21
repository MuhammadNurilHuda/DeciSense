from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ds_engine.pipeline import IntakePipelineResult
from ds_engine.reporting.analysis_report import (
    AnalysisReportResult,
    create_analysis_report,
)

PackageStatus = Literal["success", "failed"]
PackageType = Literal["analysis_only"]


@dataclass(frozen=True)
class AnalysisPackageResult:
    """Result of packaging analysis artifacts into a tarball."""

    run_id: str
    status: PackageStatus
    package_type: PackageType
    package_path: Path | None
    analysis_dir: Path | None
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
            "analysis_dir": str(self.analysis_dir) if self.analysis_dir else None,
            "included_files": [str(path) for path in self.included_files],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def package_analysis_artifacts(
    pipeline_result: IntakePipelineResult,
    *,
    analysis_report: AnalysisReportResult | None = None,
    runs_root: str | Path = "runs",
    package_name: str | None = None,
    include_pipeline_result: bool = True,
    include_source_data: bool = False,
) -> AnalysisPackageResult:
    """
    Package analysis-stage artifacts into a .tar.gz bundle.

    By default, this creates an analysis-only package without the raw dataset.
    Raw source data can be included explicitly with include_source_data=True.
    """
    report = analysis_report or create_analysis_report(pipeline_result)
    warnings: list[str] = []

    try:
        run_root = Path(runs_root).expanduser().resolve() / pipeline_result.run_id
        analysis_dir = run_root / "analysis"
        bundle_dir = run_root / "bundles"

        analysis_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        included_files: list[Path] = []

        report_json_path = analysis_dir / "analysis_report.json"
        _write_json(report_json_path, report.to_dict())
        included_files.append(report_json_path)

        report_markdown_path = analysis_dir / "analysis_report.md"
        _write_text(report_markdown_path, report.to_markdown())
        included_files.append(report_markdown_path)

        if include_pipeline_result:
            pipeline_json_path = analysis_dir / "pipeline_result.json"
            _write_json(
                pipeline_json_path,
                pipeline_result.to_dict(include_preview=True),
            )
            included_files.append(pipeline_json_path)

        if include_source_data:
            copied_source_path, source_warning = _copy_source_data_if_available(
                pipeline_result,
                analysis_dir=analysis_dir,
            )
            if copied_source_path is not None:
                included_files.append(copied_source_path)
            if source_warning is not None:
                warnings.append(source_warning)

        manifest_path = analysis_dir / "package_manifest.json"
        _write_json(
            manifest_path,
            {
                "run_id": pipeline_result.run_id,
                "package_type": "analysis_only",
                "report_status": report.status,
                "pipeline_status": pipeline_result.status,
                "include_pipeline_result": include_pipeline_result,
                "include_source_data": include_source_data,
                "included_files": [
                    _relative_path_for_manifest(path, root=run_root)
                    for path in included_files
                ],
                "warnings": warnings,
            },
        )
        included_files.append(manifest_path)

        tarball_path = bundle_dir / (
            package_name or f"{pipeline_result.run_id}_analysis_only.tar.gz"
        )
        _create_tarball(
            tarball_path,
            included_files=included_files,
            archive_root=run_root,
        )

        return AnalysisPackageResult(
            run_id=pipeline_result.run_id,
            status="success",
            package_type="analysis_only",
            package_path=tarball_path,
            analysis_dir=analysis_dir,
            included_files=included_files,
            warnings=warnings,
            errors=[],
        )

    except (OSError, TypeError, ValueError, tarfile.TarError) as exc:
        return AnalysisPackageResult(
            run_id=pipeline_result.run_id,
            status="failed",
            package_type="analysis_only",
            package_path=None,
            analysis_dir=None,
            included_files=[],
            warnings=warnings,
            errors=[f"Failed to package analysis artifacts: {exc}"],
        )


def _copy_source_data_if_available(
    pipeline_result: IntakePipelineResult,
    *,
    analysis_dir: Path,
) -> tuple[Path | None, str | None]:
    """Copy the source dataset into the package directory when explicitly requested."""
    if pipeline_result.loaded_dataset is None:
        return None, "Source data was requested but no loaded dataset is available."

    source_path = pipeline_result.loaded_dataset.file_path
    if not source_path.exists():
        return None, f"Source data was requested but file was not found: {source_path}"

    source_dir = analysis_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    destination_path = source_dir / source_path.name
    shutil.copy2(source_path, destination_path)

    return destination_path, None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload using safe defaults for analysis artifacts."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    """Write a text artifact."""
    path.write_text(content, encoding="utf-8")


def _create_tarball(
    tarball_path: Path,
    *,
    included_files: list[Path],
    archive_root: Path,
) -> None:
    """Create a .tar.gz archive from selected artifact files."""
    with tarfile.open(tarball_path, mode="w:gz") as archive:
        for file_path in included_files:
            if not file_path.exists() or not file_path.is_file():
                continue

            archive.add(
                file_path,
                arcname=_relative_path_for_manifest(file_path, root=archive_root),
            )


def _relative_path_for_manifest(path: Path, *, root: Path) -> str:
    """Return a stable relative path for manifests and tar archive names."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name
