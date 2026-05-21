from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ds_engine.pipeline import (
    IntakePipelineConfig,
    IntakePipelineResult,
    run_intake_pipeline,
)
from ds_engine.reporting.analysis_report import (
    AnalysisReportResult,
    create_analysis_report,
)
from ds_engine.reporting.packager import (
    AnalysisPackageResult,
    package_analysis_artifacts,
)

AnalysisWorkflowStatus = Literal[
    "completed",
    "requires_user_input",
    "blocked",
    "failed",
    "packaging_failed",
]


@dataclass(frozen=True)
class AnalysisWorkflowResult:
    """High-level result for the analysis-before-training workflow."""

    run_id: str
    status: AnalysisWorkflowStatus
    pipeline_result: IntakePipelineResult
    analysis_report: AnalysisReportResult
    analysis_package_result: AnalysisPackageResult | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_ready_for_training_approval(self) -> bool:
        return self.analysis_report.is_ready_for_training_approval

    @property
    def has_analysis_package(self) -> bool:
        return (
            self.analysis_package_result is not None
            and self.analysis_package_result.is_success
        )

    @property
    def package_path(self) -> Path | None:
        if self.analysis_package_result is None:
            return None
        return self.analysis_package_result.package_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "is_ready_for_training_approval": self.is_ready_for_training_approval,
            "has_analysis_package": self.has_analysis_package,
            "package_path": str(self.package_path) if self.package_path else None,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "pipeline_result": self.pipeline_result.to_dict(include_preview=True),
            "analysis_report": self.analysis_report.to_dict(),
            "analysis_package_result": (
                self.analysis_package_result.to_dict()
                if self.analysis_package_result is not None
                else None
            ),
        }

    def to_telegram_message(self) -> str:
        """Return the Telegram-friendly analysis message."""
        return self.analysis_report.to_telegram_message()


def run_analysis_workflow(
    file_path: str | Path,
    *,
    pipeline_config: IntakePipelineConfig | None = None,
    run_id: str | None = None,
    package_analysis: bool = False,
    package_runs_root: str | Path | None = None,
    package_name: str | None = None,
    include_source_data: bool = False,
) -> AnalysisWorkflowResult:
    """
    Run the complete analysis-before-training workflow.

    This is the primary local entry point for MVP 0:
    1. run intake/analysis pipeline
    2. create analysis report
    3. optionally create analysis-only package
    """
    config = pipeline_config or IntakePipelineConfig()

    pipeline_result = run_intake_pipeline(
        file_path,
        config=config,
        run_id=run_id,
    )
    analysis_report = create_analysis_report(pipeline_result)

    package_result: AnalysisPackageResult | None = None
    workflow_warnings = _deduplicate_strings(
        [
            *pipeline_result.warnings,
            *analysis_report.warnings,
        ]
    )
    workflow_errors = [
        *pipeline_result.errors,
        *analysis_report.errors,
    ]

    if package_analysis:
        package_result = package_analysis_artifacts(
            pipeline_result,
            analysis_report=analysis_report,
            runs_root=package_runs_root or config.runs_root,
            package_name=package_name,
            include_source_data=include_source_data,
        )
        workflow_warnings = _deduplicate_strings(
            [
                *workflow_warnings,
                *package_result.warnings,
            ]
        )
        workflow_errors.extend(package_result.errors)

    workflow_status = _infer_workflow_status(
        analysis_report=analysis_report,
        analysis_package_result=package_result,
    )

    return AnalysisWorkflowResult(
        run_id=pipeline_result.run_id,
        status=workflow_status,
        pipeline_result=pipeline_result,
        analysis_report=analysis_report,
        analysis_package_result=package_result,
        warnings=workflow_warnings,
        errors=workflow_errors,
    )


def _infer_workflow_status(
    *,
    analysis_report: AnalysisReportResult,
    analysis_package_result: AnalysisPackageResult | None,
) -> AnalysisWorkflowStatus:
    """Infer workflow status from report and optional packaging result."""
    if analysis_package_result is not None and not analysis_package_result.is_success:
        return "packaging_failed"

    if analysis_report.status == "ready_for_training_approval":
        return "completed"

    if analysis_report.status == "requires_user_input":
        return "requires_user_input"

    if analysis_report.status == "blocked":
        return "blocked"

    return "failed"


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
