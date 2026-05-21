from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ds_engine.pipeline import IntakePipelineResult

AnalysisReportStatus = Literal[
    "ready_for_training_approval",
    "requires_user_input",
    "blocked",
    "failed",
]


@dataclass(frozen=True)
class ReportSection:
    """A human-readable report section."""

    title: str
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "lines": list(self.lines),
        }

    def to_markdown(self) -> str:
        if not self.lines:
            return f"## {self.title}"

        body = "\n".join(f"- {line}" for line in self.lines)
        return f"## {self.title}\n{body}"


@dataclass(frozen=True)
class AnalysisReportResult:
    """Human-readable analysis report generated from the intake pipeline result."""

    run_id: str
    status: AnalysisReportStatus
    title: str
    executive_summary: str
    sections: list[ReportSection]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    training_approval_question: str | None = None

    @property
    def is_ready_for_training_approval(self) -> bool:
        return self.status == "ready_for_training_approval"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "title": self.title,
            "executive_summary": self.executive_summary,
            "sections": [section.to_dict() for section in self.sections],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "training_approval_question": self.training_approval_question,
        }

    def to_markdown(self) -> str:
        parts = [
            f"# {self.title}",
            "",
            self.executive_summary,
        ]

        for section in self.sections:
            parts.extend(["", section.to_markdown()])

        if self.warnings:
            parts.extend(
                [
                    "",
                    "## Warnings",
                    *[f"- {warning}" for warning in self.warnings],
                ]
            )

        if self.errors:
            parts.extend(
                [
                    "",
                    "## Errors",
                    *[f"- {error}" for error in self.errors],
                ]
            )

        if self.training_approval_question:
            parts.extend(
                [
                    "",
                    "## Next Step",
                    self.training_approval_question,
                ]
            )

        return "\n".join(parts).strip()

    def to_telegram_message(self) -> str:
        """
        Return a concise Telegram-friendly message.

        This intentionally keeps the message shorter than the full Markdown report.
        """
        lines = [
            self.executive_summary,
        ]

        model_section = next(
            (
                section
                for section in self.sections
                if section.title == "Model Recommendation"
            ),
            None,
        )
        if model_section is not None:
            lines.extend(["", "Model recommendation:"])
            lines.extend(f"- {line}" for line in model_section.lines[:5])

        if self.warnings:
            lines.extend(["", "Important warnings:"])
            lines.extend(f"- {warning}" for warning in self.warnings[:5])

        if self.training_approval_question:
            lines.extend(["", self.training_approval_question])

        return "\n".join(lines).strip()


def create_analysis_report(
    pipeline_result: IntakePipelineResult,
) -> AnalysisReportResult:
    """Create a human-readable report from an intake pipeline result."""
    report_status = _infer_report_status(pipeline_result)

    return AnalysisReportResult(
        run_id=pipeline_result.run_id,
        status=report_status,
        title=f"DeciSense Analysis Report - {pipeline_result.run_id}",
        executive_summary=_build_executive_summary(pipeline_result, report_status),
        sections=_build_sections(pipeline_result),
        warnings=_deduplicate_strings(pipeline_result.warnings),
        errors=list(pipeline_result.errors),
        training_approval_question=_build_training_approval_question(
            pipeline_result,
            report_status,
        ),
    )


def _infer_report_status(pipeline_result: IntakePipelineResult) -> AnalysisReportStatus:
    """Infer report-level status from pipeline result."""
    if pipeline_result.status in {"load_failed", "validation_failed", "pipeline_error"}:
        return "failed"

    task_inference = pipeline_result.task_inference_result
    if task_inference is None or task_inference.status != "ok":
        return "requires_user_input"

    recommendation = pipeline_result.model_recommendation_result
    if recommendation is not None:
        if recommendation.is_ready_for_training_approval:
            return "ready_for_training_approval"

        if recommendation.status == "blocked":
            return "blocked"

    if pipeline_result.is_ready_for_training_approval:
        return "ready_for_training_approval"

    return "blocked"


def _build_executive_summary(
    pipeline_result: IntakePipelineResult,
    report_status: AnalysisReportStatus,
) -> str:
    """Build the high-level summary for the report."""
    if report_status == "failed":
        return (
            "The dataset analysis could not be completed. "
            "Please review the reported errors before continuing."
        )

    if report_status == "requires_user_input":
        return (
            "The dataset was loaded and profiled successfully, but DeciSense could "
            "not determine the target column with enough confidence. Please provide "
            "the target column before model planning can continue."
        )

    if report_status == "ready_for_training_approval":
        recommended_model = (
            pipeline_result.model_recommendation_result.recommended_model
            if pipeline_result.model_recommendation_result is not None
            else "the recommended model"
        )
        return (
            "The dataset analysis is complete. Based on the current data profile, "
            f"DeciSense recommends trying {recommended_model} as the first model."
        )

    return (
        "The dataset analysis is complete, but training is not recommended yet "
        "because one or more blocking issues were detected."
    )


def _build_sections(
    pipeline_result: IntakePipelineResult,
) -> list[ReportSection]:
    """Build report sections from available pipeline outputs."""
    sections = [
        _build_dataset_overview_section(pipeline_result),
        _build_validation_section(pipeline_result),
    ]

    if pipeline_result.schema_profile_result is not None:
        sections.append(_build_schema_profile_section(pipeline_result))

    if pipeline_result.data_quality_result is not None:
        sections.append(_build_data_quality_section(pipeline_result))

    if pipeline_result.task_inference_result is not None:
        sections.append(_build_task_inference_section(pipeline_result))

    if pipeline_result.target_profile_result is not None:
        sections.append(_build_target_profile_section(pipeline_result))

    if pipeline_result.model_recommendation_result is not None:
        sections.append(_build_model_recommendation_section(pipeline_result))

    return sections


def _build_dataset_overview_section(
    pipeline_result: IntakePipelineResult,
) -> ReportSection:
    loaded_dataset = pipeline_result.loaded_dataset

    if loaded_dataset is None:
        return ReportSection(
            title="Dataset Overview",
            lines=[
                f"Source file: {pipeline_result.source_file}",
                "Dataset could not be loaded.",
            ],
        )

    return ReportSection(
        title="Dataset Overview",
        lines=[
            f"Source file: {loaded_dataset.file_name}",
            f"File extension: {loaded_dataset.file_extension}",
            f"Rows: {loaded_dataset.row_count}",
            f"Columns: {loaded_dataset.column_count}",
        ],
    )


def _build_validation_section(
    pipeline_result: IntakePipelineResult,
) -> ReportSection:
    validation_result = pipeline_result.validation_result

    if validation_result is None:
        return ReportSection(
            title="Validation",
            lines=["Validation was not completed."],
        )

    lines = [
        f"Valid for downstream analysis: {validation_result.is_valid}",
        f"Validation errors: {len(validation_result.errors)}",
        f"Validation warnings: {len(validation_result.warnings)}",
    ]

    lines.extend(f"Error: {error}" for error in validation_result.errors)
    lines.extend(f"Warning: {warning}" for warning in validation_result.warnings)

    return ReportSection(title="Validation", lines=lines)


def _build_schema_profile_section(
    pipeline_result: IntakePipelineResult,
) -> ReportSection:
    schema_profile = pipeline_result.schema_profile_result
    assert schema_profile is not None

    return ReportSection(
        title="Schema Profile",
        lines=[
            f"Numeric columns: {len(schema_profile.numeric_columns)}",
            f"Categorical columns: {len(schema_profile.categorical_columns)}",
            f"Boolean columns: {len(schema_profile.boolean_columns)}",
            f"Datetime columns: {len(schema_profile.datetime_columns)}",
            f"Text columns: {len(schema_profile.text_columns)}",
            f"Constant columns: {schema_profile.constant_columns or 'None'}",
            f"High-cardinality columns: {schema_profile.high_cardinality_columns or 'None'}",
        ],
    )


def _build_data_quality_section(
    pipeline_result: IntakePipelineResult,
) -> ReportSection:
    data_quality = pipeline_result.data_quality_result
    assert data_quality is not None

    return ReportSection(
        title="Data Quality",
        lines=[
            f"Missing cell ratio: {data_quality.missing_cell_ratio}",
            f"Duplicate row ratio: {data_quality.duplicate_row_ratio}",
            f"Rows with missing values: {data_quality.rows_with_missing_count}",
            f"Fully missing rows: {data_quality.fully_missing_row_count}",
            f"Possible ID columns: {data_quality.possible_id_columns or 'None'}",
            f"Quality issues: {len(data_quality.issues)}",
            f"Critical issues: {data_quality.has_critical_issues}",
        ],
    )


def _build_task_inference_section(
    pipeline_result: IntakePipelineResult,
) -> ReportSection:
    task_inference = pipeline_result.task_inference_result
    assert task_inference is not None

    return ReportSection(
        title="Task Inference",
        lines=[
            f"Status: {task_inference.status}",
            f"Candidate target: {task_inference.candidate_target or 'Unresolved'}",
            f"Task type: {task_inference.task_type or 'Unresolved'}",
            *task_inference.reasoning,
        ],
    )


def _build_target_profile_section(
    pipeline_result: IntakePipelineResult,
) -> ReportSection:
    target_profile = pipeline_result.target_profile_result
    assert target_profile is not None

    lines = [
        f"Target column: {target_profile.target_column}",
        f"Task type: {target_profile.task_type}",
        f"Missing ratio: {target_profile.missing_ratio}",
        f"Unique target values: {target_profile.unique_count}",
        f"Usable for modeling: {target_profile.is_usable_for_modeling}",
    ]

    if target_profile.task_type == "classification":
        lines.extend(
            [
                f"Majority class: {target_profile.majority_class}",
                f"Majority class ratio: {target_profile.majority_class_ratio}",
                f"Minority class: {target_profile.minority_class}",
                f"Minority class ratio: {target_profile.minority_class_ratio}",
                f"Class imbalance ratio: {target_profile.class_imbalance_ratio}",
            ]
        )

    if (
        target_profile.task_type == "regression"
        and target_profile.numeric_summary is not None
    ):
        summary = target_profile.numeric_summary
        lines.extend(
            [
                f"Target mean: {summary.mean}",
                f"Target median: {summary.median}",
                f"Target min: {summary.minimum}",
                f"Target max: {summary.maximum}",
                f"Target skewness: {summary.skewness}",
            ]
        )

    lines.extend(f"Issue: {issue.message}" for issue in target_profile.issues)

    return ReportSection(title="Target Profile", lines=lines)


def _build_model_recommendation_section(
    pipeline_result: IntakePipelineResult,
) -> ReportSection:
    recommendation = pipeline_result.model_recommendation_result
    assert recommendation is not None

    lines = [
        f"Status: {recommendation.status}",
        f"Recommended model: {recommendation.recommended_model or 'None'}",
        f"Confidence: {recommendation.confidence}",
    ]

    recommended_candidate = recommendation.recommended_candidate
    if recommended_candidate is not None:
        lines.append(f"Initial parameters: {recommended_candidate.initial_params}")
        lines.extend(f"Reason: {reason}" for reason in recommended_candidate.reasoning)
        lines.extend(
            f"Concern: {concern}" for concern in recommended_candidate.concerns
        )

    if recommendation.blocked_reasons:
        lines.extend(
            f"Blocked reason: {reason}" for reason in recommendation.blocked_reasons
        )

    return ReportSection(title="Model Recommendation", lines=lines)


def _build_training_approval_question(
    pipeline_result: IntakePipelineResult,
    report_status: AnalysisReportStatus,
) -> str | None:
    """Build the user approval question for Telegram."""
    if report_status != "ready_for_training_approval":
        return None

    recommendation = pipeline_result.model_recommendation_result
    if recommendation is None or recommendation.recommended_model is None:
        return None

    return (
        "Do you want to continue to model training with "
        f"{recommendation.recommended_model}? Reply with: yes / no"
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
