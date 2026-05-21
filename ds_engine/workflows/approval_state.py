from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ds_engine.reporting.packager import (
    AnalysisPackageResult,
    package_analysis_artifacts,
)
from ds_engine.reporting.telegram_messages import (
    TelegramMessage,
    parse_yes_no_reply,
)
from ds_engine.workflows.analysis_workflow import AnalysisWorkflowResult

ApprovalDecision = Literal["yes", "no", "invalid"]
ApprovalHandlingStatus = Literal[
    "training_approved",
    "analysis_package_created",
    "analysis_package_failed",
    "invalid_reply",
    "not_ready_for_approval",
]


@dataclass(frozen=True)
class ApprovalReplyResult:
    """Result of handling a user's yes/no training approval reply."""

    run_id: str
    decision: ApprovalDecision
    status: ApprovalHandlingStatus
    telegram_message: TelegramMessage
    analysis_package_result: AnalysisPackageResult | None = None
    errors: list[str] | None = None

    @property
    def should_continue_to_training(self) -> bool:
        return self.status == "training_approved"

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
            "decision": self.decision,
            "status": self.status,
            "should_continue_to_training": self.should_continue_to_training,
            "has_analysis_package": self.has_analysis_package,
            "package_path": str(self.package_path) if self.package_path else None,
            "telegram_message": self.telegram_message.to_dict(),
            "analysis_package_result": (
                self.analysis_package_result.to_dict()
                if self.analysis_package_result is not None
                else None
            ),
            "errors": list(self.errors or []),
        }


def handle_training_approval_reply(
    *,
    workflow_result: AnalysisWorkflowResult,
    reply_text: str,
    package_runs_root: str | Path = "runs",
    include_source_data: bool = False,
) -> ApprovalReplyResult:
    """
    Handle a strict yes/no approval reply.

    Behavior:
    - "yes" -> approve continuing to training
    - "no" -> create analysis-only package
    - anything else -> invalid reply message
    """
    decision = parse_yes_no_reply(reply_text)

    if decision == "invalid":
        return ApprovalReplyResult(
            run_id=workflow_result.run_id,
            decision="invalid",
            status="invalid_reply",
            telegram_message=_build_invalid_reply_message(workflow_result.run_id),
            analysis_package_result=None,
            errors=[],
        )

    if not workflow_result.is_ready_for_training_approval:
        return ApprovalReplyResult(
            run_id=workflow_result.run_id,
            decision=decision,
            status="not_ready_for_approval",
            telegram_message=_build_not_ready_for_approval_message(workflow_result),
            analysis_package_result=None,
            errors=[],
        )

    if decision == "yes":
        return ApprovalReplyResult(
            run_id=workflow_result.run_id,
            decision="yes",
            status="training_approved",
            telegram_message=_build_training_approved_message(workflow_result),
            analysis_package_result=None,
            errors=[],
        )

    package_result = package_analysis_artifacts(
        workflow_result.pipeline_result,
        analysis_report=workflow_result.analysis_report,
        runs_root=package_runs_root,
        include_source_data=include_source_data,
    )

    if package_result.is_success and package_result.package_path is not None:
        return ApprovalReplyResult(
            run_id=workflow_result.run_id,
            decision="no",
            status="analysis_package_created",
            telegram_message=_build_analysis_package_created_message(
                run_id=workflow_result.run_id,
                package_path=package_result.package_path,
            ),
            analysis_package_result=package_result,
            errors=[],
        )

    return ApprovalReplyResult(
        run_id=workflow_result.run_id,
        decision="no",
        status="analysis_package_failed",
        telegram_message=_build_analysis_package_failed_message(
            workflow_result.run_id,
            package_result.errors,
        ),
        analysis_package_result=package_result,
        errors=package_result.errors,
    )


def _build_invalid_reply_message(run_id: str) -> TelegramMessage:
    """Build message for invalid approval replies."""
    return TelegramMessage(
        message_type="invalid_reply",
        text=(
            "I can only accept one of these replies for this step:\n\n"
            "- yes\n"
            "- no\n\n"
            "Please choose `yes` to continue training, or `no` to stop and receive "
            "the analysis-only package."
        ),
        expects_reply=True,
        reply_hint="yes / no",
        metadata={
            "run_id": run_id,
            "reply_options": ["yes", "no"],
            "telegram_ui": {
                "type": "single_choice",
                "choices": ["yes", "no"],
            },
        },
    )


def _build_not_ready_for_approval_message(
    workflow_result: AnalysisWorkflowResult,
) -> TelegramMessage:
    """Build message when approval is received but the workflow is not approval-ready."""
    return TelegramMessage(
        message_type="blocked",
        text=(
            "This run is not ready for training approval yet.\n\n"
            f"Current workflow status: {workflow_result.status}\n"
            f"Current report status: {workflow_result.analysis_report.status}\n\n"
            "Please resolve the analysis issue first before choosing yes/no."
        ),
        expects_reply=False,
        metadata={
            "run_id": workflow_result.run_id,
            "workflow_status": workflow_result.status,
            "report_status": workflow_result.analysis_report.status,
        },
    )


def _build_training_approved_message(
    workflow_result: AnalysisWorkflowResult,
) -> TelegramMessage:
    """Build message when user approves continuing to training."""
    recommended_model = "the recommended model"
    recommendation = workflow_result.pipeline_result.model_recommendation_result

    if recommendation is not None and recommendation.recommended_model is not None:
        recommended_model = recommendation.recommended_model

    return TelegramMessage(
        message_type="training_approved",
        text=(
            "Training approval received.\n\n"
            f"I will continue with `{recommended_model}` as the first recommended "
            "model setup."
        ),
        expects_reply=False,
        metadata={
            "run_id": workflow_result.run_id,
            "recommended_model": recommended_model,
        },
    )


def _build_analysis_package_created_message(
    *,
    run_id: str,
    package_path: Path,
) -> TelegramMessage:
    """Build message when user declines training and analysis-only package is ready."""
    return TelegramMessage(
        message_type="package_ready",
        text=(
            "Understood. I will stop before training.\n\n"
            "The analysis-only package is ready. It contains the analysis report, "
            "pipeline result, and package manifest."
        ),
        expects_reply=False,
        package_path=package_path,
        metadata={
            "run_id": run_id,
            "package_label": "analysis-only package",
        },
    )


def _build_analysis_package_failed_message(
    run_id: str,
    errors: list[str],
) -> TelegramMessage:
    """Build message when user declines training but packaging fails."""
    error_lines = "\n".join(f"- {error}" for error in errors[:5])

    return TelegramMessage(
        message_type="packaging_failed",
        text=(
            "Understood. I stopped before training, but I could not create the "
            "analysis-only package.\n\n"
            f"Error detail(s):\n{error_lines}"
        ),
        expects_reply=False,
        metadata={
            "run_id": run_id,
        },
    )
