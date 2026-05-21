from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ds_engine.reporting.telegram_messages import TelegramMessage
from ds_engine.reporting.training_packager import (
    TrainingPackageResult,
    package_training_artifacts,
)
from ds_engine.workflows.analysis_workflow import AnalysisWorkflowResult
from ds_engine.workflows.approval_state import (
    ApprovalDecision,
    ApprovalReplyResult,
    handle_training_approval_reply,
)
from ds_engine.workflows.training_workflow import (
    TrainingWorkflowResult,
    run_training_workflow,
)

ApprovalWorkflowStatus = Literal[
    "training_completed",
    "training_completed_with_failures",
    "training_failed",
    "training_package_created",
    "training_package_failed",
    "analysis_package_created",
    "analysis_package_failed",
    "invalid_reply",
    "not_ready_for_approval",
]


@dataclass(frozen=True)
class ApprovalWorkflowResult:
    """Final result after handling a user training-approval reply."""

    run_id: str
    decision: ApprovalDecision
    status: ApprovalWorkflowStatus
    approval_reply_result: ApprovalReplyResult
    telegram_message: TelegramMessage
    training_workflow_result: TrainingWorkflowResult | None = None
    training_package_result: TrainingPackageResult | None = None

    @property
    def training_was_executed(self) -> bool:
        return self.training_workflow_result is not None

    @property
    def training_succeeded(self) -> bool:
        return (
            self.training_workflow_result is not None
            and self.training_workflow_result.is_success
        )

    @property
    def has_analysis_package(self) -> bool:
        return self.approval_reply_result.has_analysis_package

    @property
    def package_path(self) -> Path | None:
        if self.training_package_path is not None:
            return self.training_package_path
        return self.approval_reply_result.package_path

    @property
    def has_training_package(self) -> bool:
        return (
            self.training_package_result is not None
            and self.training_package_result.is_success
        )

    @property
    def training_package_path(self) -> Path | None:
        if self.training_package_result is None:
            return None
        return self.training_package_result.package_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decision": self.decision,
            "status": self.status,
            "training_was_executed": self.training_was_executed,
            "training_succeeded": self.training_succeeded,
            "has_analysis_package": self.has_analysis_package,
            "package_path": str(self.package_path) if self.package_path else None,
            "telegram_message": self.telegram_message.to_dict(),
            "approval_reply_result": self.approval_reply_result.to_dict(),
            "training_workflow_result": (
                self.training_workflow_result.to_dict()
                if self.training_workflow_result is not None
                else None
            ),
            "has_training_package": self.has_training_package,
            "training_package_path": (
                str(self.training_package_path) if self.training_package_path else None
            ),
            "training_package_result": (
                self.training_package_result.to_dict()
                if self.training_package_result is not None
                else None
            ),
        }


def handle_training_approval_workflow(
    *,
    analysis_workflow_result: AnalysisWorkflowResult,
    reply_text: str,
    package_runs_root: str | Path = "runs",
    include_source_data: bool = False,
    training_test_size: float | int = 0.2,
    training_random_state: int = 42,
    training_max_candidates: int | None = None,
    persist_training_artifacts: bool = True,
    training_runs_root: str | Path | None = None,
    create_training_package: bool = True,
    training_package_runs_root: str | Path | None = None,
    include_source_data_in_training_package: bool = False,
    include_fitted_pipelines_in_training_package: bool = True,
) -> ApprovalWorkflowResult:
    """
    Handle the user approval reply and execute the selected branch.

    Behavior:
    - "yes" -> run training workflow
    - "no" -> create analysis-only package
    - anything else -> invalid reply, ask user to choose yes/no
    """
    approval_reply_result = handle_training_approval_reply(
        workflow_result=analysis_workflow_result,
        reply_text=reply_text,
        package_runs_root=package_runs_root,
        include_source_data=include_source_data,
    )

    if approval_reply_result.status != "training_approved":
        return ApprovalWorkflowResult(
            run_id=analysis_workflow_result.run_id,
            decision=approval_reply_result.decision,
            status=_map_non_training_status(approval_reply_result.status),
            approval_reply_result=approval_reply_result,
            telegram_message=approval_reply_result.telegram_message,
            training_workflow_result=None,
        )

    training_result = run_training_workflow(
        analysis_workflow_result,
        test_size=training_test_size,
        random_state=training_random_state,
        max_candidates=training_max_candidates,
        persist_artifacts=persist_training_artifacts,
        runs_root=training_runs_root or package_runs_root,
    )

    if not training_result.is_success:
        return ApprovalWorkflowResult(
            run_id=analysis_workflow_result.run_id,
            decision="yes",
            status=_map_training_status(training_result),
            approval_reply_result=approval_reply_result,
            telegram_message=_build_training_result_message(training_result),
            training_workflow_result=training_result,
            training_package_result=None,
        )

    if not create_training_package:
        return ApprovalWorkflowResult(
            run_id=analysis_workflow_result.run_id,
            decision="yes",
            status=_map_training_status(training_result),
            approval_reply_result=approval_reply_result,
            telegram_message=_build_training_result_message(training_result),
            training_workflow_result=training_result,
            training_package_result=None,
        )

    training_package_result = package_training_artifacts(
        analysis_workflow_result=analysis_workflow_result,
        training_workflow_result=training_result,
        runs_root=(
            training_package_runs_root or training_runs_root or package_runs_root
        ),
        include_source_data=include_source_data_in_training_package,
        include_fitted_pipelines=include_fitted_pipelines_in_training_package,
    )

    if training_package_result.is_success:
        return ApprovalWorkflowResult(
            run_id=analysis_workflow_result.run_id,
            decision="yes",
            status="training_package_created",
            approval_reply_result=approval_reply_result,
            telegram_message=_build_training_package_ready_message(
                training_result=training_result,
                training_package_result=training_package_result,
            ),
            training_workflow_result=training_result,
            training_package_result=training_package_result,
        )

    return ApprovalWorkflowResult(
        run_id=analysis_workflow_result.run_id,
        decision="yes",
        status="training_package_failed",
        approval_reply_result=approval_reply_result,
        telegram_message=_build_training_package_failed_message(
            training_result=training_result,
            training_package_result=training_package_result,
        ),
        training_workflow_result=training_result,
        training_package_result=training_package_result,
    )


def _map_non_training_status(status: str) -> ApprovalWorkflowStatus:
    """Map approval-state status into final approval-workflow status."""
    if status == "analysis_package_created":
        return "analysis_package_created"

    if status == "analysis_package_failed":
        return "analysis_package_failed"

    if status == "invalid_reply":
        return "invalid_reply"

    return "not_ready_for_approval"


def _map_training_status(
    training_result: TrainingWorkflowResult,
) -> ApprovalWorkflowStatus:
    """Map training workflow status into approval workflow status."""
    if training_result.status == "completed":
        return "training_completed"

    if training_result.status == "completed_with_failures":
        return "training_completed_with_failures"

    return "training_failed"


def _build_training_result_message(
    training_result: TrainingWorkflowResult,
) -> TelegramMessage:
    """Build Telegram message after the training branch finishes."""
    if training_result.is_success:
        return TelegramMessage(
            message_type="training_completed",
            text=training_result.to_telegram_summary(),
            expects_reply=False,
            metadata={
                "run_id": training_result.run_id,
                "training_status": training_result.status,
                "artifact_dir": (
                    str(training_result.artifact_dir)
                    if training_result.artifact_dir
                    else None
                ),
                "artifact_paths": [
                    str(path) for path in training_result.artifact_paths
                ],
            },
        )

    return TelegramMessage(
        message_type="training_failed",
        text=training_result.to_telegram_summary(),
        expects_reply=False,
        metadata={
            "run_id": training_result.run_id,
            "training_status": training_result.status,
            "errors": list(training_result.errors),
        },
    )


def _build_training_package_ready_message(
    *,
    training_result: TrainingWorkflowResult,
    training_package_result: TrainingPackageResult,
) -> TelegramMessage:
    """Build Telegram message when training and full package creation succeed."""
    return TelegramMessage(
        message_type="training_completed",
        text=(
            f"{training_result.to_telegram_summary()}\n\n"
            "The full training package is ready. It contains the analysis report, "
            "training result, model artifacts, and package manifest."
        ),
        expects_reply=False,
        package_path=training_package_result.package_path,
        metadata={
            "run_id": training_result.run_id,
            "training_status": training_result.status,
            "package_type": training_package_result.package_type,
            "package_path": (
                str(training_package_result.package_path)
                if training_package_result.package_path
                else None
            ),
            "artifact_dir": (
                str(training_result.artifact_dir)
                if training_result.artifact_dir
                else None
            ),
            "artifact_paths": [str(path) for path in training_result.artifact_paths],
        },
    )


def _build_training_package_failed_message(
    *,
    training_result: TrainingWorkflowResult,
    training_package_result: TrainingPackageResult,
) -> TelegramMessage:
    """Build Telegram message when training succeeds but package creation fails."""
    error_lines = "\n".join(
        f"- {error}" for error in training_package_result.errors[:5]
    )

    return TelegramMessage(
        message_type="packaging_failed",
        text=(
            f"{training_result.to_telegram_summary()}\n\n"
            "Training finished, but DeciSense could not create the full training package.\n\n"
            f"Packaging error(s):\n{error_lines}"
        ),
        expects_reply=False,
        metadata={
            "run_id": training_result.run_id,
            "training_status": training_result.status,
            "package_status": training_package_result.status,
            "errors": list(training_package_result.errors),
        },
    )
