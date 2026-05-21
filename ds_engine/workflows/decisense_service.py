from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex
from typing import Any, Literal

from ds_engine.pipeline import IntakePipelineConfig
from ds_engine.reporting.telegram_messages import (
    TelegramMessage,
    build_analysis_telegram_message,
)
from ds_engine.workflows.analysis_workflow import (
    AnalysisWorkflowResult,
    run_analysis_workflow,
)
from ds_engine.workflows.approval_workflow import (
    ApprovalWorkflowResult,
    handle_training_approval_workflow,
)
from ds_engine.workflows.session_store import (
    JsonSessionStore,
    SessionRecord,
    SessionState,
    is_reset_command,
)

DeciSenseServiceStatus = Literal[
    "analysis_started",
    "active_session_exists",
    "session_reset",
    "no_active_session",
    "approval_processed",
    "target_updated",
    "requires_user_input",
    "service_error",
]


@dataclass(frozen=True)
class DeciSenseServiceConfig:
    """Configuration for the DeciSense local service layer."""

    runs_root: Path = field(default_factory=lambda: Path("runs"))
    session_store_path: Path = field(
        default_factory=lambda: Path("bot_state/sessions.json")
    )
    pipeline_config: IntakePipelineConfig = field(default_factory=IntakePipelineConfig)
    copy_uploads_to_runs: bool = True
    include_source_data_in_packages: bool = False
    persist_training_artifacts: bool = True
    create_training_package: bool = True


@dataclass(frozen=True)
class DeciSenseServiceResult:
    """Result returned by the service layer for a Telegram/OpenClaw event."""

    chat_id: str
    status: DeciSenseServiceStatus
    telegram_message: TelegramMessage
    run_id: str | None = None
    session: SessionRecord | None = None
    analysis_workflow_result: AnalysisWorkflowResult | None = None
    approval_workflow_result: ApprovalWorkflowResult | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def package_path(self) -> Path | None:
        if self.approval_workflow_result is None:
            return self.telegram_message.package_path
        return self.approval_workflow_result.package_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "status": self.status,
            "run_id": self.run_id,
            "package_path": str(self.package_path) if self.package_path else None,
            "telegram_message": self.telegram_message.to_dict(),
            "session": self.session.to_dict() if self.session is not None else None,
            "analysis_workflow_result": (
                self.analysis_workflow_result.to_dict()
                if self.analysis_workflow_result is not None
                else None
            ),
            "approval_workflow_result": (
                self.approval_workflow_result.to_dict()
                if self.approval_workflow_result is not None
                else None
            ),
            "errors": list(self.errors),
        }


class DeciSenseService:
    """
    Session-aware service layer for DeciSense.

    This class is intentionally framework-agnostic. OpenClaw or Telegram adapters
    can call this service without needing to know DeciSense internals.
    """

    def __init__(
        self,
        *,
        config: DeciSenseServiceConfig | None = None,
        session_store: JsonSessionStore | None = None,
    ) -> None:
        self.config = config or DeciSenseServiceConfig()
        self.session_store = session_store or JsonSessionStore(
            self.config.session_store_path
        )

    def start_analysis_for_upload(
        self,
        *,
        chat_id: str,
        file_path: str | Path,
        run_id: str | None = None,
    ) -> DeciSenseServiceResult:
        """
        Handle a dataset upload event.

        If the chat already has an active session, the upload is rejected until
        the user replies yes/no or sends reset.
        """
        normalized_chat_id = str(chat_id)

        active_session = self.session_store.get_session(normalized_chat_id)
        if active_session is not None and active_session.is_active:
            message = _build_active_session_exists_message(active_session)
            return DeciSenseServiceResult(
                chat_id=normalized_chat_id,
                status="active_session_exists",
                telegram_message=message,
                run_id=active_session.run_id,
                session=active_session,
            )

        actual_run_id = run_id or _generate_run_id()

        try:
            dataset_path = self._prepare_uploaded_dataset(
                file_path=file_path,
                run_id=actual_run_id,
            )
            analysis_result = run_analysis_workflow(
                dataset_path,
                pipeline_config=self.config.pipeline_config,
                run_id=actual_run_id,
            )
            message = build_analysis_telegram_message(analysis_result)

            session = self.session_store.upsert_session(
                SessionRecord(
                    chat_id=normalized_chat_id,
                    run_id=analysis_result.run_id,
                    state=_session_state_from_analysis_result(analysis_result),
                    dataset_path=str(dataset_path),
                    last_message_type=message.message_type,
                    metadata={
                        "workflow_status": analysis_result.status,
                        "report_status": analysis_result.analysis_report.status,
                    },
                )
            )

            return DeciSenseServiceResult(
                chat_id=normalized_chat_id,
                status="analysis_started",
                telegram_message=message,
                run_id=analysis_result.run_id,
                session=session,
                analysis_workflow_result=analysis_result,
            )

        except Exception as exc:
            message = TelegramMessage(
                message_type="failed",
                text=f"DeciSense could not start analysis for this upload.\n\nError: {exc}",
                expects_reply=False,
                metadata={
                    "chat_id": normalized_chat_id,
                    "run_id": actual_run_id,
                },
            )
            return DeciSenseServiceResult(
                chat_id=normalized_chat_id,
                status="service_error",
                telegram_message=message,
                run_id=actual_run_id,
                errors=[str(exc)],
            )

    def handle_text_message(
        self,
        *,
        chat_id: str,
        message_text: str,
    ) -> DeciSenseServiceResult:
        """
        Handle a text event from Telegram/OpenClaw.

        Supported commands/replies:
        - reset
        - yes / no, when waiting for training approval
        - target:<column_name>, when target inference needs user input
        """
        normalized_chat_id = str(chat_id)

        if is_reset_command(message_text):
            removed = self.session_store.reset_session(normalized_chat_id)
            message = _build_session_reset_message(removed=removed)
            return DeciSenseServiceResult(
                chat_id=normalized_chat_id,
                status="session_reset",
                telegram_message=message,
                session=None,
            )

        session = self.session_store.get_session(normalized_chat_id)
        if session is None or not session.is_active:
            message = _build_no_active_session_message()
            return DeciSenseServiceResult(
                chat_id=normalized_chat_id,
                status="no_active_session",
                telegram_message=message,
                session=session,
            )

        if session.state == "waiting_training_approval":
            return self._handle_training_approval_reply(
                chat_id=normalized_chat_id,
                session=session,
                reply_text=message_text,
            )

        if session.state == "requires_user_input":
            return self._handle_target_column_reply(
                chat_id=normalized_chat_id,
                session=session,
                message_text=message_text,
            )

        message = TelegramMessage(
            message_type="blocked",
            text=(
                "This DeciSense session is active, but it is not currently waiting "
                "for a supported text reply. Send `reset` to start over."
            ),
            expects_reply=False,
            metadata={
                "chat_id": normalized_chat_id,
                "run_id": session.run_id,
                "state": session.state,
            },
        )
        return DeciSenseServiceResult(
            chat_id=normalized_chat_id,
            status="requires_user_input",
            telegram_message=message,
            run_id=session.run_id,
            session=session,
        )

    def _handle_training_approval_reply(
        self,
        *,
        chat_id: str,
        session: SessionRecord,
        reply_text: str,
    ) -> DeciSenseServiceResult:
        """Handle yes/no reply for a session waiting for approval."""
        analysis_result = self._rebuild_analysis_workflow_result(session)

        approval_result = handle_training_approval_workflow(
            analysis_workflow_result=analysis_result,
            reply_text=reply_text,
            package_runs_root=self.config.runs_root,
            include_source_data=self.config.include_source_data_in_packages,
            persist_training_artifacts=self.config.persist_training_artifacts,
            training_runs_root=self.config.runs_root,
            create_training_package=self.config.create_training_package,
            training_package_runs_root=self.config.runs_root,
            include_source_data_in_training_package=(
                self.config.include_source_data_in_packages
            ),
        )

        updated_session = self._update_session_after_approval(
            chat_id=chat_id,
            session=session,
            approval_result=approval_result,
        )

        return DeciSenseServiceResult(
            chat_id=chat_id,
            status="approval_processed",
            telegram_message=approval_result.telegram_message,
            run_id=session.run_id,
            session=updated_session,
            analysis_workflow_result=analysis_result,
            approval_workflow_result=approval_result,
        )

    def _handle_target_column_reply(
        self,
        *,
        chat_id: str,
        session: SessionRecord,
        message_text: str,
    ) -> DeciSenseServiceResult:
        """Handle target:<column_name> reply for unresolved target inference."""
        target_column = _parse_target_column_reply(message_text)
        if target_column is None:
            message = _build_target_column_format_message(session)
            return DeciSenseServiceResult(
                chat_id=chat_id,
                status="requires_user_input",
                telegram_message=message,
                run_id=session.run_id,
                session=session,
            )

        target_pipeline_config = replace(
            self.config.pipeline_config,
            target_candidates=(target_column,),
        )
        analysis_result = run_analysis_workflow(
            session.dataset_path,
            pipeline_config=target_pipeline_config,
            run_id=session.run_id,
        )
        message = build_analysis_telegram_message(analysis_result)

        updated_session = self.session_store.upsert_session(
            replace(
                session,
                state=_session_state_from_analysis_result(analysis_result),
                last_message_type=message.message_type,
                metadata={
                    **session.metadata,
                    "manual_target_column": target_column,
                    "workflow_status": analysis_result.status,
                    "report_status": analysis_result.analysis_report.status,
                },
            )
        )

        return DeciSenseServiceResult(
            chat_id=chat_id,
            status="target_updated",
            telegram_message=message,
            run_id=session.run_id,
            session=updated_session,
            analysis_workflow_result=analysis_result,
        )

    def _rebuild_analysis_workflow_result(
        self,
        session: SessionRecord,
    ) -> AnalysisWorkflowResult:
        """Rebuild analysis workflow result from persisted session state."""
        pipeline_config = self.config.pipeline_config
        manual_target_column = session.metadata.get("manual_target_column")

        if isinstance(manual_target_column, str) and manual_target_column.strip():
            pipeline_config = replace(
                pipeline_config,
                target_candidates=(manual_target_column.strip(),),
            )

        return run_analysis_workflow(
            session.dataset_path,
            pipeline_config=pipeline_config,
            run_id=session.run_id,
        )

    def _prepare_uploaded_dataset(
        self,
        *,
        file_path: str | Path,
        run_id: str,
    ) -> Path:
        """Copy one uploaded dataset into runs/<run_id>/raw/ when configured."""
        source_path = Path(file_path).expanduser().resolve()

        if not source_path.exists():
            raise FileNotFoundError(f"Uploaded dataset file not found: {source_path}")

        if not source_path.is_file():
            raise ValueError(
                "DeciSense MVP 0 supports exactly one uploaded tabular file per session. "
                f"Expected a file path, but got: {source_path}"
            )

        if not self.config.copy_uploads_to_runs:
            return source_path

        raw_dir = Path(self.config.runs_root).expanduser().resolve() / run_id / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        destination_path = raw_dir / source_path.name

        if source_path != destination_path:
            shutil.copy2(source_path, destination_path)

        return destination_path

    def _update_session_after_approval(
        self,
        *,
        chat_id: str,
        session: SessionRecord,
        approval_result: ApprovalWorkflowResult,
    ) -> SessionRecord:
        """Persist session state after handling yes/no approval."""
        if approval_result.status == "invalid_reply":
            return self.session_store.update_session_state(
                chat_id,
                state="waiting_training_approval",
                last_message_type=approval_result.telegram_message.message_type,
            )

        if approval_result.status == "analysis_package_created":
            return self.session_store.update_session_state(
                chat_id,
                state="analysis_package_created",
                last_message_type=approval_result.telegram_message.message_type,
                analysis_package_path=(
                    str(approval_result.package_path)
                    if approval_result.package_path
                    else None
                ),
            )

        if approval_result.status == "training_package_created":
            return self.session_store.update_session_state(
                chat_id,
                state="training_completed",
                last_message_type=approval_result.telegram_message.message_type,
                training_package_path=(
                    str(approval_result.training_package_path)
                    if approval_result.training_package_path
                    else None
                ),
            )

        if approval_result.status in {
            "training_completed",
            "training_completed_with_failures",
        }:
            return self.session_store.update_session_state(
                chat_id,
                state="training_completed",
                last_message_type=approval_result.telegram_message.message_type,
            )

        return self.session_store.update_session_state(
            chat_id,
            state="failed",
            last_message_type=approval_result.telegram_message.message_type,
            metadata_updates={
                "approval_workflow_status": approval_result.status,
            },
        )


def _session_state_from_analysis_result(
    analysis_result: AnalysisWorkflowResult,
) -> SessionState:
    """Map analysis workflow status to persistent session state."""
    if (
        analysis_result.status == "completed"
        and analysis_result.is_ready_for_training_approval
    ):
        return "waiting_training_approval"

    if analysis_result.status == "requires_user_input":
        return "requires_user_input"

    if analysis_result.status == "completed":
        return "analysis_completed"

    return "failed"


def _build_active_session_exists_message(session: SessionRecord) -> TelegramMessage:
    """Build message when a new upload arrives while a session is active."""
    if session.state == "waiting_training_approval":
        reply_hint = "yes / no"
        extra = "Please reply `yes` or `no`, or send `reset` to start over."
    elif session.state == "requires_user_input":
        reply_hint = "target:<column_name>"
        extra = (
            "Please reply with `target:<column_name>`, or send `reset` to start over."
        )
    else:
        reply_hint = "reset"
        extra = "Please send `reset` to start over."

    return TelegramMessage(
        message_type="active_session_exists",
        text=(
            "You already have an active DeciSense session.\n\n"
            f"Run ID: {session.run_id}\n"
            f"Current state: {session.state}\n\n"
            f"{extra}"
        ),
        expects_reply=True,
        reply_hint=reply_hint,
        metadata={
            "run_id": session.run_id,
            "state": session.state,
        },
    )


def _build_session_reset_message(*, removed: bool) -> TelegramMessage:
    """Build message after reset command."""
    if removed:
        text = "Session reset complete.\n\nYou can upload a new tabular dataset now."
    else:
        text = (
            "No active DeciSense session was found.\n\n"
            "You can upload a tabular dataset to start."
        )

    return TelegramMessage(
        message_type="session_reset",
        text=text,
        expects_reply=False,
        metadata={"removed_existing_session": removed},
    )


def _build_no_active_session_message() -> TelegramMessage:
    """Build message when text arrives without an active session."""
    return TelegramMessage(
        message_type="no_active_session",
        text=(
            "No active DeciSense session found.\n\n"
            "Please upload a tabular dataset first, or send `reset` to clear any old state."
        ),
        expects_reply=False,
    )


def _build_target_column_format_message(session: SessionRecord) -> TelegramMessage:
    """Build message when target clarification format is invalid."""
    return TelegramMessage(
        message_type="requires_user_input",
        text=(
            "I need the target column before model planning can continue.\n\n"
            "Please reply using this exact format:\n\n"
            "target:<column_name>\n\n"
            "Example:\n"
            "target:churn"
        ),
        expects_reply=True,
        reply_hint="target:<column_name>",
        metadata={
            "run_id": session.run_id,
            "state": session.state,
        },
    )


def _parse_target_column_reply(message_text: str) -> str | None:
    """Parse target:<column_name> replies."""
    stripped = message_text.strip()
    if not stripped.lower().startswith("target:"):
        return None

    target_column = stripped.split(":", 1)[1].strip()
    return target_column or None


def _generate_run_id() -> str:
    """Generate a run ID for service-created sessions."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{token_hex(3)}"
