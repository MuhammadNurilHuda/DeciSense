from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ds_engine.workflows.analysis_workflow import AnalysisWorkflowResult

TelegramMessageType = Literal[
    "training_approval",
    "requires_user_input",
    "blocked",
    "failed",
    "packaging_failed",
    "package_ready",
    "invalid_reply",
    "training_approved",
    "training_completed",
    "training_failed",
    "session_reset",
    "active_session_exists",
    "no_active_session",
]

UserDecision = Literal["yes", "no", "invalid"]


@dataclass(frozen=True)
class TelegramMessage:
    """Telegram-facing message contract for DeciSense workflows."""

    message_type: TelegramMessageType
    text: str
    expects_reply: bool = False
    reply_hint: str | None = None
    package_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_type": self.message_type,
            "text": self.text,
            "expects_reply": self.expects_reply,
            "reply_hint": self.reply_hint,
            "package_path": str(self.package_path) if self.package_path else None,
            "metadata": dict(self.metadata),
        }


def build_analysis_telegram_message(
    workflow_result: AnalysisWorkflowResult,
) -> TelegramMessage:
    """
    Build the main Telegram message after the analysis workflow finishes.

    This is the message used before the user decides whether training should continue.
    """
    if (
        workflow_result.status == "completed"
        and workflow_result.is_ready_for_training_approval
    ):
        return TelegramMessage(
            message_type="training_approval",
            text=workflow_result.to_telegram_message(),
            expects_reply=True,
            reply_hint="yes / no",
            metadata={
                "run_id": workflow_result.run_id,
                "workflow_status": workflow_result.status,
                "report_status": workflow_result.analysis_report.status,
                "reply_options": ["yes", "no"],
                "telegram_ui": {
                    "type": "single_choice",
                    "choices": ["yes", "no"],
                },
            },
        )

    if workflow_result.status == "requires_user_input":
        return TelegramMessage(
            message_type="requires_user_input",
            text=_build_requires_user_input_text(workflow_result),
            expects_reply=True,
            reply_hint="target:<column_name>",
            metadata={
                "run_id": workflow_result.run_id,
                "workflow_status": workflow_result.status,
                "report_status": workflow_result.analysis_report.status,
            },
        )

    if workflow_result.status == "blocked":
        return TelegramMessage(
            message_type="blocked",
            text=_build_blocked_text(workflow_result),
            expects_reply=False,
            metadata={
                "run_id": workflow_result.run_id,
                "workflow_status": workflow_result.status,
                "report_status": workflow_result.analysis_report.status,
            },
        )

    if workflow_result.status == "packaging_failed":
        return TelegramMessage(
            message_type="packaging_failed",
            text=_build_packaging_failed_text(workflow_result),
            expects_reply=False,
            metadata={
                "run_id": workflow_result.run_id,
                "workflow_status": workflow_result.status,
                "report_status": workflow_result.analysis_report.status,
            },
        )

    return TelegramMessage(
        message_type="failed",
        text=_build_failed_text(workflow_result),
        expects_reply=False,
        metadata={
            "run_id": workflow_result.run_id,
            "workflow_status": workflow_result.status,
            "report_status": workflow_result.analysis_report.status,
        },
    )


def build_package_ready_telegram_message(
    *,
    run_id: str,
    package_path: str | Path,
    package_label: str = "analysis-only package",
) -> TelegramMessage:
    """Build a Telegram message for a generated package/tarball."""
    resolved_package_path = Path(package_path)

    return TelegramMessage(
        message_type="package_ready",
        text=(
            f"The {package_label} is ready for run `{run_id}`.\n"
            "You can download the generated `.tar.gz` package."
        ),
        expects_reply=False,
        package_path=resolved_package_path,
        metadata={
            "run_id": run_id,
            "package_label": package_label,
        },
    )


def parse_yes_no_reply(reply_text: str) -> UserDecision:
    """
    Strictly normalize a user approval reply.

    Only "yes" and "no" are accepted. Other values, including localized
    replies such as "ya" or "tidak", are treated as invalid so the UI can
    force a clear binary choice.
    """
    normalized = reply_text.strip().lower()

    if normalized == "yes":
        return "yes"

    if normalized == "no":
        return "no"

    return "invalid"


def _build_requires_user_input_text(
    workflow_result: AnalysisWorkflowResult,
) -> str:
    """Build message when DeciSense needs a target column before model planning."""
    return "\n".join(
        [
            workflow_result.analysis_report.executive_summary,
            "",
            "I need one more detail before I can recommend a model.",
            "Please reply with the target column using this format:",
            "",
            "target:<column_name>",
            "",
            "Example:",
            "target:churn",
        ]
    ).strip()


def _build_blocked_text(
    workflow_result: AnalysisWorkflowResult,
) -> str:
    """Build message when training approval is blocked by analysis findings."""
    lines = [
        workflow_result.analysis_report.executive_summary,
        "",
        "Training is currently blocked because DeciSense found issue(s) that should be resolved first.",
    ]

    important_items = _first_non_empty_items(
        [
            *workflow_result.analysis_report.errors,
            *workflow_result.analysis_report.warnings,
            *workflow_result.errors,
            *workflow_result.warnings,
        ],
        limit=5,
    )

    if important_items:
        lines.extend(["", "Main issue(s):"])
        lines.extend(f"- {item}" for item in important_items)

    return "\n".join(lines).strip()


def _build_failed_text(
    workflow_result: AnalysisWorkflowResult,
) -> str:
    """Build message when the analysis workflow failed."""
    lines = [
        workflow_result.analysis_report.executive_summary,
    ]

    errors = _first_non_empty_items(
        [
            *workflow_result.analysis_report.errors,
            *workflow_result.errors,
        ],
        limit=5,
    )

    if errors:
        lines.extend(["", "Error detail(s):"])
        lines.extend(f"- {error}" for error in errors)

    return "\n".join(lines).strip()


def _build_packaging_failed_text(
    workflow_result: AnalysisWorkflowResult,
) -> str:
    """Build message when analysis finished but package creation failed."""
    lines = [
        "The analysis workflow finished, but DeciSense could not create the package.",
    ]

    errors = _first_non_empty_items(workflow_result.errors, limit=5)
    if errors:
        lines.extend(["", "Packaging error(s):"])
        lines.extend(f"- {error}" for error in errors)

    return "\n".join(lines).strip()


def _first_non_empty_items(
    values: list[str],
    *,
    limit: int,
) -> list[str]:
    """Return the first non-empty unique strings up to a limit."""
    unique_values: list[str] = []
    seen: set[str] = set()

    for value in values:
        cleaned_value = value.strip()
        if not cleaned_value or cleaned_value in seen:
            continue

        seen.add(cleaned_value)
        unique_values.append(cleaned_value)

        if len(unique_values) >= limit:
            break

    return unique_values
