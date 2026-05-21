from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SessionState = Literal[
    "analysis_completed",
    "waiting_training_approval",
    "requires_user_input",
    "training_running",
    "training_completed",
    "analysis_package_created",
    "failed",
]


ACTIVE_SESSION_STATES: set[SessionState] = {
    "analysis_completed",
    "waiting_training_approval",
    "requires_user_input",
    "training_running",
}


class SessionStoreError(Exception):
    """Raised when session state cannot be read or written."""


@dataclass(frozen=True)
class SessionRecord:
    """Persistent DeciSense session state for one Telegram chat."""

    chat_id: str
    run_id: str
    state: SessionState
    dataset_path: str
    last_message_type: str | None = None
    analysis_package_path: str | None = None
    training_package_path: str | None = None
    created_at_utc: str = ""
    updated_at_utc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_SESSION_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "run_id": self.run_id,
            "state": self.state,
            "dataset_path": self.dataset_path,
            "last_message_type": self.last_message_type,
            "analysis_package_path": self.analysis_package_path,
            "training_package_path": self.training_package_path,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SessionRecord:
        return cls(
            chat_id=str(payload["chat_id"]),
            run_id=str(payload["run_id"]),
            state=payload["state"],
            dataset_path=str(payload["dataset_path"]),
            last_message_type=payload.get("last_message_type"),
            analysis_package_path=payload.get("analysis_package_path"),
            training_package_path=payload.get("training_package_path"),
            created_at_utc=str(payload.get("created_at_utc", "")),
            updated_at_utc=str(payload.get("updated_at_utc", "")),
            metadata=dict(payload.get("metadata", {})),
        )


class JsonSessionStore:
    """JSON-backed session store for DeciSense Telegram/OpenClaw state."""

    def __init__(self, store_path: str | Path = "bot_state/sessions.json") -> None:
        self.store_path = Path(store_path).expanduser().resolve()

    def get_session(self, chat_id: str) -> SessionRecord | None:
        """Return a session record for a chat ID, if present."""
        sessions = self._read_sessions()
        return sessions.get(str(chat_id))

    def has_active_session(self, chat_id: str) -> bool:
        """Return True when the chat has an active DeciSense session."""
        session = self.get_session(chat_id)
        return session is not None and session.is_active

    def upsert_session(self, session: SessionRecord) -> SessionRecord:
        """
        Create or update a session record.

        Existing created_at_utc is preserved for the same chat_id.
        updated_at_utc is refreshed on every upsert.
        """
        sessions = self._read_sessions()
        now = _utc_now_iso()
        existing_session = sessions.get(session.chat_id)

        created_at = (
            session.created_at_utc
            or (existing_session.created_at_utc if existing_session else "")
            or now
        )

        stored_session = replace(
            session,
            created_at_utc=created_at,
            updated_at_utc=now,
        )

        sessions[stored_session.chat_id] = stored_session
        self._write_sessions(sessions)

        return stored_session

    def update_session_state(
        self,
        chat_id: str,
        *,
        state: SessionState,
        last_message_type: str | None = None,
        analysis_package_path: str | None = None,
        training_package_path: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> SessionRecord:
        """Update selected fields for an existing session."""
        session = self.get_session(chat_id)
        if session is None:
            raise SessionStoreError(f"No session found for chat_id: {chat_id}")

        metadata = dict(session.metadata)
        if metadata_updates:
            metadata.update(metadata_updates)

        updated_session = replace(
            session,
            state=state,
            last_message_type=(
                session.last_message_type
                if last_message_type is None
                else last_message_type
            ),
            analysis_package_path=(
                session.analysis_package_path
                if analysis_package_path is None
                else analysis_package_path
            ),
            training_package_path=(
                session.training_package_path
                if training_package_path is None
                else training_package_path
            ),
            metadata=metadata,
        )

        return self.upsert_session(updated_session)

    def reset_session(self, chat_id: str) -> bool:
        """
        Clear the active session for a chat ID.

        This removes session state only. It does not delete run artifacts.
        Returns True when a session existed and was removed.
        """
        sessions = self._read_sessions()
        removed = sessions.pop(str(chat_id), None) is not None

        if removed:
            self._write_sessions(sessions)

        return removed

    def list_sessions(self) -> list[SessionRecord]:
        """Return all stored session records."""
        return list(self._read_sessions().values())

    def _read_sessions(self) -> dict[str, SessionRecord]:
        """Read sessions from disk."""
        if not self.store_path.exists():
            return {}

        try:
            payload = json.loads(self.store_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SessionStoreError(
                f"Session store contains invalid JSON: {self.store_path}"
            ) from exc
        except OSError as exc:
            raise SessionStoreError(
                f"Failed to read session store: {self.store_path}"
            ) from exc

        raw_sessions = payload.get("sessions", {})
        if not isinstance(raw_sessions, dict):
            raise SessionStoreError("Session store must contain a 'sessions' object.")

        return {
            chat_id: SessionRecord.from_dict(session_payload)
            for chat_id, session_payload in raw_sessions.items()
        }

    def _write_sessions(self, sessions: dict[str, SessionRecord]) -> None:
        """Write sessions to disk using atomic replacement."""
        payload = {
            "sessions": {
                chat_id: session.to_dict()
                for chat_id, session in sorted(sessions.items())
            }
        }

        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)

            temporary_path = self.store_path.with_suffix(
                f"{self.store_path.suffix}.tmp"
            )
            temporary_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(self.store_path)

        except OSError as exc:
            raise SessionStoreError(
                f"Failed to write session store: {self.store_path}"
            ) from exc


def is_reset_command(message_text: str) -> bool:
    """
    Return True only for the explicit /reset command.

    Plain 'reset' is intentionally not accepted to avoid accidental resets.
    """
    return message_text.strip().lower() == "reset"


def _utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
