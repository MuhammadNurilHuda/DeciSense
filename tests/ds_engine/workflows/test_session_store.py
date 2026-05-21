from __future__ import annotations

import json
from pathlib import Path

import pytest

from ds_engine.workflows.session_store import (
    JsonSessionStore,
    SessionRecord,
    SessionStoreError,
    is_reset_command,
)


def test_json_session_store_upserts_and_reads_session(tmp_path: Path) -> None:
    store = JsonSessionStore(tmp_path / "sessions.json")

    stored_session = store.upsert_session(
        SessionRecord(
            chat_id="chat_123",
            run_id="run_abc",
            state="waiting_training_approval",
            dataset_path="/tmp/dataset.csv",
            last_message_type="training_approval",
            metadata={"source": "telegram"},
        )
    )

    loaded_session = store.get_session("chat_123")

    assert loaded_session == stored_session
    assert loaded_session is not None
    assert loaded_session.chat_id == "chat_123"
    assert loaded_session.run_id == "run_abc"
    assert loaded_session.is_active is True
    assert loaded_session.created_at_utc
    assert loaded_session.updated_at_utc
    assert loaded_session.metadata["source"] == "telegram"


def test_json_session_store_persists_sessions_across_instances(tmp_path: Path) -> None:
    store_path = tmp_path / "sessions.json"

    first_store = JsonSessionStore(store_path)
    first_store.upsert_session(
        SessionRecord(
            chat_id="chat_123",
            run_id="run_abc",
            state="waiting_training_approval",
            dataset_path="/tmp/dataset.csv",
        )
    )

    second_store = JsonSessionStore(store_path)
    loaded_session = second_store.get_session("chat_123")

    assert loaded_session is not None
    assert loaded_session.run_id == "run_abc"
    assert loaded_session.state == "waiting_training_approval"


def test_json_session_store_updates_session_state_without_losing_metadata(
    tmp_path: Path,
) -> None:
    store = JsonSessionStore(tmp_path / "sessions.json")
    store.upsert_session(
        SessionRecord(
            chat_id="chat_123",
            run_id="run_abc",
            state="waiting_training_approval",
            dataset_path="/tmp/dataset.csv",
            metadata={"initial": "value"},
        )
    )

    updated_session = store.update_session_state(
        "chat_123",
        state="training_completed",
        last_message_type="training_completed",
        training_package_path="/tmp/full_training.tar.gz",
        metadata_updates={"final": "value"},
    )

    assert updated_session.state == "training_completed"
    assert updated_session.is_active is False
    assert updated_session.last_message_type == "training_completed"
    assert updated_session.training_package_path == "/tmp/full_training.tar.gz"
    assert updated_session.metadata == {
        "initial": "value",
        "final": "value",
    }


def test_json_session_store_reset_session_removes_state_but_not_artifacts(
    tmp_path: Path,
) -> None:
    store = JsonSessionStore(tmp_path / "sessions.json")
    store.upsert_session(
        SessionRecord(
            chat_id="chat_123",
            run_id="run_abc",
            state="waiting_training_approval",
            dataset_path="/tmp/dataset.csv",
            analysis_package_path="/tmp/analysis.tar.gz",
        )
    )

    removed = store.reset_session("chat_123")

    assert removed is True
    assert store.get_session("chat_123") is None

    removed_again = store.reset_session("chat_123")
    assert removed_again is False


def test_json_session_store_has_active_session_distinguishes_terminal_states(
    tmp_path: Path,
) -> None:
    store = JsonSessionStore(tmp_path / "sessions.json")

    store.upsert_session(
        SessionRecord(
            chat_id="active_chat",
            run_id="run_active",
            state="waiting_training_approval",
            dataset_path="/tmp/active.csv",
        )
    )
    store.upsert_session(
        SessionRecord(
            chat_id="done_chat",
            run_id="run_done",
            state="training_completed",
            dataset_path="/tmp/done.csv",
        )
    )

    assert store.has_active_session("active_chat") is True
    assert store.has_active_session("done_chat") is False
    assert store.has_active_session("missing_chat") is False


def test_json_session_store_raises_for_corrupted_json(tmp_path: Path) -> None:
    store_path = tmp_path / "sessions.json"
    store_path.write_text("{not-valid-json", encoding="utf-8")

    store = JsonSessionStore(store_path)

    with pytest.raises(SessionStoreError, match="invalid JSON"):
        store.get_session("chat_123")


def test_session_record_to_dict_is_json_serializable() -> None:
    session = SessionRecord(
        chat_id="chat_123",
        run_id="run_abc",
        state="waiting_training_approval",
        dataset_path="/tmp/dataset.csv",
        metadata={"reply_options": ["yes", "no"]},
    )

    payload = session.to_dict()
    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["chat_id"] == "chat_123"
    assert payload["metadata"]["reply_options"] == ["yes", "no"]


def test_is_reset_command_accepts_only_exact_reset() -> None:
    assert is_reset_command("reset") is True
    assert is_reset_command(" RESET ") is True
    assert is_reset_command("Reset") is True

    assert is_reset_command("/reset") is False
    assert is_reset_command("please reset") is False
    assert is_reset_command("reset now") is False
    assert is_reset_command("tolong reset") is False
