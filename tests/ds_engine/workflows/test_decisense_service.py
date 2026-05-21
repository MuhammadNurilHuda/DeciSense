from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ds_engine.workflows.decisense_service import (
    DeciSenseService,
    DeciSenseServiceConfig,
    DeciSenseServiceResult,
)


def _build_service(tmp_path: Path) -> DeciSenseService:
    config = DeciSenseServiceConfig(
        runs_root=tmp_path / "runs",
        session_store_path=tmp_path / "bot_state" / "sessions.json",
    )
    return DeciSenseService(config=config)


def test_start_analysis_for_upload_creates_waiting_approval_session(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [10, 20, 30, 40],
            "feature_b": [1.5, 2.5, 3.5, 4.5],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    service = _build_service(tmp_path)

    result = service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=file_path,
        run_id="run_service_analysis",
    )

    assert isinstance(result, DeciSenseServiceResult)
    assert result.status == "analysis_started"
    assert result.run_id == "run_service_analysis"
    assert result.telegram_message.message_type == "training_approval"
    assert result.telegram_message.expects_reply is True

    assert result.session is not None
    assert result.session.chat_id == "chat_123"
    assert result.session.run_id == "run_service_analysis"
    assert result.session.state == "waiting_training_approval"
    assert Path(result.session.dataset_path).exists()
    assert Path(result.session.dataset_path).parent.name == "raw"


def test_start_analysis_for_upload_rejects_new_upload_when_active_session_exists(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    first_file = tmp_path / "first.csv"
    second_file = tmp_path / "second.csv"
    dataframe.to_csv(first_file, index=False)
    dataframe.to_csv(second_file, index=False)

    service = _build_service(tmp_path)
    first_result = service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=first_file,
        run_id="run_first",
    )

    second_result = service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=second_file,
        run_id="run_second",
    )

    assert first_result.status == "analysis_started"
    assert second_result.status == "active_session_exists"
    assert second_result.run_id == "run_first"
    assert second_result.telegram_message.message_type == "active_session_exists"


def test_handle_text_message_reset_clears_active_session(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    service = _build_service(tmp_path)
    service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=file_path,
        run_id="run_reset",
    )

    result = service.handle_text_message(
        chat_id="chat_123",
        message_text="reset",
    )

    assert result.status == "session_reset"
    assert result.telegram_message.message_type == "session_reset"
    assert "Session reset complete" in result.telegram_message.text
    assert service.session_store.get_session("chat_123") is None


def test_handle_text_message_returns_no_active_session_without_session(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path)

    result = service.handle_text_message(
        chat_id="chat_missing",
        message_text="yes",
    )

    assert result.status == "no_active_session"
    assert result.telegram_message.message_type == "no_active_session"
    assert "No active DeciSense session" in result.telegram_message.text


def test_handle_text_message_no_creates_analysis_package_and_finishes_session(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    service = _build_service(tmp_path)
    service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=file_path,
        run_id="run_reply_no",
    )

    result = service.handle_text_message(
        chat_id="chat_123",
        message_text="no",
    )

    assert result.status == "approval_processed"
    assert result.approval_workflow_result is not None
    assert result.approval_workflow_result.status == "analysis_package_created"
    assert result.telegram_message.message_type == "package_ready"
    assert result.package_path is not None
    assert result.package_path.exists()

    session = service.session_store.get_session("chat_123")
    assert session is not None
    assert session.state == "analysis_package_created"
    assert session.is_active is False
    assert session.analysis_package_path == str(result.package_path)


def test_handle_text_message_yes_runs_training_and_creates_training_package(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": list(range(20)),
            "feature_b": [value % 3 for value in range(20)],
            "target": [0] * 10 + [1] * 10,
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    service = _build_service(tmp_path)
    service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=file_path,
        run_id="run_reply_yes",
    )

    result = service.handle_text_message(
        chat_id="chat_123",
        message_text="yes",
    )

    assert result.status == "approval_processed"
    assert result.approval_workflow_result is not None
    assert result.approval_workflow_result.status == "training_package_created"
    assert result.telegram_message.message_type == "training_completed"
    assert result.package_path is not None
    assert result.package_path.exists()

    session = service.session_store.get_session("chat_123")
    assert session is not None
    assert session.state == "training_completed"
    assert session.is_active is False
    assert session.training_package_path == str(result.package_path)


def test_handle_text_message_invalid_reply_keeps_waiting_approval_session(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    service = _build_service(tmp_path)
    service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=file_path,
        run_id="run_invalid_reply",
    )

    result = service.handle_text_message(
        chat_id="chat_123",
        message_text="tidak",
    )

    assert result.status == "approval_processed"
    assert result.approval_workflow_result is not None
    assert result.approval_workflow_result.status == "invalid_reply"
    assert result.telegram_message.message_type == "invalid_reply"

    session = service.session_store.get_session("chat_123")
    assert session is not None
    assert session.state == "waiting_training_approval"
    assert session.is_active is True


def test_handle_text_message_target_column_reply_updates_unresolved_target_session(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [10, 20, 30, 40],
            "feature_b": [1.5, 2.5, 3.5, 4.5],
            "churn_flag": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "custom_target.csv"
    dataframe.to_csv(file_path, index=False)

    service = _build_service(tmp_path)

    upload_result = service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=file_path,
        run_id="run_manual_target",
    )

    assert upload_result.session is not None
    assert upload_result.session.state == "requires_user_input"
    assert upload_result.telegram_message.message_type == "requires_user_input"

    target_result = service.handle_text_message(
        chat_id="chat_123",
        message_text="target:churn_flag",
    )

    assert target_result.status == "target_updated"
    assert target_result.telegram_message.message_type == "training_approval"
    assert target_result.telegram_message.expects_reply is True

    session = service.session_store.get_session("chat_123")
    assert session is not None
    assert session.state == "waiting_training_approval"
    assert session.metadata["manual_target_column"] == "churn_flag"


def test_decisense_service_result_to_dict_is_json_serializable(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    service = _build_service(tmp_path)
    result = service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=file_path,
        run_id="run_service_json",
    )
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["chat_id"] == "chat_123"
    assert payload["status"] == "analysis_started"
    assert payload["run_id"] == "run_service_json"


def test_start_analysis_for_upload_rejects_directory_upload_with_clear_message(
    tmp_path: Path,
) -> None:
    upload_dir = tmp_path / "uploaded_files"
    upload_dir.mkdir()

    service = _build_service(tmp_path)

    result = service.start_analysis_for_upload(
        chat_id="chat_123",
        file_path=upload_dir,
        run_id="run_directory_upload",
    )

    assert result.status == "service_error"
    assert result.telegram_message.message_type == "failed"
    assert result.session is None
    assert result.analysis_workflow_result is None
    assert result.errors
    assert "supports exactly one uploaded tabular file per session" in result.errors[0]
