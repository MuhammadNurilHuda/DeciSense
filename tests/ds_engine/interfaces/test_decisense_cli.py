from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ds_engine.interfaces.decisense_cli import main


def _read_stdout_json(capsys) -> dict:
    captured = capsys.readouterr()
    return json.loads(captured.out)


def test_decisense_cli_analyze_upload_outputs_training_approval_payload(
    tmp_path: Path,
    capsys,
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

    exit_code = main(
        [
            "analyze-upload",
            "--chat-id",
            "chat_123",
            "--file-path",
            str(file_path),
            "--run-id",
            "run_cli_analysis",
            "--runs-root",
            str(tmp_path / "runs"),
            "--session-store",
            str(tmp_path / "bot_state" / "sessions.json"),
        ]
    )

    payload = _read_stdout_json(capsys)

    assert exit_code == 0
    assert payload["status"] == "analysis_started"
    assert payload["run_id"] == "run_cli_analysis"
    assert payload["session_state"] == "waiting_training_approval"
    assert payload["message_type"] == "training_approval"
    assert payload["expects_reply"] is True
    assert payload["reply_hint"] == "yes / no"
    assert "HistGradientBoostingClassifier" in payload["text"]


def test_decisense_cli_handle_text_no_creates_analysis_package_from_existing_session(
    tmp_path: Path,
    capsys,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    common_args = [
        "--runs-root",
        str(tmp_path / "runs"),
        "--session-store",
        str(tmp_path / "bot_state" / "sessions.json"),
    ]

    analyze_exit_code = main(
        [
            "analyze-upload",
            "--chat-id",
            "chat_123",
            "--file-path",
            str(file_path),
            "--run-id",
            "run_cli_no",
            *common_args,
        ]
    )
    _ = _read_stdout_json(capsys)

    reply_exit_code = main(
        [
            "handle-text",
            "--chat-id",
            "chat_123",
            "--message-text",
            "no",
            *common_args,
        ]
    )
    payload = _read_stdout_json(capsys)

    assert analyze_exit_code == 0
    assert reply_exit_code == 0
    assert payload["status"] == "approval_processed"
    assert payload["session_state"] == "analysis_package_created"
    assert payload["message_type"] == "package_ready"
    assert payload["package_path"] is not None
    assert Path(payload["package_path"]).exists()


def test_decisense_cli_handle_text_reset_clears_existing_session(
    tmp_path: Path,
    capsys,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    common_args = [
        "--runs-root",
        str(tmp_path / "runs"),
        "--session-store",
        str(tmp_path / "bot_state" / "sessions.json"),
    ]

    main(
        [
            "analyze-upload",
            "--chat-id",
            "chat_123",
            "--file-path",
            str(file_path),
            "--run-id",
            "run_cli_reset",
            *common_args,
        ]
    )
    _ = _read_stdout_json(capsys)

    reset_exit_code = main(
        [
            "handle-text",
            "--chat-id",
            "chat_123",
            "--message-text",
            "reset",
            *common_args,
        ]
    )
    reset_payload = _read_stdout_json(capsys)

    no_session_exit_code = main(
        [
            "handle-text",
            "--chat-id",
            "chat_123",
            "--message-text",
            "yes",
            *common_args,
        ]
    )
    no_session_payload = _read_stdout_json(capsys)

    assert reset_exit_code == 0
    assert reset_payload["status"] == "session_reset"
    assert reset_payload["message_type"] == "session_reset"

    assert no_session_exit_code == 0
    assert no_session_payload["status"] == "no_active_session"
    assert no_session_payload["message_type"] == "no_active_session"


def test_decisense_cli_full_json_contains_nested_service_result(
    tmp_path: Path,
    capsys,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    exit_code = main(
        [
            "analyze-upload",
            "--chat-id",
            "chat_123",
            "--file-path",
            str(file_path),
            "--run-id",
            "run_cli_full_json",
            "--runs-root",
            str(tmp_path / "runs"),
            "--session-store",
            str(tmp_path / "bot_state" / "sessions.json"),
            "--full-json",
        ]
    )

    payload = _read_stdout_json(capsys)

    assert exit_code == 0
    assert payload["status"] == "analysis_started"
    assert payload["analysis_workflow_result"] is not None
    assert payload["telegram_message"]["message_type"] == "training_approval"
