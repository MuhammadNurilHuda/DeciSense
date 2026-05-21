from __future__ import annotations

from pathlib import Path

import pandas as pd

from ds_engine.reporting.telegram_messages import (
    TelegramMessage,
    build_analysis_telegram_message,
    build_package_ready_telegram_message,
    parse_yes_no_reply,
)
from ds_engine.workflows.analysis_workflow import run_analysis_workflow


def test_build_analysis_telegram_message_returns_training_approval_message(
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

    workflow_result = run_analysis_workflow(
        file_path,
        run_id="run_telegram_ready",
    )

    message = build_analysis_telegram_message(workflow_result)

    assert isinstance(message, TelegramMessage)
    assert message.message_type == "training_approval"
    assert message.expects_reply is True
    assert message.reply_hint == "yes / no"
    assert "HistGradientBoostingClassifier" in message.text
    assert "Reply with: yes / no" in message.text
    assert message.metadata["run_id"] == "run_telegram_ready"
    assert message.metadata["reply_options"] == ["yes", "no"]
    assert message.metadata["telegram_ui"]["type"] == "single_choice"


def test_build_analysis_telegram_message_requests_target_when_target_is_missing(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [10, 20, 30],
        }
    )
    file_path = tmp_path / "no_target.csv"
    dataframe.to_csv(file_path, index=False)

    workflow_result = run_analysis_workflow(
        file_path,
        run_id="run_telegram_missing_target",
    )

    message = build_analysis_telegram_message(workflow_result)

    assert message.message_type == "requires_user_input"
    assert message.expects_reply is True
    assert message.reply_hint == "target:<column_name>"
    assert "target:<column_name>" in message.text
    assert "target:churn" in message.text


def test_build_analysis_telegram_message_blocks_when_target_is_not_modelable(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "target": [1, 1, 1, 1],
        }
    )
    file_path = tmp_path / "single_class.csv"
    dataframe.to_csv(file_path, index=False)

    workflow_result = run_analysis_workflow(
        file_path,
        run_id="run_telegram_blocked",
    )

    message = build_analysis_telegram_message(workflow_result)

    assert message.message_type == "blocked"
    assert message.expects_reply is False
    assert "Training is currently blocked" in message.text
    assert "only one class" in message.text


def test_build_analysis_telegram_message_returns_failed_message_for_load_failure(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not a dataset", encoding="utf-8")

    workflow_result = run_analysis_workflow(
        file_path,
        run_id="run_telegram_failed",
    )

    message = build_analysis_telegram_message(workflow_result)

    assert message.message_type == "failed"
    assert message.expects_reply is False
    assert "could not be completed" in message.text
    assert "Unsupported file extension" in message.text


def test_build_package_ready_telegram_message_contains_package_path_metadata(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "run_123_analysis_only.tar.gz"

    message = build_package_ready_telegram_message(
        run_id="run_123",
        package_path=package_path,
    )

    payload = message.to_dict()

    assert message.message_type == "package_ready"
    assert message.expects_reply is False
    assert message.package_path == package_path
    assert payload["package_path"] == str(package_path)
    assert payload["metadata"]["run_id"] == "run_123"


def test_parse_yes_no_reply_accepts_only_strict_yes_no() -> None:
    assert parse_yes_no_reply("yes") == "yes"
    assert parse_yes_no_reply("YES") == "yes"
    assert parse_yes_no_reply(" yes ") == "yes"

    assert parse_yes_no_reply("no") == "no"
    assert parse_yes_no_reply("NO") == "no"
    assert parse_yes_no_reply(" no ") == "no"

    assert parse_yes_no_reply("ya") == "invalid"
    assert parse_yes_no_reply("YA") == "invalid"
    assert parse_yes_no_reply("tidak") == "invalid"
    assert parse_yes_no_reply("lanjut") == "invalid"
    assert parse_yes_no_reply("batal") == "invalid"
    assert parse_yes_no_reply("maybe") == "invalid"
