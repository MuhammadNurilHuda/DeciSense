from __future__ import annotations

from pathlib import Path

import pandas as pd

from ds_engine.workflows.analysis_workflow import run_analysis_workflow
from ds_engine.workflows.approval_state import (
    ApprovalReplyResult,
    handle_training_approval_reply,
)


def test_handle_training_approval_reply_approves_training_for_yes(
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
        run_id="run_approval_yes",
    )

    result = handle_training_approval_reply(
        workflow_result=workflow_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
    )

    assert isinstance(result, ApprovalReplyResult)
    assert result.run_id == "run_approval_yes"
    assert result.decision == "yes"
    assert result.status == "training_approved"
    assert result.should_continue_to_training is True
    assert result.has_analysis_package is False
    assert result.telegram_message.message_type == "training_approved"
    assert "HistGradientBoostingClassifier" in result.telegram_message.text


def test_handle_training_approval_reply_creates_analysis_package_for_no(
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

    workflow_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_no",
    )

    result = handle_training_approval_reply(
        workflow_result=workflow_result,
        reply_text="no",
        package_runs_root=tmp_path / "runs",
    )

    assert result.decision == "no"
    assert result.status == "analysis_package_created"
    assert result.should_continue_to_training is False
    assert result.has_analysis_package is True
    assert result.package_path is not None
    assert result.package_path.exists()
    assert result.telegram_message.message_type == "package_ready"
    assert result.telegram_message.package_path == result.package_path
    assert "stop before training" in result.telegram_message.text


def test_handle_training_approval_reply_rejects_localized_reply_tidak(
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

    workflow_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_invalid",
    )

    result = handle_training_approval_reply(
        workflow_result=workflow_result,
        reply_text="tidak",
        package_runs_root=tmp_path / "runs",
    )

    assert result.decision == "invalid"
    assert result.status == "invalid_reply"
    assert result.should_continue_to_training is False
    assert result.has_analysis_package is False
    assert result.telegram_message.message_type == "invalid_reply"
    assert result.telegram_message.expects_reply is True
    assert result.telegram_message.reply_hint == "yes / no"
    assert result.telegram_message.metadata["reply_options"] == ["yes", "no"]


def test_handle_training_approval_reply_returns_not_ready_when_workflow_requires_target_input(
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
        run_id="run_approval_not_ready",
    )

    result = handle_training_approval_reply(
        workflow_result=workflow_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
    )

    assert result.decision == "yes"
    assert result.status == "not_ready_for_approval"
    assert result.should_continue_to_training is False
    assert result.has_analysis_package is False
    assert result.telegram_message.message_type == "blocked"
    assert "not ready for training approval" in result.telegram_message.text
