from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ds_engine.workflows.analysis_workflow import run_analysis_workflow
from ds_engine.workflows.approval_workflow import (
    ApprovalWorkflowResult,
    handle_training_approval_workflow,
)


def test_handle_training_approval_workflow_runs_training_for_yes(
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

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_workflow_yes",
    )

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
        training_test_size=0.3,
        training_random_state=42,
    )

    assert isinstance(result, ApprovalWorkflowResult)
    assert result.run_id == "run_approval_workflow_yes"
    assert result.status == "training_package_created"
    assert result.training_was_executed is True
    assert result.training_succeeded is True
    assert result.has_training_package is True
    assert result.training_package_path is not None
    assert result.training_package_path.exists()

    assert result.training_workflow_result is not None
    assert result.training_workflow_result.model_training_result is not None
    assert result.training_workflow_result.best_experiment is not None

    assert result.telegram_message.message_type == "training_completed"
    assert "Training completed." in result.telegram_message.text
    assert "Best model:" in result.telegram_message.text

    assert result.training_package_result is not None
    assert result.training_package_result.package_type == "full_training"
    assert result.telegram_message.message_type == "training_completed"
    assert result.telegram_message.package_path == result.training_package_path
    assert "full training package is ready" in result.telegram_message.text


def test_handle_training_approval_workflow_persists_training_artifacts_for_yes(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": list(range(20)),
            "target": [0] * 10 + [1] * 10,
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_training_artifacts",
    )

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
        persist_training_artifacts=True,
    )

    assert result.training_workflow_result is not None
    assert result.training_workflow_result.artifact_dir is not None
    assert result.training_workflow_result.artifact_dir.exists()

    expected_files = {
        "prepared_modeling_dataset.json",
        "model_training_result.json",
        "training_summary.txt",
        "training_manifest.json",
    }
    assert expected_files.issubset(
        {path.name for path in result.training_workflow_result.artifact_paths}
    )

    assert result.telegram_message.metadata["artifact_dir"] == str(
        result.training_workflow_result.artifact_dir
    )

    assert result.has_training_package is True
    assert result.training_package_path is not None
    assert result.training_package_path.exists()


def test_handle_training_approval_workflow_creates_analysis_package_for_no(
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

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_workflow_no",
    )

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="no",
        package_runs_root=tmp_path / "runs",
    )

    assert result.decision == "no"
    assert result.status == "analysis_package_created"
    assert result.training_was_executed is False
    assert result.training_workflow_result is None
    assert result.has_analysis_package is True
    assert result.package_path is not None
    assert result.package_path.exists()
    assert result.telegram_message.message_type == "package_ready"


def test_handle_training_approval_workflow_rejects_invalid_reply_without_running_any_branch(
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

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_workflow_invalid",
    )

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="tidak",
        package_runs_root=tmp_path / "runs",
    )

    assert result.decision == "invalid"
    assert result.status == "invalid_reply"
    assert result.training_was_executed is False
    assert result.training_succeeded is False
    assert result.has_analysis_package is False
    assert result.telegram_message.message_type == "invalid_reply"
    assert result.telegram_message.expects_reply is True
    assert result.telegram_message.metadata["reply_options"] == ["yes", "no"]


def test_handle_training_approval_workflow_returns_not_ready_when_analysis_requires_target_input(
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

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_workflow_not_ready",
    )

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
    )

    assert result.decision == "yes"
    assert result.status == "not_ready_for_approval"
    assert result.training_was_executed is False
    assert result.training_succeeded is False
    assert result.telegram_message.message_type == "blocked"


def test_handle_training_approval_workflow_returns_training_failed_when_training_branch_fails(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "target_only.csv"
    dataframe.to_csv(file_path, index=False)

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_workflow_training_failed",
    )

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
    )

    assert result.decision == "yes"
    assert result.status == "training_failed"
    assert result.training_was_executed is True
    assert result.training_succeeded is False

    assert result.training_workflow_result is not None
    assert result.training_workflow_result.status == "failed"

    assert result.telegram_message.message_type == "training_failed"
    assert "Training could not be completed." in result.telegram_message.text
    assert "No usable modeling features" in result.telegram_message.text


def test_approval_workflow_result_to_dict_is_json_serializable(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": list(range(20)),
            "target": [0] * 10 + [1] * 10,
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_workflow_json",
    )

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
    )
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["run_id"] == "run_approval_workflow_json"
    assert payload["decision"] == "yes"
    assert payload["training_was_executed"] is True
    assert payload["training_workflow_result"] is not None
    assert payload["has_training_package"] is True
    assert payload["training_package_result"] is not None
    assert payload["training_package_path"] is not None


def test_handle_training_approval_workflow_can_run_training_without_creating_full_package(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": list(range(20)),
            "target": [0] * 10 + [1] * 10,
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_no_training_package",
    )

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
        create_training_package=False,
    )

    assert result.decision == "yes"
    assert result.status in {"training_completed", "training_completed_with_failures"}
    assert result.training_was_executed is True
    assert result.training_succeeded is True
    assert result.has_training_package is False
    assert result.training_package_result is None
    assert result.telegram_message.message_type == "training_completed"


def test_handle_training_approval_workflow_returns_training_package_failed_when_package_creation_fails(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": list(range(20)),
            "target": [0] * 10 + [1] * 10,
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    analysis_result = run_analysis_workflow(
        file_path,
        run_id="run_approval_training_package_failed",
    )

    invalid_runs_root = tmp_path / "runs_as_file"
    invalid_runs_root.write_text("not a directory", encoding="utf-8")

    result = handle_training_approval_workflow(
        analysis_workflow_result=analysis_result,
        reply_text="yes",
        package_runs_root=tmp_path / "runs",
        training_package_runs_root=invalid_runs_root,
    )

    assert result.decision == "yes"
    assert result.status == "training_package_failed"
    assert result.training_was_executed is True
    assert result.training_succeeded is True
    assert result.has_training_package is False
    assert result.training_package_result is not None
    assert result.training_package_result.status == "failed"
    assert result.telegram_message.message_type == "packaging_failed"
    assert "could not create the full training package" in result.telegram_message.text
