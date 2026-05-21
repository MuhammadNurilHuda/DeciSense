from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ds_engine.workflows.analysis_workflow import (
    AnalysisWorkflowResult,
    run_analysis_workflow,
)


def test_run_analysis_workflow_completes_ready_for_training_approval(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [10, 20, 30, 40],
            "feature_b": [1.5, 2.5, 3.5, 4.5],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "customer_data.csv"
    dataframe.to_csv(file_path, index=False)

    result = run_analysis_workflow(
        file_path,
        run_id="run_workflow_ready",
    )

    assert isinstance(result, AnalysisWorkflowResult)
    assert result.run_id == "run_workflow_ready"
    assert result.status == "completed"
    assert result.is_ready_for_training_approval is True
    assert result.has_analysis_package is False
    assert result.analysis_package_result is None

    assert result.pipeline_result.model_recommendation_result is not None
    assert result.pipeline_result.model_recommendation_result.recommended_model == (
        "HistGradientBoostingClassifier"
    )

    telegram_message = result.to_telegram_message()
    assert "HistGradientBoostingClassifier" in telegram_message
    assert "Reply with: yes / no" in telegram_message


def test_run_analysis_workflow_can_create_analysis_only_package(
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

    result = run_analysis_workflow(
        file_path,
        run_id="run_workflow_package",
        package_analysis=True,
        package_runs_root=tmp_path / "runs",
    )

    assert result.status == "completed"
    assert result.has_analysis_package is True
    assert result.package_path is not None
    assert result.package_path.exists()
    assert result.package_path.name == "run_workflow_package_analysis_only.tar.gz"

    assert result.analysis_package_result is not None
    assert result.analysis_package_result.package_type == "analysis_only"


def test_run_analysis_workflow_requires_user_input_when_target_is_missing(
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

    result = run_analysis_workflow(
        file_path,
        run_id="run_workflow_missing_target",
    )

    assert result.status == "requires_user_input"
    assert result.is_ready_for_training_approval is False
    assert result.pipeline_result.requires_user_target_input is True
    assert result.analysis_report.training_approval_question is None
    assert "target column" in result.analysis_report.executive_summary


def test_run_analysis_workflow_blocks_when_target_has_critical_issue(
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

    result = run_analysis_workflow(
        file_path,
        run_id="run_workflow_blocked",
    )

    assert result.status == "blocked"
    assert result.is_ready_for_training_approval is False
    assert result.pipeline_result.target_profile_result is not None
    assert result.pipeline_result.target_profile_result.has_critical_issues is True
    assert result.pipeline_result.model_recommendation_result is not None
    assert result.pipeline_result.model_recommendation_result.status == "blocked"


def test_run_analysis_workflow_returns_failed_for_load_failure(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not a supported dataset", encoding="utf-8")

    result = run_analysis_workflow(
        file_path,
        run_id="run_workflow_failed",
    )

    assert result.status == "failed"
    assert result.is_ready_for_training_approval is False
    assert result.pipeline_result.status == "load_failed"
    assert result.errors
    assert any("Unsupported file extension" in error for error in result.errors)


def test_run_analysis_workflow_returns_packaging_failed_when_package_creation_fails(
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

    invalid_runs_root = tmp_path / "runs_as_file"
    invalid_runs_root.write_text("not a directory", encoding="utf-8")

    result = run_analysis_workflow(
        file_path,
        run_id="run_workflow_packaging_failed",
        package_analysis=True,
        package_runs_root=invalid_runs_root,
    )

    assert result.status == "packaging_failed"
    assert result.is_ready_for_training_approval is True
    assert result.has_analysis_package is False
    assert result.analysis_package_result is not None
    assert result.analysis_package_result.status == "failed"
    assert result.errors
    assert any(
        "Failed to package analysis artifacts" in error for error in result.errors
    )


def test_analysis_workflow_result_to_dict_is_json_serializable(
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

    result = run_analysis_workflow(
        file_path,
        run_id="run_workflow_json",
        package_analysis=True,
        package_runs_root=tmp_path / "runs",
    )
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["run_id"] == "run_workflow_json"
    assert payload["status"] == "completed"
    assert payload["has_analysis_package"] is True
    assert payload["analysis_report"]["status"] == "ready_for_training_approval"
