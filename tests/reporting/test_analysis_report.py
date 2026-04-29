from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ds_engine.pipeline import IntakePipelineConfig, run_intake_pipeline
from ds_engine.reporting.analysis_report import (
    AnalysisReportResult,
    create_analysis_report,
)


def test_create_analysis_report_returns_training_approval_report_for_ready_pipeline(
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

    pipeline_result = run_intake_pipeline(
        file_path,
        run_id="run_ready_report",
    )

    report = create_analysis_report(pipeline_result)

    assert isinstance(report, AnalysisReportResult)
    assert report.run_id == "run_ready_report"
    assert report.status == "ready_for_training_approval"
    assert report.is_ready_for_training_approval is True
    assert "HistGradientBoostingClassifier" in report.executive_summary
    assert report.training_approval_question is not None
    assert "yes / no" in report.training_approval_question

    markdown = report.to_markdown()
    assert "# DeciSense Analysis Report - run_ready_report" in markdown
    assert "## Dataset Overview" in markdown
    assert "## Model Recommendation" in markdown
    assert "HistGradientBoostingClassifier" in markdown

    telegram_message = report.to_telegram_message()
    assert "Model recommendation:" in telegram_message
    assert "HistGradientBoostingClassifier" in telegram_message
    assert "Reply with: yes / no" in telegram_message


def test_create_analysis_report_requires_user_input_when_target_is_unresolved(
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

    pipeline_result = run_intake_pipeline(
        file_path,
        run_id="run_missing_target_report",
    )

    report = create_analysis_report(pipeline_result)

    assert report.status == "requires_user_input"
    assert report.is_ready_for_training_approval is False
    assert report.training_approval_question is None
    assert "could not determine the target column" in report.executive_summary

    markdown = report.to_markdown()
    assert "Task type: Unresolved" in markdown
    assert "Candidate target: Unresolved" in markdown


def test_create_analysis_report_blocks_training_when_model_recommendation_is_blocked(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [10, 20, 30, 40],
            "target": [1, 1, 1, 1],
        }
    )
    file_path = tmp_path / "single_class.csv"
    dataframe.to_csv(file_path, index=False)

    pipeline_result = run_intake_pipeline(
        file_path,
        run_id="run_blocked_report",
    )

    report = create_analysis_report(pipeline_result)

    assert report.status == "blocked"
    assert report.is_ready_for_training_approval is False
    assert report.training_approval_question is None
    assert "training is not recommended yet" in report.executive_summary.lower()

    markdown = report.to_markdown()
    assert "Status: blocked" in markdown
    assert "Blocked reason:" in markdown
    assert "only one class" in markdown


def test_create_analysis_report_returns_failed_report_for_load_failure(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("not a supported tabular file", encoding="utf-8")

    pipeline_result = run_intake_pipeline(
        file_path,
        run_id="run_failed_report",
    )

    report = create_analysis_report(pipeline_result)

    assert report.status == "failed"
    assert report.is_ready_for_training_approval is False
    assert report.training_approval_question is None
    assert report.errors
    assert "could not be completed" in report.executive_summary

    markdown = report.to_markdown()
    assert "Dataset could not be loaded." in markdown
    assert "Unsupported file extension" in markdown


def test_analysis_report_to_dict_is_json_serializable(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 20, 30, 40],
            "target": [0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    pipeline_result = run_intake_pipeline(
        file_path,
        run_id="run_json_report",
    )

    report = create_analysis_report(pipeline_result)
    payload = report.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["run_id"] == "run_json_report"
    assert payload["status"] == "ready_for_training_approval"
    assert payload["training_approval_question"] is not None