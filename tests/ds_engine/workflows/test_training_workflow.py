from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ds_engine.workflows.analysis_workflow import run_analysis_workflow
from ds_engine.workflows.training_workflow import (
    TrainingWorkflowResult,
    run_training_workflow,
)


def test_run_training_workflow_completes_after_ready_analysis(
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
        run_id="run_training_ready",
    )

    result = run_training_workflow(
        analysis_result,
        test_size=0.3,
        random_state=42,
    )

    assert isinstance(result, TrainingWorkflowResult)
    assert result.status in {"completed", "completed_with_failures"}
    assert result.is_success is True
    assert result.prepared_dataset is not None
    assert result.prepared_dataset.feature_count == 2

    assert result.model_training_result is not None
    assert result.model_training_result.best_experiment is not None
    assert result.best_experiment is not None
    assert result.best_experiment.primary_metric_name == "f1_macro"

    telegram_summary = result.to_telegram_summary()
    assert "Training completed." in telegram_summary
    assert "Best model:" in telegram_summary
    assert "Overfitting risk:" in telegram_summary


def test_run_training_workflow_can_persist_training_artifacts(
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
        run_id="run_training_artifacts",
    )

    result = run_training_workflow(
        analysis_result,
        persist_artifacts=True,
        runs_root=tmp_path / "runs",
    )

    assert result.status in {"completed", "completed_with_failures"}
    assert result.artifact_dir is not None
    assert result.artifact_dir.exists()

    expected_files = {
        "prepared_modeling_dataset.json",
        "model_training_result.json",
        "training_summary.txt",
        "training_manifest.json",
    }
    assert expected_files.issubset({path.name for path in result.artifact_paths})

    training_result_path = result.artifact_dir / "model_training_result.json"
    payload = json.loads(training_result_path.read_text(encoding="utf-8"))

    assert payload["status"] in {"completed", "completed_with_failures"}
    assert payload["best_experiment"] is not None


def test_run_training_workflow_returns_not_ready_when_analysis_requires_target_input(
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
        run_id="run_training_not_ready",
    )

    result = run_training_workflow(analysis_result)

    assert result.status == "not_ready"
    assert result.is_success is False
    assert result.prepared_dataset is None
    assert result.model_training_result is None
    assert result.errors
    assert "not ready for training approval" in result.errors[0]


def test_run_training_workflow_returns_failed_when_no_usable_features_remain(
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
        run_id="run_training_no_features",
    )

    result = run_training_workflow(analysis_result)

    assert result.status == "failed"
    assert result.is_success is False
    assert result.prepared_dataset is None
    assert result.model_training_result is None
    assert any("No usable modeling features" in error for error in result.errors)


def test_training_workflow_result_to_dict_is_json_serializable(
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
        run_id="run_training_json",
    )

    result = run_training_workflow(
        analysis_result,
        persist_artifacts=True,
        runs_root=tmp_path / "runs",
    )
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["run_id"] == "run_training_json"
    assert payload["status"] in {"completed", "completed_with_failures"}
    assert payload["prepared_dataset"] is not None
    assert payload["model_training_result"] is not None
