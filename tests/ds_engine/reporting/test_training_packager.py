from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd

from ds_engine.reporting.training_packager import (
    TrainingPackageResult,
    package_training_artifacts,
)
from ds_engine.workflows.analysis_workflow import run_analysis_workflow
from ds_engine.workflows.training_workflow import run_training_workflow


def test_package_training_artifacts_creates_full_training_tarball(
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
        run_id="run_full_training_package",
    )
    training_result = run_training_workflow(
        analysis_result,
        test_size=0.3,
        random_state=42,
    )

    result = package_training_artifacts(
        analysis_workflow_result=analysis_result,
        training_workflow_result=training_result,
        runs_root=tmp_path / "runs",
    )

    assert isinstance(result, TrainingPackageResult)
    assert result.status == "success"
    assert result.is_success is True
    assert result.package_type == "full_training"

    assert result.package_path is not None
    assert result.package_path.exists()
    assert result.package_path.name == "run_full_training_package_full_training.tar.gz"

    assert result.package_dir is not None
    assert (result.package_dir / "analysis" / "analysis_report.json").exists()
    assert (result.package_dir / "analysis" / "pipeline_result.json").exists()
    assert (result.package_dir / "training" / "prepared_modeling_dataset.json").exists()
    assert (result.package_dir / "training" / "model_training_result.json").exists()
    assert (result.package_dir / "training" / "training_summary.txt").exists()
    assert (result.package_dir / "package_manifest.json").exists()

    model_files = list((result.package_dir / "training" / "models").glob("*.joblib"))
    assert model_files

    manifest = json.loads(
        (result.package_dir / "package_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["package_type"] == "full_training"
    assert manifest["training_workflow_status"] in {
        "completed",
        "completed_with_failures",
    }
    assert manifest["best_model"] is not None
    assert manifest["include_source_data"] is False
    assert manifest["include_fitted_pipelines"] is True

    with tarfile.open(result.package_path, mode="r:gz") as archive:
        names = set(archive.getnames())

    assert "analysis/analysis_report.json" in names
    assert "analysis/pipeline_result.json" in names
    assert "training/prepared_modeling_dataset.json" in names
    assert "training/model_training_result.json" in names
    assert "training/training_summary.txt" in names
    assert "package_manifest.json" in names
    assert any(
        name.startswith("training/models/") and name.endswith(".joblib")
        for name in names
    )
    assert "source_data/dataset.csv" not in names


def test_package_training_artifacts_can_include_source_data(
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
        run_id="run_full_training_with_source",
    )
    training_result = run_training_workflow(analysis_result)

    result = package_training_artifacts(
        analysis_workflow_result=analysis_result,
        training_workflow_result=training_result,
        runs_root=tmp_path / "runs",
        include_source_data=True,
    )

    assert result.status == "success"
    assert result.package_path is not None
    assert result.package_dir is not None
    assert (result.package_dir / "source_data" / "dataset.csv").exists()

    with tarfile.open(result.package_path, mode="r:gz") as archive:
        names = set(archive.getnames())

    assert "source_data/dataset.csv" in names


def test_package_training_artifacts_returns_failed_for_unsuccessful_training(
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
        run_id="run_training_package_failed",
    )
    training_result = run_training_workflow(analysis_result)

    result = package_training_artifacts(
        analysis_workflow_result=analysis_result,
        training_workflow_result=training_result,
        runs_root=tmp_path / "runs",
    )

    assert result.status == "failed"
    assert result.is_success is False
    assert result.package_path is None
    assert result.errors
    assert "training did not complete successfully" in result.errors[0]


def test_package_training_artifacts_returns_failed_for_run_id_mismatch(
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

    analysis_result_a = run_analysis_workflow(
        file_path,
        run_id="run_a",
    )
    analysis_result_b = run_analysis_workflow(
        file_path,
        run_id="run_b",
    )
    training_result_b = run_training_workflow(analysis_result_b)

    result = package_training_artifacts(
        analysis_workflow_result=analysis_result_a,
        training_workflow_result=training_result_b,
        runs_root=tmp_path / "runs",
    )

    assert result.status == "failed"
    assert result.package_path is None
    assert result.errors == [
        "Run ID mismatch between analysis workflow result and training workflow result."
    ]


def test_training_package_result_to_dict_is_json_serializable(
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
        run_id="run_full_training_json",
    )
    training_result = run_training_workflow(analysis_result)

    result = package_training_artifacts(
        analysis_workflow_result=analysis_result,
        training_workflow_result=training_result,
        runs_root=tmp_path / "runs",
    )
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["run_id"] == "run_full_training_json"
    assert payload["status"] == "success"
    assert payload["package_type"] == "full_training"
    assert payload["package_path"] is not None
