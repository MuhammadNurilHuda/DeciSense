from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd

from ds_engine.pipeline import run_intake_pipeline
from ds_engine.reporting.packager import (
    AnalysisPackageResult,
    package_analysis_artifacts,
)


def test_package_analysis_artifacts_creates_analysis_only_tarball(
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
        run_id="run_package_success",
    )

    result = package_analysis_artifacts(
        pipeline_result,
        runs_root=tmp_path / "runs",
    )

    assert isinstance(result, AnalysisPackageResult)
    assert result.status == "success"
    assert result.is_success is True
    assert result.package_path is not None
    assert result.package_path.exists()
    assert result.package_path.name == "run_package_success_analysis_only.tar.gz"

    assert result.analysis_dir is not None
    assert (result.analysis_dir / "analysis_report.json").exists()
    assert (result.analysis_dir / "analysis_report.md").exists()
    assert (result.analysis_dir / "pipeline_result.json").exists()
    assert (result.analysis_dir / "package_manifest.json").exists()

    report_payload = json.loads(
        (result.analysis_dir / "analysis_report.json").read_text(encoding="utf-8")
    )
    assert report_payload["status"] == "ready_for_training_approval"
    assert report_payload["training_approval_question"] is not None

    manifest_payload = json.loads(
        (result.analysis_dir / "package_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_payload["package_type"] == "analysis_only"
    assert manifest_payload["include_source_data"] is False

    with tarfile.open(result.package_path, mode="r:gz") as archive:
        names = set(archive.getnames())

    assert "analysis/analysis_report.json" in names
    assert "analysis/analysis_report.md" in names
    assert "analysis/pipeline_result.json" in names
    assert "analysis/package_manifest.json" in names
    assert "analysis/source_data/customer_data.csv" not in names


def test_package_analysis_artifacts_can_include_source_data_when_requested(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    pipeline_result = run_intake_pipeline(
        file_path,
        run_id="run_package_with_source",
    )

    result = package_analysis_artifacts(
        pipeline_result,
        runs_root=tmp_path / "runs",
        include_source_data=True,
    )

    assert result.status == "success"
    assert result.package_path is not None
    assert result.package_path.exists()

    assert result.analysis_dir is not None
    copied_source = result.analysis_dir / "source_data" / "dataset.csv"
    assert copied_source.exists()

    with tarfile.open(result.package_path, mode="r:gz") as archive:
        names = set(archive.getnames())

    assert "analysis/source_data/dataset.csv" in names


def test_package_analysis_artifacts_succeeds_for_unresolved_target_report(
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
        run_id="run_package_requires_input",
    )

    result = package_analysis_artifacts(
        pipeline_result,
        runs_root=tmp_path / "runs",
    )

    assert result.status == "success"
    assert result.package_path is not None
    assert result.package_path.exists()

    assert result.analysis_dir is not None
    report_payload = json.loads(
        (result.analysis_dir / "analysis_report.json").read_text(encoding="utf-8")
    )
    assert report_payload["status"] == "requires_user_input"
    assert report_payload["training_approval_question"] is None


def test_package_analysis_artifacts_returns_failed_when_runs_root_is_not_directory(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )
    file_path = tmp_path / "dataset.csv"
    dataframe.to_csv(file_path, index=False)

    pipeline_result = run_intake_pipeline(
        file_path,
        run_id="run_package_failure",
    )

    invalid_runs_root = tmp_path / "runs_as_file"
    invalid_runs_root.write_text("not a directory", encoding="utf-8")

    result = package_analysis_artifacts(
        pipeline_result,
        runs_root=invalid_runs_root,
    )

    assert result.status == "failed"
    assert result.is_success is False
    assert result.package_path is None
    assert result.errors
    assert "Failed to package analysis artifacts" in result.errors[0]
