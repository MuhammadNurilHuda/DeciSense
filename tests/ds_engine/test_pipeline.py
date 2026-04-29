from __future__ import annotations

from pathlib import Path

import pandas as pd

from ds_engine.intake.infer_task import TaskInferenceResult
from ds_engine.intake.load_data import DataLoadError, LoadedDataset
from ds_engine.intake.validate_tabular import TabularValidationResult
from ds_engine.pipeline import IntakePipelineConfig, run_intake_pipeline


def _build_loaded_dataset(tmp_path: Path, dataframe: pd.DataFrame) -> LoadedDataset:
    file_path = (tmp_path / "dataset.csv").resolve()
    return LoadedDataset(
        dataframe=dataframe,
        file_path=file_path,
        file_name=file_path.name,
        file_extension=".csv",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
    )


def test_run_intake_pipeline_returns_validation_failed_and_skips_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataframe = pd.DataFrame({"target": [1]})
    loaded_dataset = _build_loaded_dataset(tmp_path, dataframe)

    def fake_load_tabular_data(file_path: str | Path) -> LoadedDataset:
        return loaded_dataset

    def fake_validate_tabular_dataset(
        dataframe: pd.DataFrame,
        *,
        min_rows: int,
        min_columns: int,
    ) -> TabularValidationResult:
        return TabularValidationResult(
            is_valid=False,
            errors=["Dataset must contain at least 2 row(s), but found 1."],
            warnings=[],
            row_count=1,
            column_count=1,
        )

    infer_called = {"value": False}

    def fake_infer_task_from_dataframe(*args, **kwargs) -> TaskInferenceResult:
        infer_called["value"] = True
        return TaskInferenceResult(
            candidate_target="target",
            task_type="classification",
            status="ok",
            reasoning=[],
        )

    monkeypatch.setattr("ds_engine.pipeline.load_tabular_data", fake_load_tabular_data)
    monkeypatch.setattr(
        "ds_engine.pipeline.validate_tabular_dataset",
        fake_validate_tabular_dataset,
    )
    monkeypatch.setattr(
        "ds_engine.pipeline.infer_task_from_dataframe",
        fake_infer_task_from_dataframe,
    )

    result = run_intake_pipeline("unused.csv", run_id="run_validation_failed")

    assert result.status == "validation_failed"
    assert result.loaded_dataset == loaded_dataset
    assert result.validation_result is not None
    assert result.validation_result.is_valid is False
    assert result.task_inference_result is None
    assert result.errors == ["Dataset must contain at least 2 row(s), but found 1."]
    assert result.is_ready_for_downstream_analysis is False
    assert result.is_ready_for_model_planning is False
    assert result.requires_user_target_input is False
    assert infer_called["value"] is False


def test_run_intake_pipeline_completes_but_requires_user_target_input_when_inference_is_unresolved(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataframe = pd.DataFrame({"feature_a": [1, 2, 3]})
    loaded_dataset = _build_loaded_dataset(tmp_path, dataframe)

    def fake_load_tabular_data(file_path: str | Path) -> LoadedDataset:
        return loaded_dataset

    def fake_validate_tabular_dataset(
        dataframe: pd.DataFrame,
        *,
        min_rows: int,
        min_columns: int,
    ) -> TabularValidationResult:
        return TabularValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            row_count=3,
            column_count=1,
        )

    def fake_infer_task_from_dataframe(
        dataframe: pd.DataFrame,
        *,
        target_candidates: tuple[str, ...],
        classification_unique_value_threshold: int,
    ) -> TaskInferenceResult:
        return TaskInferenceResult(
            candidate_target=None,
            task_type=None,
            status="not_found",
            reasoning=["No target column matched the configured candidate names."],
        )

    monkeypatch.setattr("ds_engine.pipeline.load_tabular_data", fake_load_tabular_data)
    monkeypatch.setattr(
        "ds_engine.pipeline.validate_tabular_dataset",
        fake_validate_tabular_dataset,
    )
    monkeypatch.setattr(
        "ds_engine.pipeline.infer_task_from_dataframe",
        fake_infer_task_from_dataframe,
    )

    result = run_intake_pipeline("unused.csv", run_id="run_missing_target")

    assert result.status == "completed"
    assert result.validation_result is not None
    assert result.validation_result.is_valid is True
    assert result.task_inference_result is not None
    assert result.task_inference_result.status == "not_found"
    assert result.is_ready_for_downstream_analysis is True
    assert result.is_ready_for_model_planning is False
    assert result.requires_user_target_input is True
    assert "No target column matched the configured candidate names." in result.warnings


def test_run_intake_pipeline_returns_load_failed_when_loader_raises(
    monkeypatch,
) -> None:
    def fake_load_tabular_data(file_path: str | Path) -> LoadedDataset:
        raise DataLoadError("Unsupported file extension '.txt'.")

    monkeypatch.setattr("ds_engine.pipeline.load_tabular_data", fake_load_tabular_data)

    result = run_intake_pipeline("notes.txt", run_id="run_load_failed")

    assert result.status == "load_failed"
    assert result.loaded_dataset is None
    assert result.validation_result is None
    assert result.task_inference_result is None
    assert result.errors == ["Unsupported file extension '.txt'."]
    assert result.is_ready_for_downstream_analysis is False
    assert result.is_ready_for_model_planning is False


def test_run_intake_pipeline_attaches_artifact_path_when_persistence_is_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataframe = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})
    loaded_dataset = _build_loaded_dataset(tmp_path, dataframe)

    def fake_load_tabular_data(file_path: str | Path) -> LoadedDataset:
        return loaded_dataset

    def fake_validate_tabular_dataset(
        dataframe: pd.DataFrame,
        *,
        min_rows: int,
        min_columns: int,
    ) -> TabularValidationResult:
        return TabularValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            row_count=3,
            column_count=2,
        )

    def fake_infer_task_from_dataframe(
        dataframe: pd.DataFrame,
        *,
        target_candidates: tuple[str, ...],
        classification_unique_value_threshold: int,
    ) -> TaskInferenceResult:
        return TaskInferenceResult(
            candidate_target="target",
            task_type="classification",
            status="ok",
            reasoning=["Matched target column 'target'."],
        )

    captured = {}

    def fake_save_intake_pipeline_artifacts(
        result,
        *,
        runs_root: str | Path,
        include_preview: bool,
        preview_row_count: int,
    ) -> Path:
        captured["runs_root"] = Path(runs_root)
        captured["include_preview"] = include_preview
        captured["preview_row_count"] = preview_row_count
        return tmp_path / "runs" / result.run_id / "intake" / "intake_result.json"

    monkeypatch.setattr("ds_engine.pipeline.load_tabular_data", fake_load_tabular_data)
    monkeypatch.setattr(
        "ds_engine.pipeline.validate_tabular_dataset",
        fake_validate_tabular_dataset,
    )
    monkeypatch.setattr(
        "ds_engine.pipeline.infer_task_from_dataframe",
        fake_infer_task_from_dataframe,
    )
    monkeypatch.setattr(
        "ds_engine.pipeline.save_intake_pipeline_artifacts",
        fake_save_intake_pipeline_artifacts,
    )

    config = IntakePipelineConfig(
        persist_artifacts=True,
        runs_root=tmp_path / "runs_root",
        include_preview_in_artifacts=False,
        preview_row_count=3,
    )

    result = run_intake_pipeline(
        "unused.csv",
        config=config,
        run_id="run_with_artifact",
    )

    assert result.status == "completed"
    assert result.artifact_path == tmp_path / "runs" / "run_with_artifact" / "intake" / "intake_result.json"
    assert captured["runs_root"] == tmp_path / "runs_root"
    assert captured["include_preview"] is False
    assert captured["preview_row_count"] == 3