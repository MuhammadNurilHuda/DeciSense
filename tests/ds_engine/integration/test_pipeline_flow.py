from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ds_engine.pipeline import IntakePipelineConfig, run_intake_pipeline


def test_pipeline_flow_completes_for_valid_csv_recommends_model_and_persists_artifact(
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

    config = IntakePipelineConfig(
        persist_artifacts=True,
        runs_root=tmp_path / "runs",
    )

    result = run_intake_pipeline(
        file_path,
        config=config,
        run_id="run_integration_success",
    )

    assert result.status == "completed"

    assert result.loaded_dataset is not None
    assert result.loaded_dataset.file_name == "customer_data.csv"
    assert result.loaded_dataset.row_count == 4
    assert result.loaded_dataset.column_count == 3

    assert result.validation_result is not None
    assert result.validation_result.is_valid is True

    assert result.task_inference_result is not None
    assert result.task_inference_result.status == "ok"
    assert result.task_inference_result.candidate_target == "target"
    assert result.task_inference_result.task_type == "classification"

    assert result.schema_profile_result is not None
    assert result.schema_profile_result.row_count == 4
    assert result.schema_profile_result.column_count == 3
    assert result.schema_profile_result.numeric_columns == [
        "feature_a",
        "feature_b",
        "target",
    ]

    assert result.data_quality_result is not None
    assert result.data_quality_result.row_count == 4
    assert result.data_quality_result.missing_cell_count == 0
    assert result.data_quality_result.issues == []

    assert result.target_profile_result is not None
    assert result.target_profile_result.target_column == "target"
    assert result.target_profile_result.task_type == "classification"
    assert result.target_profile_result.unique_count == 2
    assert result.target_profile_result.is_usable_for_modeling is True

    assert result.model_recommendation_result is not None
    assert result.model_recommendation_result.status == "ready"
    assert result.model_recommendation_result.recommended_model == (
        "HistGradientBoostingClassifier"
    )
    assert result.model_recommendation_result.is_ready_for_training_approval is True

    assert result.is_ready_for_downstream_analysis is True
    assert result.is_ready_for_model_planning is True
    assert result.is_ready_for_training_approval is True

    assert result.artifact_path is not None
    assert result.artifact_path.exists()

    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert payload["loaded_dataset"]["file_name"] == "customer_data.csv"
    assert payload["loaded_dataset"]["row_count"] == 4
    assert len(payload["loaded_dataset"]["preview_rows"]) == 4

    assert payload["task_inference_result"]["candidate_target"] == "target"
    assert payload["task_inference_result"]["task_type"] == "classification"

    assert payload["schema_profile_result"]["row_count"] == 4
    assert payload["data_quality_result"]["missing_cell_count"] == 0

    assert payload["target_profile_result"]["target_column"] == "target"
    assert payload["target_profile_result"]["task_type"] == "classification"

    assert payload["model_recommendation_result"]["status"] == "ready"
    assert payload["model_recommendation_result"]["recommended_model"] == (
        "HistGradientBoostingClassifier"
    )


def test_pipeline_flow_returns_load_failed_for_duplicate_csv_headers(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "duplicate_headers.csv"
    file_path.write_text(
        "target,target,feature\n0,1,10\n",
        encoding="utf-8",
    )

    result = run_intake_pipeline(
        file_path,
        run_id="run_duplicate_headers",
    )

    assert result.status == "load_failed"
    assert result.loaded_dataset is None
    assert result.validation_result is None
    assert result.task_inference_result is None
    assert result.schema_profile_result is None
    assert result.data_quality_result is None
    assert result.target_profile_result is None
    assert result.model_recommendation_result is None
    assert result.is_ready_for_downstream_analysis is False
    assert result.is_ready_for_model_planning is False
    assert result.is_ready_for_training_approval is False
    assert any("duplicate column names" in error.lower() for error in result.errors)


def test_pipeline_flow_completes_but_requires_target_input_when_no_target_candidate_is_found(
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

    result = run_intake_pipeline(
        file_path,
        run_id="run_no_target",
    )

    assert result.status == "completed"

    assert result.validation_result is not None
    assert result.validation_result.is_valid is True

    assert result.schema_profile_result is not None
    assert result.data_quality_result is not None

    assert result.task_inference_result is not None
    assert result.task_inference_result.status == "not_found"

    assert result.target_profile_result is None
    assert result.model_recommendation_result is None

    assert result.is_ready_for_downstream_analysis is True
    assert result.is_ready_for_model_planning is False
    assert result.is_ready_for_training_approval is False
    assert result.requires_user_target_input is True


def test_pipeline_flow_completes_but_blocks_training_approval_for_critical_quality_issue(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [None, None, None],
            "feature_b": [None, None, 10],
            "target": [0, 1, 0],
        }
    )
    file_path = tmp_path / "high_missingness.csv"
    dataframe.to_csv(file_path, index=False)

    config = IntakePipelineConfig(
        missing_cell_warning_threshold=0.10,
        missing_cell_critical_threshold=0.40,
    )

    result = run_intake_pipeline(
        file_path,
        config=config,
        run_id="run_critical_quality",
    )

    assert result.status == "completed"

    assert result.validation_result is not None
    assert result.validation_result.is_valid is True

    assert result.task_inference_result is not None
    assert result.task_inference_result.status == "ok"

    assert result.data_quality_result is not None
    assert result.data_quality_result.has_critical_issues is True

    assert result.target_profile_result is not None
    assert result.target_profile_result.is_usable_for_modeling is True

    assert result.model_recommendation_result is not None
    assert result.model_recommendation_result.status == "blocked"
    assert result.model_recommendation_result.recommended_model is None
    assert result.model_recommendation_result.is_ready_for_training_approval is False

    assert result.is_ready_for_downstream_analysis is True
    assert result.is_ready_for_model_planning is False
    assert result.is_ready_for_training_approval is False
    assert any("critical" in warning for warning in result.warnings)


def test_pipeline_flow_completes_but_blocks_training_approval_for_single_class_target(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [10, 20, 30, 40],
            "feature_b": [1.5, 2.5, 3.5, 4.5],
            "target": [1, 1, 1, 1],
        }
    )
    file_path = tmp_path / "single_class_target.csv"
    dataframe.to_csv(file_path, index=False)

    result = run_intake_pipeline(
        file_path,
        run_id="run_single_class_target",
    )

    assert result.status == "completed"

    assert result.validation_result is not None
    assert result.validation_result.is_valid is True

    assert result.task_inference_result is not None
    assert result.task_inference_result.status == "ok"

    assert result.target_profile_result is not None
    assert result.target_profile_result.has_critical_issues is True
    assert result.target_profile_result.is_usable_for_modeling is False

    assert result.model_recommendation_result is not None
    assert result.model_recommendation_result.status == "blocked"
    assert result.model_recommendation_result.recommended_model is None
    assert result.model_recommendation_result.is_ready_for_training_approval is False

    assert result.is_ready_for_downstream_analysis is True
    assert result.is_ready_for_model_planning is False
    assert result.is_ready_for_training_approval is False
    assert any(
        "classification target contains only one class" in warning
        for warning in result.warnings
    )


def test_pipeline_flow_recommends_catboost_when_optional_dependencies_are_allowed(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "city": ["A", "B", "C", "D", "E", "F"],
            "merchant_category": [
                "food",
                "retail",
                "food",
                "travel",
                "retail",
                "travel",
            ],
            "customer_id": ["C001", "C002", "C003", "C004", "C005", "C006"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    file_path = tmp_path / "categorical_data.csv"
    dataframe.to_csv(file_path, index=False)

    config = IntakePipelineConfig(
        allow_optional_model_dependencies=True,
        high_cardinality_min_unique_count=4,
        high_cardinality_unique_ratio_threshold=0.8,
    )

    result = run_intake_pipeline(
        file_path,
        config=config,
        run_id="run_catboost_recommendation",
    )

    assert result.status == "completed"

    assert result.model_recommendation_result is not None
    assert result.model_recommendation_result.status == "ready"
    assert result.model_recommendation_result.recommended_model == "CatBoostClassifier"
    assert result.model_recommendation_result.recommended_candidate is not None
    assert (
        result.model_recommendation_result.recommended_candidate.optional_dependency
        == ("catboost")
    )

    assert result.is_ready_for_model_planning is True
    assert result.is_ready_for_training_approval is True
