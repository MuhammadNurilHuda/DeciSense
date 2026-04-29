from __future__ import annotations

import json

import pandas as pd

from ds_engine.profiling.data_quality import (
    DataQualityResult,
    create_data_quality_report,
)
from ds_engine.profiling.schema_profile import create_schema_profile


def _issue_codes(result: DataQualityResult) -> set[str]:
    return {issue.code for issue in result.issues}


def test_create_data_quality_report_returns_clean_summary_for_simple_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "income": [1000, 2000, 3000],
            "target": [0, 1, 0],
        }
    )

    result = create_data_quality_report(dataframe)

    assert result.row_count == 3
    assert result.column_count == 3
    assert result.duplicate_row_count == 0
    assert result.missing_cell_count == 0
    assert result.missing_cell_ratio == 0.0
    assert result.issues == []
    assert result.is_usable_for_analysis is True


def test_create_data_quality_report_flags_missingness_and_fully_missing_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [20, None, 40],
            "income": [None, None, None],
            "target": [0, 1, 0],
        }
    )

    result = create_data_quality_report(
        dataframe,
        missing_cell_warning_threshold=0.10,
        missing_cell_critical_threshold=0.90,
    )

    assert result.missing_cell_count == 4
    assert result.missing_cell_ratio == 0.444444
    assert result.columns_with_missing_values == ["age", "income"]
    assert result.fully_missing_columns == ["income"]
    assert "missing_values" in _issue_codes(result)
    assert "fully_missing_columns" in _issue_codes(result)
    assert result.is_usable_for_analysis is True


def test_create_data_quality_report_flags_duplicates_and_constant_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "constant_feature": [1, 1, 1],
            "city": ["Jakarta", "Jakarta", "Jakarta"],
            "target": [0, 0, 0],
        }
    )

    result = create_data_quality_report(
        dataframe,
        duplicate_row_warning_threshold=0.01,
    )

    assert result.duplicate_row_count == 2
    assert result.duplicate_row_ratio == 0.666667
    assert set(result.constant_columns) == {"constant_feature", "city", "target"}
    assert "duplicate_rows" in _issue_codes(result)
    assert "constant_columns" in _issue_codes(result)


def test_create_data_quality_report_flags_high_cardinality_and_possible_id_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "segment": ["A", "A", "B", "B", "A"],
            "target": [0, 1, 0, 1, 0],
        }
    )
    schema_profile = create_schema_profile(
        dataframe,
        high_cardinality_min_unique_count=4,
        high_cardinality_unique_ratio_threshold=0.8,
    )

    result = create_data_quality_report(
        dataframe,
        schema_profile=schema_profile,
    )

    assert result.high_cardinality_columns == ["customer_id"]
    assert result.possible_id_columns == ["customer_id"]
    assert "high_cardinality_columns" in _issue_codes(result)
    assert "possible_id_columns" in _issue_codes(result)


def test_data_quality_result_to_dict_is_json_serializable() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [20, None, 40],
            "target": [0, 1, 0],
        }
    )

    result = create_data_quality_report(dataframe)
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["row_count"] == 3
    assert payload["columns_with_missing_values"] == ["age"]