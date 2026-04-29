from __future__ import annotations

import json

import pandas as pd

from ds_engine.profiling.schema_profile import (
    ColumnSchemaProfile,
    SchemaProfileResult,
    create_schema_profile,
)


def _columns_by_name(result: SchemaProfileResult) -> dict[str, ColumnSchemaProfile]:
    return {column.column_name: column for column in result.columns}


def test_create_schema_profile_summarizes_core_column_types() -> None:
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Jakarta", "Bandung", "Jakarta"],
            "is_active": [True, False, True],
            "signup_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "feedback": [
                "The service experience was smooth and consistently helpful.",
                "The onboarding process was clear and easy to follow.",
                "The support response was fast and very informative.",
            ],
        }
    )

    result = create_schema_profile(
        dataframe,
        text_avg_length_threshold=30,
        text_unique_ratio_threshold=0.8,
    )
    columns = _columns_by_name(result)

    assert result.row_count == 3
    assert result.column_count == 5
    assert columns["age"].inferred_type == "numeric"
    assert columns["city"].inferred_type == "categorical"
    assert columns["is_active"].inferred_type == "boolean"
    assert columns["signup_date"].inferred_type == "datetime"
    assert columns["feedback"].inferred_type == "text"
    assert result.numeric_columns == ["age"]
    assert result.boolean_columns == ["is_active"]
    assert result.datetime_columns == ["signup_date"]
    assert result.text_columns == ["feedback"]


def test_create_schema_profile_flags_missing_constant_and_all_missing_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "constant_feature": [1, 1, 1],
            "partially_missing": [10, None, 30],
            "all_missing": [None, None, None],
        }
    )

    result = create_schema_profile(dataframe)
    columns = _columns_by_name(result)

    assert columns["constant_feature"].is_constant is True
    assert columns["partially_missing"].missing_count == 1
    assert columns["partially_missing"].missing_ratio == 0.333333
    assert columns["all_missing"].is_all_missing is True
    assert columns["all_missing"].inferred_type == "empty"
    assert result.constant_columns == ["constant_feature"]
    assert result.all_missing_columns == ["all_missing"]


def test_create_schema_profile_flags_high_cardinality_categorical_column() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "segment": ["A", "A", "B", "B", "A"],
        }
    )

    result = create_schema_profile(
        dataframe,
        high_cardinality_min_unique_count=4,
        high_cardinality_unique_ratio_threshold=0.8,
    )
    columns = _columns_by_name(result)

    assert columns["customer_id"].inferred_type == "categorical"
    assert columns["customer_id"].is_high_cardinality is True
    assert columns["segment"].is_high_cardinality is False
    assert result.high_cardinality_columns == ["customer_id"]


def test_schema_profile_to_dict_is_json_serializable() -> None:
    dataframe = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "amount": [100.5, 200.25],
            "label": ["paid", "unpaid"],
        }
    )

    result = create_schema_profile(dataframe)
    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["row_count"] == 2
    assert payload["column_count"] == 3
    assert payload["columns"][0]["sample_values"][0] == "2024-01-01T00:00:00"