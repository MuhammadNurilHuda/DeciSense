from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ds_engine.intake.load_data import (
    DataLoadError,
    LoadedDataset,
    _detect_duplicate_columns,
    _find_duplicate_headers_in_delimited_file,
    _read_dataframe,
    load_tabular_data,
)


def test_load_tabular_data_csv_success(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "age": [25, 30, 35],
            "income": [5000, 7000, 9000],
        }
    )
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)

    loaded = load_tabular_data(file_path)

    assert isinstance(loaded, LoadedDataset)
    assert loaded.file_path == file_path.resolve()
    assert loaded.file_name == "sample.csv"
    assert loaded.file_extension == ".csv"
    assert loaded.row_count == 3
    assert loaded.column_count == 2
    pd.testing.assert_frame_equal(loaded.dataframe, df)


def test_load_tabular_data_tsv_success(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "city": ["Jakarta", "Surabaya"],
            "sales": [100, 200],
        }
    )
    file_path = tmp_path / "sample.tsv"
    df.to_csv(file_path, sep="\t", index=False)

    loaded = load_tabular_data(file_path)

    assert loaded.file_extension == ".tsv"
    assert loaded.row_count == 2
    assert loaded.column_count == 2
    pd.testing.assert_frame_equal(loaded.dataframe, df)


def test_load_tabular_data_xlsx_success(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")

    df = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": ["x", "y"],
        }
    )
    file_path = tmp_path / "sample.xlsx"
    df.to_excel(file_path, index=False)

    loaded = load_tabular_data(file_path)

    assert loaded.file_extension == ".xlsx"
    assert loaded.row_count == 2
    assert loaded.column_count == 2
    pd.testing.assert_frame_equal(loaded.dataframe, df)


def test_load_tabular_data_parquet_success(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    df = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.3],
            "label": [0, 1, 0],
        }
    )
    file_path = tmp_path / "sample.parquet"
    df.to_parquet(file_path, index=False)

    loaded = load_tabular_data(file_path)

    assert loaded.file_extension == ".parquet"
    assert loaded.row_count == 3
    assert loaded.column_count == 2
    pd.testing.assert_frame_equal(loaded.dataframe, df)


def test_load_tabular_data_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_tabular_data("this_file_does_not_exist.csv")


def test_load_tabular_data_raises_for_directory(tmp_path: Path) -> None:
    folder_path = tmp_path / "some_folder"
    folder_path.mkdir()

    with pytest.raises(DataLoadError, match="Expected a file path"):
        load_tabular_data(folder_path)


def test_load_tabular_data_raises_for_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    with pytest.raises(DataLoadError, match="Unsupported file extension"):
        load_tabular_data(file_path)


def test_load_tabular_data_raises_for_duplicate_columns_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "duplicate_columns.csv"
    file_path.write_text("age,age,income\n25,30,5000\n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="duplicate column names"):
        load_tabular_data(file_path)


def test_detect_duplicate_columns_for_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "duplicate_columns.csv"
    file_path.write_text("age,age,income\n25,30,5000\n", encoding="utf-8")

    duplicates = _detect_duplicate_columns(file_path, ".csv")

    assert duplicates == ["age"]


def test_detect_duplicate_columns_for_tsv(tmp_path: Path) -> None:
    file_path = tmp_path / "duplicate_columns.tsv"
    file_path.write_text("name\tname\tlabel\nA\tB\t1\n", encoding="utf-8")

    duplicates = _detect_duplicate_columns(file_path, ".tsv")

    assert duplicates == ["name"]


def test_detect_duplicate_columns_returns_empty_for_unsupported_raw_check(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.parquet"
    file_path.write_text("dummy", encoding="utf-8")

    duplicates = _detect_duplicate_columns(file_path, ".parquet")

    assert duplicates == []


def test_find_duplicate_headers_in_delimited_file_returns_duplicates_once(tmp_path: Path) -> None:
    file_path = tmp_path / "duplicate_headers.csv"
    file_path.write_text("a,b,a,a,c\n1,2,3,4,5\n", encoding="utf-8")

    duplicates = _find_duplicate_headers_in_delimited_file(file_path, delimiter=",")

    assert duplicates == ["a"]


def test_find_duplicate_headers_in_delimited_file_strips_whitespace(tmp_path: Path) -> None:
    file_path = tmp_path / "duplicate_headers.csv"
    file_path.write_text(" age , income , age \n25,5000,30\n", encoding="utf-8")

    duplicates = _find_duplicate_headers_in_delimited_file(file_path, delimiter=",")

    assert duplicates == ["age"]


def test_read_dataframe_csv_success(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)

    loaded_df = _read_dataframe(file_path, ".csv")

    pd.testing.assert_frame_equal(loaded_df, df)


def test_read_dataframe_tsv_success(tmp_path: Path) -> None:
    df = pd.DataFrame({"col1": ["a", "b"], "col2": [10, 20]})
    file_path = tmp_path / "sample.tsv"
    df.to_csv(file_path, sep="\t", index=False)

    loaded_df = _read_dataframe(file_path, ".tsv")

    pd.testing.assert_frame_equal(loaded_df, df)


def test_read_dataframe_xlsx_success(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")

    df = pd.DataFrame({"name": ["A", "B"], "value": [1.5, 2.5]})
    file_path = tmp_path / "sample.xlsx"
    df.to_excel(file_path, index=False)

    loaded_df = _read_dataframe(file_path, ".xlsx")

    pd.testing.assert_frame_equal(loaded_df, df)


def test_read_dataframe_parquet_success(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    df = pd.DataFrame({"f1": [1, 2, 3], "f2": ["a", "b", "c"]})
    file_path = tmp_path / "sample.parquet"
    df.to_parquet(file_path, index=False)

    loaded_df = _read_dataframe(file_path, ".parquet")

    pd.testing.assert_frame_equal(loaded_df, df)


def test_read_dataframe_wraps_reader_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "broken.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    def mock_read_csv(*args, **kwargs):
        raise ValueError("simulated pandas failure")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)

    with pytest.raises(DataLoadError, match="Failed to load dataset from 'broken.csv'"):
        _read_dataframe(file_path, ".csv")


def test_read_dataframe_raises_when_no_loader_is_defined(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.unknown"
    file_path.write_text("dummy", encoding="utf-8")

    with pytest.raises(DataLoadError, match="No loader is defined"):
        _read_dataframe(file_path, ".unknown")