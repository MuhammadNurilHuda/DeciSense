from __future__ import annotations
import csv
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SUPPORTED_EXTENSIONS: Final [set[str]] = {
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet"
}

class DataLoadError(Exception):
    """Raised when a dataset can't be loaded as a supported tabular file"""

@dataclass(frozen=True)
class LoadedDataset:
    """Container for a loaded dataset and its basic source metadata"""

    dataframe: pd.DataFrame
    file_path: Path
    file_name: str
    file_extension: str
    row_count: int
    column_count : int


def load_tabular_data(file_path: str | Path) -> LoadedDataset:
    """
    Load a supported tabular dataset into a pandas DataFrame.

    Supported formats:
    - .csv
    - .tsv
    - .xlsx
    - .xls
    - .parquet

    Returns
    -------
    LoadedDataset
        Loaded dataset and basic metadata

    Raises
    -------
    FileNotFoundError
        If file doesn't exists
    DataLoadError
        If the file extension is unsupported or loading fails
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if not path.is_file():
        raise DataLoadError(f"Expected a file path, but got: {path}")
    
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DataLoadError(f"Unsupported file extension '{extension}'. Supported formats: {supported}")
    
    duplicate_columns = _detect_duplicate_columns(path, extension)
    if duplicate_columns:
        raise DataLoadError(
            "Dataset contains duplicate column names, which is not supported. "
            f"Duplicated columns: {duplicate_columns}"
        )

    dataframe = _read_dataframe(path, extension)
    return LoadedDataset(dataframe=dataframe,
                         file_path=path,
                         file_name=path.name,
                         file_extension=extension,
                         row_count=len(dataframe),
                         column_count=len(dataframe.columns))


def _detect_duplicate_columns(path: Path, extension: str) -> list[str]:
    """
    Detect duplicate column names from raw file headers for delimited files.

    Currently implemented for:
    - .csv
    - .tsv

    For other file types, this function returns an empty list.
    """
    if extension == ".csv":
        return _find_duplicate_headers_in_delimited_file(path, delimiter=",")
    if extension == ".tsv":
        return _find_duplicate_headers_in_delimited_file(path, delimiter="\t")
    return []

def _find_duplicate_headers_in_delimited_file(path: Path, delimiter:str) -> list[str]:
    """
    Read the first row of a delimited file and return duplicate header names.
    """
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, delimiter=delimiter)
            header = next(reader, None)
    except Exception as exc:
        raise DataLoadError(
            f"Failed to inspect dataset header from '{path.name}': {exc}"
        ) from exc

    if header is None:
        return []

    duplicates: list[str] = []
    seen: set[str] = set()

    for column_name in header:
        normalized_name = column_name.strip()
        if normalized_name in seen and normalized_name not in duplicates:
            duplicates.append(normalized_name)
        seen.add(normalized_name)

    return duplicates

def _read_dataframe(path: Path, extension: str) -> pd.DataFrame:
    """
    Dispatch file reading based on file extension.
    """
    try:
        if extension == ".csv":
            return pd.read_csv(path)
        if extension ==".tsv":
            return pd.read_csv(path, sep="\t")
        if extension in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if extension == ".parquet":
            return pd.read_parquet(path)

    except Exception as exc:
        raise DataLoadError(
            f"Failed to load dataset from '{path.name}': {exc}"
        ) from exc

    raise DataLoadError(
        f"No loader is defined for extension '{extension}'"
    )
