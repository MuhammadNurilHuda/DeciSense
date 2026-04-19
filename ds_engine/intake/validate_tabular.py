from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

@dataclass(frozen=True)
class TabularValidationResult:
    """Validation outcome for a loaded tabular dataset"""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0

def validate_tabular_dataset(
    dataframe: pd.DataFrame,
    *,
    min_rows: int = 2,
    min_columns: int = 1) -> TabularValidationResult:
    """
    Validate whether a loaded dataframe is suitable for downstream tabular analysis

    Returns
    -------
    TabularValidationResult
        Structures validation result containing errors and warnings
    """

    errors: list[str] = []
    warnings: list[str] = []

    row_count = len(dataframe)
    column_count = len(dataframe.columns) 

    if column_count < min_columns:
        errors.append(f"Dataset must contain at least {min_columns} column(s), but found {column_count}.")

    if row_count < min_rows:
        errors.append(f"Dataset must contain at least {min_rows} row(s), but found {row_count}.")

    if dataframe.empty:
        errors.append("Dataset is empty.")

    if column_count > 0 and dataframe.isna().all(axis=0).all():
        errors.append("All dataset columns contain only missing values.")

    if row_count > 0 and dataframe.isna().all(axis=1).any():
        warnings.append("Some rows contain only missing values.")

    if column_count > 0:
        fully_missing_columns = dataframe.columns[dataframe.isna().all(axis=0)].tolist()
        if fully_missing_columns:
            warnings.append(
                "Some columns contain only missing values: "
                f"{fully_missing_columns}"
            )
    
    return TabularValidationResult(
        is_valid= not errors,
        errors=errors,
        warnings=warnings,
        row_count=row_count,
        column_count=column_count,
    )