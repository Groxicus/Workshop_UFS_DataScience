"""
utils.helpers
=============

Shared helper utilities for ML Analytics Studio.

Provides:

- Data type inference
- Formatting utilities
- Validation helpers
- Memory optimization
- Sampling helpers
- Progress utilities
- General reusable functions

Author: ML Analytics Studio
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

################################################################################
# General Utilities
################################################################################


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Safely divide two numbers.
    """

    if denominator == 0:

        return default

    return numerator / denominator


################################################################################


def percentage(
    value: float,
    total: float,
) -> float:
    """
    Calculate a percentage.
    """

    return round(
        safe_divide(value, total) * 100,
        2,
    )


################################################################################


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Restrict value to a given range.
    """

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


################################################################################
# Formatting Helpers
################################################################################


def format_number(
    value: float | int,
    decimals: int = 2,
) -> str:
    """
    Nicely format numeric values.
    """

    if isinstance(value, int):

        return f"{value:,}"

    return f"{value:,.{decimals}f}"


################################################################################


def format_bytes(
    num_bytes: int,
) -> str:
    """
    Human-readable byte formatting.
    """

    units = [

        "B",

        "KB",

        "MB",

        "GB",

        "TB",

    ]

    value = float(num_bytes)

    for unit in units:

        if value < 1024:

            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PB"


################################################################################


def format_duration(
    seconds: float,
) -> str:
    """
    Format elapsed time.
    """

    if seconds < 60:

        return f"{seconds:.2f} sec"

    minutes = seconds / 60

    if minutes < 60:

        return f"{minutes:.2f} min"

    hours = minutes / 60

    return f"{hours:.2f} hr"


################################################################################
# DataFrame Helpers
################################################################################


def memory_usage(
    dataframe: pd.DataFrame,
) -> float:
    """
    DataFrame memory usage (MB).
    """

    return (

        dataframe.memory_usage(
            deep=True,
        ).sum()

        / 1024**2

    )


################################################################################


def optimise_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage by
    downcasting numeric columns.
    """

    df = dataframe.copy()

    for column in df.columns:

        dtype = df[column].dtype

        if pd.api.types.is_integer_dtype(dtype):

            df[column] = pd.to_numeric(

                df[column],

                downcast="integer",

            )

        elif pd.api.types.is_float_dtype(dtype):

            df[column] = pd.to_numeric(

                df[column],

                downcast="float",

            )

    return df


################################################################################


def sample_dataframe(
    dataframe: pd.DataFrame,
    max_rows: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample large datasets for visualization.
    """

    if len(dataframe) <= max_rows:

        return dataframe

    return dataframe.sample(

        max_rows,

        random_state=random_state,

    )


################################################################################
# Validation Helpers
################################################################################


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[bool, str]:
    """
    Validate uploaded dataset.
    """

    if dataframe is None:

        return False, "Dataset is None."

    if dataframe.empty:

        return False, "Dataset is empty."

    if dataframe.shape[1] == 0:

        return False, "Dataset has no columns."

    return True, ""


################################################################################


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> tuple[bool, list[str]]:
    """
    Verify required columns exist.
    """

    missing = [

        col

        for col in required_columns

        if col not in dataframe.columns

    ]

    return (

        len(missing) == 0,

        missing,

    )


################################################################################
# Text Utilities
################################################################################


def clean_column_name(
    column: str,
) -> str:
    """
    Standardize column names.
    """

    column = column.strip()

    column = column.lower()

    column = re.sub(

        r"[^\w]+",

        "_",

        column,

    )

    column = re.sub(

        r"_+",

        "_",

        column,

    )

    return column.strip("_")


################################################################################


def clean_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean all DataFrame column names.
    """

    df = dataframe.copy()

    df.columns = [

        clean_column_name(col)

        for col in df.columns

    ]

    return df
################################################################################
# Type Inference Helpers
################################################################################


def infer_column_type(
    series: pd.Series,
) -> str:
    """
    Infer the logical type of a DataFrame column.
    """

    if pd.api.types.is_bool_dtype(series):

        return "boolean"

    if pd.api.types.is_numeric_dtype(series):

        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):

        return "datetime"

    unique = series.nunique(dropna=True)

    if unique <= 20:

        return "categorical"

    return "text"


################################################################################


def infer_dataframe_types(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    """
    Infer logical types for all columns.
    """

    return {

        column: infer_column_type(dataframe[column])

        for column in dataframe.columns

    }


################################################################################
# Random Seed
################################################################################


def set_random_seed(
    seed: int = 42,
) -> None:
    """
    Set NumPy random seed.
    """

    np.random.seed(seed)


################################################################################
# Progress Helper
################################################################################


def progress_generator(
    total_steps: int,
):
    """
    Yield incremental progress values.

    Example
    -------
    for progress in progress_generator(5):
        progress_bar.progress(progress)
    """

    if total_steps <= 0:

        yield 1.0

        return

    for step in range(1, total_steps + 1):

        yield step / total_steps


################################################################################
# Timing Utilities
################################################################################


class Timer:
    """
    Simple execution timer.

    Example
    -------
    timer = Timer()

    ...

    elapsed = timer.elapsed()
    """

    def __init__(self):

        self.start_time = time.perf_counter()

    def reset(self):

        self.start_time = time.perf_counter()

    def elapsed(self):

        return time.perf_counter() - self.start_time


################################################################################
# File Helpers
################################################################################


def file_size(
    filepath: str | Path,
) -> int:
    """
    Return file size in bytes.
    """

    return Path(filepath).stat().st_size


################################################################################


def file_extension(
    filepath: str | Path,
) -> str:
    """
    Return lowercase file extension.
    """

    return Path(filepath).suffix.lower()


################################################################################


def ensure_directory(
    directory: str | Path,
) -> Path:
    """
    Create a directory if it does not exist.
    """

    path = Path(directory)

    path.mkdir(

        parents=True,

        exist_ok=True,

    )

    return path


################################################################################
# Dictionary Helpers
################################################################################


def flatten_dict(
    dictionary: dict,
    parent_key: str = "",
    separator: str = ".",
):
    """
    Flatten nested dictionaries.
    """

    items = []

    for key, value in dictionary.items():

        new_key = (

            f"{parent_key}{separator}{key}"

            if parent_key

            else key

        )

        if isinstance(value, dict):

            items.extend(

                flatten_dict(

                    value,

                    new_key,

                    separator,

                ).items()

            )

        else:

            items.append(

                (

                    new_key,

                    value,

                )

            )

    return dict(items)


################################################################################
# Missing Value Summary
################################################################################


def missing_value_percentages(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Percentage of missing values per column.
    """

    return (

        dataframe.isna()

        .mean()

        .mul(100)

        .sort_values(

            ascending=False,

        )

    )


################################################################################
# Public API
################################################################################


__all__ = [

    "safe_divide",

    "percentage",

    "clamp",

    "format_number",

    "format_bytes",

    "format_duration",

    "memory_usage",

    "optimise_dataframe",

    "sample_dataframe",

    "validate_dataframe",

    "validate_columns",

    "clean_column_name",

    "clean_column_names",

    "infer_column_type",

    "infer_dataframe_types",

    "set_random_seed",

    "progress_generator",

    "Timer",

    "file_size",

    "file_extension",

    "ensure_directory",

    "flatten_dict",

    "missing_value_percentages",

]
