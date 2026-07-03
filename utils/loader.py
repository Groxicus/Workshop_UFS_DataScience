"""
utils.loader
============

Dataset loading, validation, and metadata extraction utilities.

This module provides:

- CSV loading
- Excel loading
- Automatic file type detection
- Streamlit caching
- Dataset validation
- File metadata
- Preview generation
- Basic schema information

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

###############################################################################
# Supported File Types
###############################################################################

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
}

###############################################################################
# Data Classes
###############################################################################


@dataclass(slots=True)
class FileInformation:
    """
    Metadata describing an uploaded dataset.
    """

    file_name: str
    extension: str
    size_bytes: int
    rows: int
    columns: int
    memory_usage: int


###############################################################################
# Internal Helpers
###############################################################################


def _extension(uploaded_file) -> str:
    """
    Return the file extension in lowercase.
    """
    return Path(uploaded_file.name).suffix.lower()


def _validate_extension(uploaded_file) -> None:
    """
    Validate the uploaded file extension.
    """

    extension = _extension(uploaded_file)

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            f"Unsupported file type '{extension}'. "
            "Supported formats are CSV and XLSX."
        )


###############################################################################
# Cached Dataset Loading
###############################################################################


@st.cache_data(show_spinner=False)
def _read_csv(uploaded_file) -> pd.DataFrame:
    """
    Read a CSV file with fallback encodings.
    """

    encodings = (
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",
    )

    last_exception = None

    for encoding in encodings:

        try:

            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding=encoding,
            )

        except Exception as exc:

            last_exception = exc

    raise RuntimeError(
        "Unable to read CSV using supported encodings."
    ) from last_exception


@st.cache_data(show_spinner=False)
def _read_excel(uploaded_file) -> pd.DataFrame:
    """
    Read an Excel workbook.
    """

    uploaded_file.seek(0)

    return pd.read_excel(uploaded_file)


###############################################################################
# Public Loading Function
###############################################################################


@st.cache_data(show_spinner=True)
def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load a dataset from an uploaded Streamlit file.

    Parameters
    ----------
    uploaded_file
        UploadedFile object returned by st.file_uploader()

    Returns
    -------
    pandas.DataFrame
    """

    if uploaded_file is None:

        raise ValueError("No file has been uploaded.")

    _validate_extension(uploaded_file)

    extension = _extension(uploaded_file)

    if extension == ".csv":

        df = _read_csv(uploaded_file)

    elif extension == ".xlsx":

        df = _read_excel(uploaded_file)

    else:

        raise RuntimeError(
            "Unexpected file type."
        )

    if df.empty:

        raise ValueError(
            "The uploaded dataset contains no rows."
        )

    return df.copy()


###############################################################################
# Dataset Validation
###############################################################################


def validate_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate a loaded dataset.

    Returns
    -------
    (valid, messages)
    """

    issues: list[str] = []

    if df.empty:

        issues.append("Dataset is empty.")

    if df.shape[1] == 0:

        issues.append("Dataset contains no columns.")

    duplicated_columns = df.columns[df.columns.duplicated()]

    if len(duplicated_columns):

        issues.append(
            "Duplicate column names detected."
        )

    completely_empty = df.columns[df.isna().all()]

    if len(completely_empty):

        issues.append(
            f"{len(completely_empty)} completely empty column(s)."
        )

    return (
        len(issues) == 0,
        issues,
    )


###############################################################################
# File Metadata
###############################################################################


def get_file_info(
    uploaded_file,
    df: pd.DataFrame,
) -> FileInformation:
    """
    Return metadata describing the uploaded dataset.
    """

    return FileInformation(
        file_name=uploaded_file.name,
        extension=_extension(uploaded_file),
        size_bytes=uploaded_file.size,
        rows=df.shape[0],
        columns=df.shape[1],
        memory_usage=int(
            df.memory_usage(deep=True).sum()
        ),
    )


###############################################################################
# Preview Helpers
###############################################################################


def preview_dataset(
    df: pd.DataFrame,
    rows: int = 10,
) -> pd.DataFrame:
    """
    Return the first rows of a dataset.
    """

    return df.head(rows).copy()


def sample_dataset(
    df: pd.DataFrame,
    n: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Return a sampled dataset.

    Useful for visualisations on very
    large datasets.
    """

    if len(df) <= n:

        return df.copy()

    return df.sample(
        n=n,
        random_state=random_state,
    ).copy()
###############################################################################
# Data Type Optimisation
###############################################################################

def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage by selecting more efficient dtypes.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
    """

    data = df.copy()

    for column in data.columns:

        series = data[column]

        # Integer columns
        if pd.api.types.is_integer_dtype(series):

            data[column] = pd.to_numeric(
                series,
                downcast="integer",
            )

        # Float columns
        elif pd.api.types.is_float_dtype(series):

            data[column] = pd.to_numeric(
                series,
                downcast="float",
            )

        # Boolean-like object columns
        elif pd.api.types.is_object_dtype(series):

            unique = (
                series.dropna()
                .astype(str)
                .str.lower()
                .unique()
            )

            bool_values = {
                "true",
                "false",
                "yes",
                "no",
                "0",
                "1",
            }

            if (
                len(unique) <= 2
                and set(unique).issubset(bool_values)
            ):

                mapping = {
                    "true": True,
                    "false": False,
                    "yes": True,
                    "no": False,
                    "1": True,
                    "0": False,
                }

                data[column] = (
                    series.astype(str)
                    .str.lower()
                    .map(mapping)
                )

    return data


###############################################################################
# Automatic Date Parsing
###############################################################################

def detect_datetime_columns(
    df: pd.DataFrame,
    sample_size: int = 1000,
    threshold: float = 0.80,
) -> list[str]:
    """
    Detect columns that appear to contain dates.

    Parameters
    ----------
    threshold
        Fraction of successfully parsed values required.
    """

    detected: list[str] = []

    sample = df.head(sample_size)

    for column in sample.columns:

        if pd.api.types.is_datetime64_any_dtype(sample[column]):

            detected.append(column)
            continue

        if not pd.api.types.is_object_dtype(sample[column]):

            continue

        try:

            parsed = pd.to_datetime(
                sample[column],
                errors="coerce",
            )

            ratio = parsed.notna().mean()

            if ratio >= threshold:

                detected.append(column)

        except Exception:

            continue

    return detected


def parse_datetime_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert detected date columns to datetime.
    """

    data = df.copy()

    for column in detect_datetime_columns(data):

        data[column] = pd.to_datetime(
            data[column],
            errors="coerce",
        )

    return data


###############################################################################
# Basic Cleaning
###############################################################################

def basic_cleaning(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform lightweight cleaning.

    - Trim column names
    - Remove duplicate columns
    - Parse dates
    - Optimise dtypes
    """

    data = df.copy()

    data.columns = (
        data.columns.astype(str)
        .str.strip()
    )

    data = data.loc[
        :,
        ~data.columns.duplicated(),
    ]

    data = parse_datetime_columns(data)

    data = optimise_dtypes(data)

    return data


###############################################################################
# Memory Information
###############################################################################

def memory_usage_mb(
    df: pd.DataFrame,
) -> float:
    """
    Return dataframe memory usage in MB.
    """

    return (
        df.memory_usage(deep=True)
        .sum()
        / 1024
        / 1024
    )


###############################################################################
# Dataset Summary
###############################################################################

def dataset_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Return a lightweight summary describing the dataset.
    """

    return {

        "rows": len(df),

        "columns": df.shape[1],

        "missing_values":
            int(df.isna().sum().sum()),

        "duplicate_rows":
            int(df.duplicated().sum()),

        "memory_mb":
            round(memory_usage_mb(df), 2),

        "numeric_columns":
            len(
                df.select_dtypes(
                    include="number"
                ).columns
            ),

        "categorical_columns":
            len(
                df.select_dtypes(
                    include=["object", "category"]
                ).columns
            ),

        "boolean_columns":
            len(
                df.select_dtypes(
                    include="bool"
                ).columns
            ),

        "datetime_columns":
            len(
                df.select_dtypes(
                    include="datetime"
                ).columns
            ),
    }


###############################################################################
# Convenience Loader
###############################################################################

def load_and_prepare_dataset(
    uploaded_file,
) -> tuple[pd.DataFrame, FileInformation]:
    """
    Complete loading pipeline.

    Returns
    -------
    (clean_dataframe, file_information)
    """

    df = load_dataset(uploaded_file)

    valid, issues = validate_dataset(df)

    if not valid:

        raise ValueError(
            "\n".join(issues)
        )

    df = basic_cleaning(df)

    info = get_file_info(
        uploaded_file,
        df,
    )

    return df, info


###############################################################################
# Public Exports
###############################################################################

__all__ = [

    "FileInformation",

    "SUPPORTED_EXTENSIONS",

    "load_dataset",

    "load_and_prepare_dataset",

    "validate_dataset",

    "get_file_info",

    "preview_dataset",

    "sample_dataset",

    "optimise_dtypes",

    "detect_datetime_columns",

    "parse_datetime_columns",

    "basic_cleaning",

    "memory_usage_mb",

    "dataset_summary",

]
