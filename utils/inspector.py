"""
utils.inspector
===============

Dataset inspection and profiling utilities.

This module performs automatic analysis of uploaded datasets,
including feature type detection, missing values, duplicates,
constant columns, high-cardinality features, and dataset quality
assessment.

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

###############################################################################
# Data Classes
###############################################################################


@dataclass(slots=True)
class DatasetProfile:
    """
    Complete dataset profile.
    """

    rows: int
    columns: int

    numeric_features: list[str]
    categorical_features: list[str]
    boolean_features: list[str]
    datetime_features: list[str]
    text_features: list[str]

    missing_values: dict[str, int]
    missing_percentages: dict[str, float]

    duplicate_rows: int

    constant_columns: list[str]

    high_cardinality_columns: list[str]

    memory_usage_mb: float

    health_score: float


###############################################################################
# Feature Type Detection
###############################################################################


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    """
    Return numeric columns.
    """

    return (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


def get_boolean_features(df: pd.DataFrame) -> list[str]:
    """
    Return boolean columns.
    """

    return (
        df.select_dtypes(
            include="bool"
        )
        .columns
        .tolist()
    )


def get_datetime_features(df: pd.DataFrame) -> list[str]:
    """
    Return datetime columns.
    """

    return (
        df.select_dtypes(
            include="datetime"
        )
        .columns
        .tolist()
    )


def get_categorical_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return categorical features.
    """

    categorical = df.select_dtypes(
        include=["object", "category"]
    )

    return categorical.columns.tolist()


def get_text_features(
    df: pd.DataFrame,
    unique_ratio: float = 0.70,
) -> list[str]:
    """
    Detect free-text columns.

    A column is considered text if:

    - object dtype
    - high uniqueness ratio
    """

    detected: list[str] = []

    for column in df.select_dtypes(
        include="object"
    ):

        series = df[column]

        if len(series) == 0:

            continue

        ratio = (
            series.nunique(dropna=True)
            / max(len(series), 1)
        )

        if ratio >= unique_ratio:

            detected.append(column)

    return detected


###############################################################################
# Missing Values
###############################################################################


def missing_value_counts(
    df: pd.DataFrame,
) -> dict[str, int]:
    """
    Missing values per column.
    """

    return (
        df.isna()
        .sum()
        .astype(int)
        .to_dict()
    )


def missing_value_percentages(
    df: pd.DataFrame,
) -> dict[str, float]:
    """
    Missing percentage per column.
    """

    if len(df) == 0:

        return {
            column: 0.0
            for column in df.columns
        }

    percentages = (
        df.isna()
        .mean()
        .mul(100)
        .round(2)
    )

    return percentages.to_dict()


###############################################################################
# Duplicate Records
###############################################################################


def duplicate_row_count(
    df: pd.DataFrame,
) -> int:
    """
    Count duplicate rows.
    """

    return int(
        df.duplicated().sum()
    )


###############################################################################
# Constant Columns
###############################################################################


def constant_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Columns containing only one value.
    """

    constants = []

    for column in df.columns:

        if (
            df[column]
            .nunique(dropna=False)
            <= 1
        ):

            constants.append(column)

    return constants


###############################################################################
# High Cardinality
###############################################################################


def high_cardinality_columns(
    df: pd.DataFrame,
    threshold: int = 50,
) -> list[str]:
    """
    Detect categorical columns with many unique values.
    """

    detected = []

    categorical = df.select_dtypes(
        include=["object", "category"]
    )

    for column in categorical.columns:

        unique = categorical[column].nunique()

        if unique >= threshold:

            detected.append(column)

    return detected
###############################################################################
# Memory Usage
###############################################################################

def memory_usage_mb(
    df: pd.DataFrame,
) -> float:
    """
    Return dataframe memory usage in megabytes.
    """

    return round(
        df.memory_usage(deep=True).sum() / (1024 ** 2),
        2,
    )


###############################################################################
# Feature Statistics
###############################################################################

def feature_statistics(
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """
    Return summary statistics for each column.
    """

    statistics: dict[str, dict[str, Any]] = {}

    for column in df.columns:

        series = df[column]

        statistics[column] = {

            "dtype": str(series.dtype),

            "non_null": int(series.count()),

            "missing": int(series.isna().sum()),

            "unique": int(series.nunique(dropna=True)),

        }

        if pd.api.types.is_numeric_dtype(series):

            statistics[column].update(

                {

                    "min": float(series.min())
                    if series.count()
                    else None,

                    "max": float(series.max())
                    if series.count()
                    else None,

                    "mean": float(series.mean())
                    if series.count()
                    else None,

                    "std": float(series.std())
                    if series.count()
                    else None,

                }

            )

    return statistics


###############################################################################
# Correlation Summary
###############################################################################

def correlation_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return correlation matrix for numeric columns.

    Empty dataframe is returned when fewer than two
    numeric columns exist.
    """

    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:

        return pd.DataFrame()

    return numeric.corr(numeric_only=True)


###############################################################################
# Data Quality Checks
###############################################################################

def quality_issues(
    df: pd.DataFrame,
) -> list[str]:
    """
    Identify potential data quality issues.
    """

    issues: list[str] = []

    if duplicate_row_count(df):

        issues.append(
            f"{duplicate_row_count(df)} duplicate rows detected."
        )

    constants = constant_columns(df)

    if constants:

        issues.append(
            f"{len(constants)} constant-value columns detected."
        )

    missing = df.isna().sum()

    problematic = missing[missing > 0]

    if len(problematic):

        issues.append(
            f"{len(problematic)} columns contain missing values."
        )

    high_card = high_cardinality_columns(df)

    if high_card:

        issues.append(
            f"{len(high_card)} high-cardinality columns detected."
        )

    return issues


###############################################################################
# Dataset Health Score
###############################################################################

def dataset_health_score(
    df: pd.DataFrame,
) -> float:
    """
    Compute a simple health score between 0 and 100.

    The score is heuristic-based and penalizes:

    - Missing values
    - Duplicate rows
    - Constant columns
    - High-cardinality categorical columns
    """

    score = 100.0

    total_cells = max(df.shape[0] * df.shape[1], 1)

    missing_ratio = df.isna().sum().sum() / total_cells

    duplicate_ratio = duplicate_row_count(df) / max(len(df), 1)

    constant_ratio = (
        len(constant_columns(df))
        / max(df.shape[1], 1)
    )

    cardinality_ratio = (
        len(high_cardinality_columns(df))
        / max(df.shape[1], 1)
    )

    score -= missing_ratio * 40
    score -= duplicate_ratio * 20
    score -= constant_ratio * 20
    score -= cardinality_ratio * 20

    return round(max(score, 0), 2)


###############################################################################
# Main Inspection Pipeline
###############################################################################

def inspect_dataset(
    df: pd.DataFrame,
) -> DatasetProfile:
    """
    Perform a comprehensive inspection of a dataset.
    """

    return DatasetProfile(

        rows=len(df),

        columns=df.shape[1],

        numeric_features=get_numeric_features(df),

        categorical_features=get_categorical_features(df),

        boolean_features=get_boolean_features(df),

        datetime_features=get_datetime_features(df),

        text_features=get_text_features(df),

        missing_values=missing_value_counts(df),

        missing_percentages=missing_value_percentages(df),

        duplicate_rows=duplicate_row_count(df),

        constant_columns=constant_columns(df),

        high_cardinality_columns=high_cardinality_columns(df),

        memory_usage_mb=memory_usage_mb(df),

        health_score=dataset_health_score(df),

    )


###############################################################################
# Public API
###############################################################################

__all__ = [

    "DatasetProfile",

    "inspect_dataset",

    "feature_statistics",

    "correlation_summary",

    "quality_issues",

    "dataset_health_score",

    "memory_usage_mb",

    "get_numeric_features",

    "get_categorical_features",

    "get_boolean_features",

    "get_datetime_features",

    "get_text_features",

    "missing_value_counts",

    "missing_value_percentages",

    "duplicate_row_count",

    "constant_columns",

    "high_cardinality_columns",

]
