"""
utils.problem_detection
=======================

Automatic machine learning problem detection.

This module determines whether the selected target variable
represents a:

- Classification problem
- Regression problem
- Time Series Forecasting problem

The decision is based on both the target variable and the
overall dataset structure.

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

################################################################################
# Dataclass
################################################################################


@dataclass(slots=True)
class ProblemDetectionResult:
    """
    Result returned by detect_problem_type().
    """

    problem_type: str
    confidence: float
    reason: str
    is_time_series: bool


################################################################################
# Helper Functions
################################################################################


def has_datetime_features(df: pd.DataFrame) -> bool:
    """
    Determine whether the dataset contains
    datetime columns.
    """

    return any(
        is_datetime64_any_dtype(df[column])
        for column in df.columns
    )


def datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    Return detected datetime columns.
    """

    return [
        column
        for column in df.columns
        if is_datetime64_any_dtype(df[column])
    ]


################################################################################
# Time Series Detection
################################################################################


def detect_time_series(
    df: pd.DataFrame,
    target: str,
) -> bool:
    """
    Determine whether this appears to be
    a time series forecasting dataset.

    Conditions:

    • At least one datetime column
    • Numeric target
    """

    if target not in df.columns:

        return False

    if not has_datetime_features(df):

        return False

    return is_numeric_dtype(df[target])


################################################################################
# Classification Detection
################################################################################


def classification_confidence(
    series: pd.Series,
) -> float:
    """
    Estimate the confidence that the target
    represents a classification problem.
    """

    score = 0.0

    unique = series.nunique(dropna=True)

    rows = max(len(series), 1)

    ratio = unique / rows

    if is_bool_dtype(series):

        score += 60

    if is_object_dtype(series):

        score += 40

    if is_categorical_dtype(series):

        score += 40

    if unique == 2:

        score += 35

    elif unique <= 5:

        score += 25

    elif unique <= 10:

        score += 15

    elif unique <= 20:

        score += 8

    if ratio < 0.05:

        score += 20

    elif ratio < 0.10:

        score += 10

    return min(score, 100.0)


################################################################################
# Regression Detection
################################################################################


def regression_confidence(
    series: pd.Series,
) -> float:
    """
    Estimate the confidence that the target
    represents a regression problem.
    """

    if not is_numeric_dtype(series):

        return 0.0

    score = 0.0

    unique = series.nunique(dropna=True)

    rows = max(len(series), 1)

    ratio = unique / rows

    if unique > 20:

        score += 30

    if ratio > 0.25:

        score += 35

    if series.std(skipna=True) > 0:

        score += 20

    if series.min(skipna=True) != series.max(skipna=True):

        score += 15

    return min(score, 100.0)


################################################################################
# Main Detection
################################################################################


def detect_problem_type(
    df: pd.DataFrame,
    target: str,
) -> ProblemDetectionResult:
    """
    Determine the machine learning problem type.

    Parameters
    ----------
    df
        Dataset

    target
        Selected target variable

    Returns
    -------
    ProblemDetectionResult
    """

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' does not exist."
        )

    target_series = df[target]

    # ---------------------------------------------------------
    # Time Series Detection
    # ---------------------------------------------------------

    if detect_time_series(df, target):

        return ProblemDetectionResult(

            problem_type="time_series",

            confidence=95.0,

            reason=(
                "Detected datetime feature(s) "
                "with a numeric target."
            ),

            is_time_series=True,

        )

    # ---------------------------------------------------------
    # Classification vs Regression
    # ---------------------------------------------------------

    classification = classification_confidence(
        target_series
    )

    regression = regression_confidence(
        target_series
    )

    if classification >= regression:

        return ProblemDetectionResult(

            problem_type="classification",

            confidence=round(classification, 2),

            reason=(
                "Target has relatively few unique "
                "values and resembles class labels."
            ),

            is_time_series=False,

        )

    return ProblemDetectionResult(

        problem_type="regression",

        confidence=round(regression, 2),

        reason=(
            "Target is continuous with sufficient "
            "numeric variation."
        ),

        is_time_series=False,

    )


################################################################################
# Convenience Functions
################################################################################


def is_classification(
    df: pd.DataFrame,
    target: str,
) -> bool:
    """
    Convenience wrapper.
    """

    return (
        detect_problem_type(
            df,
            target,
        ).problem_type
        == "classification"
    )


def is_regression(
    df: pd.DataFrame,
    target: str,
) -> bool:
    """
    Convenience wrapper.
    """

    return (
        detect_problem_type(
            df,
            target,
        ).problem_type
        == "regression"
    )


def is_time_series(
    df: pd.DataFrame,
    target: str,
) -> bool:
    """
    Convenience wrapper.
    """

    return (
        detect_problem_type(
            df,
            target,
        ).problem_type
        == "time_series"
    )


################################################################################
# Public API
################################################################################

__all__ = [

    "ProblemDetectionResult",

    "detect_problem_type",

    "classification_confidence",

    "regression_confidence",

    "detect_time_series",

    "datetime_columns",

    "has_datetime_features",

    "is_classification",

    "is_regression",

    "is_time_series",

]
