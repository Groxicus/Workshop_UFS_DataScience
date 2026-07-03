"""
utils.target_detection
======================

Intelligent target variable detection.

This module attempts to identify the most likely target column(s)
using a weighted heuristic scoring system. It supports both
classification and regression datasets without relying on
hard-coded schemas.

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

from config import TARGET_KEYWORDS

###############################################################################
# Dataclasses
###############################################################################


@dataclass(slots=True)
class TargetCandidate:
    """
    Represents a possible target variable.
    """

    column: str
    confidence: float
    reason: str
    problem_hint: str


###############################################################################
# Internal Scoring Utilities
###############################################################################


def _normalise(name: str) -> str:
    """
    Standardise a column name for comparisons.
    """

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
    )


def _keyword_score(column: str) -> float:
    """
    Score based on known target keywords.
    """

    name = _normalise(column)

    score = 0.0

    for keyword in TARGET_KEYWORDS:

        if keyword == name:

            score += 50

        elif keyword in name:

            score += 30

    return score


def _position_score(
    column_index: int,
    total_columns: int,
) -> float:
    """
    Many datasets place the target near the end.
    """

    if total_columns <= 1:
        return 0.0

    ratio = column_index / (total_columns - 1)

    return ratio * 15


def _missing_penalty(series: pd.Series) -> float:
    """
    Penalise columns with excessive missing values.
    """

    missing_ratio = series.isna().mean()

    if missing_ratio >= 0.75:
        return -30

    if missing_ratio >= 0.50:
        return -20

    if missing_ratio >= 0.25:
        return -10

    return 0


###############################################################################
# Classification Heuristics
###############################################################################


def _classification_score(
    series: pd.Series,
) -> float:
    """
    Estimate suitability as a classification target.
    """

    score = 0.0

    unique = series.nunique(dropna=True)

    rows = max(len(series), 1)

    ratio = unique / rows

    if unique == 2:

        score += 40

    elif unique <= 5:

        score += 30

    elif unique <= 10:

        score += 20

    elif unique <= 20:

        score += 10

    if ratio < 0.10:

        score += 15

    if (
        is_object_dtype(series)
        or is_bool_dtype(series)
        or is_categorical_dtype(series)
    ):

        score += 10

    return score


###############################################################################
# Regression Heuristics
###############################################################################


def _regression_score(
    series: pd.Series,
) -> float:
    """
    Estimate suitability as a regression target.
    """

    if not is_numeric_dtype(series):

        return 0

    score = 0.0

    unique = series.nunique(dropna=True)

    rows = max(len(series), 1)

    ratio = unique / rows

    if ratio > 0.30:

        score += 25

    if unique > 20:

        score += 20

    if series.std(skipna=True) > 0:

        score += 10

    return score


###############################################################################
# Candidate Scoring
###############################################################################


def score_target_candidate(
    df: pd.DataFrame,
    column: str,
) -> TargetCandidate:
    """
    Compute the confidence score for a candidate target.
    """

    series = df[column]

    score = 0.0

    reasons: list[str] = []

    keyword = _keyword_score(column)

    if keyword:

        score += keyword
        reasons.append("Target keyword")

    position = _position_score(
        df.columns.get_loc(column),
        len(df.columns),
    )

    score += position

    if position >= 10:
        reasons.append("Near end of dataset")

    penalty = _missing_penalty(series)

    score += penalty

    if penalty:
        reasons.append("Missing value penalty")

    cls_score = _classification_score(series)

    reg_score = _regression_score(series)

    if cls_score >= reg_score:

        score += cls_score

        problem = "classification"

        reasons.append("Low-cardinality target")

    else:

        score += reg_score

        problem = "regression"

        reasons.append("Continuous numeric target")

    score = max(0.0, min(score, 100.0))

    return TargetCandidate(
        column=column,
        confidence=round(score, 2),
        reason=", ".join(reasons),
        problem_hint=problem,
    )
  ###############################################################################
# Candidate Ranking
###############################################################################

def rank_target_candidates(
    df: pd.DataFrame,
    minimum_confidence: float = 20.0,
) -> list[TargetCandidate]:
    """
    Rank all columns by their likelihood of being the target variable.

    Parameters
    ----------
    df : pandas.DataFrame

    minimum_confidence : float
        Minimum confidence required for a candidate to be returned.

    Returns
    -------
    list[TargetCandidate]
    """

    candidates: list[TargetCandidate] = []

    for column in df.columns:

        # Datetime columns are almost never prediction targets.
        if is_datetime64_any_dtype(df[column]):
            continue

        candidate = score_target_candidate(df, column)

        if candidate.confidence >= minimum_confidence:
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )

    return candidates


###############################################################################
# Target Suggestion
###############################################################################

def suggest_target_variables(
    df: pd.DataFrame,
    top_k: int = 5,
) -> list[TargetCandidate]:
    """
    Return the highest-ranked target candidates.

    Parameters
    ----------
    df : pandas.DataFrame

    top_k : int
        Maximum number of candidates returned.

    Returns
    -------
    list[TargetCandidate]
    """

    ranked = rank_target_candidates(df)

    if ranked:

        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Fallback Strategy
    #
    # If no candidate exceeds the minimum confidence threshold,
    # generate scores for every non-datetime column and return the
    # highest-scoring candidates.
    # ------------------------------------------------------------------

    fallback: list[TargetCandidate] = []

    for column in df.columns:

        if is_datetime64_any_dtype(df[column]):
            continue

        fallback.append(
            score_target_candidate(df, column)
        )

    fallback.sort(
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )

    return fallback[:top_k]


###############################################################################
# Convenience Helpers
###############################################################################

def suggested_target(
    df: pd.DataFrame,
) -> TargetCandidate | None:
    """
    Return the single best target candidate.

    Returns
    -------
    TargetCandidate | None
    """

    candidates = suggest_target_variables(
        df,
        top_k=1,
    )

    if candidates:

        return candidates[0]

    return None


def suggested_problem_type(
    df: pd.DataFrame,
) -> str:
    """
    Return the inferred ML problem type.

    Returns
    -------
    str

        "classification"

        "regression"
    """

    candidate = suggested_target(df)

    if candidate is None:

        # Safe default

        numeric = sum(
            is_numeric_dtype(df[column])
            for column in df.columns
        )

        if numeric >= len(df.columns) / 2:

            return "regression"

        return "classification"

    return candidate.problem_hint


###############################################################################
# Candidate DataFrame
###############################################################################

def candidate_dataframe(
    candidates: list[TargetCandidate],
) -> pd.DataFrame:
    """
    Convert candidates into a displayable DataFrame.
    """

    if not candidates:

        return pd.DataFrame(
            columns=[
                "Column",
                "Confidence",
                "Problem Type",
                "Reason",
            ]
        )

    return pd.DataFrame(

        {

            "Column": [
                candidate.column
                for candidate in candidates
            ],

            "Confidence": [
                candidate.confidence
                for candidate in candidates
            ],

            "Problem Type": [
                candidate.problem_hint.title()
                for candidate in candidates
            ],

            "Reason": [
                candidate.reason
                for candidate in candidates
            ],

        }

    )


###############################################################################
# Utility Functions
###############################################################################

def is_good_target_candidate(
    candidate: TargetCandidate,
    threshold: float = 60.0,
) -> bool:
    """
    Determine whether a candidate has sufficiently
    high confidence to recommend automatically.
    """

    return candidate.confidence >= threshold


###############################################################################
# Public API
###############################################################################

__all__ = [

    "TargetCandidate",

    "score_target_candidate",

    "rank_target_candidates",

    "suggest_target_variables",

    "suggested_target",

    "suggested_problem_type",

    "candidate_dataframe",

    "is_good_target_candidate",

]
