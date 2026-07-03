"""
utils.eda
=========

Automated Exploratory Data Analysis (EDA) utilities.

This module provides:

- Dataset summary
- Feature type detection
- Missing value analysis
- Duplicate analysis
- Descriptive statistics
- Correlation analysis
- Automatic Plotly visualisations
- Dataset health scoring

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
    is_categorical_dtype,
)

################################################################################
# Dataclasses
################################################################################


@dataclass(slots=True)
class DatasetSummary:
    """
    Summary information describing a dataset.
    """

    rows: int

    columns: int

    memory_usage_mb: float

    numeric_features: list[str]

    categorical_features: list[str]

    boolean_features: list[str]

    datetime_features: list[str]

    text_features: list[str]

    duplicate_rows: int

    missing_values: int

    constant_columns: list[str]

    high_cardinality_columns: list[str]


@dataclass(slots=True)
class DatasetHealth:
    """
    Overall health assessment.
    """

    score: float

    warnings: list[str] = field(default_factory=list)

    recommendations: list[str] = field(default_factory=list)


################################################################################
# Feature Detection
################################################################################


def detect_feature_types(
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Automatically classify dataframe columns.
    """

    numeric = []

    categorical = []

    boolean = []

    datetime = []

    text = []

    for column in df.columns:

        series = df[column]

        if is_bool_dtype(series):

            boolean.append(column)

            continue

        if is_numeric_dtype(series):

            numeric.append(column)

            continue

        if is_datetime64_any_dtype(series):

            datetime.append(column)

            continue

        if (
            is_object_dtype(series)
            or is_categorical_dtype(series)
        ):

            unique_ratio = (
                series.nunique(dropna=True)
                / max(len(series), 1)
            )

            avg_length = (
                series.fillna("")
                .astype(str)
                .str.len()
                .mean()
            )

            if (
                unique_ratio > 0.70
                or avg_length > 25
            ):

                text.append(column)

            else:

                categorical.append(column)

    return {

        "numeric": numeric,

        "categorical": categorical,

        "boolean": boolean,

        "datetime": datetime,

        "text": text,

    }


################################################################################
# Missing Values
################################################################################


def missing_value_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Missing values by feature.
    """

    summary = pd.DataFrame(

        {

            "Missing Values": df.isna().sum(),

            "Percentage": (
                df.isna().mean() * 100
            ).round(2),

        }

    )

    return summary.sort_values(
        "Percentage",
        ascending=False,
    )


################################################################################
# Duplicate Summary
################################################################################


def duplicate_summary(
    df: pd.DataFrame,
) -> dict:
    """
    Duplicate statistics.
    """

    duplicates = df.duplicated()

    return {

        "duplicate_rows": int(
            duplicates.sum()
        ),

        "duplicate_percentage": round(

            duplicates.mean() * 100,

            2,

        ),

    }


################################################################################
# Constant Columns
################################################################################


def constant_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect constant-value columns.
    """

    return [

        column

        for column in df.columns

        if df[column].nunique(
            dropna=False
        ) <= 1

    ]


################################################################################
# High Cardinality
################################################################################


def high_cardinality_columns(
    df: pd.DataFrame,
    threshold: int = 100,
) -> list[str]:
    """
    Detect high-cardinality categorical features.
    """

    columns = []

    for column in df.select_dtypes(

        include=[
            "object",
            "category",
        ]

    ):

        if df[column].nunique() >= threshold:

            columns.append(column)

    return columns


################################################################################
# Dataset Summary
################################################################################


def summarize_dataset(
    df: pd.DataFrame,
) -> DatasetSummary:
    """
    Produce a complete dataset summary.
    """

    feature_types = detect_feature_types(df)

    return DatasetSummary(

        rows=len(df),

        columns=df.shape[1],

        memory_usage_mb=round(

            df.memory_usage(
                deep=True
            ).sum()
            / (1024 ** 2),

            2,

        ),

        numeric_features=feature_types["numeric"],

        categorical_features=feature_types["categorical"],

        boolean_features=feature_types["boolean"],

        datetime_features=feature_types["datetime"],

        text_features=feature_types["text"],

        duplicate_rows=int(
            df.duplicated().sum()
        ),

        missing_values=int(
            df.isna().sum().sum()
        ),

        constant_columns=constant_columns(df),

        high_cardinality_columns=high_cardinality_columns(df),

    )


################################################################################
# Descriptive Statistics
################################################################################


def descriptive_statistics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extended descriptive statistics.
    """

    return df.describe(
        include="all",
        datetime_is_numeric=True,
    ).transpose()


################################################################################
# Correlation Matrix
################################################################################


def correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Numeric feature correlation matrix.
    """

    numeric = df.select_dtypes(
        include=np.number,
    )

    if numeric.empty:

        return pd.DataFrame()

    return numeric.corr(
        numeric_only=True,
    )
################################################################################
# Interactive Plotly Visualizations
################################################################################

def histogram_plot(
    df: pd.DataFrame,
    column: str,
):
    """
    Interactive histogram for numeric features.
    """

    return px.histogram(
        df,
        x=column,
        marginal="box",
        template="plotly_white",
        title=f"Distribution of {column}",
    )


def box_plot(
    df: pd.DataFrame,
    column: str,
):
    """
    Interactive box plot.
    """

    return px.box(
        df,
        y=column,
        points="outliers",
        template="plotly_white",
        title=f"Box Plot - {column}",
    )


def violin_plot(
    df: pd.DataFrame,
    column: str,
):
    """
    Interactive violin plot.
    """

    return px.violin(
        df,
        y=column,
        box=True,
        points="outliers",
        template="plotly_white",
        title=f"Violin Plot - {column}",
    )


def count_plot(
    df: pd.DataFrame,
    column: str,
):
    """
    Interactive categorical count plot.
    """

    counts = (
        df[column]
        .fillna("Missing")
        .value_counts()
        .reset_index()
    )

    counts.columns = ["Category", "Count"]

    return px.bar(
        counts,
        x="Category",
        y="Count",
        template="plotly_white",
        title=f"{column} Distribution",
    )


################################################################################
# Scatter Plots
################################################################################

def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
):
    """
    Interactive scatter plot.
    """

    return px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        template="plotly_white",
        title=f"{x} vs {y}",
    )


################################################################################
# Correlation Heatmap
################################################################################

def correlation_heatmap(
    df: pd.DataFrame,
):
    """
    Interactive correlation heatmap.
    """

    corr = correlation_matrix(df)

    if corr.empty:

        return go.Figure()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
    )

    fig.update_layout(
        title="Correlation Heatmap"
    )

    return fig


################################################################################
# Missing Value Heatmap
################################################################################

def missing_value_heatmap(
    df: pd.DataFrame,
):
    """
    Missing value visualization.
    """

    fig = px.imshow(
        df.isna(),
        aspect="auto",
        color_continuous_scale=[
            "#2ecc71",
            "#e74c3c",
        ],
    )

    fig.update_layout(
        title="Missing Value Heatmap"
    )

    return fig


################################################################################
# Pair Plot (Sampled)
################################################################################

def pair_plot(
    df: pd.DataFrame,
    sample_size: int = 500,
):
    """
    Scatter matrix for numeric variables.

    Automatically samples large datasets.
    """

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] < 2:

        return go.Figure()

    if len(numeric) > sample_size:

        numeric = numeric.sample(
            sample_size,
            random_state=42,
        )

    return px.scatter_matrix(
        numeric,
        template="plotly_white",
    )


################################################################################
# Target Distribution
################################################################################

def target_distribution(
    df: pd.DataFrame,
    target: str,
):
    """
    Automatically visualize the selected target.
    """

    if target not in df.columns:

        return go.Figure()

    if is_numeric_dtype(df[target]):

        return histogram_plot(
            df,
            target,
        )

    return count_plot(
        df,
        target,
    )


################################################################################
# Dataset Health
################################################################################

def dataset_health(
    df: pd.DataFrame,
) -> DatasetHealth:
    """
    Compute an overall dataset health score.
    """

    score = 100.0

    warnings = []

    recommendations = []

    missing_pct = (
        df.isna()
        .sum()
        .sum()
        / max(df.size, 1)
    )

    if missing_pct > 0.20:

        score -= 20

        warnings.append(
            "High proportion of missing values."
        )

        recommendations.append(
            "Consider imputing missing values."
        )

    duplicates = df.duplicated().sum()

    if duplicates:

        score -= 10

        warnings.append(
            f"{duplicates} duplicate rows detected."
        )

        recommendations.append(
            "Remove duplicate observations."
        )

    constants = constant_columns(df)

    if constants:

        score -= 5

        warnings.append(
            "Constant-value columns detected."
        )

        recommendations.append(
            "Remove constant features."
        )

    high_card = high_cardinality_columns(df)

    if high_card:

        score -= 5

        warnings.append(
            "High-cardinality categorical columns found."
        )

        recommendations.append(
            "Consider frequency encoding or removal."
        )

    score = max(score, 0)

    return DatasetHealth(
        score=round(score, 1),
        warnings=warnings,
        recommendations=recommendations,
    )


################################################################################
# EDA Report
################################################################################

def eda_report(
    df: pd.DataFrame,
) -> dict:
    """
    Generate a complete EDA report.
    """

    return {

        "summary": summarize_dataset(df),

        "health": dataset_health(df),

        "missing": missing_value_summary(df),

        "duplicates": duplicate_summary(df),

        "statistics": descriptive_statistics(df),

        "correlation": correlation_matrix(df),

    }


################################################################################
# Public API
################################################################################

__all__ = [

    "DatasetSummary",

    "DatasetHealth",

    "detect_feature_types",

    "missing_value_summary",

    "duplicate_summary",

    "constant_columns",

    "high_cardinality_columns",

    "summarize_dataset",

    "descriptive_statistics",

    "correlation_matrix",

    "histogram_plot",

    "box_plot",

    "violin_plot",

    "count_plot",

    "scatter_plot",

    "correlation_heatmap",

    "missing_value_heatmap",

    "pair_plot",

    "target_distribution",

    "dataset_health",

    "eda_report",

]
