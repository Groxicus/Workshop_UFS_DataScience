"""
utils.feature_engineering
=========================

Automatic feature engineering for ML Analytics Studio.

This module provides reusable feature engineering utilities that
operate dynamically on arbitrary datasets.

Implemented in this part:

- Configuration dataclasses
- Feature engineering metadata
- Numeric feature detection
- Interaction features
- Ratio features
- Aggregate statistics
- Safe mathematical operations

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

################################################################################
# Dataclasses
################################################################################


@dataclass(slots=True)
class FeatureEngineeringConfig:
    """
    Configuration controlling automatic feature engineering.
    """

    create_interactions: bool = True

    create_ratios: bool = True

    create_aggregates: bool = True

    create_polynomial: bool = False

    polynomial_degree: int = 2

    max_numeric_features: int = 20

    max_interaction_pairs: int = 50

    drop_invalid_features: bool = True


@dataclass(slots=True)
class FeatureEngineeringReport:
    """
    Metadata describing generated features.
    """

    interaction_features: list[str] = field(default_factory=list)

    ratio_features: list[str] = field(default_factory=list)

    aggregate_features: list[str] = field(default_factory=list)

    removed_features: list[str] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)


################################################################################
# Numeric Feature Detection
################################################################################


def numeric_columns(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
) -> list[str]:
    """
    Return numeric feature columns.
    """

    exclude = exclude or []

    return [

        column

        for column in df.select_dtypes(
            include=np.number,
        ).columns

        if column not in exclude

    ]


################################################################################
# Safe Operations
################################################################################


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divide safely while avoiding division-by-zero.
    """

    denominator = denominator.replace(
        0,
        np.nan,
    )

    return numerator / denominator


################################################################################
# Interaction Features
################################################################################


def create_interaction_features(
    df: pd.DataFrame,
    columns: list[str],
    max_pairs: int = 50,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate multiplicative interaction features.

    Example

    A * B
    Height * Weight
    Age * Salary
    """

    data = df.copy()

    created: list[str] = []

    pairs = list(
        combinations(columns, 2)
    )[:max_pairs]

    for left, right in pairs:

        name = f"{left}_x_{right}"

        data[name] = (
            data[left]
            * data[right]
        )

        created.append(name)

    return data, created


################################################################################
# Ratio Features
################################################################################


def create_ratio_features(
    df: pd.DataFrame,
    columns: list[str],
    max_pairs: int = 50,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate ratio features.

    Example

    Salary / Age

    Weight / Height
    """

    data = df.copy()

    created: list[str] = []

    pairs = list(
        combinations(columns, 2)
    )[:max_pairs]

    for left, right in pairs:

        name = f"{left}_div_{right}"

        data[name] = safe_divide(

            data[left],

            data[right],

        )

        created.append(name)

    return data, created


################################################################################
# Aggregate Numeric Features
################################################################################


def create_numeric_aggregates(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate row-wise aggregate statistics.
    """

    data = df.copy()

    created: list[str] = []

    if len(columns) < 2:

        return data, created

    aggregates = {

        "row_sum":
            data[columns].sum(axis=1),

        "row_mean":
            data[columns].mean(axis=1),

        "row_std":
            data[columns].std(axis=1),

        "row_min":
            data[columns].min(axis=1),

        "row_max":
            data[columns].max(axis=1),

        "row_range":
            data[columns].max(axis=1)
            - data[columns].min(axis=1),

    }

    for feature, values in aggregates.items():

        data[feature] = values

        created.append(feature)

    return data, created


################################################################################
# Invalid Feature Cleanup
################################################################################


def remove_invalid_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove unusable engineered features.

    Removes columns containing:

    - only NaN
    - infinite values
    - constant values
    """

    data = df.copy()

    removed: list[str] = []

    for column in list(data.columns):

        series = data[column]

        if series.isna().all():

            removed.append(column)

            data.drop(
                columns=[column],
                inplace=True,
            )

            continue

        if np.isinf(
            pd.to_numeric(
                series,
                errors="coerce",
            )
        ).any():

            data[column] = (
                pd.to_numeric(
                    series,
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

        if data[column].nunique(
            dropna=False
        ) <= 1:

            removed.append(column)

            data.drop(
                columns=[column],
                inplace=True,
            )

    return data, removed
################################################################################
# Polynomial Feature Generation
################################################################################

from sklearn.preprocessing import PolynomialFeatures


def create_polynomial_features(
    df: pd.DataFrame,
    columns: list[str],
    degree: int = 2,
    interaction_only: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate polynomial features for numeric columns.

    Parameters
    ----------
    df
        Source dataframe.

    columns
        Numeric columns to transform.

    degree
        Polynomial degree.

    interaction_only
        If True, only interaction terms are generated.
    """

    data = df.copy()

    created: list[str] = []

    if len(columns) == 0:

        return data, created

    transformer = PolynomialFeatures(

        degree=degree,

        include_bias=False,

        interaction_only=interaction_only,

    )

    transformed = transformer.fit_transform(
        data[columns]
    )

    feature_names = transformer.get_feature_names_out(
        columns
    )

    existing = set(data.columns)

    for index, name in enumerate(feature_names):

        if name in existing:

            continue

        data[name] = transformed[:, index]

        created.append(name)

    return data, created


################################################################################
# Logarithmic Features
################################################################################

def create_log_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create log1p transformed features.

    Only non-negative values are transformed.
    """

    data = df.copy()

    created: list[str] = []

    for column in columns:

        minimum = data[column].min(skipna=True)

        if pd.isna(minimum):

            continue

        if minimum < 0:

            continue

        feature = f"{column}_log"

        data[feature] = np.log1p(
            data[column]
        )

        created.append(feature)

    return data, created


################################################################################
# Square Root Features
################################################################################

def create_sqrt_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create square-root transformed features.

    Negative values are skipped.
    """

    data = df.copy()

    created: list[str] = []

    for column in columns:

        minimum = data[column].min(skipna=True)

        if pd.isna(minimum):

            continue

        if minimum < 0:

            continue

        feature = f"{column}_sqrt"

        data[feature] = np.sqrt(
            data[column]
        )

        created.append(feature)

    return data, created


################################################################################
# Squared Features
################################################################################

def create_squared_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create squared versions of numeric features.
    """

    data = df.copy()

    created: list[str] = []

    for column in columns:

        feature = f"{column}_squared"

        data[feature] = data[column] ** 2

        created.append(feature)

    return data, created


################################################################################
# Cubed Features
################################################################################

def create_cubed_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create cubed versions of numeric features.
    """

    data = df.copy()

    created: list[str] = []

    for column in columns:

        feature = f"{column}_cubed"

        data[feature] = data[column] ** 3

        created.append(feature)

    return data, created


################################################################################
# Absolute Value Features
################################################################################

def create_absolute_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate absolute-value features.
    """

    data = df.copy()

    created: list[str] = []

    for column in columns:

        feature = f"{column}_abs"

        data[feature] = data[column].abs()

        created.append(feature)

    return data, created


################################################################################
# Reciprocal Features
################################################################################

def create_reciprocal_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create reciprocal (1/x) features.

    Zero values are converted to NaN.
    """

    data = df.copy()

    created: list[str] = []

    for column in columns:

        feature = f"{column}_reciprocal"

        values = data[column].replace(
            0,
            np.nan,
        )

        data[feature] = 1.0 / values

        created.append(feature)

    return data, created


################################################################################
# Numeric Transformation Pipeline
################################################################################

def engineer_numeric_features(
    df: pd.DataFrame,
    config: FeatureEngineeringConfig,
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    """
    Generate engineered numeric features.

    Returns
    -------
    (dataframe, report)
    """

    exclude = exclude or []

    data = df.copy()

    report = FeatureEngineeringReport()

    columns = numeric_columns(
        data,
        exclude=exclude,
    )

    if len(columns) > config.max_numeric_features:

        columns = columns[
            : config.max_numeric_features
        ]

        report.warnings.append(
            "Numeric feature generation limited "
            "to the configured maximum number of columns."
        )

    if config.create_interactions:

        data, created = create_interaction_features(

            data,

            columns,

            config.max_interaction_pairs,

        )

        report.interaction_features.extend(created)

    if config.create_ratios:

        data, created = create_ratio_features(

            data,

            columns,

            config.max_interaction_pairs,

        )

        report.ratio_features.extend(created)

    if config.create_aggregates:

        data, created = create_numeric_aggregates(

            data,

            columns,

        )

        report.aggregate_features.extend(created)

    if config.create_polynomial:

        data, created = create_polynomial_features(

            data,

            columns,

            degree=config.polynomial_degree,

        )

        report.aggregate_features.extend(created)

    data, created = create_log_features(
        data,
        columns,
    )

    report.aggregate_features.extend(created)

    data, created = create_sqrt_features(
        data,
        columns,
    )

    report.aggregate_features.extend(created)

    data, created = create_squared_features(
        data,
        columns,
    )

    report.aggregate_features.extend(created)

    data, created = create_cubed_features(
        data,
        columns,
    )

    report.aggregate_features.extend(created)

    data, created = create_absolute_features(
        data,
        columns,
    )

    report.aggregate_features.extend(created)

    data, created = create_reciprocal_features(
        data,
        columns,
    )

    report.aggregate_features.extend(created)

    if config.drop_invalid_features:

        data, removed = remove_invalid_features(
            data,
        )

        report.removed_features.extend(
            removed
        )

    return data, report
################################################################################
# Cyclical Feature Encoding
################################################################################

def encode_cyclical(
    series: pd.Series,
    period: int,
    prefix: str,
) -> pd.DataFrame:
    """
    Encode cyclical features using sine and cosine transforms.

    Parameters
    ----------
    series
        Numeric values representing a repeating cycle.

    period
        Length of the cycle.

    prefix
        Prefix for generated column names.
    """

    radians = (2 * np.pi * series) / period

    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(radians),
            f"{prefix}_cos": np.cos(radians),
        },
        index=series.index,
    )


def engineer_cyclical_datetime_features(
    df: pd.DataFrame,
    datetime_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create cyclical encodings for common datetime components.
    """

    data = df.copy()

    created: list[str] = []

    periods = {

        "month": 12,

        "day": 31,

        "dayofweek": 7,

        "dayofyear": 365,

        "hour": 24,

        "minute": 60,

    }

    for column in datetime_columns:

        if column not in data.columns:

            continue

        dt = pd.to_datetime(
            data[column],
            errors="coerce",
        )

        components = {

            "month": dt.dt.month,

            "day": dt.dt.day,

            "dayofweek": dt.dt.dayofweek,

            "dayofyear": dt.dt.dayofyear,

            "hour": dt.dt.hour,

            "minute": dt.dt.minute,

        }

        for component, values in components.items():

            encoded = encode_cyclical(

                values.fillna(0),

                periods[component],

                f"{column}_{component}",

            )

            data = pd.concat(
                [data, encoded],
                axis=1,
            )

            created.extend(
                encoded.columns.tolist()
            )

    return data, created


################################################################################
# Frequency Encoding
################################################################################

def frequency_encode(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Frequency encode categorical features.
    """

    data = df.copy()

    created: list[str] = []

    for column in columns:

        frequency = (
            data[column]
            .value_counts(dropna=False)
            .to_dict()
        )

        feature = f"{column}_frequency"

        data[feature] = (
            data[column]
            .map(frequency)
            .astype(float)
        )

        created.append(feature)

    return data, created


################################################################################
# Count Encoding
################################################################################

def count_encode(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Count encode categorical variables.
    """

    data = df.copy()

    created: list[str] = []

    for column in columns:

        counts = data[column].value_counts()

        feature = f"{column}_count"

        data[feature] = data[column].map(counts)

        created.append(feature)

    return data, created


################################################################################
# Length-Based Text Features
################################################################################

def create_text_length_features(
    df: pd.DataFrame,
    text_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create simple text-derived features.
    """

    data = df.copy()

    created: list[str] = []

    for column in text_columns:

        data[column] = data[column].fillna("").astype(str)

        length_name = f"{column}_length"

        words_name = f"{column}_word_count"

        data[length_name] = data[column].str.len()

        data[words_name] = (
            data[column]
            .str.split()
            .str.len()
        )

        created.extend(
            [
                length_name,
                words_name,
            ]
        )

    return data, created


################################################################################
# Datetime Expansion
################################################################################

def expand_datetime_features(
    df: pd.DataFrame,
    datetime_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Expand datetime columns into useful numeric features.
    """

    data = df.copy()

    created: list[str] = []

    for column in datetime_columns:

        dt = pd.to_datetime(
            data[column],
            errors="coerce",
        )

        derived = {

            f"{column}_year": dt.dt.year,

            f"{column}_quarter": dt.dt.quarter,

            f"{column}_month": dt.dt.month,

            f"{column}_week": dt.dt.isocalendar().week.astype("Int64"),

            f"{column}_day": dt.dt.day,

            f"{column}_dayofweek": dt.dt.dayofweek,

            f"{column}_dayofyear": dt.dt.dayofyear,

            f"{column}_is_weekend": (
                dt.dt.dayofweek >= 5
            ).astype(int),

        }

        for name, values in derived.items():

            data[name] = values

            created.append(name)

    return data, created


################################################################################
# Feature Validation
################################################################################

def validate_engineered_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove duplicate feature names and empty columns.
    """

    data = df.copy()

    removed: list[str] = []

    # Remove duplicated column names
    if data.columns.duplicated().any():

        duplicated = data.columns[data.columns.duplicated()]

        removed.extend(duplicated.tolist())

        data = data.loc[:, ~data.columns.duplicated()]

    # Remove all-null columns
    for column in list(data.columns):

        if data[column].isna().all():

            removed.append(column)

            data.drop(
                columns=[column],
                inplace=True,
            )

    return data, removed
################################################################################
# Complete Feature Engineering Pipeline
################################################################################

def engineer_features(
    df: pd.DataFrame,
    config: FeatureEngineeringConfig | None = None,
    target: str | None = None,
) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    """
    Execute the complete feature engineering workflow.

    Parameters
    ----------
    df
        Input dataframe.

    config
        Feature engineering configuration.

    target
        Target column to exclude from feature generation.

    Returns
    -------
    tuple
        (
            engineered_dataframe,
            engineering_report,
        )
    """

    if config is None:
        config = FeatureEngineeringConfig()

    data = df.copy()

    report = FeatureEngineeringReport()

    exclude = []

    if target and target in data.columns:
        exclude.append(target)

    ###########################################################################
    # Numeric Features
    ###########################################################################

    data, numeric_report = engineer_numeric_features(
        data,
        config,
        exclude=exclude,
    )

    report.interaction_features.extend(
        numeric_report.interaction_features
    )

    report.ratio_features.extend(
        numeric_report.ratio_features
    )

    report.aggregate_features.extend(
        numeric_report.aggregate_features
    )

    report.removed_features.extend(
        numeric_report.removed_features
    )

    report.warnings.extend(
        numeric_report.warnings
    )

    ###########################################################################
    # Datetime Features
    ###########################################################################

    datetime_columns = [

        column

        for column in data.columns

        if pd.api.types.is_datetime64_any_dtype(
            data[column]
        )

    ]

    if datetime_columns:

        data, created = expand_datetime_features(
            data,
            datetime_columns,
        )

        report.aggregate_features.extend(created)

        data, created = engineer_cyclical_datetime_features(
            data,
            datetime_columns,
        )

        report.aggregate_features.extend(created)

    ###########################################################################
    # Text Features
    ###########################################################################

    text_columns = [

        column

        for column in data.select_dtypes(
            include=["object"]
        ).columns

        if column not in exclude

    ]

    if text_columns:

        data, created = create_text_length_features(
            data,
            text_columns,
        )

        report.aggregate_features.extend(created)

    ###########################################################################
    # Frequency Encoding
    ###########################################################################

    categorical_columns = [

        column

        for column in data.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns

        if column not in exclude

    ]

    if categorical_columns:

        data, created = frequency_encode(
            data,
            categorical_columns,
        )

        report.aggregate_features.extend(created)

        data, created = count_encode(
            data,
            categorical_columns,
        )

        report.aggregate_features.extend(created)

    ###########################################################################
    # Validation
    ###########################################################################

    data, removed = validate_engineered_features(
        data
    )

    report.removed_features.extend(
        removed
    )

    return data, report


################################################################################
# Reporting Utilities
################################################################################

def feature_engineering_summary(
    report: FeatureEngineeringReport,
) -> pd.DataFrame:
    """
    Convert the feature engineering report into a dataframe.
    """

    summary = [

        {

            "Metric": "Interaction Features",

            "Value": len(
                report.interaction_features
            ),

        },

        {

            "Metric": "Ratio Features",

            "Value": len(
                report.ratio_features
            ),

        },

        {

            "Metric": "Aggregate Features",

            "Value": len(
                report.aggregate_features
            ),

        },

        {

            "Metric": "Removed Features",

            "Value": len(
                report.removed_features
            ),

        },

        {

            "Metric": "Warnings",

            "Value": len(
                report.warnings
            ),

        },

    ]

    return pd.DataFrame(summary)


def generated_feature_names(
    report: FeatureEngineeringReport,
) -> list[str]:
    """
    Return every generated feature name.
    """

    return sorted(

        set(

            report.interaction_features

            + report.ratio_features

            + report.aggregate_features

        )

    )


################################################################################
# Metadata Export
################################################################################

def report_to_dict(
    report: FeatureEngineeringReport,
) -> dict[str, Any]:
    """
    Convert a report object into a serializable dictionary.
    """

    return {

        "interaction_features":
            report.interaction_features,

        "ratio_features":
            report.ratio_features,

        "aggregate_features":
            report.aggregate_features,

        "removed_features":
            report.removed_features,

        "warnings":
            report.warnings,

    }


################################################################################
# Public API
################################################################################

__all__ = [

    # Dataclasses

    "FeatureEngineeringConfig",

    "FeatureEngineeringReport",

    # Utilities

    "numeric_columns",

    "safe_divide",

    # Numeric Engineering

    "create_interaction_features",

    "create_ratio_features",

    "create_numeric_aggregates",

    "create_polynomial_features",

    "create_log_features",

    "create_sqrt_features",

    "create_squared_features",

    "create_cubed_features",

    "create_absolute_features",

    "create_reciprocal_features",

    "engineer_numeric_features",

    # Datetime Engineering

    "encode_cyclical",

    "engineer_cyclical_datetime_features",

    "expand_datetime_features",

    # Categorical Engineering

    "frequency_encode",

    "count_encode",

    # Text Engineering

    "create_text_length_features",

    # Validation

    "remove_invalid_features",

    "validate_engineered_features",

    # Orchestration

    "engineer_features",

    "feature_engineering_summary",

    "generated_feature_names",

    "report_to_dict",

]
