"""
utils.preprocessing
===================

Dynamic preprocessing pipeline for ML Analytics Studio.

This module provides:

- Automatic feature type detection
- Missing value handling
- Duplicate removal
- Feature classification
- Preprocessing configuration
- Data validation

Later parts of this module will add:

- Outlier handling
- Date feature engineering
- Text preprocessing
- Encoding
- Scaling
- Feature selection
- ColumnTransformer pipeline generation

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

################################################################################
# Dataclasses
################################################################################


@dataclass(slots=True)
class FeatureGroups:
    """
    Stores detected feature groups.
    """

    numeric: list[str] = field(default_factory=list)

    categorical: list[str] = field(default_factory=list)

    boolean: list[str] = field(default_factory=list)

    datetime: list[str] = field(default_factory=list)

    text: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PreprocessingConfig:
    """
    User-configurable preprocessing options.
    """

    numeric_missing_strategy: str = "median"

    categorical_missing_strategy: str = "most_frequent"

    remove_duplicates: bool = True

    drop_constant_columns: bool = True

    one_hot_encode: bool = True

    scale_numeric: bool = True

    scaler: str = "standard"

    parse_dates: bool = True

    extract_date_parts: bool = True

    detect_text: bool = True

    remove_outliers: bool = False

    outlier_method: str = "iqr"

    feature_selection: bool = False


################################################################################
# Feature Detection
################################################################################


def detect_feature_groups(
    df: pd.DataFrame,
    target: str | None = None,
) -> FeatureGroups:
    """
    Automatically classify dataset features.

    Parameters
    ----------
    df
        Dataset

    target
        Optional target column to exclude.
    """

    working = df.copy()

    if target and target in working.columns:

        working = working.drop(columns=[target])

    groups = FeatureGroups()

    for column in working.columns:

        series = working[column]

        if pd.api.types.is_bool_dtype(series):

            groups.boolean.append(column)

            continue

        if pd.api.types.is_numeric_dtype(series):

            groups.numeric.append(column)

            continue

        if pd.api.types.is_datetime64_any_dtype(series):

            groups.datetime.append(column)

            continue

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):

            unique_ratio = (
                series.nunique(dropna=True)
                / max(len(series), 1)
            )

            if unique_ratio >= 0.70:

                groups.text.append(column)

            else:

                groups.categorical.append(column)

    return groups


################################################################################
# Missing Value Handling
################################################################################


def create_numeric_imputer(
    strategy: str = "median",
) -> SimpleImputer:
    """
    Create an imputer for numeric columns.
    """

    return SimpleImputer(
        strategy=strategy,
    )


def create_categorical_imputer(
    strategy: str = "most_frequent",
) -> SimpleImputer:
    """
    Create an imputer for categorical columns.
    """

    return SimpleImputer(
        strategy=strategy,
    )


def fill_missing_values(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    target: str | None = None,
) -> pd.DataFrame:
    """
    Fill missing values using configurable strategies.
    """

    data = df.copy()

    groups = detect_feature_groups(
        data,
        target,
    )

    if groups.numeric:

        numeric_imputer = create_numeric_imputer(
            config.numeric_missing_strategy
        )

        data[groups.numeric] = numeric_imputer.fit_transform(
            data[groups.numeric]
        )

    if groups.categorical:

        categorical_imputer = create_categorical_imputer(
            config.categorical_missing_strategy
        )

        data[groups.categorical] = categorical_imputer.fit_transform(
            data[groups.categorical]
        )

    if groups.boolean:

        for column in groups.boolean:

            mode = data[column].mode(dropna=True)

            if not mode.empty:

                data[column] = data[column].fillna(mode.iloc[0])

    return data


################################################################################
# Duplicate Handling
################################################################################


def remove_duplicate_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """

    return df.drop_duplicates().reset_index(drop=True)


################################################################################
# Constant Columns
################################################################################


def remove_constant_columns(
    df: pd.DataFrame,
    target: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove columns containing only a single unique value.
    """

    data = df.copy()

    removed: list[str] = []

    for column in list(data.columns):

        if column == target:

            continue

        if data[column].nunique(dropna=False) <= 1:

            removed.append(column)

            data.drop(columns=[column], inplace=True)

    return data, removed


################################################################################
# Data Validation
################################################################################


def validate_preprocessing_input(
    df: pd.DataFrame,
    target: str | None = None,
) -> None:
    """
    Validate preprocessing inputs.
    """

    if df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    if target:

        if target not in df.columns:

            raise ValueError(
                f"Target column '{target}' does not exist."
            )
################################################################################
# Datetime Feature Engineering
################################################################################

def extract_datetime_features(
    df: pd.DataFrame,
    groups: FeatureGroups,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Extract useful machine learning features from datetime columns.

    Generated features include:

    - year
    - quarter
    - month
    - week
    - day
    - day_of_week
    - day_of_year
    - hour
    - minute
    - weekend indicator
    """

    data = df.copy()

    for column in groups.datetime:

        if column not in data.columns:
            continue

        data[column] = pd.to_datetime(
            data[column],
            errors="coerce",
        )

        prefix = column

        data[f"{prefix}_year"] = data[column].dt.year

        data[f"{prefix}_quarter"] = data[column].dt.quarter

        data[f"{prefix}_month"] = data[column].dt.month

        data[f"{prefix}_week"] = (
            data[column]
            .dt.isocalendar()
            .week
            .astype("Int64")
        )

        data[f"{prefix}_day"] = data[column].dt.day

        data[f"{prefix}_dayofweek"] = data[column].dt.dayofweek

        data[f"{prefix}_dayofyear"] = data[column].dt.dayofyear

        data[f"{prefix}_hour"] = data[column].dt.hour

        data[f"{prefix}_minute"] = data[column].dt.minute

        data[f"{prefix}_is_weekend"] = (
            data[column].dt.dayofweek >= 5
        ).astype(int)

        if drop_original:

            data.drop(columns=[column], inplace=True)

    return data


################################################################################
# Text Detection Utilities
################################################################################

def average_text_length(
    series: pd.Series,
) -> float:
    """
    Compute the average string length.
    """

    if series.dropna().empty:

        return 0.0

    return (
        series.dropna()
        .astype(str)
        .str.len()
        .mean()
    )


def detect_text_columns(
    df: pd.DataFrame,
    minimum_average_length: int = 20,
) -> list[str]:
    """
    Detect free-text columns.
    """

    detected: list[str] = []

    for column in df.select_dtypes(
        include=["object", "category"]
    ):

        avg = average_text_length(df[column])

        unique_ratio = (
            df[column].nunique(dropna=True)
            / max(len(df), 1)
        )

        if (
            avg >= minimum_average_length
            or unique_ratio >= 0.70
        ):

            detected.append(column)

    return detected


################################################################################
# Lightweight Text Preprocessing
################################################################################

def clean_text(
    series: pd.Series,
) -> pd.Series:
    """
    Basic text normalization.

    - lowercase
    - trim whitespace
    - collapse repeated spaces
    """

    return (

        series.fillna("")

        .astype(str)

        .str.lower()

        .str.strip()

        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )

    )


def preprocess_text_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Apply lightweight preprocessing to text columns.
    """

    data = df.copy()

    for column in columns:

        if column in data.columns:

            data[column] = clean_text(
                data[column]
            )

    return data


################################################################################
# Outlier Detection
################################################################################

def iqr_outlier_mask(
    series: pd.Series,
) -> pd.Series:
    """
    Detect outliers using the IQR rule.
    """

    q1 = series.quantile(0.25)

    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)

    upper = q3 + (1.5 * iqr)

    return (
        (series < lower)
        | (series > upper)
    )


def zscore_outlier_mask(
    series: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Detect outliers using z-score.
    """

    std = series.std()

    if std == 0 or pd.isna(std):

        return pd.Series(
            False,
            index=series.index,
        )

    z = (
        series - series.mean()
    ) / std

    return z.abs() > threshold


################################################################################
# Outlier Removal
################################################################################

def remove_outliers(
    df: pd.DataFrame,
    groups: FeatureGroups,
    method: str = "iqr",
) -> pd.DataFrame:
    """
    Remove rows containing numeric outliers.
    """

    data = df.copy()

    if not groups.numeric:

        return data

    keep = pd.Series(
        True,
        index=data.index,
    )

    for column in groups.numeric:

        if method == "zscore":

            mask = zscore_outlier_mask(
                data[column]
            )

        else:

            mask = iqr_outlier_mask(
                data[column]
            )

        keep &= ~mask.fillna(False)

    return data.loc[keep].reset_index(drop=True)


################################################################################
# Missing Value Summary
################################################################################

def preprocessing_report(
    original: pd.DataFrame,
    processed: pd.DataFrame,
) -> dict[str, int]:
    """
    Generate a lightweight preprocessing report.
    """

    return {

        "original_rows":
            len(original),

        "processed_rows":
            len(processed),

        "rows_removed":
            len(original) - len(processed),

        "original_columns":
            original.shape[1],

        "processed_columns":
            processed.shape[1],

        "missing_before":
            int(original.isna().sum().sum()),

        "missing_after":
            int(processed.isna().sum().sum()),

    }
  ################################################################################
# Encoding Utilities
################################################################################

def create_encoder(
    use_one_hot: bool = True,
):
    """
    Create a categorical encoder.

    Returns
    -------
    sklearn transformer
    """

    if use_one_hot:

        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    return OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )


################################################################################
# Scaling Utilities
################################################################################

def create_scaler(
    scaler: str = "standard",
):
    """
    Factory for numeric scalers.
    """

    scaler = scaler.lower()

    if scaler == "minmax":

        return MinMaxScaler()

    if scaler == "robust":

        return RobustScaler()

    return StandardScaler()


################################################################################
# Feature Selection
################################################################################

def remove_high_missing_features(
    df: pd.DataFrame,
    threshold: float = 0.50,
    target: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove columns with excessive missing values.

    Parameters
    ----------
    threshold
        Fraction of missing values required
        before removing the feature.
    """

    data = df.copy()

    removed: list[str] = []

    for column in list(data.columns):

        if column == target:

            continue

        ratio = data[column].isna().mean()

        if ratio >= threshold:

            removed.append(column)

            data.drop(
                columns=[column],
                inplace=True,
            )

    return data, removed


def remove_high_cardinality_features(
    df: pd.DataFrame,
    threshold: int = 100,
    target: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove categorical columns with excessive
    cardinality.

    Extremely high-cardinality columns often
    produce sparse matrices that provide
    limited predictive value.
    """

    data = df.copy()

    removed: list[str] = []

    categorical = data.select_dtypes(
        include=[
            "object",
            "category",
        ]
    )

    for column in categorical.columns:

        if column == target:

            continue

        if categorical[column].nunique() >= threshold:

            removed.append(column)

            data.drop(
                columns=[column],
                inplace=True,
            )

    return data, removed


################################################################################
# ColumnTransformer Builder
################################################################################

def build_preprocessing_pipeline(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    target: str | None = None,
) -> tuple[ColumnTransformer, FeatureGroups]:
    """
    Construct a reusable sklearn preprocessing pipeline.
    """

    groups = detect_feature_groups(
        df,
        target,
    )

    transformers = []

    # --------------------------------------------------------
    # Numeric Pipeline
    # --------------------------------------------------------

    if groups.numeric:

        numeric_pipeline = Pipeline(

            steps=[

                (
                    "imputer",

                    create_numeric_imputer(
                        config.numeric_missing_strategy
                    ),

                ),

                (
                    "scaler",

                    create_scaler(
                        config.scaler
                    )
                    if config.scale_numeric
                    else "passthrough",

                ),

            ]

        )

        transformers.append(

            (

                "numeric",

                numeric_pipeline,

                groups.numeric,

            )

        )

    # --------------------------------------------------------
    # Categorical Pipeline
    # --------------------------------------------------------

    if groups.categorical:

        categorical_pipeline = Pipeline(

            steps=[

                (

                    "imputer",

                    create_categorical_imputer(
                        config.categorical_missing_strategy
                    ),

                ),

                (

                    "encoder",

                    create_encoder(
                        config.one_hot_encode
                    ),

                ),

            ]

        )

        transformers.append(

            (

                "categorical",

                categorical_pipeline,

                groups.categorical,

            )

        )

    # --------------------------------------------------------
    # Boolean Pipeline
    # --------------------------------------------------------

    if groups.boolean:

        transformers.append(

            (

                "boolean",

                "passthrough",

                groups.boolean,

            )

        )

    preprocessor = ColumnTransformer(

        transformers=transformers,

        remainder="drop",

        verbose_feature_names_out=False,

    )

    return preprocessor, groups


################################################################################
# Feature Name Recovery
################################################################################

def transformed_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """
    Recover transformed feature names from a fitted
    ColumnTransformer.
    """

    try:

        return list(
            preprocessor.get_feature_names_out()
        )

    except Exception:

        return []


################################################################################
# Pipeline Validation
################################################################################

def validate_pipeline(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> bool:
    """
    Verify that a preprocessing pipeline can
    successfully fit the supplied dataset.
    """

    try:

        preprocessor.fit(X)

        return True

    except Exception:

        return False
################################################################################
# Complete Preprocessing Workflow
################################################################################

def preprocess_dataset(
    df: pd.DataFrame,
    config: PreprocessingConfig | None = None,
    target: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Execute the complete preprocessing workflow.

    Parameters
    ----------
    df
        Input dataframe.

    config
        Preprocessing configuration. If None,
        default settings are used.

    target
        Target column that should be preserved.

    Returns
    -------
    tuple

        (
            processed_dataframe,
            preprocessing_metadata,
        )
    """

    if config is None:

        config = PreprocessingConfig()

    validate_preprocessing_input(
        df,
        target,
    )

    original = df.copy()

    data = df.copy()

    metadata: dict[str, Any] = {

        "removed_duplicates": 0,

        "removed_constant_columns": [],

        "removed_high_missing_columns": [],

        "removed_high_cardinality_columns": [],

        "text_columns": [],

        "datetime_columns": [],

        "feature_groups": None,

    }

    # --------------------------------------------------------
    # Duplicate Removal
    # --------------------------------------------------------

    if config.remove_duplicates:

        before = len(data)

        data = remove_duplicate_rows(data)

        metadata["removed_duplicates"] = (
            before - len(data)
        )

    # --------------------------------------------------------
    # Remove Constant Columns
    # --------------------------------------------------------

    if config.drop_constant_columns:

        data, removed = remove_constant_columns(
            data,
            target,
        )

        metadata[
            "removed_constant_columns"
        ] = removed

    # --------------------------------------------------------
    # Remove High Missing Features
    # --------------------------------------------------------

    data, removed = remove_high_missing_features(
        data,
        target=target,
    )

    metadata[
        "removed_high_missing_columns"
    ] = removed

    # --------------------------------------------------------
    # Remove High Cardinality Features
    # --------------------------------------------------------

    data, removed = remove_high_cardinality_features(
        data,
        target=target,
    )

    metadata[
        "removed_high_cardinality_columns"
    ] = removed

    # --------------------------------------------------------
    # Feature Detection
    # --------------------------------------------------------

    groups = detect_feature_groups(
        data,
        target,
    )

    metadata["feature_groups"] = groups

    metadata["text_columns"] = groups.text

    metadata["datetime_columns"] = groups.datetime

    # --------------------------------------------------------
    # Missing Values
    # --------------------------------------------------------

    data = fill_missing_values(
        data,
        config,
        target,
    )

    # --------------------------------------------------------
    # Datetime Feature Engineering
    # --------------------------------------------------------

    if (
        config.parse_dates
        and config.extract_date_parts
        and groups.datetime
    ):

        data = extract_datetime_features(
            data,
            groups,
            drop_original=False,
        )

    # --------------------------------------------------------
    # Text Processing
    # --------------------------------------------------------

    if (
        config.detect_text
        and groups.text
    ):

        data = preprocess_text_columns(
            data,
            groups.text,
        )

    # --------------------------------------------------------
    # Outlier Removal
    # --------------------------------------------------------

    if config.remove_outliers:

        groups = detect_feature_groups(
            data,
            target,
        )

        data = remove_outliers(

            data,

            groups,

            method=config.outlier_method,

        )

    metadata["report"] = preprocessing_report(
        original,
        data,
    )

    return data, metadata


################################################################################
# Train/Test Preparation
################################################################################

def prepare_training_data(
    df: pd.DataFrame,
    target: str,
    config: PreprocessingConfig | None = None,
):
    """
    Prepare feature matrix and target vector.
    """

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' not found."
        )

    processed, metadata = preprocess_dataset(

        df,

        config,

        target,

    )

    X = processed.drop(
        columns=[target],
    )

    y = processed[target]

    pipeline, groups = build_preprocessing_pipeline(

        processed,

        config or PreprocessingConfig(),

        target,

    )

    return (

        X,

        y,

        pipeline,

        metadata,

        groups,

    )


################################################################################
# Prediction Dataset Preparation
################################################################################

def prepare_prediction_data(
    df: pd.DataFrame,
    training_columns: list[str],
) -> pd.DataFrame:
    """
    Align prediction datasets with training features.
    """

    data = df.copy()

    for column in training_columns:

        if column not in data.columns:

            data[column] = np.nan

    extra = [

        column

        for column in data.columns

        if column not in training_columns

    ]

    if extra:

        data.drop(
            columns=extra,
            inplace=True,
        )

    return data[
        training_columns
    ]


################################################################################
# Preprocessing Summary
################################################################################

def preprocessing_summary(
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """
    Convert preprocessing metadata into
    a dataframe suitable for Streamlit.
    """

    rows = [

        {

            "Step": "Duplicate Rows Removed",

            "Value": metadata.get(
                "removed_duplicates",
                0,
            ),

        },

        {

            "Step": "Constant Columns Removed",

            "Value": len(

                metadata.get(
                    "removed_constant_columns",
                    [],
                )

            ),

        },

        {

            "Step": "High Missing Columns Removed",

            "Value": len(

                metadata.get(
                    "removed_high_missing_columns",
                    [],
                )

            ),

        },

        {

            "Step": "High Cardinality Columns Removed",

            "Value": len(

                metadata.get(
                    "removed_high_cardinality_columns",
                    [],
                )

            ),

        },

        {

            "Step": "Detected Datetime Columns",

            "Value": len(

                metadata.get(
                    "datetime_columns",
                    [],
                )

            ),

        },

        {

            "Step": "Detected Text Columns",

            "Value": len(

                metadata.get(
                    "text_columns",
                    [],
                )

            ),

        },

    ]

    return pd.DataFrame(rows)


################################################################################
# Public API
################################################################################

__all__ = [

    "FeatureGroups",

    "PreprocessingConfig",

    "detect_feature_groups",

    "fill_missing_values",

    "remove_duplicate_rows",

    "remove_constant_columns",

    "extract_datetime_features",

    "detect_text_columns",

    "preprocess_text_columns",

    "iqr_outlier_mask",

    "zscore_outlier_mask",

    "remove_outliers",

    "create_encoder",

    "create_scaler",

    "remove_high_missing_features",

    "remove_high_cardinality_features",

    "build_preprocessing_pipeline",

    "transformed_feature_names",

    "validate_pipeline",

    "preprocess_dataset",

    "prepare_training_data",

    "prepare_prediction_data",

    "preprocessing_summary",

]
