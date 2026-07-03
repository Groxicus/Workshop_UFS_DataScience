"""
utils.predict
=============

Prediction utilities for ML Analytics Studio.

This module provides:

- Prediction dataset loading
- Prediction dataset validation
- Feature alignment
- Batch inference
- Prediction export
- Download-ready CSV generation

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

################################################################################
# Dataclasses
################################################################################


@dataclass(slots=True)
class PredictionResult:
    """
    Stores prediction outputs and associated metadata.
    """

    predictions: np.ndarray

    probabilities: np.ndarray | None = None

    prediction_dataframe: pd.DataFrame | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


################################################################################
# Prediction Dataset Loading
################################################################################


def load_prediction_dataset(
    file,
) -> pd.DataFrame:
    """
    Load prediction dataset from CSV or Excel.
    """

    filename = file.name.lower()

    if filename.endswith(".csv"):

        return pd.read_csv(file)

    if filename.endswith(".xlsx"):

        return pd.read_excel(file)

    raise ValueError(
        "Unsupported prediction file format."
    )


################################################################################
# Dataset Validation
################################################################################


def validate_prediction_dataset(
    prediction_df: pd.DataFrame,
    required_features: list[str],
) -> tuple[bool, list[str]]:
    """
    Validate that all required model features exist.
    """

    missing = [

        feature

        for feature in required_features

        if feature not in prediction_df.columns

    ]

    return (

        len(missing) == 0,

        missing,

    )


################################################################################
# Feature Alignment
################################################################################


def align_prediction_features(
    prediction_df: pd.DataFrame,
    required_features: list[str],
) -> pd.DataFrame:
    """
    Reorder and align prediction features to match
    the training feature order.
    """

    aligned = prediction_df.copy()

    aligned = aligned[required_features]

    return aligned


################################################################################
# Missing Column Creation
################################################################################


def add_missing_features(
    prediction_df: pd.DataFrame,
    required_features: list[str],
    fill_value: Any = 0,
) -> pd.DataFrame:
    """
    Add missing model features with default values.
    """

    data = prediction_df.copy()

    for feature in required_features:

        if feature not in data.columns:

            data[feature] = fill_value

    return data


################################################################################
# Remove Unexpected Columns
################################################################################


def remove_extra_columns(
    prediction_df: pd.DataFrame,
    required_features: list[str],
) -> pd.DataFrame:
    """
    Remove columns that were not used during training.
    """

    keep = [

        column

        for column in prediction_df.columns

        if column in required_features

    ]

    return prediction_df[keep]
################################################################################
# Batch Prediction
################################################################################


def generate_predictions(
    model: BaseEstimator,
    prediction_df: pd.DataFrame,
    required_features: list[str],
) -> PredictionResult:
    """
    Generate predictions from a trained model.

    Parameters
    ----------
    model
        Trained estimator.

    prediction_df
        Dataset to predict.

    required_features
        Features expected by the trained model.

    Returns
    -------
    PredictionResult
    """

    data = prediction_df.copy()

    # Add any missing features
    data = add_missing_features(
        data,
        required_features,
    )

    # Remove unexpected columns
    data = remove_extra_columns(
        data,
        required_features,
    )

    # Ensure feature order matches training
    data = align_prediction_features(
        data,
        required_features,
    )

    predictions = model.predict(data)

    probabilities = None

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(data)

        except Exception:

            probabilities = None

    output = prediction_df.copy()

    output["Prediction"] = predictions

    if probabilities is not None:

        if probabilities.ndim == 2:

            for index in range(probabilities.shape[1]):

                output[
                    f"Probability_Class_{index}"
                ] = probabilities[:, index]

    metadata = {

        "rows": len(output),

        "features_used": required_features,

        "prediction_columns": output.columns.tolist(),

    }

    return PredictionResult(

        predictions=predictions,

        probabilities=probabilities,

        prediction_dataframe=output,

        metadata=metadata,

    )


################################################################################
# Prediction Summary
################################################################################


def prediction_summary(
    result: PredictionResult,
) -> pd.DataFrame:
    """
    Create a summary of prediction results.
    """

    summary = [

        {

            "Metric": "Predictions Generated",

            "Value": len(result.predictions),

        },

        {

            "Metric": "Probability Estimates",

            "Value": (
                "Yes"
                if result.probabilities is not None
                else "No"
            ),

        },

        {

            "Metric": "Output Columns",

            "Value": len(
                result.prediction_dataframe.columns
            ),

        },

    ]

    return pd.DataFrame(summary)


################################################################################
# CSV Export
################################################################################


def predictions_to_csv(
    result: PredictionResult,
) -> bytes:
    """
    Convert prediction dataframe to CSV bytes.

    Suitable for Streamlit download_button().
    """

    if result.prediction_dataframe is None:

        raise ValueError(
            "No prediction dataframe available."
        )

    return result.prediction_dataframe.to_csv(
        index=False,
    ).encode("utf-8")


################################################################################
# Excel Export
################################################################################


def predictions_to_excel(
    result: PredictionResult,
) -> bytes:
    """
    Export predictions as an Excel workbook.

    Returns
    -------
    bytes
        Excel file bytes for download.
    """

    if result.prediction_dataframe is None:

        raise ValueError(
            "No prediction dataframe available."
        )

    from io import BytesIO

    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:

        result.prediction_dataframe.to_excel(

            writer,

            index=False,

            sheet_name="Predictions",

        )

    buffer.seek(0)

    return buffer.getvalue()


################################################################################
# Save Predictions
################################################################################


def save_predictions(
    result: PredictionResult,
    filepath: str | Path,
) -> Path:
    """
    Save predictions to disk.

    File extension determines output format.

    Supported:
        .csv
        .xlsx
    """

    path = Path(filepath)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    dataframe = result.prediction_dataframe

    if dataframe is None:

        raise ValueError(
            "Prediction dataframe is empty."
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":

        dataframe.to_csv(

            path,

            index=False,

        )

    elif suffix == ".xlsx":

        dataframe.to_excel(

            path,

            index=False,

        )

    else:

        raise ValueError(
            "Unsupported output format."
        )

    return path


################################################################################
# Public API
################################################################################


__all__ = [

    "PredictionResult",

    "load_prediction_dataset",

    "validate_prediction_dataset",

    "align_prediction_features",

    "add_missing_features",

    "remove_extra_columns",

    "generate_predictions",

    "prediction_summary",

    "predictions_to_csv",

    "predictions_to_excel",

    "save_predictions",

]
