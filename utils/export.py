"""
utils.export
============

Export utilities for ML Analytics Studio.

Provides centralized exporting of:

- Cleaned datasets
- Prediction datasets
- Evaluation metrics
- Model metadata
- Training summaries
- Plotly charts
- Trained models
- JSON reports
- Excel reports
- ZIP project exports

Author: ML Analytics Studio
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go

################################################################################
# Dataset Export
################################################################################


def dataframe_to_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    """
    Convert a DataFrame into downloadable CSV bytes.
    """

    return dataframe.to_csv(
        index=False,
    ).encode("utf-8")


################################################################################


def dataframe_to_excel_bytes(
    dataframe: pd.DataFrame,
    sheet_name: str = "Data",
) -> bytes:
    """
    Convert a DataFrame into an Excel workbook.
    """

    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:

        dataframe.to_excel(

            writer,

            sheet_name=sheet_name,

            index=False,

        )

    buffer.seek(0)

    return buffer.getvalue()


################################################################################
# JSON Export
################################################################################


def dict_to_json_bytes(
    dictionary: dict[str, Any],
    indent: int = 4,
) -> bytes:
    """
    Serialize a dictionary to JSON bytes.
    """

    return json.dumps(

        dictionary,

        indent=indent,

        default=str,

    ).encode("utf-8")


################################################################################
# Save DataFrame
################################################################################


def save_dataframe(
    dataframe: pd.DataFrame,
    filepath: str | Path,
) -> Path:
    """
    Save a DataFrame to disk.

    Supported formats:
        *.csv
        *.xlsx
    """

    path = Path(filepath)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

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

            "Unsupported export format."

        )

    return path


################################################################################
# Save JSON
################################################################################


def save_json(
    dictionary: dict[str, Any],
    filepath: str | Path,
) -> Path:
    """
    Save dictionary as JSON.
    """

    path = Path(filepath)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(

        path,

        "w",

        encoding="utf-8",

    ) as file:

        json.dump(

            dictionary,

            file,

            indent=4,

            default=str,

        )

    return path


################################################################################
# Model Export
################################################################################


def save_model(
    model,
    filepath: str | Path,
) -> Path:
    """
    Save a trained model using Joblib.
    """

    path = Path(filepath)

    path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    joblib.dump(

        model,

        path,

    )

    return path


################################################################################
# Model Download
################################################################################


def model_to_bytes(
    model,
) -> bytes:
    """
    Convert a model to downloadable bytes.
    """

    buffer = BytesIO()

    joblib.dump(

        model,

        buffer,

    )

    buffer.seek(0)

    return buffer.getvalue()
################################################################################
# Plotly Figure Export
################################################################################


def figure_to_html_bytes(
    figure: go.Figure,
) -> bytes:
    """
    Convert a Plotly figure into standalone HTML bytes.
    """

    html = figure.to_html(
        include_plotlyjs="cdn",
        full_html=True,
    )

    return html.encode("utf-8")


################################################################################


def save_figure(
    figure: go.Figure,
    filepath: str | Path,
) -> Path:
    """
    Save a Plotly figure as an HTML file.
    """

    path = Path(filepath)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(path)

    return path


################################################################################
# Metrics Export
################################################################################


def metrics_to_dataframe(
    metrics: dict[str, Any],
) -> pd.DataFrame:
    """
    Convert evaluation metrics into a DataFrame.
    """

    return pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        }
    )


################################################################################


def metrics_to_csv_bytes(
    metrics: dict[str, Any],
) -> bytes:
    """
    Export metrics as CSV bytes.
    """

    return dataframe_to_csv_bytes(
        metrics_to_dataframe(metrics)
    )


################################################################################


def metrics_to_excel_bytes(
    metrics: dict[str, Any],
) -> bytes:
    """
    Export metrics as Excel bytes.
    """

    return dataframe_to_excel_bytes(
        metrics_to_dataframe(metrics),
        sheet_name="Metrics",
    )


################################################################################
# ZIP Export
################################################################################


def create_zip_archive(
    files: dict[str, bytes],
) -> bytes:
    """
    Create a ZIP archive from an in-memory dictionary.

    Parameters
    ----------
    files
        Dictionary where:
            key   = filename
            value = file bytes
    """

    buffer = BytesIO()

    with zipfile.ZipFile(

        buffer,

        mode="w",

        compression=zipfile.ZIP_DEFLATED,

    ) as archive:

        for filename, content in files.items():

            archive.writestr(

                filename,

                content,

            )

    buffer.seek(0)

    return buffer.getvalue()


################################################################################
# Complete Project Export
################################################################################


def create_project_export(
    cleaned_dataset: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    model=None,
    figures: dict[str, go.Figure] | None = None,
) -> bytes:
    """
    Create a downloadable ZIP containing all project outputs.
    """

    export_files: dict[str, bytes] = {}

    if cleaned_dataset is not None:

        export_files["cleaned_dataset.csv"] = (
            dataframe_to_csv_bytes(cleaned_dataset)
        )

    if predictions is not None:

        export_files["predictions.csv"] = (
            dataframe_to_csv_bytes(predictions)
        )

    if metrics is not None:

        export_files["metrics.json"] = (
            dict_to_json_bytes(metrics)
        )

        export_files["metrics.csv"] = (
            metrics_to_csv_bytes(metrics)
        )

    if model is not None:

        export_files["trained_model.joblib"] = (
            model_to_bytes(model)
        )

    if figures:

        for name, figure in figures.items():

            export_files[f"{name}.html"] = (
                figure_to_html_bytes(figure)
            )

    return create_zip_archive(export_files)


################################################################################
# Export Summary
################################################################################


def export_summary(
    files: dict[str, bytes],
) -> pd.DataFrame:
    """
    Generate a summary of exported artifacts.
    """

    rows = []

    for filename, content in files.items():

        rows.append(

            {

                "File": filename,

                "Size (KB)": round(
                    len(content) / 1024,
                    2,
                ),

            }

        )

    return pd.DataFrame(rows)


################################################################################
# Public API
################################################################################


__all__ = [

    "dataframe_to_csv_bytes",

    "dataframe_to_excel_bytes",

    "dict_to_json_bytes",

    "save_dataframe",

    "save_json",

    "save_model",

    "model_to_bytes",

    "figure_to_html_bytes",

    "save_figure",

    "metrics_to_dataframe",

    "metrics_to_csv_bytes",

    "metrics_to_excel_bytes",

    "create_zip_archive",

    "create_project_export",

    "export_summary",

]
