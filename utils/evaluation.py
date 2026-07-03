"""
utils.evaluation
================

Comprehensive model evaluation utilities.

Features
--------
- Classification metrics
- Regression metrics
- Confusion matrix
- ROC Curve
- Precision-Recall Curve
- Residual analysis
- Prediction plots
- Feature importance extraction
- SHAP support
- Permutation importance
- Interactive Plotly visualisations

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.inspection import permutation_importance

try:

    import shap

    SHAP_AVAILABLE = True

except ImportError:

    SHAP_AVAILABLE = False


################################################################################
# Dataclasses
################################################################################


@dataclass(slots=True)
class EvaluationResult:
    """
    Container for evaluation outputs.
    """

    metrics: dict[str, Any] = field(default_factory=dict)

    classification_report: pd.DataFrame | None = None

    confusion_matrix: np.ndarray | None = None

    feature_importance: pd.DataFrame | None = None

    permutation_importance: pd.DataFrame | None = None

    shap_values: Any = None

    additional_outputs: dict[str, Any] = field(default_factory=dict)


################################################################################
# Classification Metrics
################################################################################


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> EvaluationResult:
    """
    Compute common classification metrics.
    """

    result = EvaluationResult()

    metrics = {

        "Accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

    }

    if probabilities is not None:

        try:

            if probabilities.ndim == 2:

                if probabilities.shape[1] == 2:

                    metrics["ROC AUC"] = roc_auc_score(
                        y_true,
                        probabilities[:, 1],
                    )

                else:

                    metrics["ROC AUC"] = roc_auc_score(
                        y_true,
                        probabilities,
                        multi_class="ovr",
                    )

        except Exception:

            pass

    result.metrics = metrics

    result.confusion_matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    result.classification_report = (
        pd.DataFrame(report)
        .transpose()
        .round(4)
    )

    return result


################################################################################
# Regression Metrics
################################################################################


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> EvaluationResult:
    """
    Compute regression metrics.
    """

    result = EvaluationResult()

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    result.metrics = {

        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),

        "MSE": mse,

        "RMSE": np.sqrt(mse),

        "R²": r2_score(
            y_true,
            y_pred,
        ),

    }

    return result


################################################################################
# Confusion Matrix Plot
################################################################################


def confusion_matrix_plot(
    matrix: np.ndarray,
):
    """
    Interactive confusion matrix.
    """

    fig = px.imshow(

        matrix,

        text_auto=True,

        aspect="auto",

        color_continuous_scale="Blues",

    )

    fig.update_layout(

        title="Confusion Matrix",

        xaxis_title="Predicted",

        yaxis_title="Actual",

    )

    return fig


################################################################################
# ROC Curve
################################################################################


def roc_curve_plot(
    y_true,
    probabilities,
):
    """
    ROC Curve for binary classification.
    """

    if probabilities is None:

        return go.Figure()

    if probabilities.ndim != 2:

        return go.Figure()

    if probabilities.shape[1] != 2:

        return go.Figure()

    fpr, tpr, _ = roc_curve(

        y_true,

        probabilities[:, 1],

    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=fpr,

            y=tpr,

            mode="lines",

            name="ROC",

        )

    )

    fig.add_trace(

        go.Scatter(

            x=[0, 1],

            y=[0, 1],

            mode="lines",

            name="Random",

        )

    )

    fig.update_layout(

        title="ROC Curve",

        xaxis_title="False Positive Rate",

        yaxis_title="True Positive Rate",

    )

    return fig
################################################################################
# Precision-Recall Curve
################################################################################

def precision_recall_curve_plot(
    y_true,
    probabilities,
):
    """
    Interactive Precision-Recall curve for binary classification.
    """

    if probabilities is None:
        return go.Figure()

    if probabilities.ndim != 2:
        return go.Figure()

    if probabilities.shape[1] != 2:
        return go.Figure()

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities[:, 1],
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Precision-Recall",
        )
    )

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )

    return fig


################################################################################
# Residual Plot
################################################################################

def residual_plot(
    y_true,
    y_pred,
):
    """
    Interactive residual plot.
    """

    residuals = y_true - y_pred

    fig = px.scatter(
        x=y_pred,
        y=residuals,
        labels={
            "x": "Predicted",
            "y": "Residual",
        },
        title="Residual Plot",
        template="plotly_white",
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    return fig


################################################################################
# Prediction vs Actual
################################################################################

def prediction_plot(
    y_true,
    y_pred,
):
    """
    Predicted versus actual values.
    """

    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={
            "x": "Actual",
            "y": "Predicted",
        },
        title="Predicted vs Actual",
        template="plotly_white",
    )

    minimum = min(
        np.min(y_true),
        np.min(y_pred),
    )

    maximum = max(
        np.max(y_true),
        np.max(y_pred),
    )

    fig.add_trace(

        go.Scatter(

            x=[minimum, maximum],

            y=[minimum, maximum],

            mode="lines",

            name="Ideal",

        )

    )

    return fig


################################################################################
# Feature Importance
################################################################################

def feature_importance(
    model,
    feature_names: list[str],
) -> pd.DataFrame | None:
    """
    Extract model feature importance where supported.
    """

    importance = None

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

    elif hasattr(model, "coef_"):

        coef = np.asarray(model.coef_)

        if coef.ndim > 1:
            coef = np.mean(
                np.abs(coef),
                axis=0,
            )
        else:
            coef = np.abs(coef)

        importance = coef

    if importance is None:
        return None

    result = pd.DataFrame(

        {

            "Feature": feature_names,

            "Importance": importance,

        }

    )

    result = result.sort_values(

        "Importance",

        ascending=False,

    ).reset_index(drop=True)

    return result


################################################################################
# Feature Importance Plot
################################################################################

def feature_importance_plot(
    importance_df: pd.DataFrame,
):
    """
    Interactive feature importance plot.
    """

    if (
        importance_df is None
        or importance_df.empty
    ):
        return go.Figure()

    return px.bar(

        importance_df,

        x="Importance",

        y="Feature",

        orientation="h",

        template="plotly_white",

        title="Feature Importance",

    )


################################################################################
# Permutation Importance
################################################################################

def permutation_importance_analysis(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance.
    """

    result = permutation_importance(

        model,

        X,

        y,

        random_state=random_state,

        n_repeats=10,

    )

    return (

        pd.DataFrame(

            {

                "Feature": X.columns,

                "Importance": result.importances_mean,

            }

        )

        .sort_values(

            "Importance",

            ascending=False,

        )

        .reset_index(drop=True)

    )


################################################################################
# SHAP Analysis
################################################################################

def shap_analysis(
    model,
    X: pd.DataFrame,
):
    """
    Compute SHAP values if available.
    """

    if not SHAP_AVAILABLE:
        return None

    try:

        explainer = shap.Explainer(model)

        values = explainer(X)

        return values

    except Exception:

        return None


################################################################################
# Complete Evaluation
################################################################################

def evaluate_model(
    problem_type: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> EvaluationResult:
    """
    Execute a complete evaluation workflow.
    """

    if problem_type.lower() == "classification":

        result = classification_metrics(
            y_test,
            predictions,
            probabilities,
        )

    else:

        result = regression_metrics(
            y_test,
            predictions,
        )

    importance = feature_importance(
        model,
        X_test.columns.tolist(),
    )

    result.feature_importance = importance

    try:

        result.permutation_importance = (
            permutation_importance_analysis(
                model,
                X_test,
                y_test,
            )
        )

    except Exception:

        result.permutation_importance = None

    try:

        result.shap_values = shap_analysis(
            model,
            X_test,
        )

    except Exception:

        result.shap_values = None

    return result


################################################################################
# Public API
################################################################################

__all__ = [

    "EvaluationResult",

    "classification_metrics",

    "regression_metrics",

    "evaluate_model",

    "confusion_matrix_plot",

    "roc_curve_plot",

    "precision_recall_curve_plot",

    "prediction_plot",

    "residual_plot",

    "feature_importance",

    "feature_importance_plot",

    "permutation_importance_analysis",

    "shap_analysis",

    "SHAP_AVAILABLE",

]
