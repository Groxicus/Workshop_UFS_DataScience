"""
utils.trainer
=============

Model training utilities for ML Analytics Studio.

Features
--------
- Train/test splitting
- Cross-validation
- Hyperparameter tuning
- Training time measurement
- Progress tracking
- Model persistence
- Training report generation

Author: ML Analytics Studio
"""

from __future__ import annotations

import time
import joblib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)

from sklearn.pipeline import Pipeline

################################################################################
# Dataclasses
################################################################################


@dataclass(slots=True)
class TrainingConfig:
    """
    Configuration used during model training.
    """

    test_size: float = 0.20

    random_state: int = 42

    shuffle: bool = True

    cross_validation: bool = True

    cv_folds: int = 5

    stratified: bool = True

    hyperparameter_tuning: bool = False

    tuning_method: str = "grid"

    tuning_iterations: int = 25

    scoring: str | None = None

    n_jobs: int = -1


@dataclass(slots=True)
class TrainingResult:
    """
    Stores the results of model training.
    """

    model: BaseEstimator

    training_time: float

    x_train: pd.DataFrame

    x_test: pd.DataFrame

    y_train: pd.Series

    y_test: pd.Series

    predictions: np.ndarray

    probabilities: np.ndarray | None = None

    cross_validation_scores: dict[str, Any] = field(
        default_factory=dict
    )

    best_parameters: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


################################################################################
# Train/Test Split
################################################################################


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
    problem_type: str,
):
    """
    Split dataset into training and testing partitions.
    """

    stratify = None

    if (
        problem_type.lower() == "classification"
        and config.stratified
    ):
        stratify = y

    return train_test_split(

        X,

        y,

        test_size=config.test_size,

        random_state=config.random_state,

        shuffle=config.shuffle,

        stratify=stratify,

    )


################################################################################
# Cross Validation
################################################################################


def perform_cross_validation(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
    problem_type: str,
) -> dict[str, Any]:
    """
    Perform cross-validation.
    """

    if problem_type.lower() == "classification":

        cv = StratifiedKFold(

            n_splits=config.cv_folds,

            shuffle=True,

            random_state=config.random_state,

        )

    else:

        cv = KFold(

            n_splits=config.cv_folds,

            shuffle=True,

            random_state=config.random_state,

        )

    results = cross_validate(

        estimator,

        X,

        y,

        cv=cv,

        scoring=config.scoring,

        n_jobs=config.n_jobs,

        return_train_score=True,

    )

    return results


################################################################################
# Hyperparameter Search
################################################################################


def tune_model(
    estimator: BaseEstimator,
    parameter_grid: dict,
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
):
    """
    Perform optional hyperparameter optimisation.
    """

    if not parameter_grid:

        estimator.fit(X, y)

        return estimator, {}

    if config.tuning_method.lower() == "random":

        search = RandomizedSearchCV(

            estimator,

            parameter_grid,

            n_iter=config.tuning_iterations,

            cv=config.cv_folds,

            random_state=config.random_state,

            scoring=config.scoring,

            n_jobs=config.n_jobs,

        )

    else:

        search = GridSearchCV(

            estimator,

            parameter_grid,

            cv=config.cv_folds,

            scoring=config.scoring,

            n_jobs=config.n_jobs,

        )

    search.fit(X, y)

    return (

        search.best_estimator_,

        search.best_params_,

    )
################################################################################
# Model Training
################################################################################


def train_model(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
    problem_type: str,
    parameter_grid: dict | None = None,
) -> TrainingResult:
    """
    Train a machine learning model and return a TrainingResult.
    """

    parameter_grid = parameter_grid or {}

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        X,
        y,
        config,
        problem_type,
    )

    model = estimator

    best_params = {}

    if config.hyperparameter_tuning:

        model, best_params = tune_model(
            estimator=model,
            parameter_grid=parameter_grid,
            X=X_train,
            y=y_train,
            config=config,
        )

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    predictions = model.predict(
        X_test
    )

    probabilities = None

    if (
        problem_type.lower() == "classification"
        and hasattr(model, "predict_proba")
    ):

        try:

            probabilities = model.predict_proba(
                X_test
            )

        except Exception:

            probabilities = None

    cv_scores = {}

    if config.cross_validation:

        try:

            cv_scores = perform_cross_validation(

                estimator=model,

                X=X,

                y=y,

                config=config,

                problem_type=problem_type,

            )

        except Exception:

            cv_scores = {}

    metadata = {

        "problem_type": problem_type,

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "features": X.columns.tolist(),

        "target": getattr(
            y,
            "name",
            "target",
        ),

    }

    return TrainingResult(

        model=model,

        training_time=training_time,

        x_train=X_train,

        x_test=X_test,

        y_train=y_train,

        y_test=y_test,

        predictions=predictions,

        probabilities=probabilities,

        cross_validation_scores=cv_scores,

        best_parameters=best_params,

        metadata=metadata,

    )


################################################################################
# Model Persistence
################################################################################


def save_model(
    model: BaseEstimator,
    filepath: str | Path,
) -> Path:
    """
    Save a trained model using joblib.
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


def load_model(
    filepath: str | Path,
) -> BaseEstimator:
    """
    Load a saved model.
    """

    return joblib.load(
        Path(filepath)
    )


################################################################################
# Training Report
################################################################################


def training_summary(
    result: TrainingResult,
) -> pd.DataFrame:
    """
    Generate a summary DataFrame.
    """

    rows = [

        {

            "Metric": "Training Time (seconds)",

            "Value": round(
                result.training_time,
                4,
            ),

        },

        {

            "Metric": "Training Samples",

            "Value": len(
                result.x_train
            ),

        },

        {

            "Metric": "Testing Samples",

            "Value": len(
                result.x_test
            ),

        },

        {

            "Metric": "Number of Features",

            "Value": len(
                result.x_train.columns
            ),

        },

    ]

    if result.best_parameters:

        rows.append(

            {

                "Metric": "Hyperparameter Tuning",

                "Value": "Enabled",

            }

        )

    else:

        rows.append(

            {

                "Metric": "Hyperparameter Tuning",

                "Value": "Disabled",

            }

        )

    return pd.DataFrame(rows)


################################################################################
# Cross Validation Summary
################################################################################


def cross_validation_summary(
    cv_scores: dict[str, Any],
) -> pd.DataFrame:
    """
    Summarize cross-validation metrics.
    """

    if not cv_scores:

        return pd.DataFrame()

    rows = []

    for key, value in cv_scores.items():

        if isinstance(
            value,
            np.ndarray,
        ):

            rows.append(

                {

                    "Metric": key,

                    "Mean": np.mean(value),

                    "Std": np.std(value),

                }

            )

    return pd.DataFrame(rows)


################################################################################
# Pipeline Helpers
################################################################################


def train_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """
    Fit a complete sklearn pipeline.
    """

    pipeline.fit(
        X,
        y,
    )

    return pipeline


################################################################################
# Public API
################################################################################


__all__ = [

    "TrainingConfig",

    "TrainingResult",

    "split_dataset",

    "perform_cross_validation",

    "tune_model",

    "train_model",

    "save_model",

    "load_model",

    "training_summary",

    "cross_validation_summary",

    "train_pipeline",

]
