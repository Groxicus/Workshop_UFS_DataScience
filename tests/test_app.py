"""
Basic integration tests for ML Analytics Studio.

These tests verify that the core application modules import correctly and that
the primary machine learning workflow functions as expected.

Run:
    pytest tests/

Author: ML Analytics Studio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import (
    load_iris,
    load_diabetes,
)

from utils.problem_detection import detect_problem_type
from utils.model_factory import (
    create_model,
    available_models,
)
from utils.trainer import (
    TrainingConfig,
    train_model,
)
from utils.evaluation import evaluate_model
from utils.predict import generate_predictions


###############################################################################
# Fixtures
###############################################################################


@pytest.fixture
def iris_dataset():
    """
    Classification dataset.
    """

    dataset = load_iris(as_frame=True)

    X = dataset.data

    y = dataset.target

    return X, y


@pytest.fixture
def diabetes_dataset():
    """
    Regression dataset.
    """

    dataset = load_diabetes(as_frame=True)

    X = dataset.data

    y = dataset.target

    return X, y


###############################################################################
# Model Factory
###############################################################################


def test_available_classification_models():

    models = available_models(
        "classification"
    )

    assert isinstance(models, list)

    assert len(models) > 0


def test_available_regression_models():

    models = available_models(
        "regression"
    )

    assert isinstance(models, list)

    assert len(models) > 0


###############################################################################
# Problem Detection
###############################################################################


def test_problem_detection_classification(
    iris_dataset,
):

    _, y = iris_dataset

    problem = detect_problem_type(y)

    assert problem == "classification"


def test_problem_detection_regression(
    diabetes_dataset,
):

    _, y = diabetes_dataset

    problem = detect_problem_type(y)

    assert problem == "regression"


###############################################################################
# Classification Workflow
###############################################################################


def test_classification_training(
    iris_dataset,
):

    X, y = iris_dataset

    model = create_model(

        "Random Forest",

        "classification",

    )

    config = TrainingConfig()

    result = train_model(

        estimator=model,

        X=X,

        y=y,

        config=config,

        problem_type="classification",

    )

    assert result.model is not None

    assert len(result.predictions) > 0

    evaluation = evaluate_model(

        problem_type="classification",

        model=result.model,

        X_test=result.x_test,

        y_test=result.y_test,

        predictions=result.predictions,

        probabilities=result.probabilities,

    )

    assert "Accuracy" in evaluation.metrics

    assert evaluation.metrics["Accuracy"] >= 0.5


###############################################################################
# Regression Workflow
###############################################################################


def test_regression_training(
    diabetes_dataset,
):

    X, y = diabetes_dataset

    model = create_model(

        "Random Forest Regressor",

        "regression",

    )

    config = TrainingConfig()

    result = train_model(

        estimator=model,

        X=X,

        y=y,

        config=config,

        problem_type="regression",

    )

    evaluation = evaluate_model(

        problem_type="regression",

        model=result.model,

        X_test=result.x_test,

        y_test=result.y_test,

        predictions=result.predictions,

    )

    assert "RMSE" in evaluation.metrics

    assert evaluation.metrics["RMSE"] >= 0


###############################################################################
# Prediction Workflow
###############################################################################


def test_prediction_pipeline(
    iris_dataset,
):

    X, y = iris_dataset

    model = create_model(

        "Random Forest",

        "classification",

    )

    config = TrainingConfig()

    result = train_model(

        estimator=model,

        X=X,

        y=y,

        config=config,

        problem_type="classification",

    )

    prediction_data = X.head(10).copy()

    prediction_result = generate_predictions(

        model=result.model,

        prediction_df=prediction_data,

        required_features=X.columns.tolist(),

    )

    assert len(

        prediction_result.predictions

    ) == 10

    assert prediction_result.prediction_dataframe is not None


###############################################################################
# Missing Feature Handling
###############################################################################


def test_prediction_missing_feature(
    iris_dataset,
):

    X, y = iris_dataset

    model = create_model(

        "Random Forest",

        "classification",

    )

    config = TrainingConfig()

    result = train_model(

        estimator=model,

        X=X,

        y=y,

        config=config,

        problem_type="classification",

    )

    prediction_data = X.drop(

        columns=X.columns[0]

    ).head()

    prediction_result = generate_predictions(

        model=result.model,

        prediction_df=prediction_data,

        required_features=X.columns.tolist(),

    )

    assert len(

        prediction_result.predictions

    ) == len(prediction_data)


###############################################################################
# Feature Importance
###############################################################################


def test_feature_importance_available(
    iris_dataset,
):

    X, y = iris_dataset

    model = create_model(

        "Random Forest",

        "classification",

    )

    config = TrainingConfig()

    result = train_model(

        estimator=model,

        X=X,

        y=y,

        config=config,

        problem_type="classification",

    )

    evaluation = evaluate_model(

        "classification",

        result.model,

        result.x_test,

        result.y_test,

        result.predictions,

        result.probabilities,

    )

    assert evaluation.feature_importance is not None

    assert len(

        evaluation.feature_importance

    ) == X.shape[1]


###############################################################################
# Invalid Model
###############################################################################


def test_invalid_model():

    with pytest.raises(ValueError):

        create_model(

            "This Model Does Not Exist",

            "classification",

        )


###############################################################################
# End of File
###############################################################################
