"""
utils.model_factory
===================

Dynamic Machine Learning Model Factory.

This module provides:

- Automatic model discovery
- Classification model registry
- Regression model registry
- Time-series placeholder support
- Default hyperparameter management
- Unified model creation interface

Author: ML Analytics Studio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ============================================================================
# Scikit-Learn Models
# ============================================================================

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)

from sklearn.svm import (
    SVC,
    SVR,
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)

from sklearn.naive_bayes import (
    GaussianNB,
)

from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor,
)

# Optional libraries
try:

    from xgboost import (
        XGBClassifier,
        XGBRegressor,
    )

    XGBOOST_AVAILABLE = True

except ImportError:

    XGBOOST_AVAILABLE = False


try:

    from lightgbm import (
        LGBMClassifier,
        LGBMRegressor,
    )

    LIGHTGBM_AVAILABLE = True

except ImportError:

    LIGHTGBM_AVAILABLE = False


try:

    from catboost import (
        CatBoostClassifier,
        CatBoostRegressor,
    )

    CATBOOST_AVAILABLE = True

except ImportError:

    CATBOOST_AVAILABLE = False


###############################################################################
# Dataclasses
###############################################################################


@dataclass(slots=True)
class ModelDefinition:
    """
    Metadata describing an available model.
    """

    name: str

    estimator: Any

    problem_type: str

    category: str

    default_parameters: dict[str, Any] = field(default_factory=dict)

    supports_feature_importance: bool = False

    supports_probability: bool = False

    supports_shap: bool = False


###############################################################################
# Classification Models
###############################################################################

CLASSIFICATION_MODELS: dict[str, ModelDefinition] = {

    "Logistic Regression": ModelDefinition(

        name="Logistic Regression",

        estimator=LogisticRegression,

        problem_type="classification",

        category="Linear",

        default_parameters={

            "max_iter": 1000,

            "random_state": 42,

        },

        supports_probability=True,

    ),

    "Decision Tree": ModelDefinition(

        name="Decision Tree",

        estimator=DecisionTreeClassifier,

        problem_type="classification",

        category="Tree",

        default_parameters={

            "random_state": 42,

        },

        supports_feature_importance=True,

        supports_probability=True,

        supports_shap=True,

    ),

    "Random Forest": ModelDefinition(

        name="Random Forest",

        estimator=RandomForestClassifier,

        problem_type="classification",

        category="Ensemble",

        default_parameters={

            "n_estimators": 300,

            "random_state": 42,

            "n_jobs": -1,

        },

        supports_feature_importance=True,

        supports_probability=True,

        supports_shap=True,

    ),

    "Extra Trees": ModelDefinition(

        name="Extra Trees",

        estimator=ExtraTreesClassifier,

        problem_type="classification",

        category="Ensemble",

        default_parameters={

            "n_estimators": 300,

            "random_state": 42,

            "n_jobs": -1,

        },

        supports_feature_importance=True,

        supports_probability=True,

        supports_shap=True,

    ),

    "Gradient Boosting": ModelDefinition(

        name="Gradient Boosting",

        estimator=GradientBoostingClassifier,

        problem_type="classification",

        category="Boosting",

        default_parameters={

            "random_state": 42,

        },

        supports_feature_importance=True,

        supports_probability=True,

        supports_shap=True,

    ),

    "AdaBoost": ModelDefinition(

        name="AdaBoost",

        estimator=AdaBoostClassifier,

        problem_type="classification",

        category="Boosting",

        default_parameters={

            "random_state": 42,

        },

        supports_probability=True,

    ),

    "Support Vector Machine": ModelDefinition(

        name="Support Vector Machine",

        estimator=SVC,

        problem_type="classification",

        category="Kernel",

        default_parameters={

            "probability": True,

            "random_state": 42,

        },

        supports_probability=True,

    ),

    "K-Nearest Neighbours": ModelDefinition(

        name="K-Nearest Neighbours",

        estimator=KNeighborsClassifier,

        problem_type="classification",

        category="Distance",

        default_parameters={

            "n_neighbors": 5,

        },

        supports_probability=True,

    ),

    "Naive Bayes": ModelDefinition(

        name="Naive Bayes",

        estimator=GaussianNB,

        problem_type="classification",

        category="Probabilistic",

        supports_probability=True,

    ),

    "Multi-layer Perceptron": ModelDefinition(

        name="Multi-layer Perceptron",

        estimator=MLPClassifier,

        problem_type="classification",

        category="Neural Network",

        default_parameters={

            "hidden_layer_sizes": (100,),

            "max_iter": 500,

            "random_state": 42,

        },

        supports_probability=True,

    ),

}
###############################################################################
# Regression Models
###############################################################################

REGRESSION_MODELS: dict[str, ModelDefinition] = {

    "Linear Regression": ModelDefinition(

        name="Linear Regression",

        estimator=LinearRegression,

        problem_type="regression",

        category="Linear",

    ),

    "Ridge Regression": ModelDefinition(

        name="Ridge Regression",

        estimator=Ridge,

        problem_type="regression",

        category="Linear",

        default_parameters={

            "alpha": 1.0,

            "random_state": 42,

        },

    ),

    "Lasso Regression": ModelDefinition(

        name="Lasso Regression",

        estimator=Lasso,

        problem_type="regression",

        category="Linear",

        default_parameters={

            "alpha": 1.0,

            "random_state": 42,

        },

    ),

    "Elastic Net": ModelDefinition(

        name="Elastic Net",

        estimator=ElasticNet,

        problem_type="regression",

        category="Linear",

        default_parameters={

            "alpha": 1.0,

            "l1_ratio": 0.5,

            "random_state": 42,

        },

    ),

    "Decision Tree Regressor": ModelDefinition(

        name="Decision Tree Regressor",

        estimator=DecisionTreeRegressor,

        problem_type="regression",

        category="Tree",

        default_parameters={

            "random_state": 42,

        },

        supports_feature_importance=True,

        supports_shap=True,

    ),

    "Random Forest Regressor": ModelDefinition(

        name="Random Forest Regressor",

        estimator=RandomForestRegressor,

        problem_type="regression",

        category="Ensemble",

        default_parameters={

            "n_estimators": 300,

            "random_state": 42,

            "n_jobs": -1,

        },

        supports_feature_importance=True,

        supports_shap=True,

    ),

    "Extra Trees Regressor": ModelDefinition(

        name="Extra Trees Regressor",

        estimator=ExtraTreesRegressor,

        problem_type="regression",

        category="Ensemble",

        default_parameters={

            "n_estimators": 300,

            "random_state": 42,

            "n_jobs": -1,

        },

        supports_feature_importance=True,

        supports_shap=True,

    ),

    "Gradient Boosting Regressor": ModelDefinition(

        name="Gradient Boosting Regressor",

        estimator=GradientBoostingRegressor,

        problem_type="regression",

        category="Boosting",

        default_parameters={

            "random_state": 42,

        },

        supports_feature_importance=True,

        supports_shap=True,

    ),

    "AdaBoost Regressor": ModelDefinition(

        name="AdaBoost Regressor",

        estimator=AdaBoostRegressor,

        problem_type="regression",

        category="Boosting",

        default_parameters={

            "random_state": 42,

        },

    ),

    "Support Vector Regressor": ModelDefinition(

        name="Support Vector Regressor",

        estimator=SVR,

        problem_type="regression",

        category="Kernel",

    ),

    "K-Nearest Neighbours Regressor": ModelDefinition(

        name="K-Nearest Neighbours Regressor",

        estimator=KNeighborsRegressor,

        problem_type="regression",

        category="Distance",

        default_parameters={

            "n_neighbors": 5,

        },

    ),

    "Multi-layer Perceptron Regressor": ModelDefinition(

        name="Multi-layer Perceptron Regressor",

        estimator=MLPRegressor,

        problem_type="regression",

        category="Neural Network",

        default_parameters={

            "hidden_layer_sizes": (100,),

            "max_iter": 500,

            "random_state": 42,

        },

    ),

}

###############################################################################
# Optional Libraries
###############################################################################

if XGBOOST_AVAILABLE:

    CLASSIFICATION_MODELS["XGBoost"] = ModelDefinition(

        name="XGBoost",

        estimator=XGBClassifier,

        problem_type="classification",

        category="Boosting",

        default_parameters={

            "random_state": 42,

            "eval_metric": "logloss",

            "n_estimators": 300,

        },

        supports_feature_importance=True,

        supports_probability=True,

        supports_shap=True,

    )

    REGRESSION_MODELS["XGBoost Regressor"] = ModelDefinition(

        name="XGBoost Regressor",

        estimator=XGBRegressor,

        problem_type="regression",

        category="Boosting",

        default_parameters={

            "random_state": 42,

            "n_estimators": 300,

        },

        supports_feature_importance=True,

        supports_shap=True,

    )

if LIGHTGBM_AVAILABLE:

    CLASSIFICATION_MODELS["LightGBM"] = ModelDefinition(

        name="LightGBM",

        estimator=LGBMClassifier,

        problem_type="classification",

        category="Boosting",

        default_parameters={

            "random_state": 42,

        },

        supports_feature_importance=True,

        supports_probability=True,

        supports_shap=True,

    )

    REGRESSION_MODELS["LightGBM Regressor"] = ModelDefinition(

        name="LightGBM Regressor",

        estimator=LGBMRegressor,

        problem_type="regression",

        category="Boosting",

        default_parameters={

            "random_state": 42,

        },

        supports_feature_importance=True,

        supports_shap=True,

    )

if CATBOOST_AVAILABLE:

    CLASSIFICATION_MODELS["CatBoost"] = ModelDefinition(

        name="CatBoost",

        estimator=CatBoostClassifier,

        problem_type="classification",

        category="Boosting",

        default_parameters={

            "verbose": False,

            "random_state": 42,

        },

        supports_feature_importance=True,

        supports_probability=True,

        supports_shap=True,

    )

    REGRESSION_MODELS["CatBoost Regressor"] = ModelDefinition(

        name="CatBoost Regressor",

        estimator=CatBoostRegressor,

        problem_type="regression",

        category="Boosting",

        default_parameters={

            "verbose": False,

            "random_state": 42,

        },

        supports_feature_importance=True,

        supports_shap=True,

    )

###############################################################################
# Factory Functions
###############################################################################

def available_models(
    problem_type: str,
) -> list[str]:
    """
    Return available model names.
    """

    problem_type = problem_type.lower()

    if problem_type == "classification":

        return sorted(CLASSIFICATION_MODELS.keys())

    if problem_type == "regression":

        return sorted(REGRESSION_MODELS.keys())

    raise ValueError(
        f"Unsupported problem type: {problem_type}"
    )


def get_model_definition(
    model_name: str,
    problem_type: str,
) -> ModelDefinition:
    """
    Retrieve a model definition.
    """

    registry = (
        CLASSIFICATION_MODELS
        if problem_type.lower() == "classification"
        else REGRESSION_MODELS
    )

    if model_name not in registry:

        raise ValueError(
            f"Unknown model '{model_name}'."
        )

    return registry[model_name]


def create_model(
    model_name: str,
    problem_type: str,
    **overrides,
):
    """
    Instantiate a model using default parameters
    combined with user overrides.
    """

    definition = get_model_definition(
        model_name,
        problem_type,
    )

    parameters = (
        definition.default_parameters.copy()
    )

    parameters.update(overrides)

    return definition.estimator(
        **parameters
    )


def model_supports_feature_importance(
    model_name: str,
    problem_type: str,
) -> bool:
    """
    Determine whether a model exposes feature importance.
    """

    return get_model_definition(
        model_name,
        problem_type,
    ).supports_feature_importance


def model_supports_probability(
    model_name: str,
    problem_type: str,
) -> bool:
    """
    Determine whether a classifier supports predict_proba().
    """

    return get_model_definition(
        model_name,
        problem_type,
    ).supports_probability


def model_supports_shap(
    model_name: str,
    problem_type: str,
) -> bool:
    """
    Determine whether SHAP explanations are supported.
    """

    return get_model_definition(
        model_name,
        problem_type,
    ).supports_shap


###############################################################################
# Public API
###############################################################################

__all__ = [

    "ModelDefinition",

    "CLASSIFICATION_MODELS",

    "REGRESSION_MODELS",

    "available_models",

    "get_model_definition",

    "create_model",

    "model_supports_feature_importance",

    "model_supports_probability",

    "model_supports_shap",

    "XGBOOST_AVAILABLE",

    "LIGHTGBM_AVAILABLE",

    "CATBOOST_AVAILABLE",

]
