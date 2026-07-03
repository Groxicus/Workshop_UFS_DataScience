"""
config.py

Global configuration for ML Analytics Studio.

This module contains:
- Application metadata
- UI configuration
- Default ML settings
- Supported algorithms
- File handling
- Target detection heuristics
- Feature detection thresholds
- Directory management
"""

from __future__ import annotations

from pathlib import Path

################################################################################
# Project Paths
################################################################################

PROJECT_ROOT = Path(__file__).resolve().parent

ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = PROJECT_ROOT / "models"
EXPORTS_DIR = PROJECT_ROOT / "exports"
REPORTS_DIR = PROJECT_ROOT / "reports"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
TEMP_DIR = PROJECT_ROOT / "temp"

# Automatically create required directories
for directory in [
    ASSETS_DIR,
    MODELS_DIR,
    EXPORTS_DIR,
    REPORTS_DIR,
    UPLOADS_DIR,
    TEMP_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

################################################################################
# Application Information
################################################################################

APP_NAME = "ML Analytics Studio"

APP_VERSION = "1.0.0"

APP_ICON = "🚀"

PAGE_TITLE = "ML Analytics Studio"

PAGE_LAYOUT = "wide"

INITIAL_SIDEBAR_STATE = "expanded"

AUTHOR = "OpenAI + User"

################################################################################
# Upload Settings
################################################################################

SUPPORTED_FILE_TYPES = [
    "csv",
    "xlsx",
]

MAX_PREVIEW_ROWS = 10

MAX_EXPORT_ROWS = 500000

################################################################################
# Dataset Inspection
################################################################################

HIGH_CARDINALITY_THRESHOLD = 50

TEXT_UNIQUE_RATIO = 0.70

DATE_PARSE_SAMPLE = 1000

PAIRPLOT_SAMPLE_SIZE = 500

EDA_SAMPLE_SIZE = 10000

################################################################################
# Default ML Settings
################################################################################

DEFAULT_RANDOM_STATE = 42

DEFAULT_TEST_SIZE = 0.20

DEFAULT_CV_FOLDS = 5

DEFAULT_N_JOBS = -1

################################################################################
# Default Preprocessing
################################################################################

DEFAULT_MISSING_NUMERIC = "median"

DEFAULT_MISSING_CATEGORICAL = "most_frequent"

DEFAULT_SCALER = "StandardScaler"

DEFAULT_ENCODING = "OneHotEncoder"

DEFAULT_OUTLIER_METHOD = "IQR"

################################################################################
# Target Detection
################################################################################

TARGET_KEYWORDS = [

    "target",
    "label",
    "class",
    "outcome",
    "result",
    "prediction",
    "sales",
    "price",
    "profit",
    "income",
    "revenue",
    "quality",
    "species",
    "survived",
    "diagnosis",
    "default",
    "churn",
    "fraud",
    "score",
    "rating",
    "risk",
    "response",
    "y"

]

################################################################################
# Classification Algorithms
################################################################################

CLASSIFICATION_MODELS = {

    "Logistic Regression":
        "logistic_regression",

    "Decision Tree":
        "decision_tree",

    "Random Forest":
        "random_forest",

    "Support Vector Machine":
        "svm",

    "K-Nearest Neighbours":
        "knn",

    "Naive Bayes":
        "naive_bayes",

    "XGBoost":
        "xgboost",

    "LightGBM":
        "lightgbm",

    "CatBoost":
        "catboost",

}

################################################################################
# Regression Algorithms
################################################################################

REGRESSION_MODELS = {

    "Linear Regression":
        "linear_regression",

    "Decision Tree Regressor":
        "decision_tree",

    "Random Forest Regressor":
        "random_forest",

    "Support Vector Regressor":
        "svr",

    "KNN Regressor":
        "knn",

    "XGBoost Regressor":
        "xgboost",

    "LightGBM Regressor":
        "lightgbm",

    "CatBoost Regressor":
        "catboost",

}

################################################################################
# Plot Configuration
################################################################################

PLOT_HEIGHT = 500

PLOT_WIDTH = 900

PLOT_TEMPLATE = "plotly_white"

CORRELATION_METHOD = "pearson"

################################################################################
# Export Filenames
################################################################################

CLEAN_DATASET_NAME = "cleaned_dataset.csv"

PREDICTIONS_NAME = "predictions.csv"

METRICS_NAME = "evaluation_metrics.csv"

MODEL_NAME = "trained_model.joblib"

################################################################################
# Session State Keys
################################################################################

SESSION_KEYS = {

    "dataset": "dataset",

    "clean_dataset": "clean_dataset",

    "target": "target",

    "problem_type": "problem_type",

    "preprocessor": "preprocessor",

    "model": "model",

    "metrics": "metrics",

    "predictions": "predictions",

    "feature_names": "feature_names",

}

################################################################################
# Theme Colours
################################################################################

PRIMARY_COLOR = "#1976D2"

SUCCESS_COLOR = "#2E7D32"

WARNING_COLOR = "#F9A825"

ERROR_COLOR = "#C62828"

################################################################################
# Miscellaneous
################################################################################

RANDOM_SEED_OPTIONS = list(range(0, 101))

TEST_SIZE_OPTIONS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

CV_OPTIONS = [
    3,
    5,
    10,
]

################################################################################
# Footer
################################################################################

FOOTER_TEXT = (
    "ML Analytics Studio | Built with Streamlit, "
    "Scikit-learn, Plotly, XGBoost, LightGBM and CatBoost."
)
