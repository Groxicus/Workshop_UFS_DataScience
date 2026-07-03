"""
utils/__init__.py

ML Analytics Studio Utility Package
===================================

This package contains reusable modules that power the
ML Analytics Studio application.

Modules
-------
loader
    Dataset loading and validation.

inspector
    Automatic dataset inspection and profiling.

target_detection
    Intelligent target variable inference.

problem_detection
    Automatic classification/regression detection.

preprocessing
    Data cleaning and preprocessing pipelines.

feature_engineering
    Automatic feature engineering.

eda
    Exploratory data analysis.

visualization
    Plotly and Matplotlib visualisations.

model_factory
    Machine learning model creation.

trainer
    Model training utilities.

evaluator
    Model evaluation.

predictor
    Prediction generation.

exporter
    Export utilities.

helpers
    General helper functions.
"""

from .loader import (
    load_dataset,
    validate_dataset,
    get_file_info,
)

from .inspector import (
    inspect_dataset,
)

from .target_detection import (
    suggest_target_variables,
)

from .problem_detection import (
    detect_problem_type,
)

from .preprocessing import (
    build_preprocessing_pipeline,
)

from .feature_engineering import (
    engineer_features,
)

from .eda import (
    generate_dataset_summary,
)

from .visualization import (
    create_visualisations,
)

from .model_factory import (
    create_model,
)

from .trainer import (
    train_model,
)

from .evaluator import (
    evaluate_model,
)

from .predictor import (
    generate_predictions,
)

from .exporter import (
    export_dataset,
    export_model,
    export_predictions,
    export_metrics,
)

from .helpers import (
    reset_session_state,
    format_bytes,
    timer,
)

__all__ = [

    # Loader
    "load_dataset",
    "validate_dataset",
    "get_file_info",

    # Inspector
    "inspect_dataset",

    # Target Detection
    "suggest_target_variables",

    # Problem Detection
    "detect_problem_type",

    # Preprocessing
    "build_preprocessing_pipeline",

    # Feature Engineering
    "engineer_features",

    # EDA
    "generate_dataset_summary",

    # Visualisation
    "create_visualisations",

    # Models
    "create_model",
    "train_model",
    "evaluate_model",

    # Predictions
    "generate_predictions",

    # Export
    "export_dataset",
    "export_model",
    "export_predictions",
    "export_metrics",

    # Helpers
    "reset_session_state",
    "format_bytes",
    "timer",
]
