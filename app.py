"""
app.py

ML Analytics Studio
-------------------
Main entry point for the Streamlit application.

Part 1:
- Imports
- Page configuration
- Session State Initialization
- Utility Functions
- Reusable UI Components
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import (
    APP_ICON,
    APP_NAME,
    APP_VERSION,
    FOOTER_TEXT,
    PAGE_LAYOUT,
    PAGE_TITLE,
    INITIAL_SIDEBAR_STATE,
    SESSION_KEYS,
)

###############################################################################
# Streamlit Configuration
###############################################################################

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)

###############################################################################
# Custom CSS
###############################################################################

st.markdown(
    """
<style>

.main > div {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.metric-card{
    background-color:#F8F9FA;
    border-radius:12px;
    padding:18px;
    border:1px solid #E5E5E5;
    box-shadow:0 2px 8px rgba(0,0,0,0.05);
}

.section-header{
    font-size:30px;
    font-weight:700;
    margin-bottom:10px;
}

.small-text{
    color:#777;
    font-size:14px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:13px;
    margin-top:60px;
}

</style>
""",
    unsafe_allow_html=True,
)

###############################################################################
# Session State Initialisation
###############################################################################


def initialise_session_state() -> None:
    """
    Initialise all Streamlit session variables.

    This prevents KeyErrors throughout the application.
    """

    defaults = {
        SESSION_KEYS["dataset"]: None,
        SESSION_KEYS["clean_dataset"]: None,
        SESSION_KEYS["target"]: None,
        SESSION_KEYS["problem_type"]: None,
        SESSION_KEYS["preprocessor"]: None,
        SESSION_KEYS["model"]: None,
        SESSION_KEYS["metrics"]: None,
        SESSION_KEYS["predictions"]: None,
        SESSION_KEYS["feature_names"]: None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialise_session_state()

###############################################################################
# Helper Functions
###############################################################################


def get_dataset() -> pd.DataFrame | None:
    """Return currently loaded dataset."""
    return st.session_state.get(SESSION_KEYS["dataset"])


def set_dataset(df: pd.DataFrame) -> None:
    """Store dataset."""
    st.session_state[SESSION_KEYS["dataset"]] = df


def dataset_loaded() -> bool:
    """Determine whether a dataset has been uploaded."""
    df = get_dataset()
    return df is not None and not df.empty


def safe_execute(function, *args, **kwargs):
    """
    Execute any callable with graceful exception handling.
    """

    try:
        return function(*args, **kwargs)

    except Exception as ex:

        st.error("An unexpected error occurred.")

        with st.expander("Technical Details"):

            st.code(traceback.format_exc())

        return None


###############################################################################
# Reusable Components
###############################################################################


def application_header():

    col1, col2 = st.columns([1, 7])

    with col1:
        st.markdown("# 🚀")

    with col2:

        st.title(APP_NAME)

        st.caption(f"Version {APP_VERSION}")

        st.write(
            """
A professional Machine Learning platform for data exploration,
preprocessing, model development, evaluation and prediction.
"""
        )


def horizontal_rule():

    st.markdown("---")


def info_box(message: str):

    st.info(message, icon="ℹ️")


def success_box(message: str):

    st.success(message, icon="✅")


def warning_box(message: str):

    st.warning(message, icon="⚠️")


def error_box(message: str):

    st.error(message, icon="❌")


###############################################################################
# Dashboard Metrics
###############################################################################


def dashboard_metrics():

    if not dataset_loaded():

        info_box("Upload a dataset to begin.")

        return

    df = get_dataset()

    rows, cols = df.shape

    missing = int(df.isna().sum().sum())

    duplicates = int(df.duplicated().sum())

    numeric = len(df.select_dtypes(include="number").columns)

    categorical = len(
        df.select_dtypes(include=["object", "category"]).columns
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Rows",
            f"{rows:,}",
        )

        st.metric(
            "Numeric Features",
            numeric,
        )

    with m2:

        st.metric(
            "Columns",
            cols,
        )

        st.metric(
            "Categorical Features",
            categorical,
        )

    with m3:

        st.metric(
            "Missing Values",
            missing,
        )

        st.metric(
            "Duplicate Rows",
            duplicates,
        )


###############################################################################
# Dataset Preview Component
###############################################################################


def dataset_preview():

    if not dataset_loaded():

        return

    df = get_dataset()

    with st.expander("📄 Dataset Preview", expanded=True):

        st.dataframe(
            df.head(15),
            use_container_width=True,
        )


###############################################################################
# Footer
###############################################################################


def footer():

    st.markdown("---")

    st.markdown(
        f"""
<div class="footer">

{FOOTER_TEXT}

</div>
""",
        unsafe_allow_html=True,
    )
  ###############################################################################
# Sidebar Navigation
###############################################################################

def build_sidebar() -> str:
    """
    Build the application sidebar.

    Returns
    -------
    str
        The selected navigation option.
    """

    with st.sidebar:

        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/examples/dog.jpg", width=110)

        st.title("🚀 ML Analytics Studio")

        st.caption("End-to-End Machine Learning Platform")

        st.markdown("---")

        navigation = st.radio(
            "Navigation",
            [
                "🏠 Home",
                "📂 Upload Dataset",
                "📊 Exploratory Data Analysis",
                "🛠 Data Preparation",
                "🤖 Model Training",
                "📈 Evaluation",
                "🔮 Predictions",
                "💾 Export Results",
            ],
            label_visibility="visible",
        )

        st.markdown("---")

        st.subheader("📋 Session Status")

        if dataset_loaded():

            df = get_dataset()

            st.success("Dataset Loaded")

            st.write(f"**Rows:** {df.shape[0]:,}")

            st.write(f"**Columns:** {df.shape[1]}")

            if st.session_state[SESSION_KEYS["target"]]:

                st.write(
                    f"**Target:** {st.session_state[SESSION_KEYS['target']]}"
                )

            if st.session_state[SESSION_KEYS["problem_type"]]:

                st.write(
                    f"**Problem:** {st.session_state[SESSION_KEYS['problem_type']]}"
                )

        else:

            st.warning("No dataset uploaded.")
            st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

        st.markdown("---")

        st.subheader("ℹ Application")

        st.write(f"Version **{APP_VERSION}**")

        st.caption(
            """
Supports:

- CSV
- Excel
- Classification
- Regression
- Time Series (basic)
"""
        )

    return navigation


###############################################################################
# Welcome Dashboard
###############################################################################

def home_page() -> None:
    """
    Display the landing page.
    """

    application_header()

    horizontal_rule()

    st.markdown(
        """
## Welcome

Welcome to **ML Analytics Studio**.

This application provides an end-to-end workflow for building machine
learning solutions without writing code.

Simply upload your dataset and follow the guided workflow.
"""
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
### 📂 Upload

- CSV
- Excel
- Automatic inspection
"""
        )

    with col2:

        st.success(
            """
### 🤖 Machine Learning

- Classification
- Regression
- Feature Engineering
"""
        )

    with col3:

        st.warning(
            """
### 📈 Results

- Evaluation
- Predictions
- Model Export
"""
        )

    horizontal_rule()

    st.subheader("Workflow")

    workflow_cols = st.columns(6)

    workflow = [
        ("📂", "Upload"),
        ("📊", "EDA"),
        ("🛠", "Prepare"),
        ("🤖", "Train"),
        ("📈", "Evaluate"),
        ("🔮", "Predict"),
    ]

    for column, (icon, label) in zip(workflow_cols, workflow):

        with column:

            st.markdown(
                f"""
<div style="text-align:center;
padding:15px;
border-radius:10px;
border:1px solid #DDDDDD;">

<h1>{icon}</h1>

<b>{label}</b>

</div>
""",
                unsafe_allow_html=True,
            )

    horizontal_rule()

    dashboard_metrics()

    dataset_preview()


###############################################################################
# Placeholder Pages
###############################################################################

def upload_placeholder():

    st.header("📂 Upload Dataset")
st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    st.info(
        """
The Upload page will be implemented in
`pages/1_Upload.py`.

Features include:

- CSV upload
- Excel upload
- File validation
- Automatic inspection
- Preview
"""
    )


def eda_placeholder():

    st.header("📊 Exploratory Data Analysis")

    st.info(
        """
The EDA page will provide:

- Statistics
- Histograms
- Boxplots
- Heatmaps
- Scatterplots
- Pairplots
- Correlation analysis
- Missing value analysis
"""
    )


def preprocessing_placeholder():

    st.header("🛠 Data Preparation")

    st.info(
        """
The preprocessing page will provide:

- Missing value handling
- Encoding
- Scaling
- Outlier detection
- Feature selection
- Date processing
- Text preprocessing
"""
    )


def training_placeholder():

    st.header("🤖 Model Training")

    st.info(
        """
The training page will support:

- Automatic problem detection
- Model selection
- Hyperparameter tuning
- Cross-validation
- Progress tracking
"""
    )


def evaluation_placeholder():

    st.header("📈 Model Evaluation")

    st.info(
        """
The evaluation page will include:

Classification:

- Accuracy
- Precision
- Recall
- F1
- ROC
- Confusion Matrix

Regression:

- MAE
- RMSE
- R²
- Residual plots
"""
    )


def prediction_placeholder():

    st.header("🔮 Predictions")

    st.info(
        """
Prediction page features:

- Upload prediction dataset
- Validate columns
- Generate predictions
- Download CSV
"""
    )


def export_placeholder():

    st.header("💾 Export Results")

    st.info(
        """
Export page:

- Clean dataset
- Trained model
- Metrics
- Charts
- Prediction results
"""
    )
###############################################################################
# Page Router
###############################################################################

def render_page(navigation: str) -> None:
    """
    Render the selected page.

    During development, placeholder pages are displayed.
    Once the dedicated pages are created under /pages,
    this router can be updated to import and execute them.
    """

    if navigation == "🏠 Home":
        home_page()

    elif navigation == "📂 Upload Dataset":
        upload_placeholder()

    elif navigation == "📊 Exploratory Data Analysis":
        eda_placeholder()

    elif navigation == "🛠 Data Preparation":
        preprocessing_placeholder()

    elif navigation == "🤖 Model Training":
        training_placeholder()

    elif navigation == "📈 Evaluation":
        evaluation_placeholder()

    elif navigation == "🔮 Predictions":
        prediction_placeholder()

    elif navigation == "💾 Export Results":
        export_placeholder()

    else:

        st.error("Unknown page selected.")


###############################################################################
# Startup Checks
###############################################################################

def startup_checks() -> None:
    """
    Perform application startup validation.
    """

    required_directories = [
        "assets",
        "models",
        "exports",
        "reports",
        "uploads",
        "temp",
    ]

    missing = []

    project_root = Path(__file__).resolve().parent

    for directory in required_directories:

        path = project_root / directory

        if not path.exists():

            missing.append(directory)

            path.mkdir(parents=True, exist_ok=True)

    if missing:

        st.toast(
            "Created missing project directories.",
            icon="📁",
        )


###############################################################################
# About Section
###############################################################################

def about_dialog() -> None:
    """
    Display application information.
    """

    with st.expander("ℹ About ML Analytics Studio"):

        st.markdown(
            f"""
### {APP_NAME}

Version **{APP_VERSION}**

ML Analytics Studio is a modular end-to-end
machine learning platform designed to simplify:

- Dataset exploration
- Automated EDA
- Feature engineering
- Model training
- Evaluation
- Prediction
- Export

The application is intentionally dataset-agnostic,
allowing it to work with virtually any structured
CSV or Excel dataset.

---

#### Supported Machine Learning Tasks

✅ Classification

✅ Regression

✅ Basic Time-Series Forecasting

---

#### Supported Algorithms

- Logistic Regression
- Linear Regression
- Decision Trees
- Random Forests
- XGBoost
- LightGBM
- CatBoost
- Support Vector Machines
- KNN
- Naive Bayes

---

Built with

- Streamlit
- Scikit-Learn
- Plotly
- Pandas
- NumPy
- XGBoost
- LightGBM
- CatBoost
- SHAP
"""
        )


###############################################################################
# Global Status Area
###############################################################################

def global_status() -> None:
    """
    Display the current application status.
    """

    if dataset_loaded():

        st.toast(
            "Dataset ready.",
            icon="✅",
        )

    else:

        st.toast(
            "Awaiting dataset upload.",
            icon="📂",
        )


###############################################################################
# Main Entry Point
###############################################################################

def main() -> None:
    """
    Main application entry point.
    """

    startup_checks()

    navigation = build_sidebar()

    global_status()

    render_page(navigation)

    horizontal_rule()

    about_dialog()

    footer()


###############################################################################
# Application Bootstrap
###############################################################################

if __name__ == "__main__":

    safe_execute(main)
