from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
TRAIN_PATH = APP_DIR / "train.csv"
TARGET = "Credit_Score"
EXPECTED_CLASSES = ["Poor", "Standard", "Good"]
RANDOM_STATE = 42


# ----------------------------- Model utilities ----------------------------- #
def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply light, safe cleanup without assuming a specific CSV schema."""
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace(r"^\s*$", np.nan, regex=True)
    # Normalize common placeholder values found in credit-score datasets.
    placeholders = ["_______", "_", "!@9#%8", "NM", "nan", "None"]
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].replace(placeholders, np.nan)
        df[col] = df[col].astype("string").str.strip()
    return df


def _make_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    classifier = LogisticRegression(
        C=0.1,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


@st.cache_resource(show_spinner=False)
def load_or_train_model() -> tuple[Any, str, pd.DataFrame | None, str]:
    """
    Load model.joblib if available. Otherwise train a compatible model from
    train.csv and save it to model.joblib. The saved object is a full sklearn
    Pipeline, including imputers, encoding, scaling, and classifier.
    """
    if MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            if isinstance(artifact, dict) and "model" in artifact:
                artifact = artifact["model"]
            if not hasattr(artifact, "predict"):
                raise ValueError("The loaded artifact does not expose predict().")
            return artifact, "Loaded existing model.joblib", None, ""
        except Exception as exc:
            raise RuntimeError(f"Could not load {MODEL_PATH.name}: {exc}") from exc

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Neither {MODEL_PATH.name} nor {TRAIN_PATH.name} was found. "
            "Place one of them in the same folder as app.py."
        )

    train = _clean_frame(pd.read_csv(TRAIN_PATH, low_memory=False))
    if TARGET not in train.columns:
        raise ValueError(
            f"train.csv must contain the target column '{TARGET}'. "
            f"Columns found: {', '.join(train.columns[:30])}"
        )

    train = train.dropna(subset=[TARGET])
    y = train[TARGET].astype(str).str.strip()
    X = train.drop(columns=[TARGET])

    # Identifiers and free-text personal identifiers usually do not generalize.
    drop_cols = [
        c for c in X.columns
        if c.lower() in {"id", "customer_id", "name", "ssn"}
    ]
    X = X.drop(columns=drop_cols, errors="ignore")
    if X.empty:
        raise ValueError("No usable feature columns remain after removing identifiers.")

    model = _make_pipeline(X)
    model.fit(X, y)
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Model trained but could not be saved: {exc}") from exc

    return model, f"Trained from train.csv and saved to {MODEL_PATH.name}", train, ""


def get_model_classes(model: Any) -> list[str]:
    try:
        return [str(x) for x in model.classes_]
    except Exception:
        try:
            return [str(x) for x in model.named_steps["classifier"].classes_]
        except Exception:
            return EXPECTED_CLASSES.copy()


def prepare_input_frame(data: dict[str, Any] | pd.DataFrame, model: Any) -> pd.DataFrame:
    """Align input columns to the fitted pipeline's expected feature order."""
    if isinstance(data, dict):
        frame = pd.DataFrame([data])
    else:
        frame = data.copy()

    # If this is the app's own Pipeline, use the original training feature names.
    try:
        expected = list(model.feature_names_in_)
    except Exception:
        try:
            expected = list(model.named_steps["preprocessor"].feature_names_in_)
        except Exception:
            expected = list(frame.columns)

    frame.columns = [str(c).strip() for c in frame.columns]
    frame = _clean_frame(frame)
    for col in expected:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[expected]


def predict_frame(model: Any, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    aligned = prepare_input_frame(data, model)
    prediction = model.predict(aligned).astype(str)
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(aligned)
    return prediction, probabilities


def class_color(label: str) -> str:
    return {"Good": "#16803c", "Standard": "#c77700", "Poor": "#c62828"}.get(label, "#4b5563")


def show_prediction(label: str) -> None:
    color = class_color(label)
    st.markdown(
        f"""
        <div style="padding:16px 20px;border-radius:12px;background:{color};
                    color:white;text-align:center;font-size:1.35rem;font-weight:700;">
            Predicted Credit Score: {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_table(classes: list[str], probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"Credit score": classes, "Probability": np.asarray(probabilities, dtype=float)}
    ).sort_values("Probability", ascending=False)


# --------------------------------- UI -------------------------------------- #
st.set_page_config(
    page_title="Credit Score Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25);
      padding: 12px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("💳 Credit Score Intelligence")
st.caption("Predict credit-score categories, review model probabilities, and score CSV files.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a workspace",
        ["Single Customer Prediction", "Batch CSV Prediction", "Model Insights"],
    )
    st.divider()
    st.caption("Model: Logistic Regression · C=0.1 · balanced class weights")
    st.caption("Missing numeric values: median imputation")
    st.caption("Missing categorical values: most-frequent imputation")

try:
    with st.spinner("Preparing model..."):
        model, model_status, training_data, _ = load_or_train_model()
    classes = get_model_classes(model)
    st.sidebar.success(model_status)
except Exception as exc:
    st.error(f"Model setup failed: {exc}")
    st.info(
        "Put model.joblib or train.csv beside app.py. For automatic training, "
        "train.csv must contain a Credit_Score target column."
    )
    st.stop()

# Determine the features and category choices from the training data when available.
try:
    feature_names = list(model.feature_names_in_)
except Exception:
    try:
        feature_names = list(model.named_steps["preprocessor"].feature_names_in_)
    except Exception:
        feature_names = []

if training_data is None and TRAIN_PATH.exists():
    try:
        training_data = _clean_frame(pd.read_csv(TRAIN_PATH, low_memory=False))
    except Exception:
        training_data = None

if TRAIN_PATH.exists():
    try:
        training_data = pd.read_csv(TRAIN_PATH, low_memory=False)
    except Exception as e:
        st.error(f"Could not load training data: {e}")

if not feature_names and training_data is not None:
    feature_names = [c for c in training_data.columns if c != TARGET]
    feature_names = [
        c for c in feature_names
        if c.lower() not in {"id", "customer_id", "name", "ssn"}
    ]

if not feature_names:
    st.warning(
        "The loaded model does not expose its input feature names. Batch prediction "
        "can still work if the CSV matches the model, but the dynamic single-customer "
        "form cannot be built reliably."
    )

# ---------------------- Single customer prediction ------------------------ #
if page == "Single Customer Prediction":
    st.subheader("Single customer assessment")
    st.write("Enter the customer's financial information, then request a prediction.")

    with st.form("single_prediction_form"):
        values: dict[str, Any] = {}
        left, right = st.columns(2)

        for index, col in enumerate(feature_names):
            container = left if index % 2 == 0 else right
            with container:
                dtype = None
                if training_data is not None and col in training_data.columns:
                    dtype = training_data[col].dtype

                if dtype is not None and (
                    pd.api.types.is_numeric_dtype(dtype)
                    or pd.api.types.is_bool_dtype(dtype)
                ):
                    series = pd.to_numeric(training_data[col], errors="coerce").dropna()
                    default = float(series.median()) if not series.empty else 0.0
                    minimum = float(series.min()) if not series.empty else 0.0
                    maximum = float(series.max()) if not series.empty else max(100.0, default)
                    if minimum == maximum:
                        minimum, maximum = min(0.0, minimum), max(1.0, maximum)
                    is_integer = pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
                    if is_integer:
                        values[col] = st.number_input(
                            col.replace("_", " "),
                            min_value=int(np.floor(minimum)),
                            max_value=int(np.ceil(maximum)),
                            value=int(np.clip(round(default), np.floor(minimum), np.ceil(maximum))),
                            step=1,
                        )
                    else:
                        values[col] = st.number_input(
                            col.replace("_", " "),
                            min_value=float(minimum),
                            max_value=float(maximum),
                            value=float(np.clip(default, minimum, maximum)),
                            step=0.1,
                            format="%.3f",
                        )
                else:
                    choices = []
                    if training_data is not None and col in training_data.columns:
                        choices = sorted(
                            training_data[col].dropna().astype(str).unique().tolist()
                        )
                    if choices:
                        values[col] = st.selectbox(col.replace("_", " "), choices)
                    else:
                        values[col] = st.text_input(col.replace("_", " "), value="")

        submitted = st.form_submit_button("Predict credit score", type="primary", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([values])
            prediction, probability = predict_frame(model, input_df)
            label = str(prediction[0])
            show_prediction(label)

            if probability is not None:
                prob_df = probability_table(classes, probability[0])
                st.subheader("Probability breakdown")
                chart_col, table_col = st.columns([2, 1])
                with chart_col:
                    fig = px.bar(
                        prob_df,
                        x="Credit score",
                        y="Probability",
                        color="Credit score",
                        color_discrete_map={
                            "Good": "#16803c",
                            "Standard": "#f0a22e",
                            "Poor": "#c62828",
                        },
                        range_y=[0, 1],
                        text=prob_df["Probability"].map(lambda x: f"{x:.1%}"),
                    )
                    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                with table_col:
                    st.dataframe(
                        prob_df.assign(Probability=prob_df["Probability"].map(lambda x: f"{x:.2%}")),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.metric("Top-class confidence", f"{float(np.max(probability[0])):.1%}")
            else:
                st.info("This model does not expose predict_proba(), so probability charts are unavailable.")
        except Exception as exc:
            st.error(f"Prediction failed. Check that the supplied features match the model. Details: {exc}")

# ------------------------- Batch prediction ------------------------------- #
elif page == "Batch CSV Prediction":
    st.subheader("Batch CSV upload and prediction")
    st.write(
        "Upload a CSV with the same feature columns used during model training. "
        "The target column is optional."
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            batch = _clean_frame(pd.read_csv(uploaded, low_memory=False))
            st.write(f"Rows: **{len(batch):,}** · Columns: **{len(batch.columns):,}**")
            st.dataframe(batch.head(10), use_container_width=True)

            if st.button("Run bulk predictions", type="primary"):
                with st.spinner("Scoring uploaded rows..."):
                    source = batch.drop(columns=[TARGET], errors="ignore")
                    predictions, probabilities = predict_frame(model, source)
                    result = batch.copy()
                    result["Predicted_Credit_Score"] = predictions
                    if probabilities is not None:
                        for idx, cls in enumerate(classes):
                            result[f"Probability_{cls}"] = probabilities[:, idx]
                        result["Prediction_Confidence"] = probabilities.max(axis=1)

                st.success(f"Successfully predicted {len(result):,} rows.")
                counts = result["Predicted_Credit_Score"].value_counts().rename_axis("Credit score").reset_index(name="Rows")
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(result.head(25), use_container_width=True)
                with c2:
                    fig = px.pie(
                        counts, names="Credit score", values="Rows",
                        color="Credit score",
                        color_discrete_map={"Good": "#16803c", "Standard": "#f0a22e", "Poor": "#c62828"},
                        title="Predicted class distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                csv_bytes = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="credit_score_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"Could not read or score this CSV: {exc}")

# -------------------------- Model insights ------------------------------- #
else:
    st.subheader("Model insights and analytics")
    classifier = None
    preprocessor = None
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        preprocessor = model.named_steps.get("preprocessor")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Model type", type(classifier or model).__name__)
    metric_cols[1].metric("Input features", len(feature_names) if feature_names else "Unknown")
    metric_cols[2].metric("Classes", len(classes))

    if training_data is not None and TARGET in training_data.columns:
        st.markdown("#### Training class distribution")
        dist = training_data[TARGET].astype(str).value_counts().rename_axis("Credit score").reset_index(name="Rows")
        dist["Share"] = dist["Rows"] / dist["Rows"].sum()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                dist, x="Credit score", y="Rows", color="Credit score",
                color_discrete_map={"Good": "#16803c", "Standard": "#f0a22e", "Poor": "#c62828"},
                text="Rows",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(
                dist.assign(Share=dist["Share"].map(lambda x: f"{x:.1%}")),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("Training class distribution is shown when train.csv is available beside app.py.")

    if classifier is not None and hasattr(classifier, "coef_") and preprocessor is not None:
        st.markdown("#### Logistic Regression coefficient insights")
        try:
            feature_out = preprocessor.get_feature_names_out()
            coef = np.asarray(classifier.coef_)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            selected_class = st.selectbox("Show coefficients for class", classes)
            class_idx = classes.index(selected_class)
            if coef.shape[0] == 1 and len(classes) == 2:
                coef_values = coef[0]
            elif class_idx < coef.shape[0]:
                coef_values = coef[class_idx]
            else:
                coef_values = coef[0]
            coef_df = pd.DataFrame({"Feature": feature_out, "Coefficient": coef_values})
            coef_df["Absolute coefficient"] = coef_df["Coefficient"].abs()
            coef_df = coef_df.sort_values("Absolute coefficient", ascending=False).head(25)
            fig = px.bar(
                coef_df.sort_values("Coefficient"),
                x="Coefficient",
                y="Feature",
                orientation="h",
                color="Coefficient",
                color_continuous_scale="RdYlGn",
                title=f"Top coefficients for {selected_class}",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Coefficient magnitude is model-specific and does not imply causation.")
        except Exception as exc:
            st.warning(f"Could not extract feature coefficients: {exc}")
    elif hasattr(model, "feature_importances_"):
        st.markdown("#### Feature importance")
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(25)
        st.plotly_chart(
            px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h"),
            use_container_width=True,
        )
    else:
        st.info("Coefficient or feature-importance charts are available for supported fitted model types.")

    st.markdown("#### Model file")
    st.code(str(MODEL_PATH))
    st.caption("The model is loaded from model.joblib, or trained from train.csv when no model file exists.")
