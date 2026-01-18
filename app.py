"""
Streamlit app for Patient Survival Prediction.

Note:
- To avoid "attempted relative import" errors when Streamlit runs the script,
  this file inserts the project root into sys.path at runtime (local dev convenience).
"""

# Shim: make project root importable (helps when Streamlit changes cwd)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.config import DATA_FILE, TARGET_COL
from src.deployment import predict_single, predict_batch
from src.preprocessing import data_load

# Cache loading sample data to build UI
@st.cache_data
def load_sample_data() -> pd.DataFrame:
    df = data_load(DATA_FILE)
    return df


def build_input_form(df_sample: pd.DataFrame) -> Dict[str, Any]:
    """
    Build input widgets based on df_sample columns.
    """
    st.subheader("Enter Patient Details")
    feature_cols = [c for c in df_sample.columns if c != TARGET_COL]

    input_data: Dict[str, Any] = {}
    for col in feature_cols:
        series = df_sample[col]

        if pd.api.types.is_numeric_dtype(series):
            default = float(series.median()) if not series.isna().all() else 0.0
            input_data[col] = st.number_input(label=col, value=default)
        else:
            options = series.dropna().unique().tolist()
            if not options:
                options = ["Unknown"]
            input_data[col] = st.selectbox(label=col, options=options, index=0)
    return input_data


def main() -> None:
    st.title("Patient Survival Prediction")

    st.markdown(
        """
        Input patient information and predict whether the patient is likely to survive.
        The model uses preprocessing pipelines (imputer + scaler, and imputer + OHE).
        """
    )

    df_sample = load_sample_data()
    st.sidebar.markdown("### Data Snapshot")
    st.sidebar.dataframe(df_sample.head(5))

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    # --- Single Prediction Tab ---
    with tab1:
        input_data = build_input_form(df_sample)
        st.write("Input preview:")
        st.json(input_data)

        if st.button("Predict Survival Chances"):
            try:
                result = predict_single(input_data)
                surv_prob = result["survival_probability"]

                # Show numeric result
                st.write("**Predicted class:**", result["prediction"])
                st.write("**Survival probability:**", f"{surv_prob:.4f}")

                # Add a progress bar gauge
                st.progress(surv_prob)

                # Add a metric widget
                st.metric(label="Survival Probability", value=f"{surv_prob:.2%}")

                # Friendly message
                if surv_prob > 0.5:
                    st.success("High chance of survival")
                else:
                    st.error("Low chance of survival")

            except Exception as exc:
                st.error("Prediction failed — check logs for details.")
                st.write(f"Error: {str(exc)}")

    # --- Batch Prediction Tab ---
    with tab2:
        st.subheader("Upload a cleaned CSV for batch predictions (must contain same columns as training)")
        uploaded = st.file_uploader("Choose a CSV", type=["csv"])
        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(uploaded_df.head())
            if st.button("Run batch prediction"):
                try:
                    result_df = predict_batch(uploaded_df)
                    st.write("Prediction results (first 10 rows):")
                    st.dataframe(result_df.head(10))
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions",
                        data=csv,
                        file_name="survival_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as exc:
                    st.error("Batch prediction failed — check logs for details.")
                    st.write(f"Error: {str(exc)}")


if __name__ == "__main__":
    main()