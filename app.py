"""
Streamlit app for Patient Survival Prediction.

Note:
- To avoid "attempted relative import" errors when Streamlit runs the script,
  this file inserts the project root into sys.path at runtime (local dev convenience).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "src").exists():
    alt_root = Path(__file__).resolve().parent
    if (alt_root / "src").exists():
        PROJECT_ROOT = alt_root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.config import DATA_FILE, TARGET_COL
from src.deployment import (
    predict_single,
    predict_batch,
    missing_required_columns,
    load_model,
    get_input_schema,
    _get_feature_columns_from_pipeline,
)
from src.preprocessing import data_load

st.set_page_config(page_title="Patient survival prediction", page_icon="🩺", layout="centered")

# Reasonable clinical bounds for the numeric fields we know about. Anything
# not listed here just gets an unbounded number_input rather than breaking.
NUMERIC_BOUNDS = {
    "Age at Diagnosis": (0.0, 100.0, 0.5),
    "Tumor Size": (0.0, 200.0, 0.5),
    "Tumor Stage": (0, 4, 1),
    "Neoplasm Histologic Grade": (1, 3, 1),
    "Lymph nodes examined positive": (0, 50, 1),
    "Mutation Count": (0, 100, 1),
    "Nottingham prognostic index": (0.0, 10.0, 0.1),
}
INTEGER_COLS = {"Tumor Stage", "Neoplasm Histologic Grade", "Lymph nodes examined positive", "Mutation Count"}

# Groups fields into short, scannable sections instead of one long list.
# Any feature not named here falls into "Other" automatically, so a schema
# change doesn't silently drop a field from the form.
FIELD_GROUPS = {
    "Patient and tumor": [
        "Age at Diagnosis", "Type of Breast Surgery", "Tumor Size", "Tumor Stage",
        "Neoplasm Histologic Grade", "Primary Tumor Laterality",
        "Lymph nodes examined positive", "Nottingham prognostic index",
        "Mutation Count", "Oncotree Code", "Tumor Other Histologic Subtype",
    ],
    "Biomarkers and subtype": [
        "ER Status", "HER2 Status", "PR Status", "Cellularity",
        "Pam50 + Claudin-low subtype", "3-Gene classifier subtype",
        "Integrative Cluster", "Inferred Menopausal State",
    ],
    "Treatment": ["Chemotherapy", "Hormone Therapy", "Radio Therapy"],
}


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return data_load(DATA_FILE)


@st.cache_resource
def get_model_and_schema():
    model = load_model()
    schema = get_input_schema(model)
    return model, schema


def _ordered_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    grouped = {name: [c for c in cols if c in feature_cols] for name, cols in FIELD_GROUPS.items()}
    placed = {c for cols in grouped.values() for c in cols}
    leftover = [c for c in feature_cols if c not in placed]
    if leftover:
        grouped["Other"] = leftover
    return {k: v for k, v in grouped.items() if v}


def _numeric_default(df_sample: pd.DataFrame, col: str):
    if col in df_sample.columns and not df_sample[col].isna().all():
        val = df_sample[col].median()
    else:
        val = 0
    return int(round(val)) if col in INTEGER_COLS else float(val)


def _categorical_default(df_sample: pd.DataFrame, col: str, options: List[str]) -> str:
    if col in df_sample.columns and not df_sample[col].dropna().empty:
        mode_val = df_sample[col].mode()[0]
        if mode_val in options:
            return mode_val
    return options[0] if options else "Unknown"


def build_input_form(schema: Dict[str, Dict[str, Any]], df_sample: pd.DataFrame) -> Dict[str, Any]:
    st.subheader("Enter patient details")

    preset = st.session_state.get("preset_values", {})
    input_data: Dict[str, Any] = {}
    groups = _ordered_groups(list(schema.keys()))

    for group_name, cols in groups.items():
        st.markdown(
            f'<p style="font-size:11px;font-weight:600;color:#888;'
            f'text-transform:uppercase;letter-spacing:0.03em;margin:14px 0 4px">{group_name}</p>',
            unsafe_allow_html=True,
        )
        widget_cols = st.columns(3)
        for i, col in enumerate(cols):
            spec = schema[col]
            with widget_cols[i % 3]:
                if spec["type"] == "numeric":
                    # A preset value (from "Load example patient") is a raw
                    # value straight from the DataFrame -- it may be a NaN
                    # (numeric columns can still have real missing values at
                    # this stage, since imputation happens in the pipeline,
                    # not in cleaning) or a plain float even for columns that
                    # should be integer-stepped. Route it through the same
                    # int-casting as the no-preset path instead of using it
                    # raw, or Streamlit's number_input rejects a float
                    # `value` paired with int `min_value`/`max_value`/`step`.
                    preset_val = preset.get(col)
                    if preset_val is None or (isinstance(preset_val, float) and pd.isna(preset_val)):
                        default = _numeric_default(df_sample, col)
                    else:
                        default = int(round(preset_val)) if col in INTEGER_COLS else float(preset_val)

                    bounds = NUMERIC_BOUNDS.get(col)
                    kwargs = {}
                    if bounds:
                        lo, hi, step = bounds
                        kwargs = {"min_value": lo, "max_value": hi, "step": step}
                        default = max(lo, min(hi, default))
                    input_data[col] = st.number_input(col, value=default, **kwargs)
                else:
                    options = spec.get("categories") or ["Unknown"]
                    default_val = preset.get(col, _categorical_default(df_sample, col, options))
                    idx = options.index(default_val) if default_val in options else 0
                    input_data[col] = st.selectbox(col, options=options, index=idx)

    return input_data


def risk_label(survival_prob: float):
    if survival_prob >= 0.7:
        return "High chance of survival", "success"
    if survival_prob >= 0.4:
        return "Moderate risk, unclear outcome", "warning"
    return "Low chance of survival", "error"


def render_result(result: Dict[str, Any]) -> None:
    surv_prob = result["survival_probability"]
    label, tone = risk_label(surv_prob)
    tone_colors = {
        "success": ("#0F6E56", "#E1F5EE"),
        "warning": ("#854F0B", "#FAEEDA"),
        "error": ("#A32D2D", "#FCEBEB"),
    }
    text_color, bg_color = tone_colors[tone]

    with st.container(border=True):
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown(f'<p style="font-size:34px;font-weight:600;margin:0;line-height:1">{surv_prob:.0%}</p>', unsafe_allow_html=True)
            st.caption("predicted survival probability")
        with col_b:
            st.markdown(
                f'<div style="display:inline-block;background:{bg_color};color:{text_color};'
                f'font-size:12px;font-weight:600;padding:6px 14px;border-radius:999px;margin-top:8px">{label}</div>',
                unsafe_allow_html=True,
            )
        st.progress(surv_prob)
        st.caption(f"Predicted class: {result['prediction']}")


def main() -> None:
    st.title("Patient survival prediction")
    st.caption("Estimate survival likelihood from clinical and tumor features.")

    model, schema = get_model_and_schema()
    df_sample = load_sample_data()

    tab1, tab2 = st.tabs(["Single prediction", "Batch prediction"])

    with tab1:
        top_cols = st.columns([1, 3])
        with top_cols[0]:
            if st.button("Load example patient"):
                st.session_state["preset_values"] = df_sample.sample(1).iloc[0].to_dict()
                st.rerun()

        input_data = build_input_form(schema, df_sample)

        if st.button("Predict survival chances", type="primary"):
            try:
                with st.spinner("Running prediction..."):
                    result = predict_single(input_data)
                render_result(result)
            except Exception as exc:
                st.error("Prediction failed — check logs for details.")
                st.caption(f"Error: {str(exc)}")

    with tab2:
        st.subheader("Upload a cleaned CSV for batch predictions")
        st.caption("Must contain the same columns used during training.")
        uploaded = st.file_uploader("Choose a CSV", type=["csv"], label_visibility="collapsed")

        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(uploaded_df.head())

            expected_cols = _get_feature_columns_from_pipeline(model) or []
            missing_cols = missing_required_columns(uploaded_df, expected_cols)
            if missing_cols:
                st.warning(
                    "Missing expected columns: " + ", ".join(missing_cols) +
                    ". Predictions will still run, but the model will fill these in "
                    "with a single fixed value for every row, which reduces accuracy."
                )

            if st.button("Run batch prediction", type="primary"):
                try:
                    with st.spinner("Scoring uploaded patients..."):
                        result_df = predict_batch(uploaded_df)
                    st.dataframe(result_df.head(10))
                    csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions",
                        data=csv,
                        file_name="survival_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error("Batch prediction failed — check logs for details.")
                    st.caption(f"Error: {str(exc)}")

    st.divider()
    st.caption(
        "For educational and demonstration purposes only. Not a substitute for "
        "professional medical advice, diagnosis, or treatment. Trained on a "
        "historical research cohort and may not generalize to individual patients."
    )


if __name__ == "__main__":
    main()