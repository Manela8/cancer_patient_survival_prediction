"""
Deployment utilities.

- load_model: lazy-load pipeline
- _get_feature_columns_from_pipeline: try to infer feature list or load from models/feature_columns.json
- predict_single: align single input dict to expected features, coerce types, predict
- predict_batch: align DataFrame and predict
"""

from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BEST_MODEL_PATH, FEATURES_PATH, MODEL_DIR

# The event of clinical interest is death, not survival. Mirrors the same
# constant/reasoning used in training.py -- kept in sync deliberately, since
# this is exactly the assumption that was hardcoded (and wrong-by-luck) here
# before: predict_proba(...)[:, 1] only happened to equal P(Living) because
# "Deceased" < "Living" alphabetically. Deriving the index from classes_
# at runtime removes that fragile assumption.
POSITIVE_CLASS = "Deceased"

_model = None
_feature_cols_cache: Optional[List[str]] = None
_pos_idx_cache: Optional[int] = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(BEST_MODEL_PATH)
    return _model


def _get_positive_class_index(model) -> int:
    """
    Column index of POSITIVE_CLASS within predict_proba's output, looked up
    from the fitted model's classes_ rather than assumed. Cached after first
    lookup since it can't change without reloading the model.
    """
    global _pos_idx_cache
    if _pos_idx_cache is not None:
        return _pos_idx_cache

    if not hasattr(model, "classes_"):
        raise ValueError(
            "Loaded model has no classes_ attribute -- can't determine which "
            "predict_proba column corresponds to the positive class."
        )

    classes = list(model.classes_)
    if POSITIVE_CLASS not in classes:
        raise ValueError(
            f"POSITIVE_CLASS='{POSITIVE_CLASS}' not found in model.classes_={classes}. "
            "Check that TARGET_COL / label spelling still matches what the model was trained on."
        )

    _pos_idx_cache = classes.index(POSITIVE_CLASS)
    return _pos_idx_cache


def _get_feature_columns_from_pipeline(model) -> Optional[List[str]]:
    # 1) pipeline.feature_names_in_
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    except Exception:
        pass

    # 2) preprocessor.feature_names_in_
    try:
        pre = model.named_steps.get("preprocess", None)
        if pre is not None and hasattr(pre, "feature_names_in_"):
            return list(pre.feature_names_in_)
    except Exception:
        pass

    # 3) pre.transformers_ columns list
    try:
        pre = model.named_steps.get("preprocess", None)
        if pre is not None and hasattr(pre, "transformers_"):
            cols = []
            for name, trans, cols_spec in pre.transformers_:
                if isinstance(cols_spec, (list, tuple)):
                    cols.extend([c for c in cols_spec if isinstance(c, str)])
            if cols:
                return cols
    except Exception:
        pass

    # 4) models/feature_columns.json
    try:
        if Path(FEATURES_PATH).exists():
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass

    return None


def _get_categorical_columns_from_pipeline(model) -> List[str]:
    """
    Column names routed to the categorical ("cat") branch of the fitted
    ColumnTransformer. Used to coerce inputs to string so a stray int/float
    typed into a categorical field doesn't silently fail to match any
    category the OneHotEncoder learned (handle_unknown='ignore' means a
    mismatched value becomes an all-zero row instead of raising -- a quiet
    accuracy hit rather than a loud error).
    """
    try:
        pre = model.named_steps.get("preprocess", None)
        if pre is not None and hasattr(pre, "transformers_"):
            for name, trans, cols in pre.transformers_:
                if name == "cat" and isinstance(cols, (list, tuple)):
                    return list(cols)
    except Exception:
        pass
    return []


def get_input_schema(model) -> Dict[str, Dict[str, Any]]:
    """
    Describe what each expected input feature looks like, for building a form
    (e.g. in Streamlit) without hardcoding column lists/types by hand.

    Returns
    -------
    dict of {column_name: {"type": "numeric" | "categorical", "categories": [...] }}
    "categories" is only present for categorical columns, taken from the
    OneHotEncoder's learned categories_ (so the form only ever offers values
    the model actually saw during training).
    """
    feature_cols = _get_feature_columns_from_pipeline(model) or []
    cat_cols = set(_get_categorical_columns_from_pipeline(model))
    schema: Dict[str, Dict[str, Any]] = {}

    categories_by_col: Dict[str, List[str]] = {}
    try:
        pre = model.named_steps.get("preprocess", None)
        cat_pipeline = pre.named_transformers_.get("cat", None) if pre is not None else None
        if cat_pipeline is not None:
            onehot = cat_pipeline.named_steps.get("onehot", None)
            if onehot is not None and hasattr(onehot, "categories_"):
                ordered_cat_cols = [c for name, _, cols in pre.transformers_ if name == "cat" for c in cols]
                for col, cats in zip(ordered_cat_cols, onehot.categories_):
                    categories_by_col[col] = [c for c in cats.tolist() if pd.notna(c)]
    except Exception:
        pass

    for col in feature_cols:
        if col in cat_cols:
            schema[col] = {"type": "categorical", "categories": categories_by_col.get(col, [])}
        else:
            schema[col] = {"type": "numeric"}
    return schema


def _ensure_input_frame(input_obj: Any, feature_cols: List[str]) -> pd.DataFrame:
    if isinstance(input_obj, dict):
        row = {c: input_obj.get(c, np.nan) for c in feature_cols}
        return pd.DataFrame([row])
    elif isinstance(input_obj, pd.DataFrame):
        return input_obj.reindex(columns=feature_cols)
    else:
        raise TypeError("Input must be dict or pandas DataFrame")


def _coerce_column_types(df: pd.DataFrame, model) -> pd.DataFrame:
    try:
        pre = model.named_steps.get("preprocess", None)
        if pre is not None and hasattr(pre, "transformers_"):
            for name, trans, cols in pre.transformers_:
                if not isinstance(cols, (list, tuple)):
                    continue
                if name == "num":
                    for c in cols:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                elif name == "cat":
                    for c in cols:
                        if c in df.columns:
                            # Coerce to string but keep real NaNs as NaN (not the
                            # literal string "nan") so the categorical imputer
                            # still treats them as missing rather than a category.
                            df[c] = df[c].apply(lambda v: v if pd.isna(v) else str(v))
    except Exception:
        pass
    return df


def missing_required_columns(input_df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    """
    Feature columns entirely absent from an uploaded/incoming DataFrame.
    Reindexing silently fills these with NaN for every row, which the
    pipeline's imputer then fills with a single constant for the whole
    column -- fabricating a feature rather than raising. Callers (e.g. the
    Streamlit batch-upload flow) should surface this list to the user
    instead of predicting silently.
    """
    return [c for c in feature_cols if c not in input_df.columns]


def predict_single(input_data: Dict[str, Any]) -> Dict[str, Any]:
    model = load_model()
    global _feature_cols_cache
    if _feature_cols_cache is None:
        _feature_cols_cache = _get_feature_columns_from_pipeline(model)

    if _feature_cols_cache is None:
        raise ValueError("Could not determine feature columns expected by the model. Ensure feature_columns.json exists.")

    pos_idx = _get_positive_class_index(model)

    df_input = _ensure_input_frame(input_data, _feature_cols_cache)
    df_input = _coerce_column_types(df_input, model)

    try:
        proba = model.predict_proba(df_input)[0, pos_idx]
        pred = model.predict(df_input)[0]
    except Exception as e:
        raise ValueError("Model transform/predict failed: " + str(e)) from e

    return {
        "prediction": pred,
        # Explicit about what this number means: P(the modeled event =
        # POSITIVE_CLASS), not P(survival) in general -- if POSITIVE_CLASS
        # is ever changed, this key's meaning changes with it.
        f"probability_{POSITIVE_CLASS.lower()}": float(proba),
        "survival_probability": float(1 - proba) if POSITIVE_CLASS == "Deceased" else float(proba),
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    global _feature_cols_cache
    if _feature_cols_cache is None:
        _feature_cols_cache = _get_feature_columns_from_pipeline(model)

    if _feature_cols_cache is None:
        raise ValueError("Could not determine feature columns expected by the model. Ensure feature_columns.json exists.")

    pos_idx = _get_positive_class_index(model)

    missing_cols = missing_required_columns(df, _feature_cols_cache)

    df_input = df.reindex(columns=_feature_cols_cache)
    df_input = _coerce_column_types(df_input, model)

    try:
        probas_deceased = model.predict_proba(df_input)[:, pos_idx]
        preds = model.predict(df_input)
    except Exception as e:
        raise ValueError("Model transform/predict failed: " + str(e)) from e

    out = df.copy()
    out["survival_prediction"] = preds
    out["survival_probability"] = 1 - probas_deceased if POSITIVE_CLASS == "Deceased" else probas_deceased
    if missing_cols:
        # Attach rather than raise: batch prediction still runs (useful for
        # partial data), but the caller can see exactly which columns were
        # fabricated by the imputer so they can decide whether to trust it.
        out.attrs["missing_input_columns"] = missing_cols
    return out