"""
Training module.

- loads cleaned_data.csv
- splits into train/test
- builds preprocessor (imputer+scaler / imputer+ohe)
- trains multiple models with GridSearchCV
- saves per-model best and overall best model
- saves feature column list to models/feature_columns.json
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import DATA_FILE, MODEL_DIR, BEST_MODEL_PATH, FEATURES_PATH, CV_FOLDS, SCORING, N_JOBS
from src.preprocessing import data_load, split_data, build_preprocessor

def get_models_and_params() -> List[Tuple[str, object, Dict[str, List]]]:
    models_and_params = []

    log_reg = LogisticRegression(max_iter=1000)
    log_reg_params = {"clf__C": [0.01, 0.1, 1.0], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]}
    models_and_params.append(("log_reg", log_reg, log_reg_params))

    rf = RandomForestClassifier(random_state=42)
    rf_params = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10]}
    models_and_params.append(("random_forest", rf, rf_params))

    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]}
    models_and_params.append(("gradient_boosting", gb, gb_params))

    svc = SVC(probability=True)
    svc_params = {"clf__C": [0.1, 1.0], "clf__kernel": ["rbf", "linear"]}
    models_and_params.append(("svc", svc, svc_params))

    dtc_model = DecisionTreeClassifier(random_state=42)
    dtc_params = {"clf__max_depth": [None, 5, 10], "clf__criterion": ["gini", "entropy"]}
    models_and_params.append(("decision_tree", dtc_model, dtc_params))

    return models_and_params

def train_and_select_model() -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pre-cleaned data ->", DATA_FILE)
    df = data_load(DATA_FILE)
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Save feature columns (before any accidental modifications) and split
    X_train, X_test, y_train, y_test = split_data(df)
    # Save the training feature list to disk for runtime alignment
    feature_cols = X_train.columns.tolist()
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"Saved feature list ({len(feature_cols)}) to {FEATURES_PATH}")

    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    preprocessor = build_preprocessor(X_train)
    print("Preprocessor built.")

    models_and_params = get_models_and_params()
    best_overall = None
    best_score = -np.inf
    best_name = None
    results = []

    for name, estimator, param_grid in models_and_params:
        print(f"\n=== Training {name} ===")
        pipeline = Pipeline([("preprocess", preprocessor), ("clf", estimator)])

        grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=CV_FOLDS, scoring=SCORING, n_jobs=N_JOBS, verbose=1)
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        cv_score = grid.best_score_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test_roc = metrics.roc_auc_score(y_test, y_proba)

        print(f"{name} best params: {grid.best_params_}")
        print(f"{name} CV {SCORING}: {cv_score:.4f}")
        print(f"{name} Test ROC-AUC: {test_roc:.4f}")
        print(metrics.classification_report(y_test, y_pred))

        model_path = MODEL_DIR / f"{name}_best_model.joblib"
        joblib.dump(best, model_path)
        print(f"Saved {name} -> {model_path}")

        results.append({"model": name, "best_params": grid.best_params_, "cv_score": cv_score, "test_roc_auc": test_roc})

        if test_roc > best_score:
            best_score = test_roc
            best_overall = best
            best_name = name

    if best_overall is not None:
        joblib.dump(best_overall, BEST_MODEL_PATH)
        print(f"\nOverall best model: {best_name} (ROC-AUC: {best_score:.4f}) -> {BEST_MODEL_PATH}")
    else:
        print("No model trained successfully.")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_results = train_and_select_model()
    print("\nSummary:\n", df_results)
