"""
Training module.

- loads cleaned_data.csv
- splits into train/test
- builds preprocessor (imputer+scaler / imputer+ohe)
- trains multiple models with GridSearchCV
- saves per-model best and overall best model (selected by CV score, not test score)
- saves feature column list to models/feature_columns.json
- saves a training results table to models/training_results.csv
- checks whether the winning model's probabilities are trustworthy (calibration),
  not just whether they rank patients correctly (ROC-AUC) -- and calibrates it
  if that turns out to matter, since the app displays the raw probability to users
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless -- this runs as a script/CI job, not in a notebook
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import (
    DATA_FILE, MODEL_DIR, BEST_MODEL_PATH, FEATURES_PATH,
    CV_FOLDS, SCORING, N_JOBS, RANDOM_STATE,
)
from src.preprocessing import data_load, split_data, build_preprocessor

# The event of clinical interest is death, not survival -- fix the "positive
# class" explicitly rather than relying on sklearn's alphabetical class
# ordering (which happens to put "Living" in predict_proba column 1). This
# way risk scores and ROC-AUC always mean "probability of Deceased",
# regardless of how classes_ happens to sort.
POSITIVE_CLASS = "Deceased"

CALIBRATION_REPORT_PATH = None  # set inside train_and_select_model, needs MODEL_DIR


def get_models_and_params() -> List[Tuple[str, object, Dict[str, List]]]:
    models_and_params = []

    log_reg = LogisticRegression(max_iter=1000)
    log_reg_params = {"clf__C": [0.01, 0.1, 1.0], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]}
    models_and_params.append(("log_reg", log_reg, log_reg_params))

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_params = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10]}
    models_and_params.append(("random_forest", rf, rf_params))

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gb_params = {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]}
    models_and_params.append(("gradient_boosting", gb, gb_params))

    svc = SVC(probability=True, random_state=RANDOM_STATE)
    svc_params = {"clf__C": [0.1, 1.0], "clf__kernel": ["rbf", "linear"]}
    models_and_params.append(("svc", svc, svc_params))

    dtc_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dtc_params = {"clf__max_depth": [None, 5, 10], "clf__criterion": ["gini", "entropy"]}
    models_and_params.append(("decision_tree", dtc_model, dtc_params))

    return models_and_params


def _positive_class_proba(model, X, positive_class: str = POSITIVE_CLASS) -> np.ndarray:
    """
    predict_proba column for `positive_class`, looked up from classes_ rather
    than assumed to be a fixed index -- see POSITIVE_CLASS comment above.
    """
    pos_idx = list(model.classes_).index(positive_class)
    return model.predict_proba(X)[:, pos_idx]


def _calibration_summary(y_true_binary: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Brier score + reliability-curve points for one set of predictions.

    Brier score is the calibration analogue of MSE: mean((predicted_prob -
    actual_outcome)^2). Lower is better, 0 is perfect, 0.25 is what a
    constant 50% prediction gets you on a balanced problem. Unlike ROC-AUC,
    it penalizes a probability that's confidently wrong, which is exactly
    what "is this number trustworthy" needs to measure.

    `strategy="quantile"` (equal-count bins) is used instead of equal-width
    bins because predicted probabilities are rarely spread evenly across
    [0, 1] -- equal-width bins often leave some near-empty and produce a
    noisy, hard-to-read curve on a dataset this size (~1900 rows).
    """
    brier = float(metrics.brier_score_loss(y_true_binary, y_proba))
    frac_pos, mean_pred = calibration_curve(y_true_binary, y_proba, n_bins=n_bins, strategy="quantile")
    return {
        "brier_score": brier,
        "mean_predicted_prob": mean_pred.tolist(),
        "fraction_of_positives": frac_pos.tolist(),
    }


def _plot_calibration(curves: Dict[str, Dict], out_path: Path, title: str) -> None:
    """
    Reliability diagram: perfectly calibrated predictions sit on the y=x
    diagonal. A curve below the diagonal means the model is overconfident in
    that probability range (predicts e.g. 80% but is only right ~60% of the
    time); above the diagonal means it's underconfident.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    for label, curve in curves.items():
        ax.plot(curve["mean_predicted_prob"], curve["fraction_of_positives"], marker="o",
                 label=f"{label} (Brier={curve['brier_score']:.3f})")
    ax.set_xlabel(f"Mean predicted probability of {POSITIVE_CLASS}")
    ax.set_ylabel(f"Observed fraction of {POSITIVE_CLASS}")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_and_select_model() -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    calibration_report_path = MODEL_DIR / "calibration_report.json"
    calibration_plot_path = MODEL_DIR / "calibration_curve.png"

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

    y_test_binary = (y_test == POSITIVE_CLASS).astype(int)

    models_and_params = get_models_and_params()
    best_overall = None
    best_cv_score = -np.inf
    best_name = None
    results = []
    calibration_curves: Dict[str, Dict] = {}

    for name, estimator, param_grid in models_and_params:
        print(f"\n=== Training {name} ===")
        pipeline = Pipeline([("preprocess", preprocessor), ("clf", estimator)])

        try:
            grid = GridSearchCV(
                estimator=pipeline, param_grid=param_grid, cv=CV_FOLDS,
                scoring=SCORING, n_jobs=N_JOBS, verbose=1,
            )
            grid.fit(X_train, y_train)
        except Exception as exc:
            # Don't let one model's failure (e.g. a slow/unstable SVC fit)
            # take down the whole training run -- log it and move on.
            print(f"!! {name} failed to train: {exc}")
            results.append({"model": name, "status": "failed", "error": str(exc)})
            continue

        best = grid.best_estimator_
        cv_score = grid.best_score_

        y_proba = _positive_class_proba(best, X_test)
        # roc_auc_score doesn't take a pos_label argument -- binarize y_test
        # against POSITIVE_CLASS explicitly so the score always means
        # "how well does this model rank Deceased above Living", regardless
        # of label spelling or sort order.
        test_roc = metrics.roc_auc_score(y_test_binary, y_proba)

        # Calibration is a separate question from ranking (ROC-AUC): a model
        # can rank patients correctly while still being wrong about what its
        # own probability numbers mean. Checked for every model, not just the
        # winner, since CV-score selection alone says nothing about this.
        cal = _calibration_summary(y_test_binary, y_proba)
        calibration_curves[name] = cal

        y_pred = best.predict(X_test)
        report_text = metrics.classification_report(y_test, y_pred)
        report_dict = metrics.classification_report(y_test, y_pred, output_dict=True)
        pos_report = report_dict.get(POSITIVE_CLASS, {})

        print(f"{name} best params: {grid.best_params_}")
        print(f"{name} CV {SCORING}: {cv_score:.4f}")
        print(f"{name} Test ROC-AUC (P({POSITIVE_CLASS})): {test_roc:.4f}")
        print(f"{name} Brier score (lower=better calibrated): {cal['brier_score']:.4f}")
        print(report_text)

        model_path = MODEL_DIR / f"{name}_best_model.joblib"
        joblib.dump(best, model_path)
        print(f"Saved {name} -> {model_path}")

        results.append({
            "model": name,
            "status": "ok",
            "best_params": grid.best_params_,
            "cv_score": cv_score,
            "test_roc_auc": test_roc,
            "test_precision": pos_report.get("precision"),
            "test_recall": pos_report.get("recall"),
            "test_f1": pos_report.get("f1-score"),
            "test_brier_score": cal["brier_score"],
            "test_report": report_dict,
        })

        # Select the overall best model by cross-validated score, not test
        # score. Using the test set to choose *between* models would make it
        # part of model selection rather than a clean final check -- the test
        # set is only used above to report the chosen/each model's
        # performance, never to decide which model wins.
        if cv_score > best_cv_score:
            best_cv_score = cv_score
            best_overall = best
            best_name = name

    final_model = best_overall
    calibration_note = "not calibrated -- raw model was already as good or better"

    if best_overall is not None:
        # The winning model was picked purely on ROC-AUC / CV score, which
        # only measures ranking. Before shipping it as BEST_MODEL_PATH --
        # the exact object app.py loads and displays a probability from --
        # check whether its probabilities are actually trustworthy, and
        # calibrate if not.
        #
        # method="sigmoid" (Platt scaling) rather than "isotonic": isotonic
        # regression is nonparametric and needs more data to avoid
        # overfitting the calibration mapping itself -- with ~1900 rows this
        # dataset is on the smaller side for it, sigmoid's 2-parameter fit is
        # the safer default here.
        #
        # cv=5 makes CalibratedClassifierCV internally cross-validate on
        # X_train rather than reusing the same data the pipeline was already
        # fit on -- calibrating on training predictions the model has
        # already memorized would systematically look better than it is.
        print(f"\n=== Checking calibration for winning model: {best_name} ===")
        calibrated = CalibratedClassifierCV(best_overall, method="sigmoid", cv=5)
        calibrated.fit(X_train, y_train)

        raw_proba = _positive_class_proba(best_overall, X_test)
        calibrated_proba = _positive_class_proba(calibrated, X_test)

        raw_cal = calibration_curves[best_name]
        calibrated_cal = _calibration_summary(y_test_binary, calibrated_proba)

        print(f"Raw Brier score:        {raw_cal['brier_score']:.4f}")
        print(f"Calibrated Brier score: {calibrated_cal['brier_score']:.4f}")

        # Only switch to the calibrated version if it's a real improvement,
        # not noise -- an arbitrary but reasonable margin. Calibration can't
        # improve ranking (ROC-AUC is monotone-invariant), so ROC-AUC is not
        # re-checked here; ranking was already decided by CV score above.
        improvement = raw_cal["brier_score"] - calibrated_cal["brier_score"]
        if improvement > 0.005:
            final_model = calibrated
            calibration_note = (
                f"calibrated (sigmoid) -- Brier improved from "
                f"{raw_cal['brier_score']:.4f} to {calibrated_cal['brier_score']:.4f}"
            )
            print(f"Using calibrated model -- {calibration_note}")
        else:
            calibration_note = (
                f"kept raw model -- calibration did not meaningfully improve Brier score "
                f"({raw_cal['brier_score']:.4f} -> {calibrated_cal['brier_score']:.4f})"
            )
            print(calibration_note)

        _plot_calibration(
            {f"{best_name} (raw)": raw_cal, f"{best_name} (calibrated)": calibrated_cal},
            calibration_plot_path,
            title=f"Reliability diagram -- {best_name}",
        )
        print(f"Saved calibration plot -> {calibration_plot_path}")

        with open(calibration_report_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_model": best_name,
                "decision": calibration_note,
                "all_models": calibration_curves,
                "winning_model_raw": raw_cal,
                "winning_model_calibrated": calibrated_cal,
            }, f, indent=2)
        print(f"Saved calibration report -> {calibration_report_path}")

        joblib.dump(final_model, BEST_MODEL_PATH)
        best_test_roc = next(r["test_roc_auc"] for r in results if r["model"] == best_name)
        print(
            f"\nOverall best model: {best_name} "
            f"(CV {SCORING}: {best_cv_score:.4f}, Test ROC-AUC: {best_test_roc:.4f}) -> {BEST_MODEL_PATH}"
        )
        print(f"Calibration: {calibration_note}")
    else:
        print("No model trained successfully.")

    results_df = pd.DataFrame(results)
    results_path = MODEL_DIR / "training_results.csv"
    # test_report/best_params are nested dicts -- keep the CSV to the
    # summary columns and leave the full structured results in the returned
    # DataFrame for anyone working with this interactively.
    summary_cols = [c for c in [
        "model", "status", "cv_score", "test_roc_auc",
        "test_precision", "test_recall", "test_f1", "test_brier_score",
    ] if c in results_df.columns]
    if summary_cols:
        results_df[summary_cols].to_csv(results_path, index=False)
        print(f"Saved training results -> {results_path}")

    return results_df


if __name__ == "__main__":
    df_results = train_and_select_model()
    print("\nSummary:\n", df_results)