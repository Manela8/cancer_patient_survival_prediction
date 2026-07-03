# cancer_patient_survival_prediction
Built an end‑to‑end ML pipeline for healthcare data. Conducted EDA and data cleaning, developed modular preprocessing, training, and deployment scripts with scikit‑learn, automated model selection (Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Tree) via GridSearchCV, checked model calibration, deployed the best model through a Streamlit app, and built a set of business-facing SQL queries on top of the cleaned data.

# 🩺 Patient Survival Prediction System

An end‑to‑end machine learning pipeline for healthcare data, designed to predict patient survival chances (Living / Deceased) from clinical and genomic features.
This project covers the full lifecycle: **EDA → data cleaning → preprocessing → model training & calibration → deployment → Streamlit app → SQL business intelligence**.

---

## 📖 Overview

Built an end-to-end machine learning pipeline on the METABRIC breast cancer dataset to predict patient survival status (Living/Deceased) from clinical and genomic features, deployed as an interactive web application.

### Data Cleaning & EDA
Cleaned a 2,509-row clinical dataset with structured, documented decisions rather than blanket imputation — standardized inconsistent column formatting, corrected data-entry errors, removed out-of-scope records (non-breast-carcinoma cases mislabeled in the cohort), and distinguished between randomly missing values versus values missing by design (an entire block of treatment/genomic columns tied to specific patient cohorts that were never profiled). That distinction shaped how missingness was handled downstream, preserving data integrity instead of fabricating values for untested patients.

### Preprocessing & Modeling
Built a modular, reusable preprocessing pipeline (median imputation + scaling for numeric features, imputation + one-hot encoding for categorical features) using scikit-learn's `ColumnTransformer`. Trained and compared five candidate models — Logistic Regression, Random Forest, Gradient Boosting, SVC, and Decision Tree — via automated hyperparameter tuning with `GridSearchCV`, selecting the best performer by cross-validated ROC-AUC. Went beyond ranking performance alone by checking model calibration (Brier score), ensuring the probabilities the app displays to users are actually trustworthy, not just well-ranked.

### Deployment
Deployed the selected model through a Streamlit application supporting both single-patient prediction (interactive form) and batch prediction (CSV upload with downloadable results). Designed the interface around real usability: grouped clinical inputs into logical sections, bounded numeric fields to clinically valid ranges, and added a one-click example-patient loader for fast demoing.

### Business Intelligence
Translated the cleaned dataset into a set of analyst-facing SQL queries answering concrete business questions: overall and subgroup survival rates, treatment pattern analysis, high-risk patient segmentation, molecular subtype breakdowns, and ongoing data-quality monitoring — each framed around the actual decision it supports (e.g., where to prioritize clinical resources, whether treatment protocols are applied consistently, which patient segments need closer follow-up).

### Impact
The project demonstrates a complete, production-oriented ML workflow — from raw clinical data to a usable clinical decision-support demo — with attention to data integrity, model trustworthiness, and business relevance at every stage, not just predictive accuracy in isolation.

---

## 📂 Project Structure
```
CapstoneProject/
│
├── data/
│   ├── Breast_Cancer_METABRIC.csv   <-- raw data
│   └── cleaned_data.csv             <-- output of Data_Cleaning.ipynb
│
├── models/
│   ├── *_best_model.joblib          <-- one saved model per candidate (log_reg, random_forest, ...)
│   ├── best_model.joblib            <-- overall winner (selected by CV score, calibrated if it helped)
│   ├── feature_columns.json         <-- feature list the pipeline expects, for runtime alignment
│   ├── training_results.csv         <-- CV score / ROC-AUC / precision / recall / F1 / Brier per model
│   ├── calibration_report.json      <-- Brier scores + reliability-curve data, raw vs calibrated
│   └── calibration_curve.png        <-- reliability diagram for the winning model
│
├── notebooks/
│   ├── Data_Cleaning.ipynb    <-- data cleaning workflow (run first)
│   └── EDA.ipynb              <-- exploratory data analysis (run on cleaned data)
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── training.py
│   └── deployment.py
│
├── app.py         <-- Streamlit app
├── script.sql     <-- analyst-facing SQL queries on the cleaned data
├── run_all.bat    <-- full pipeline (preprocess → train → deploy → app)
└── run_app.bat    <-- launch app only
```

---

## ⚙️ Features
- **EDA & Data Cleaning**: Jupyter notebooks documenting dataset exploration and preprocessing decisions, including how missing-by-design vs. randomly-missing values were told apart.
- **Modular Scripts**: Preprocessing, training, and deployment separated for clarity and reproducibility, all reading shared settings from `src/config.py`.
- **Model Selection**: Automated evaluation of Logistic Regression, Random Forest, Gradient Boosting, SVC, and Decision Tree using `GridSearchCV`, selected by cross-validated ROC-AUC (not test-set score, to keep the test set an unbiased final check).
- **Calibration Check**: The winning model's probabilities are checked for trustworthiness (Brier score, reliability diagram) and calibrated with Platt scaling if that measurably improves them — since the app shows a raw probability to the user, not just a ranked class.
- **Deployment**: Best model saved and loaded lazily; feature/category schema is derived from the fitted pipeline itself (`get_input_schema`) rather than hardcoded, so the app form stays in sync with whatever the model actually expects.
- **Streamlit App**: Interactive interface for single-patient prediction (grouped input form, clinically-bounded fields, one-click example-patient loader) and batch prediction (CSV upload, missing-column warnings, downloadable results).
- **SQL Business Intelligence**: `script.sql` turns the cleaned dataset into analyst-facing queries on survival rates, treatment patterns, tumor/subtype characteristics, risk segmentation, and data-quality monitoring.
- **Automation**: Batch files for one‑click execution of pipeline and app launch.

---

## 📊 Business Intelligence — SQL Analysis

> *"Each query turns a raw clinical variable into a business-relevant subgroup comparison — the goal isn't just descriptive stats, it's identifying which patient populations need more attention, whether treatment is being applied consistently, and whether the underlying data is even reliable enough to trust those conclusions."*

### Survival & outcomes
**Need:** Hospital administrators and research teams need to know if the patient population's outcomes are improving or need intervention — this is the single most fundamental question in oncology data.
**Impact:** The overall survival rate becomes a baseline metric leadership tracks over time (like a KPI dashboard). The stage-by-stage and biomarker-combination breakdowns tell clinicians *which specific patient subgroups* are underperforming — e.g., if triple-negative or stage 4 patients have a much lower survival rate, that's where research funding, specialist referrals, or trial recruitment should be prioritized. Without this, resources get spread evenly instead of where they're needed most.

### Treatment patterns
**Need:** Hospitals need to know whether treatment protocols are actually being followed consistently, and whether certain patients are being under- or over-treated.
**Impact:** If the "treatment combinations" query shows a large group of high-risk patients receiving *no* chemo, that's a red flag worth investigating — is it clinical judgment, or a gap in care access? The stage-controlled chemo comparison helps separate "chemo works" from "sicker patients just get more chemo" — that distinction matters because it stops leadership from drawing the wrong causal conclusion from raw numbers. This is the difference between a query that looks like insight and one that's actually decision-safe.

### Tumor & disease characteristics
**Need:** Clinical and research teams need to understand *which* biological subtype of cancer is most common and most dangerous in their patient population, since treatment protocols differ significantly by subtype.
**Impact:** Identifying the triple-negative segment specifically matters because it's the most aggressive, hardest-to-treat subtype with fewer targeted therapy options — knowing its size and survival rate directly informs whether the hospital needs more specialists or clinical trial access for that specific group. The subtype breakdown also feeds directly into the actual capstone goal: it's the same reasoning that justified building the ML model in the first place — different subtypes need different risk models.

### Demographics & risk segmentation
**Need:** Population health teams need to know which age groups are most affected and where to focus early-detection or outreach campaigns.
**Impact:** If survival rates drop sharply after a certain age band, that tells a hospital where to invest in screening programs or age-specific care pathways. The "high-risk cohort" query is a concrete, actionable patient list — this is the kind of query that in a real hospital system would feed a case-management team's worklist, i.e., "here are the specific patients who need closer follow-up."

### Data quality
**Need:** This one isn't for a business stakeholder — it's for whoever maintains this pipeline. Any analysis built on top of columns with heavy missingness is only as trustworthy as that missingness is understood.
**Impact:** This is what stops someone from confidently reporting "68% of patients received chemo" when actually 30% of that column is just unrecorded, not "No." It protects every other query above from being quietly wrong.

---

## 📊 Example Workflow
1. Explore the raw dataset in `data/`.
2. Clean it with `notebooks/Data_Cleaning.ipynb`, then explore it with `notebooks/EDA.ipynb` — both write/read `data/cleaned_data.csv`.
3. Train and calibrate models with `src/training.py` (saves everything under `models/`).
4. Serve predictions with `src/deployment.py` (`predict_single`, `predict_batch`, schema helpers).
5. Interact via `app.py` (Streamlit) — single-patient form or batch CSV upload.
6. Run analyst queries in `script.sql` against the cleaned data loaded into a database.

---

## 🛠️ Tech Stack
- **Python** (pandas, scikit‑learn, joblib, matplotlib)
- **Streamlit** (interactive app)
- **Jupyter Notebooks** (EDA & cleaning)
- **SQL** (MySQL — business-intelligence queries in `script.sql`)
- **Batch scripting** (automation)

---

## ⚠️ Disclaimer
For educational and demonstration purposes only. Not a substitute for professional medical advice, diagnosis, or treatment. Trained on a historical research cohort and may not generalize to individual patients.