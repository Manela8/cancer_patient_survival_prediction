# cancer_patient_survival_prediction
Built an end‑to‑end ML pipeline for healthcare data. Conducted EDA and data cleaning, developed modular preprocessing, training, and deployment scripts with scikit‑learn, automated model selection (Logistic Regression, Random Forest, Gradient Boosting, SVC, Decision Tree) via GridSearchCV, and deployed the best model through a Streamlit app.

# 🩺 Patient Survival Prediction System

An end‑to‑end machine learning pipeline for healthcare data, designed to predict patient survival chances.  
This project covers the full lifecycle: **EDA → data cleaning → model training → deployment → Streamlit app interface → automation**.

---

## 📂 Project Structure
```
CapstoneProject/
│
├── data/
│   └── cleaned_data.csv
 |    └── raw_data.csv
│
├── models/
│   ├── contains the models
│
│
├── notebooks/
│   ├── 01_EDA.ipynb           <-- Exploratory Data Analysis
│   └── 02_DataCleaning.ipynb  <-- Data cleaning workflow
│
├── src/
│   ├── _init_.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── training.py
│   └── deployment.py
│
├── app.py        <-- Streamlit app
├── run_all.bat   <-- Full pipeline (preprocess → train → deploy → app)
└── run_app.bat   <-- Launch app only
```

---

## ⚙️ Features
- **EDA & Data Cleaning**: Jupyter notebooks documenting dataset exploration and preprocessing.
- **Modular Scripts**: Preprocessing, training, and deployment separated for clarity and reproducibility.
- **Model Selection**: Automated evaluation of Logistic Regression, Random Forest, Gradient Boosting, SVC, and Decision Tree using GridSearchCV.
- **Deployment**: Best model saved and loaded lazily for predictions.
- **Streamlit App**: Interactive interface for single and batch patient survival predictions.
- **Automation**: Batch files for one‑click execution of pipeline and app launch.

---

📊 Example Workflow
- Explore dataset in data/.
- Clean and save dataset in data, using notebooks/DataCleaning.ipynb and EDA.ipynb.
- Train models with src/training.py.
- Deploy predictions with src/deployment.py.
- Interact via app.py (Streamlit).

🛠️ Tech Stack
- Python (pandas, scikit‑learn, joblib)
- Streamlit (interactive app)
- Jupyter Notebooks (EDA & cleaning)
- Batch scripting (automation)



