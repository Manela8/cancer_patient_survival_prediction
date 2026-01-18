"""
Project configuration and paths.
"""

from pathlib import Path

# Project root (parent of src/)
BASE_DIR: Path = Path(__file__).resolve().parents[1]

# Data and model paths
DATA_DIR: Path = BASE_DIR / "data"
DATA_FILE: Path = DATA_DIR / "cleaned_data.csv"

MODEL_DIR: Path = BASE_DIR / "models"
BEST_MODEL_PATH: Path = MODEL_DIR / "best_model.joblib"
FEATURES_PATH: Path = MODEL_DIR / "feature_columns.json"

# Target column (set exactly to your cleaned_data.csv column name)
TARGET_COL: str = "Overall_Survival_Status"

# Train/test & CV settings
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
SCORING: str = "roc_auc"
N_JOBS: int = -1
MODEL_DIR.mkdir(parents=True, exist_ok=True)
