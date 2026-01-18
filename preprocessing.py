"""
Preprocessing utilities.

- data_load: read cleaned CSV and normalize column names.
- build_preprocessor: build ColumnTransformer including:
    * numeric pipeline: SimpleImputer(mean) -> StandardScaler
    * categorical pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
  Accepts either a DataFrame with the target column or features-only DataFrame.
- split_data: stratified train_test_split (returns X_train, X_test, y_train, y_test)
"""

from pathlib import Path
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


from src.config import DATA_FILE, TARGET_COL, TEST_SIZE, RANDOM_STATE


def data_load(data_path: Path = DATA_FILE) -> pd.DataFrame:
    """
    Load the pre-cleaned dataset and normalize column names (strip whitespace).

    Returns
    -------
    pd.DataFrame
        The cleaned dataset loaded from CSV.
    """
    df = pd.read_csv(data_path)
    # normalize column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _infer_feature_columns_from_df(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical feature column lists from a DataFrame.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(df_or_X: pd.DataFrame):
    """
    Build a ColumnTransformer for imputation + scaling/encoding.

    Parameters
    ----------
    df_or_X : pd.DataFrame
        Either:
          - the full cleaned DataFrame (including TARGET_COL)
          - OR features-only DataFrame (no target)

    Returns
    -------
    ColumnTransformer
        transformer applying numeric and categorical pipelines to respective columns.
    """
    # Work on a copy
    df = df_or_X.copy()

    # If target present, drop rows with missing target and remove target column
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df

    # Infer columns
    num_columns, cat_columns = _infer_feature_columns_from_df(X)

    # Numeric pipeline: impute -> scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute -> one-hot
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if num_columns:
        transformers.append(("num", numeric_transformer, num_columns))
    if cat_columns:
        transformers.append(("cat", categorical_transformer, cat_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train-test split. Expects target column to exist.

    Returns (X_train, X_test, y_train, y_test)
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in DataFrame columns: {df.columns.tolist()}")

    # ✅ Drop rows with missing target values
    df = df.dropna(subset=[TARGET_COL])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test

# quick module-level print for verification when imported (optional)
print("preprocessing module loaded")
