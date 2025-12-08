# src/modelling/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# 1. Load data
# -----------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data with appropriate dtype handling.
    """
    dtype_dict = {
        'Column4': 'str',   # replace with actual column name
        'Column37': 'str'   # replace with actual column name
    }
    df = pd.read_csv(file_path, dtype=dtype_dict, low_memory=False)
    return df


# -----------------------------
# 2. Filter claims
# -----------------------------
def filter_claims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows with claims > 0.
    """
    return df[df['TotalClaims'] > 0].copy()


# -----------------------------
# 3. Handle missing values
# -----------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - Drop fully empty columns
    - Numeric columns: fill with median
    - Categorical columns: fill with mode
    """
    df = df.copy()

    # Drop fully empty columns
    fully_empty_cols = df.columns[df.isnull().all()]
    if len(fully_empty_cols) > 0:
        print(f"Dropping fully empty columns: {list(fully_empty_cols)}")
        df = df.drop(columns=fully_empty_cols)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill numeric columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns
    for col in categorical_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# -----------------------------
# 4. Feature engineering
# -----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'PolicyLossRatio' not in df.columns:
        df['PolicyLossRatio'] = df['TotalClaims'] / df['TotalPremium']
    return df


# -----------------------------
# 5. Encode categorical variables
# -----------------------------
def get_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(include=['object']).columns.tolist()


def encode_categoricals(df: pd.DataFrame, categorical_cols):
    df = df.copy()
    if categorical_cols:
        # Updated for sklearn 1.7.2: use sparse_output=False
        encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)
    return df


# -----------------------------
# 6. Train-test split
# -----------------------------
def train_test_split_data(df: pd.DataFrame, target: str, test_size: float = 0.3, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
