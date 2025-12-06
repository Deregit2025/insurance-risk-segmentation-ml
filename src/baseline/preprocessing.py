# src/baseline/preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path
import os


# -------------------------------------
# 1️⃣ Load Raw Data
# -------------------------------------
def load_raw_data(raw_path: str) -> pd.DataFrame:
    """
    Load raw insurance data from a pipe-delimited text file.
    Save an unclean copy for future reference.
    """
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"{raw_file} does not exist.")
    
    df = pd.read_csv(raw_file, delimiter="|", low_memory=False)
    print("✅ Raw data loaded successfully!")

    # Save unclean version
    unclean_path = Path("data/raw/unclean.csv")
    unclean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(unclean_path, index=False)
    print(f"✅ Unclean data saved to {unclean_path}")

    return df


# -------------------------------------
# 2️⃣ Outlier Handling
# -------------------------------------
def cap_outliers(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return np.clip(series, lower, upper)


# -------------------------------------
# 3️⃣ Clean & Preprocess Data
# -------------------------------------
# -------------------------------------
# 3️⃣ Clean & Preprocess Data (Safe Version)
# -------------------------------------
def clean_data_safe(df: pd.DataFrame) -> pd.DataFrame:

    # ---------- Numeric Cleaning ----------
    def clean_numeric(col):
        return (
            df[col]
            .astype(str)
            .str.replace(",", "")       # remove commas
            .str.strip()
            .replace({"": "0", "nan": "0"})  # empty → 0
        )

    numeric_cols_to_clean = ["TotalClaims", "TotalPremium"]
    for col in numeric_cols_to_clean:
        df[col] = clean_numeric(col)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---------- Other Numeric Columns ----------
    numeric_cols = [
        "Cylinders", "cubiccapacity", "kilowatts",
        "NumberOfDoors", "CapitalOutstanding", "CustomValueEstimate", "SumInsured"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # ---------- Categorical Missing ----------
    categorical_cols = [
        "Gender", "MaritalStatus", "LegalType", "Citizenship",
        "Title", "Language", "Bank", "AccountType",
        "CoverType", "CoverCategory", "CoverGroup", "Section", "Product", "Province"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].fillna("Unknown")

    # ---------- Convert Dates ----------
    date_cols = {
        "TransactionMonth": None,
        "RegistrationYear": "%Y",
        "VehicleIntroDate": None
    }
    for col, fmt in date_cols.items():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")

    # ---------- Remove Duplicates ----------
    df.drop_duplicates(
        subset=["PolicyID", "UnderwrittenCoverID", "TransactionMonth"],
        inplace=True
    )

    # ---------- Feature Engineering ----------
    if "TransactionMonth" in df.columns and "RegistrationYear" in df.columns:
        df["VehicleAge"] = (
            df["TransactionMonth"].dt.year - df["RegistrationYear"].dt.year
        ).fillna(0).astype(int)

    df["PolicyLossRatio"] = np.where(
        df["TotalPremium"] > 0,
        df["TotalClaims"] / df["TotalPremium"],
        0
    )

    # ---------- Done ----------
    print(f"Total rows after cleaning: {len(df)}")
    return df



# -------------------------------------
# 4️⃣ Save Cleaned Data
# -------------------------------------
def save_cleaned_data(df: pd.DataFrame, save_path: str):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Cleaned data saved to {path}")

    
# -------------------------------------
# 5️⃣ Main Execution
# -------------------------------------
if __name__ == "__main__":
    raw_path = os.path.join(os.getcwd(), "data", "raw", "insurance_data.txt")
    save_path = os.path.join(os.getcwd(), "data", "processed", "insurance_data_cleaned.csv")
    
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data_safe(df_raw)  # <- use safe version
    save_cleaned_data(df_clean, save_path)

