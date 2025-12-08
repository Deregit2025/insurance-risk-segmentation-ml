# src/hypothesis_testing/segmentation.py

import pandas as pd


def clean_gender(df: pd.DataFrame, column: str = "Gender") -> pd.DataFrame:
    """
    Cleans the Gender column for hypothesis testing.
    Replaces inconsistent or missing values with standardized categories:
    - 'Male'
    - 'Female'
    Drops or renames any other values like 'Not Specified' or 'Nan' to NaN.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name for gender. Default is 'Gender'.

    Returns:
        pd.DataFrame: DataFrame with cleaned Gender column.
    """
    df = df.copy()
    df[column] = df[column].replace({"Nan": pd.NA, "Not Specified": pd.NA})
    df[column] = df[column].dropna().map(lambda x: x if x in ["Male", "Female"] else pd.NA)
    df[column] = df[column].astype("category")
    return df


def segment_two_groups(df: pd.DataFrame, column: str, group_a_value, group_b_value):
    """
    Creates Group A and Group B based on two selected categories.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to segment.
        group_a_value: Value for Group A.
        group_b_value: Value for Group B.
    
    Returns:
        tuple: (group_a_df, group_b_df)
    """
    group_a = df[df[column] == group_a_value].copy()
    group_b = df[df[column] == group_b_value].copy()
    return group_a, group_b


def binary_split(df: pd.DataFrame, column: str):
    """
    For columns that naturally have two classes (e.g., Gender: Male/Female),
    automatically split into Group A and Group B.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to split.
    
    Returns:
        tuple: (group_a_df, group_b_df)
    
    Raises:
        ValueError: If column does not have exactly two unique values.
    """
    categories = df[column].dropna().unique()
    if len(categories) != 2:
        raise ValueError(f"Column {column} does not have exactly two unique values. Consider cleaning it first.")
    
    return segment_two_groups(df, column, categories[0], categories[1])
