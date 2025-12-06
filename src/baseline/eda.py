# =====================================
# EDA.py – Exploratory Data Analysis for Insurance Dataset
# =====================================

# -----------------------------
# Step 0: Imports
# -----------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.1)


# -----------------------------
# Step 1: Import and Load Cleaned Data
# -----------------------------

import os
import pandas as pd



import os
import pandas as pd

def load_cleaned_data(processed_folder="data/processed", filename="insurance_data_cleaned.csv"):
    """
    Load cleaned insurance data.
    Works from notebook or script.
    """
    # Get current working directory (notebook location or project root if run from ML)
    cwd = os.getcwd()  # e.g., ML/notebooks/baseline or ML/

    # Try to locate ML root dynamically
    # If cwd ends with 'notebooks/baseline', move up two levels
    if cwd.endswith(os.path.join("notebooks", "baseline")):
        project_root = os.path.abspath(os.path.join(cwd, "..", ".."))
    else:
        project_root = cwd  # assume cwd is project root

    # Construct full path
    cleaned_file_path = os.path.join(project_root, processed_folder, filename)

    if not os.path.exists(cleaned_file_path):
        raise FileNotFoundError(f"File not found: {cleaned_file_path}")

    import pandas as pd
    df = pd.read_csv(cleaned_file_path)
    print(f"Cleaned data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
















# -----------------------------
# Step 2: Portfolio Overview
# -----------------------------
def portfolio_summary(df):
    summary = {}
    summary["Total Policies"] = df["PolicyID"].nunique()
    summary["Total Premium"] = df["TotalPremium"].sum()
    summary["Total Claims"] = df["TotalClaims"].sum()
    summary["Average Sum Insured"] = df["SumInsured"].mean()
    summary["Overall Loss Ratio"] = summary["Total Claims"] / summary["Total Premium"]
    return summary


def loss_ratio_by_group(df, group_cols):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    grouped = df.groupby(group_cols).agg(
        TotalClaims=pd.NamedAgg(column="TotalClaims", aggfunc="sum"),
        TotalPremium=pd.NamedAgg(column="TotalPremium", aggfunc="sum")
    )
    grouped["LossRatio"] = grouped["TotalClaims"] / grouped["TotalPremium"]
    return grouped.sort_values("LossRatio", ascending=False)


def policy_counts_by_group(df, group_cols):
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    counts = df.groupby(group_cols)["PolicyID"].nunique().reset_index()
    counts.rename(columns={"PolicyID": "PolicyCount"}, inplace=True)
    return counts.sort_values("PolicyCount", ascending=False)


# -----------------------------
# Step 3: Data Summarization & Quality Assessment
# -----------------------------
def numeric_summary(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    summary = df[numeric_cols].describe().T
    summary["missing"] = df[numeric_cols].isna().sum()
    summary["missing_pct"] = df[numeric_cols].isna().mean() * 100
    return summary


def categorical_summary(df, cat_cols=None):
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    summary = pd.DataFrame(columns=["unique", "top", "freq", "missing", "missing_pct"])
    for col in cat_cols:
        summary.loc[col, "unique"] = df[col].nunique()
        summary.loc[col, "top"] = df[col].mode().values[0] if not df[col].mode().empty else None
        summary.loc[col, "freq"] = df[col].value_counts().values[0] if not df[col].value_counts().empty else None
        summary.loc[col, "missing"] = df[col].isna().sum()
        summary.loc[col, "missing_pct"] = df[col].isna().mean() * 100
    return summary


def data_type_overview(df):
    return df.dtypes


def missing_values_overview(df):
    missing = df.isna().sum()
    missing_pct = df.isna().mean() * 100
    return pd.DataFrame({"missing": missing, "missing_pct": missing_pct}).sort_values("missing", ascending=False)


# -----------------------------
# Step 4: Univariate Analysis
# -----------------------------
def plot_numeric_distribution(df, numeric_cols=None, bins=30, save_fig=False):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col].dropna(), bins=bins, kde=True, color="skyblue")
        plt.axvline(df[col].mean(), color='red', linestyle='--', label=f"Mean: {df[col].mean():.2f}")
        plt.axvline(df[col].median(), color='green', linestyle='-', label=f"Median: {df[col].median():.2f}")
        plt.title(f'Distribution of {col}')
        plt.legend()
        if save_fig: plt.savefig(f"{col}_hist.png", bbox_inches="tight")
        plt.show()


def plot_numeric_boxplots(df, numeric_cols=None, save_fig=False):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[col], color="lightcoral")
        plt.title(f'Boxplot of {col}')
        if save_fig: plt.savefig(f"{col}_boxplot.png", bbox_inches="tight")
        plt.show()


def plot_categorical_distribution(df, cat_cols=None, top_n=10, save_fig=False):
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    for col in cat_cols:
        plt.figure(figsize=(10,5))
        top_values = df[col].value_counts().nlargest(top_n)
        sns.barplot(x=top_values.values, y=top_values.index, palette="viridis")
        plt.title(f'Top {top_n} Values in {col}')
        plt.xlabel("Count")
        plt.ylabel(col)
        if save_fig: plt.savefig(f"{col}_bar.png", bbox_inches="tight")
        plt.show()


# -----------------------------
# Step 5: Bivariate & Multivariate Analysis
# -----------------------------
def loss_ratio_overall(df):
    total_claims = df["TotalClaims"].sum()
    total_premium = df["TotalPremium"].sum()
    loss_ratio = total_claims / total_premium
    print(f"Overall Loss Ratio: {loss_ratio:.4f}")
    return loss_ratio


def loss_ratio_by_category(df, category_col):
    lr = df.groupby(category_col).apply(lambda x: x["TotalClaims"].sum() / x["TotalPremium"].sum()).sort_values(ascending=False)
    print(f"\nLoss Ratio by {category_col}:\n{lr}")
    
    plt.figure(figsize=(10,6))
    lr.plot(kind='bar', color='teal')
    plt.title(f'Loss Ratio by {category_col}')
    plt.ylabel('Loss Ratio')
    plt.xlabel(category_col)
    plt.xticks(rotation=45, ha='right')
    plt.show()
    return lr


def plot_correlation_matrix(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.show()
    return corr


def scatter_trends(df, x_col, y_col, hue_col=None):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.7, palette="Set2")
    plt.title(f"{y_col} vs {x_col}" + (f" by {hue_col}" if hue_col else ""))
    plt.show()


# -----------------------------
# Step 6: Temporal Trends and Monthly Analysis
# -----------------------------
def monthly_totals(df):
    monthly = df.groupby("TransactionMonth").agg({
        "TotalPremium": "sum",
        "TotalClaims": "sum"
    }).reset_index()
    monthly["LossRatio"] = monthly["TotalClaims"] / monthly["TotalPremium"]
    
    plt.figure(figsize=(12,6))
    plt.plot(monthly["TransactionMonth"], monthly["TotalPremium"], label="TotalPremium", marker='o')
    plt.plot(monthly["TransactionMonth"], monthly["TotalClaims"], label="TotalClaims", marker='o')
    plt.title("Monthly TotalPremium and TotalClaims")
    plt.xlabel("Transaction Month")
    plt.ylabel("Amount")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
    plt.figure(figsize=(12,6))
    plt.plot(monthly["TransactionMonth"], monthly["LossRatio"], color="red", marker='o')
    plt.title("Monthly Loss Ratio Trend")
    plt.xlabel("Transaction Month")
    plt.ylabel("Loss Ratio")
    plt.xticks(rotation=45)
    plt.show()
    
    return monthly


def monthly_trend_by_category(df, category_col):
    monthly_cat = df.groupby([category_col, "TransactionMonth"]).agg({
        "TotalPremium": "sum",
        "TotalClaims": "sum"
    }).reset_index()
    monthly_cat["LossRatio"] = monthly_cat["TotalClaims"] / monthly_cat["TotalPremium"]
    
    plt.figure(figsize=(14,6))
    sns.lineplot(data=monthly_cat, x="TransactionMonth", y="LossRatio", hue=category_col, marker='o')
    plt.title(f"Monthly Loss Ratio Trend by {category_col}")
    plt.xlabel("Transaction Month")
    plt.ylabel("Loss Ratio")
    plt.xticks(rotation=45)
    plt.legend(title=category_col, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.show()
    
    return monthly_cat


# -----------------------------
# Step 7: Outlier Detection and Distribution Analysis
# -----------------------------
def plot_distributions(df, num_cols):
    for col in num_cols:
        plt.figure(figsize=(10,4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
        
        plt.figure(figsize=(10,4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()


def plot_distributions_by_category(df, num_cols, cat_col):
    for col in num_cols:
        plt.figure(figsize=(12,6))
        sns.boxplot(x=cat_col, y=col, data=df)
        plt.title(f'{col} Distribution by {cat_col}')
        plt.xticks(rotation=45)
        plt.show()


# -----------------------------
# Step 8: Vehicle Makes/Models Analysis
# -----------------------------
def vehicle_risk_summary(df, top_n=10):
    vehicle_summary = df.groupby(["make", "Model"]).agg(
        TotalPremium=pd.NamedAgg(column="TotalPremium", aggfunc="sum"),
        TotalClaims=pd.NamedAgg(column="TotalClaims", aggfunc="sum"),
        CountPolicies=pd.NamedAgg(column="PolicyID", aggfunc="count")
    ).reset_index()
    
    # Avoid chained assignment warning
    vehicle_summary["LossRatio"] = vehicle_summary["TotalClaims"] / vehicle_summary["TotalPremium"]
    vehicle_summary["LossRatio"] = vehicle_summary["LossRatio"].replace([np.inf, -np.inf], np.nan)
    vehicle_summary = vehicle_summary.dropna(subset=["LossRatio"])
    
    vehicle_summary = vehicle_summary.sort_values("LossRatio", ascending=False)
    
    top_vehicles = vehicle_summary.head(top_n)
    bottom_vehicles = vehicle_summary.tail(top_n)
    
    # Return only top & bottom vehicles
    return top_vehicles, bottom_vehicles




def plot_vehicle_risk(top_vehicles, bottom_vehicles):
    plt.figure(figsize=(12,6))
    sns.barplot(x="LossRatio", y="make", hue="Model", data=top_vehicles)
    plt.title("Top High-Risk Vehicles by Loss Ratio")
    plt.xlabel("Loss Ratio")
    plt.ylabel("Vehicle Make")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    plt.figure(figsize=(12,6))
    sns.barplot(x="LossRatio", y="make", hue="Model", data=bottom_vehicles)
    plt.title("Top Low-Risk Vehicles by Loss Ratio")
    plt.xlabel("Loss Ratio")
    plt.ylabel("Vehicle Make")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


# -----------------------------
# Step 9: Temporal Trend Analysis
# -----------------------------
def temporal_trend_summary(df, segment=None):
    df_temp = df.copy()
    
    if segment and segment in df_temp.columns:
        trend = df_temp.groupby([segment, pd.Grouper(key="TransactionMonth", freq="M")]).agg(
            TotalPremium=pd.NamedAgg(column="TotalPremium", aggfunc="sum"),
            TotalClaims=pd.NamedAgg(column="TotalClaims", aggfunc="sum"),
            PolicyCount=pd.NamedAgg(column="PolicyID", aggfunc="count")
        ).reset_index()
        
        # Apply rolling per segment
        trend["LossRatio"] = trend["TotalClaims"] / trend["TotalPremium"]
        trend["LossRatio"] = trend["LossRatio"].replace([np.inf, -np.inf], np.nan)
        
        trend["TotalPremium_Roll"] = trend.groupby(segment)["TotalPremium"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        trend["TotalClaims_Roll"] = trend.groupby(segment)["TotalClaims"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        trend["LossRatio_Roll"] = trend.groupby(segment)["LossRatio"].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
    else:
        trend = df_temp.groupby(pd.Grouper(key="TransactionMonth", freq="M")).agg(
            TotalPremium=pd.NamedAgg(column="TotalPremium", aggfunc="sum"),
            TotalClaims=pd.NamedAgg(column="TotalClaims", aggfunc="sum"),
            PolicyCount=pd.NamedAgg(column="PolicyID", aggfunc="count")
        ).reset_index()
        
        trend["LossRatio"] = trend["TotalClaims"] / trend["TotalPremium"]
        trend["LossRatio"] = trend["LossRatio"].replace([np.inf, -np.inf], np.nan)
        trend["TotalPremium_Roll"] = trend["TotalPremium"].rolling(3, min_periods=1).mean()
        trend["TotalClaims_Roll"] = trend["TotalClaims"].rolling(3, min_periods=1).mean()
        trend["LossRatio_Roll"] = trend["LossRatio"].rolling(3, min_periods=1).mean()
    
    return trend





def plot_temporal_trends(trend_df, segment=None):
    import matplotlib.pyplot as plt
    
    if segment and segment in trend_df.columns:
        segments = trend_df[segment].unique()
        for seg in segments:
            seg_df = trend_df[trend_df[segment] == seg]
            plt.figure(figsize=(12,4))
            plt.plot(seg_df["TransactionMonth"], seg_df["TotalPremium_Roll"], label="TotalPremium")
            plt.plot(seg_df["TransactionMonth"], seg_df["TotalClaims_Roll"], label="TotalClaims")
            plt.plot(seg_df["TransactionMonth"], seg_df["LossRatio_Roll"], label="LossRatio")
            plt.title(f"Temporal Trends for {segment} = {seg}")
            plt.xlabel("Transaction Month")
            plt.ylabel("Amount / Ratio")
            plt.legend()
            plt.show()
    else:
        # No segment filtering — use the full df
        plt.figure(figsize=(12,4))
        plt.plot(trend_df["TransactionMonth"], trend_df["TotalPremium_Roll"], label="TotalPremium")
        plt.plot(trend_df["TransactionMonth"], trend_df["TotalClaims_Roll"], label="TotalClaims")
        plt.plot(trend_df["TransactionMonth"], trend_df["LossRatio_Roll"], label="LossRatio")
        plt.title("Overall Temporal Trends")
        plt.xlabel("Transaction Month")
        plt.ylabel("Amount / Ratio")
        plt.legend()
        plt.show()


# -----------------------------
# Step 10: Correlation & Feature Relationships
# -----------------------------
def numerical_correlations(df, target_cols=None, top_n=None):
    num_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = num_df.corr()
    
    if target_cols:
        for target in target_cols:
            if target in corr_matrix.columns:
                corr_sorted = corr_matrix[target].sort_values(ascending=False)
                print(f"\nTop correlations with {target}:")
                print(corr_sorted.head(top_n) if top_n else corr_sorted)
    
    plt.figure(figsize=(12,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()
    
    return corr_matrix


def categorical_relationships(df, cat_cols, num_cols):
    for cat in cat_cols:
        if cat not in df.columns:
            continue
        for num in num_cols:
            if num not in df.columns:
                continue
            plt.figure(figsize=(10,4))
            sns.boxplot(x=cat, y=num, data=df)
            plt.xticks(rotation=45)
            plt.title(f"{num} by {cat}")
            plt.show()


# -----------------------------
# Step 11: Creative & Insightful Visualizations
# -----------------------------
def plot_loss_ratio_by_cover(df):
    cover_lr = df.groupby("CoverType").apply(lambda x: x["TotalClaims"].sum() / x["TotalPremium"].sum())
    cover_lr = cover_lr.sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    sns.barplot(x=cover_lr.values, y=cover_lr.index, palette="viridis")
    plt.xlabel("Loss Ratio")
    plt.ylabel("Cover Type")
    plt.title("Loss Ratio by Cover Type")
    plt.show()


def plot_claims_vs_premium_by_vehicle(df):
    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df, 
        x="TotalPremium", 
        y="TotalClaims", 
        hue="VehicleType", 
        size="PolicyLossRatio", 
        alpha=0.7, 
        palette="tab10"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total Premium (log scale)")
    plt.ylabel("Total Claims (log scale)")
    plt.title("Total Claims vs Total Premium by Vehicle Type")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


def plot_monthly_claims_trend(df):
    monthly = df.groupby("TransactionMonth")[["TotalClaims", "TotalPremium"]].sum().reset_index()
    plt.figure(figsize=(12,5))
    sns.lineplot(data=monthly, x="TransactionMonth", y="TotalClaims", marker="o", label="Total Claims")
    sns.lineplot(data=monthly, x="TransactionMonth", y="TotalPremium", marker="o", label="Total Premium")
    plt.title("Monthly Claims & Premium Trend")
    plt.xlabel("Transaction Month")
    plt.ylabel("Amount")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def plot_top_provinces_by_loss_ratio(df, top_n=10):
    province_lr = df.groupby("Province").apply(lambda x: x["TotalClaims"].sum() / x["TotalPremium"].sum())
    top_provinces = province_lr.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_provinces.values, y=top_provinces.index, palette="magma")
    plt.xlabel("Loss Ratio")
    plt.ylabel("Province")
    plt.title(f"Top {top_n} Provinces by Loss Ratio")
    plt.show()



