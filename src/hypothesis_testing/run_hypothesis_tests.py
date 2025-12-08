# src/hypothesis_testing/run_hypothesis_tests.py

import os
import pandas as pd
import warnings

from .metrics import (
    calculate_claim_frequency,
    calculate_claim_severity,
    calculate_margin
)
from .segmentation import segment_two_groups, binary_split, clean_gender
from .statistical_tests import t_test_numeric, proportion_test
from .reporting import interpret_p_value


def load_data(path: str) -> pd.DataFrame:
    # read with low_memory=False to avoid mixed-type warnings
    return pd.read_csv(path, low_memory=False)


def choose_province_pair(df: pd.DataFrame, min_count: int = 50):
    # compute loss ratio per province and choose highest vs lowest (subject to sample size)
    agg = df.groupby("Province").agg(
        count=("PolicyID", "count"),
        loss_ratio=("PolicyLossRatio", "mean")
    ).dropna(subset=["loss_ratio"])
    if agg.empty:
        raise RuntimeError("No province data available to choose from.")
    # filter by min_count
    eligible = agg[agg["count"] >= min_count]
    if eligible.shape[0] < 2:
        # relax threshold if too few provinces
        eligible = agg.sort_values("count", ascending=False).head(2)
    highest = eligible["loss_ratio"].idxmax()
    lowest = eligible["loss_ratio"].idxmin()
    return highest, lowest


def choose_postalcode_pair(df: pd.DataFrame, min_count: int = 50):
    # choose postal codes with extremes in policy loss ratio but ensure sample size
    agg = df.groupby("PostalCode").agg(count=("PolicyID", "count"),
                                      loss_ratio=("PolicyLossRatio", "mean")).dropna()
    # keep postal codes with at least min_count
    eligible = agg[agg["count"] >= min_count]
    if eligible.shape[0] >= 2:
        highest = eligible["loss_ratio"].idxmax()
        lowest = eligible["loss_ratio"].idxmin()
        return highest, lowest
    # fallback: use two most common postal codes
    top2 = agg.sort_values("count", ascending=False).head(2).index.tolist()
    if len(top2) < 2:
        raise RuntimeError("Not enough PostalCode groups with sufficient data.")
    return top2[0], top2[1]


def test_gender_risk(df: pd.DataFrame):
    print("\n=== Hypothesis: No risk difference between genders ===")
    # clean and drop missing genders
    df_clean = clean_gender(df, column="Gender")
    df_clean = df_clean.dropna(subset=["Gender"])
    try:
        group_a, group_b = binary_split(df_clean, "Gender")
    except ValueError as e:
        print("Skipping gender test:", e)
        return

    # Claim frequency (proportion test)
    z_stat, p_value = proportion_test(group_a, group_b, target="TotalClaims")
    print("Claim Frequency:", interpret_p_value(p_value))

    # Claim severity (t-test) on only policies with claims
    sev_a = group_a[group_a["TotalClaims"] > 0]["TotalClaims"]
    sev_b = group_b[group_b["TotalClaims"] > 0]["TotalClaims"]
    if len(sev_a) < 5 or len(sev_b) < 5:
        print("Claim Severity: insufficient sample size to run t-test.")
    else:
        t_stat, p_value = t_test_numeric(sev_a, sev_b)
        print("Claim Severity:", interpret_p_value(p_value))


def test_province_risk(df: pd.DataFrame):
    print("\n=== Hypothesis: No risk differences across provinces ===")
    try:
        prov_a, prov_b = choose_province_pair(df, min_count=50)
    except RuntimeError as e:
        print("Skipping province test:", e)
        return

    print(f"Comparing provinces: {prov_a} (high loss ratio) vs {prov_b} (low loss ratio)")
    group_a, group_b = segment_two_groups(df, "Province", prov_a, prov_b)

    # Claim Frequency
    z_stat, p_value = proportion_test(group_a, group_b, target="TotalClaims")
    print("Claim Frequency:", interpret_p_value(p_value))

    # Claim Severity (only on records with claims)
    sev_a = group_a[group_a["TotalClaims"] > 0]["TotalClaims"]
    sev_b = group_b[group_b["TotalClaims"] > 0]["TotalClaims"]
    if len(sev_a) < 5 or len(sev_b) < 5:
        print("Claim Severity: insufficient sample size to run t-test.")
    else:
        t_stat, p_value = t_test_numeric(sev_a, sev_b)
        print("Claim Severity:", interpret_p_value(p_value))


def test_postalcode_risk(df: pd.DataFrame):
    print("\n=== Hypothesis: No risk differences between postal codes ===")
    try:
        pc_a, pc_b = choose_postalcode_pair(df, min_count=50)
    except RuntimeError as e:
        print("Skipping postal code risk test:", e)
        return

    print(f"Comparing postal codes: {pc_a} vs {pc_b}")
    group_a, group_b = segment_two_groups(df, "PostalCode", pc_a, pc_b)

    # Claim Frequency (proportion)
    z_stat, p_value = proportion_test(group_a, group_b, target="TotalClaims")
    print("Claim Frequency:", interpret_p_value(p_value))

    # Claim Severity
    sev_a = group_a[group_a["TotalClaims"] > 0]["TotalClaims"]
    sev_b = group_b[group_b["TotalClaims"] > 0]["TotalClaims"]
    if len(sev_a) < 5 or len(sev_b) < 5:
        print("Claim Severity: insufficient sample size to run t-test.")
    else:
        t_stat, p_value = t_test_numeric(sev_a, sev_b)
        print("Claim Severity:", interpret_p_value(p_value))


def test_margin_zipcodes(df: pd.DataFrame):
    print("\n=== Hypothesis: No significant margin difference between postal codes ===")
    # create margin column
    df = df.copy()
    df["margin"] = df["TotalPremium"] - df["TotalClaims"]

    try:
        pc_a, pc_b = choose_postalcode_pair(df, min_count=50)
    except RuntimeError as e:
        print("Skipping postal code margin test:", e)
        return

    print(f"Comparing postal codes (margin): {pc_a} vs {pc_b}")
    group_a, group_b = segment_two_groups(df, "PostalCode", pc_a, pc_b)

    # margin t-test
    m_a = group_a["margin"].dropna()
    m_b = group_b["margin"].dropna()
    if len(m_a) < 5 or len(m_b) < 5:
        print("Margin test: insufficient sample size to run t-test.")
    else:
        t_stat, p_value = t_test_numeric(m_a, m_b)
        print("Margin (mean difference):", interpret_p_value(p_value))


def main():
    # detect project root and build path (works when running from repo root)
    project_root = os.getcwd()
    # climb up if notebook location; prefer presence of data folder
    while "data" not in os.listdir(project_root):
        project_root = os.path.dirname(project_root)
        if project_root == "" or project_root == os.path.sep:
            raise FileNotFoundError("Project root with 'data' folder not found.")
    data_path = os.path.join(project_root, "data", "processed", "insurance_data_cleaned.csv")

    df = load_data(data_path)
    print("Dataset loaded. Running hypothesis tests...")

    # run tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_gender_risk(df)
        test_province_risk(df)
        test_postalcode_risk(df)
        test_margin_zipcodes(df)


if __name__ == "__main__":
    main()
