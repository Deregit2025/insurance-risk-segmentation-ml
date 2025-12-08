# src/hypothesis_testing/metrics.py

import pandas as pd
import numpy as np


def calculate_claim_frequency(df: pd.DataFrame) -> float:
    """
    Claim Frequency = proportion of policies with at least one claim.
    """
    return (df["TotalClaims"] > 0).mean()


def calculate_claim_severity(df: pd.DataFrame) -> float:
    """
    Claim Severity = average claim amount, given a claim occurred.
    """
    claims = df[df["TotalClaims"] > 0]
    if len(claims) == 0:
        return 0.0
    return claims["TotalClaims"].mean()


def calculate_margin(df: pd.DataFrame) -> float:
    """
    Margin = TotalPremium - TotalClaims (mean value)
    """
    return (df["TotalPremium"] - df["TotalClaims"]).mean()


def calculate_loss_ratio(df: pd.DataFrame) -> float:
    """
    Loss Ratio = TotalClaims / TotalPremium (mean value)
    """
    if (df["TotalPremium"] == 0).any():
        return np.nan
    return (df["TotalClaims"] / df["TotalPremium"]).mean()
