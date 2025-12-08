# src/hypothesis_testing/statistical_tests.py

import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest



def t_test_numeric(group_a: pd.Series, group_b: pd.Series):
    """
    Two-sample independent t-test for numerical variables.
    """
    return ttest_ind(group_a.dropna(), group_b.dropna(), equal_var=False)


def chi_square_test(df: pd.DataFrame, column: str, target: str):
    """
    Chi-square test for independence between categorical column and binary target.
    """
    contingency_table = pd.crosstab(df[column], df[target])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    return chi2, p_value


def proportion_test(group_a: pd.DataFrame, group_b: pd.DataFrame, target: str):
    """
    Test difference in proportions.
    Example: Claim Frequency between two groups.
    """
    count = [
        (group_a[target] > 0).sum(),
        (group_b[target] > 0).sum()
    ]
    nobs = [len(group_a), len(group_b)]

    z_stat, p_value = proportions_ztest(count, nobs)
    return z_stat, p_value
