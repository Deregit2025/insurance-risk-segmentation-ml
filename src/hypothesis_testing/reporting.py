# src/hypothesis_testing/reporting.py

def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    Interprets the p-value from a statistical test.
    
    Args:
        p_value (float): p-value from a test
        alpha (float): significance level (default 0.05)
    
    Returns:
        str: Interpretation message
    """
    if p_value < alpha:
        return f"Reject the null hypothesis (p = {p_value:.4f}). Significant difference detected."
    else:
        return f"Fail to reject the null hypothesis (p = {p_value:.4f}). No significant difference detected."
