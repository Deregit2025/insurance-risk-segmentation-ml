# src/modeling/evaluate_models.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate regression model using RMSE and R-squared.
    Returns a dictionary with metrics.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {'RMSE': rmse, 'R2': r2}

def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate multiple regression models and return a summary DataFrame.
    """
    results = []
    for name, model in models.items():
        metrics = evaluate_regression_model(model, X_test, y_test)
        metrics['Model'] = name
        results.append(metrics)
    results_df = pd.DataFrame(results).set_index('Model')
    return results_df

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance for tree-based models (Random Forest / XGBoost).
    Returns a DataFrame sorted by importance descending.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)
        return fi_df
    else:
        raise ValueError("Model does not have feature_importances_ attribute")

def shap_summary_plot(model, X_train, max_display=10):
    """
    Generate SHAP summary plot for the model.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    shap.summary_plot(shap_values, X_train, max_display=max_display, show=True)

def save_shap_summary_plot(model, X_train, file_path: str, max_display=10):
    """
    Save SHAP summary plot as an image file.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    plt.figure()
    shap.summary_plot(shap_values, X_train, max_display=max_display, show=False)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
