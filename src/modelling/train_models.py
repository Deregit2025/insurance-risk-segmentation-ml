# src/modeling/train_models.py

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor.
    """
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
    """
    Train an XGBoost Regressor.
    """
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_all_models(X_train, y_train):
    """
    Train Linear Regression, Random Forest, and XGBoost models.
    Return a dictionary of trained models.
    """
    models = {}
    
    print("Training Linear Regression...")
    models['LinearRegression'] = train_linear_regression(X_train, y_train)
    
    print("Training Random Forest Regressor...")
    models['RandomForest'] = train_random_forest(X_train, y_train)
    
    print("Training XGBoost Regressor...")
    models['XGBoost'] = train_xgboost(X_train, y_train)
    
    return models
