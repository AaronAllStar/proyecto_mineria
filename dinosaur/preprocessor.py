import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DinoCleaner(BaseEstimator, TransformerMixin):
    """
    Automated data cleaner for the Dinosaur library.
    Handles numeric extraction from strings and basic missing value imputation.
    """
    def __init__(self, target_cols=None):
        self.target_cols = target_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Standardize column names (lowercase, no spaces)
        X.columns = [col.strip().replace(' ', '_').lower() for col in X.columns]
        
        # 2. Smart numeric extraction
        # Example: '43 days' -> 43
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to extract numbers from strings if they look like quantities
                if X[col].str.contains('day|year|kg|cm', case=False, na=False).any():
                    X[col] = X[col].str.extract(r'(\d+)').astype(float)
        
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Professional feature engineering for any dataset.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Add basic time features if dates exist (placeholder for future expansion)
        # For this specific dataset, we might want to group ages into bins
        if 'player_age' in X.columns:
            X['age_group'] = pd.cut(X['player_age'], 
                                   bins=[0, 20, 25, 30, 35, 100], 
                                   labels=['Junior', 'Prime', 'Experienced', 'Veteran', 'Legend'])
        
        return X
