import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from dinosaur.preprocessor import DinoCleaner
from config import Config

class InjuryPipeline:
    """
    State-of-the-Art ML Pipeline for Injury Prediction.
    Designed by Senior AI Engineer.
    """
    def __init__(self):
        self.config = Config()
        self.model = None
        self.pipeline = None

    def build_pipeline(self):
        """Builds a robust sklearn pipeline with proper encoding and scaling."""
        
        # Define feature types
        categorical_features = ['player_position', 'club', 'injury']
        numeric_features = ['player_age']

        # Preprocessing for numerical data
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data
        # handle_unknown='ignore' is crucial for production
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Define the full pipeline
        self.pipeline = Pipeline(steps=[
            ('cleaner', DinoCleaner()),
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])

        return self.pipeline

    def run(self):
        print(f"[AI Pipeline] Initializing with dataset: {self.config.DATASET_PATH}")
        
        # Load raw data
        try:
            df = pd.read_csv(self.config.DATASET_PATH)
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load dataset. {e}")
            return

        # Pre-cleaning to get the target (days)
        # Note: DinoCleaner is inside the pipeline, but we need 'days' for 'y'
        cleaner = DinoCleaner()
        df_clean = cleaner.transform(df)
        
        # Ensure we have our target and features
        if 'days' not in df_clean.columns:
            print("ERROR: Target column 'days' not found after initial cleaning.")
            return
            
        # Handle NAs in target
        df_clean = df_clean.dropna(subset=['days'])
        
        X = df_clean.drop(columns=['days'])
        y = df_clean['days']

        # Split - Proper methodology
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train
        print("Training Random Forest model...")
        self.build_pipeline()
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n--- Model Evaluation ---")
        print(f"MAE: {mae:.2f} days")
        print(f"R2 : {r2:.4f}")

        # Save model
        joblib.dump(self.pipeline, self.config.MODEL_PATH)
        print(f"DONE: Model saved to {self.config.MODEL_PATH}")
        
        return {
            "mae": mae,
            "r2": r2,
            "samples": len(df_clean)
        }

if __name__ == "__main__":
    ai_pipeline = InjuryPipeline()
    results = ai_pipeline.run()
