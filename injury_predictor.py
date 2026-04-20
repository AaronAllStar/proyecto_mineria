"""
injury_predictor.py
===================
Senior AI Engineer: ML Pipeline — Random Forest + XGBoost Ensemble
con clasificación de riesgo, calibración de probabilidades y
preparación de tensores para la capa Deep Learning.

Outputs:
  - models/ml_ensemble.joblib
  - models/risk_classifier.joblib
  - reports/ml_evaluation.json
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    classification_report,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Try xgboost (optional but recommended)
# ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost not installed. Falling back to GradientBoosting.")

from dinosaur.preprocessor import DinoCleaner
from dinosaur.feature_engineer import InjuryFeatureEngineer
from config import Config


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "low": 7,       # < 7 days → Low risk
    "medium": 21,   # 7-21 days → Medium risk
    "high": 45,     # 21-45 days → High risk
                    # > 45 days → Critical risk
}

RISK_LABELS = ["Low", "Medium", "High", "Critical"]


# ─────────────────────────────────────────────────────────
# Helper: days → risk label
# ─────────────────────────────────────────────────────────
def days_to_risk_category(days: float) -> str:
    if days < SEVERITY_THRESHOLDS["low"]:
        return "Low"
    elif days < SEVERITY_THRESHOLDS["medium"]:
        return "Medium"
    elif days < SEVERITY_THRESHOLDS["high"]:
        return "High"
    else:
        return "Critical"


def days_to_risk_index(days: float) -> int:
    return RISK_LABELS.index(days_to_risk_category(days))


# ─────────────────────────────────────────────────────────
# Preprocessing column spec
# ─────────────────────────────────────────────────────────
NUMERIC_FEATURES   = ["player_age", "age_risk_score", "injury_risk_score",
                       "position_load", "season_phase", "composite_risk_index"]
CATEGORICAL_FEATURES = ["player_position", "injury_severity", "age_risk_group"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ─────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_regression_pipeline() -> Pipeline:
    """Ensemble regressor: RF + XGB/GB + Ridge."""
    regressors = [
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
        ("ridge", Ridge(alpha=10.0)),
    ]
    if HAS_XGB:
        regressors.append(
            ("xgb", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0))
        )
    else:
        regressors.append(
            ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                              learning_rate=0.05, random_state=42))
        )

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", VotingRegressor(estimators=regressors)),
    ])


def build_classifier_pipeline() -> Pipeline:
    """Risk category classifier (Low/Medium/High/Critical) with probability calibration."""
    if HAS_XGB:
        base_clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42, verbosity=0
        )
    else:
        base_clf = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        )

    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", CalibratedClassifierCV(base_clf, method="isotonic", cv=3)),
    ])


# ─────────────────────────────────────────────────────────
# Main InjuryPredictor class
# ─────────────────────────────────────────────────────────
class InjuryPredictor:
    """
    Production-grade injury predictor.

    Provides:
      1. Regression: predict days of recovery
      2. Classification: predict risk category (Low/Medium/High/Critical)
      3. Probability output: vector of probabilities per risk class
      4. DL-ready export: numpy tensors for downstream training
    """

    def __init__(self):
        self.config = Config()
        self.cleaner = DinoCleaner()
        self.feature_engineer = InjuryFeatureEngineer()
        self.regressor: Pipeline | None = None
        self.classifier: Pipeline | None = None
        self.label_encoder = LabelEncoder()
        self._is_trained = False
        self.metrics: dict = {}

    # --------------------------------------------------
    # Data Loading & Preprocessing
    # --------------------------------------------------
    def load_data(self, path: str | None = None) -> pd.DataFrame:
        path = path or self.config.DATASET_PATH
        df = pd.read_csv(path)
        df = self.cleaner.transform(df)
        print(f"[InjuryPredictor] Loaded {df.shape[0]} records × {df.shape[1]} columns")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Returns (X, y_days, y_risk) ready for training or inference.
        """
        df = df.dropna(subset=["days", "player_age"]).copy()
        df = self.feature_engineer.transform(df)

        y_days = df["days"].values.astype(float)
        y_risk = np.array([days_to_risk_index(d) for d in y_days])

        # Keep only needed features
        available = [c for c in ALL_FEATURES if c in df.columns]
        X = df[available].copy()

        # Fill remaining NaN
        for col in NUMERIC_FEATURES:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna("Unknown")

        return X, y_days, y_risk

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def train(self, path: str | None = None) -> dict:
        print("\n[InjuryPredictor] ═══ TRAINING PHASE ═══")
        df = self.load_data(path)
        X, y_days, y_risk = self.prepare_features(df)

        X_train, X_test, yd_train, yd_test, yr_train, yr_test = train_test_split(
            X, y_days, y_risk, test_size=0.2, random_state=42, stratify=y_risk
        )

        # ---- Regression ----
        print("  Training ensemble regressor ...")
        self.regressor = build_regression_pipeline()
        self.regressor.fit(X_train, yd_train)

        yd_pred = self.regressor.predict(X_test)
        reg_metrics = {
            "MAE": float(mean_absolute_error(yd_test, yd_pred)),
            "RMSE": float(root_mean_squared_error(yd_test, yd_pred)),
            "R2": float(r2_score(yd_test, yd_pred)),
        }
        print(f"  Regressor  →  MAE={reg_metrics['MAE']:.2f}d | RMSE={reg_metrics['RMSE']:.2f}d | R²={reg_metrics['R2']:.4f}")

        # ---- Classification ----
        print("  Training risk classifier ...")
        self.classifier = build_classifier_pipeline()
        self.classifier.fit(X_train, yr_train)

        yr_pred = self.classifier.predict(X_test)
        yr_proba = self.classifier.predict_proba(X_test)

        clf_report = classification_report(yr_test, yr_pred,
                                           target_names=RISK_LABELS, output_dict=True)
        try:
            auc = float(roc_auc_score(yr_test, yr_proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = None

        clf_metrics = {
            "accuracy": float(clf_report["accuracy"]),
            "macro_f1": float(clf_report["macro avg"]["f1-score"]),
            "auc_ovr": auc,
            "per_class": {k: clf_report[k] for k in RISK_LABELS if k in clf_report},
        }
        auc_str = f"{auc:.4f}" if auc else "N/A"
        print(f"  Classifier ->  Accuracy={clf_metrics['accuracy']:.4f} | F1={clf_metrics['macro_f1']:.4f} | AUC={auc_str}")

        self.metrics = {
            "regression": reg_metrics,
            "classification": clf_metrics,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "trained_at": datetime.now().isoformat(),
            "xgboost_enabled": HAS_XGB,
        }
        self._is_trained = True

        # Save models & metrics
        self._save(X_test, yd_test, yr_test, yd_pred, yr_pred, yr_proba, X)
        return self.metrics

    # --------------------------------------------------
    # Inference (single player)
    # --------------------------------------------------
    def predict_player(self, player_data: dict) -> dict:
        """
        Given a dict with player features, return:
          - predicted_days (float)
          - risk_category (str)
          - risk_probabilities (dict)
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() or load()")

        df_input = pd.DataFrame([player_data])
        df_input = self.cleaner.transform(df_input) if "days" not in df_input.columns else df_input
        df_input = self.feature_engineer.transform(df_input)

        available = [c for c in ALL_FEATURES if c in df_input.columns]
        X = df_input[available].copy()

        for col in NUMERIC_FEATURES:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna("Unknown")

        days_pred = float(self.regressor.predict(X)[0])
        risk_idx = int(self.classifier.predict(X)[0])
        risk_proba = self.classifier.predict_proba(X)[0]

        return {
            "predicted_days": round(max(0, days_pred), 1),
            "risk_category": RISK_LABELS[risk_idx],
            "risk_probabilities": {
                label: round(float(prob), 4)
                for label, prob in zip(RISK_LABELS, risk_proba)
            },
            "composite_risk_index": float(df_input.get("composite_risk_index", pd.Series([0]))[0])
            if "composite_risk_index" in df_input.columns else None,
        }

    # --------------------------------------------------
    # DL-ready tensor export
    # --------------------------------------------------
    def export_dl_tensors(self, path: str | None = None) -> dict:
        """
        Export preprocessed numpy arrays ready for Deep Learning training.
        Saves to models/dl_tensors.npz
        """
        df = self.load_data(path)
        X, y_days, y_risk = self.prepare_features(df)

        # Use the fitted preprocessor from regressor pipeline
        preprocessor = self.regressor.named_steps["preprocessor"]
        X_tensor = preprocessor.transform(X).astype(np.float32)
        y_days_tensor = y_days.astype(np.float32)
        y_risk_tensor = y_risk.astype(np.int64)

        tensor_path = os.path.join(os.path.dirname(self.config.MODEL_PATH), "dl_tensors.npz")
        np.savez(tensor_path,
                 X=X_tensor,
                 y_days=y_days_tensor,
                 y_risk=y_risk_tensor)

        print(f"[InjuryPredictor] DL tensors exported → {tensor_path}")
        print(f"  X shape: {X_tensor.shape} | y_days: {y_days_tensor.shape} | y_risk: {y_risk_tensor.shape}")

        return {
            "path": tensor_path,
            "X_shape": list(X_tensor.shape),
            "input_features": X_tensor.shape[1],
            "n_classes": len(RISK_LABELS),
        }

    # --------------------------------------------------
    # Save / Load
    # --------------------------------------------------
    def _save(self, X_test, yd_test, yr_test, yd_pred, yr_pred, yr_proba, X_full):
        os.makedirs(os.path.dirname(self.config.MODEL_PATH), exist_ok=True)
        joblib.dump(self.regressor, self.config.MODEL_PATH.replace("dino_model", "ml_regressor"))
        joblib.dump(self.classifier, self.config.MODEL_PATH.replace("dino_model", "ml_classifier"))

        metrics_path = os.path.join(self.config.REPORT_OUTPUT, "ml_evaluation.json")
        os.makedirs(self.config.REPORT_OUTPUT, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"  Models & metrics saved → {self.config.MODEL_PATH.replace('dino_model.joblib','ml_*.joblib')}")

    def load(self):
        reg_path = self.config.MODEL_PATH.replace("dino_model", "ml_regressor")
        clf_path = self.config.MODEL_PATH.replace("dino_model", "ml_classifier")
        if os.path.exists(reg_path) and os.path.exists(clf_path):
            self.regressor = joblib.load(reg_path)
            self.classifier = joblib.load(clf_path)
            self._is_trained = True
            print("[InjuryPredictor] Models loaded from disk.")
        else:
            raise FileNotFoundError("Trained models not found. Run train() first.")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = InjuryPredictor()
    metrics = predictor.train()

    print("\n[InjuryPredictor] ═══ SAMPLE PREDICTION ═══")
    sample = {
        "player_age": 29,
        "player_position": "Central Midfield",
        "injury": "Hamstring injury",
        "season": "23/24",
        "club": "Real Madrid",
        "league": "LaLiga",
    }
    result = predictor.predict_player(sample)
    print(f"  Player age 29 | Position: Central Midfield | Injury: Hamstring")
    print(f"  → Predicted recovery: {result['predicted_days']} days")
    print(f"  → Risk category: {result['risk_category']}")
    print(f"  → Probabilities: {result['risk_probabilities']}")

    print("\n[InjuryPredictor] Exporting DL tensors...")
    tensor_info = predictor.export_dl_tensors()
    print(f"  → Input dimension for DL: {tensor_info['input_features']}")
    print(f"  → Classes: {tensor_info['n_classes']}")
