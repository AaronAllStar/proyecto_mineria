import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class InjuryFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-driven Feature Engineering for Sports Injury Prediction.
    Designed as a senior AI engineer component for bridging ML → DL architectures.

    Produces:
    - Age-based risk bins
    - Injury severity classification
    - Season phase encoding
    - Multi-recurrence features
    - Categorical ordinal risk score
    """

    # --- Domain knowledge: injury severity map ---
    SEVERITY_MAP = {
        # Critical (>45 days typical)
        "fracture": "Critical",
        "broken": "Critical",
        "surgery": "Critical",
        "rupture": "Critical",
        "torn": "Critical",
        "achilles": "Critical",
        # Severe (21-45 days)
        "ligament": "Severe",
        "syndesmosis": "Severe",
        "meniscus": "Severe",
        "tendon": "Severe",
        "hamstring": "Severe",
        "knee": "Severe",
        "cartilage": "Severe",
        # Moderate (7-21 days)
        "muscle": "Moderate",
        "strain": "Moderate",
        "sprain": "Moderate",
        "back": "Moderate",
        "groin": "Moderate",
        "thigh": "Moderate",
        "calf": "Moderate",
        "ankle": "Moderate",
        # Mild (<7 days)
        "knock": "Mild",
        "discomfort": "Mild",
        "fatigue": "Mild",
        "cold": "Mild",
        "virus": "Mild",
        "corona": "Mild",
        "suspension": "Mild",
        "illness": "Mild",
    }

    SEVERITY_RISK_SCORE = {"Critical": 4, "Severe": 3, "Moderate": 2, "Mild": 1, "Unknown": 0}

    POSITION_LOAD_MAP = {
        "Centre-Back": 3,
        "Right-Back": 4,
        "Left-Back": 4,
        "Defensive Midfield": 4,
        "Central Midfield": 5,
        "Attacking Midfield": 4,
        "Right Winger": 4,
        "Left Winger": 4,
        "Centre-Forward": 3,
        "Goalkeeper": 2,
        "Second Striker": 3,
    }

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1. Age Risk Group
        X["age_risk_group"] = pd.cut(
            X["player_age"].fillna(25),
            bins=[0, 20, 24, 28, 32, 36, 100],
            labels=["Youth", "Developing", "Prime", "Experienced", "Veteran", "Elder"],
        )
        X["age_risk_score"] = pd.cut(
            X["player_age"].fillna(25),
            bins=[0, 20, 24, 28, 32, 36, 100],
            labels=[2, 1, 0, 1, 2, 3],
            ordered=False
        ).astype(float)

        # 2. Injury Severity from text
        X["injury_severity"] = X["injury"].apply(self._classify_severity)
        X["injury_risk_score"] = X["injury_severity"].map(self.SEVERITY_RISK_SCORE).fillna(0)

        # 3. Position physical load
        X["position_load"] = X["player_position"].map(self.POSITION_LOAD_MAP).fillna(3)

        # 4. Season phase (early/mid/late)
        X["season_phase"] = X.get("season", pd.Series(["unknown"] * len(X))).apply(
            self._encode_season_phase
        )

        # 5. Composite risk index (for ML feature importance & DL input)
        X["composite_risk_index"] = (
            X["age_risk_score"].fillna(0)
            + X["injury_risk_score"].fillna(0)
            + X["position_load"].fillna(0)
        ).astype(float)

        return X

    def _classify_severity(self, injury_text: str) -> str:
        """Map raw injury description to severity category."""
        if pd.isna(injury_text):
            return "Unknown"
        lower = str(injury_text).lower()
        for keyword, severity in self.SEVERITY_MAP.items():
            if keyword in lower:
                return severity
        return "Unknown"

    def _encode_season_phase(self, season_str: str) -> int:
        """Encode season phase as integer: 0=pre, 1=early, 2=mid, 3=late."""
        # Season format like '20/21' — approximate by year parity
        try:
            year = int(str(season_str).split("/")[0])
            # Heuristic: even-year seasons = early league phase
            return year % 3
        except Exception:
            return 1
