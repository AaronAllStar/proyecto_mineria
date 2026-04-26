import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from injury_predictor import InjuryPredictor

def test_injury_predictor_inference():
    predictor = InjuryPredictor()
    # Mocking trained models
    predictor._is_trained = True
    predictor.regressor = MagicMock()
    predictor.regressor.predict.return_value = [10.0]
    
    predictor.classifier = MagicMock()
    predictor.classifier.predict.return_value = [0]
    predictor.classifier.predict_proba.return_value = [[0.8, 0.1, 0.05, 0.05]]
    
    # Mock clean and feature engineer outputs
    predictor.cleaner.transform = MagicMock(return_value=pd.DataFrame([{"days": 10.0, "player_age": 25}]))
    predictor.feature_engineer.transform = MagicMock(return_value=pd.DataFrame([{"days": 10.0, "player_age": 25, "composite_risk_index": 2.5}]))

    sample = {
        "player_age": 29,
        "player_position": "Central Midfield",
        "injury": "Hamstring injury",
    }
    
    result = predictor.predict_player(sample)
    assert result["predicted_days"] == 10.0
    assert result["risk_category"] == "Low"
    assert "Low" in result["risk_probabilities"]
    assert result["composite_risk_index"] == 2.5
