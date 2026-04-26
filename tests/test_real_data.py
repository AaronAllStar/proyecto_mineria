import pytest
import pandas as pd
import numpy as np
import os
from dinosaur.preprocessor import DinoCleaner
from dinosaur.data_loader import DataLoader
from config import Config

def test_real_dataset_loading():
    """Test loading the actual thesis dataset."""
    config = Config()
    if os.path.exists(config.DATASET_PATH):
        df = DataLoader.load(config.DATASET_PATH)
        assert not df.empty
        assert len(df) > 0
        print(f"\nReal dataset loaded with {len(df)} rows.")
    else:
        pytest.skip("Real dataset not found at the configured path.")

def test_pipeline_on_real_data():
    """Test the cleaner on the real dataset format."""
    config = Config()
    if os.path.exists(config.DATASET_PATH):
        df = DataLoader.load(config.DATASET_PATH)
        cleaner = DinoCleaner(extraction_patterns=['day', 'year', 'kg', 'cm'])
        df_clean = cleaner.transform(df)
        
        # Check if standardized columns exist
        # Based on the previous insights, we expect 'days', 'player_age', etc.
        cols = df_clean.columns
        assert 'days' in cols
        assert 'player_age' in cols
        
        # Verify numeric types
        assert pd.api.types.is_numeric_dtype(df_clean['days'])
        assert pd.api.types.is_numeric_dtype(df_clean['player_age'])
    else:
        pytest.skip("Real dataset not found.")

def test_dino_cleaner_edge_cases():
    """Test cleaner with edge case inputs."""
    cleaner = DinoCleaner(extraction_patterns=['mg', 'kg'])
    df = pd.DataFrame({
        'Complex Col name (Units)': ['100mg', '200kg', 'none'],
        'Empty': [np.nan, np.nan, np.nan]
    })
    
    cleaned = cleaner.transform(df)
    assert 'complex_col_name_(units)' in cleaned.columns
    assert cleaned['complex_col_name_(units)'].iloc[0] == 100
