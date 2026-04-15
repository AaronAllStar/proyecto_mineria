import pytest
import pandas as pd
import numpy as np
import os
from dinosaur.preprocessor import DinoCleaner
from dinosaur.reader import Reader

def test_reader_csv():
    # Create dummy csv
    df = pd.DataFrame({'test': [1, 2, 3]})
    df.to_csv('dummy.csv', index=False)
    
    loaded = Reader.read('dummy.csv')
    assert loaded.shape == (3, 1)
    os.remove('dummy.csv')

def test_dino_cleaner_numeric_extraction():
    cleaner = DinoCleaner()
    df = pd.DataFrame({
        'Age': ['20 years', '25', '30 years'],
        'Duration': ['10 days', '5 days', 'Unknown']
    })
    
    cleaned = cleaner.transform(df)
    
    # Check column names (standardized)
    assert 'age' in cleaned.columns
    assert 'duration' in cleaned.columns
    
    # Check numeric extraction
    assert cleaned['age'].iloc[0] == 20
    assert cleaned['duration'].iloc[0] == 10
    assert np.isnan(cleaned['duration'].iloc[2]) # 'Unknown' should be NaN

def test_reader_filenotfound():
    with pytest.raises(FileNotFoundError):
        Reader.read('non_existent.csv')
