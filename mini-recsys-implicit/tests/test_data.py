"""
Tests for data loading and preprocessing functions.
"""
import pytest
import pandas as pd
from pathlib import Path
from recsys.data import load_movielens_data, create_id_mappings, train_test_split

def test_load_movielens_data(tmp_path):
    """Test data loading."""
    # TODO: Create test data and verify loading
    pass

def test_create_id_mappings():
    """Test ID mapping creation."""
    df = pd.DataFrame({
        'userId': [1, 2, 1, 3],
        'movieId': [101, 102, 103, 101]
    })
    
    user_map, item_map = create_id_mappings(df)
    
    assert len(user_map) == 3  # Unique users
    assert len(item_map) == 3  # Unique items

def test_train_test_split():
    """Test data splitting."""
    df = pd.DataFrame({
        'userId': range(100),
        'movieId': range(100),
        'rating': [1] * 100
    })
    
    train, test = train_test_split(df, test_size=0.2)
    
    assert len(train) + len(test) == len(df)
    assert abs(len(test) / len(df) - 0.2) < 0.05  # Allow small deviation from exact split