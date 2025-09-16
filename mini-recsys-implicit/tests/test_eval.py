"""
Tests for evaluation metrics.
"""
import pytest
from recsys.eval import precision_at_k, recall_at_k, ndcg_at_k

def test_precision_at_k():
    actual = [1, 2, 3, 4, 5]
    predicted = [1, 3, 2, 6, 7]
    
    assert precision_at_k(actual, predicted, k=3) == pytest.approx(1.0)
    assert precision_at_k(actual, predicted, k=5) == pytest.approx(0.6)

def test_recall_at_k():
    actual = [1, 2, 3, 4, 5]
    predicted = [1, 3, 6, 7, 8]
    
    assert recall_at_k(actual, predicted, k=2) == pytest.approx(0.4)
    assert recall_at_k(actual, predicted, k=5) == pytest.approx(0.4)

def test_ndcg_at_k():
    actual = [1, 2, 3]
    predicted = [1, 3, 2]
    
    assert ndcg_at_k(actual, predicted, k=3) > 0.9  # High but not perfect due to order