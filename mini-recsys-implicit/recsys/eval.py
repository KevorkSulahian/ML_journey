"""
Evaluation metrics for recommender system performance.
"""
from typing import List, Tuple
import numpy as np

def precision_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """Calculate precision@k."""
    pass

def recall_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """Calculate recall@k."""
    pass

def ndcg_at_k(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """Calculate normalized discounted cumulative gain (NDCG) at k."""
    pass