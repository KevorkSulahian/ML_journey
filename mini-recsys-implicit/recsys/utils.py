"""
Utility functions for the recommender system.
"""
from pathlib import Path
import pickle
from typing import Any

def save_pickle(obj: Any, path: Path) -> None:
    """Save object to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: Path) -> Any:
    """Load object from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def ensure_path_exists(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path