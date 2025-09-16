"""
Train and save the ALS model.
"""
from pathlib import Path
import pandas as pd
from recsys.data import load_movielens_data, create_id_mappings
from recsys.model_als import MovieRecommender
from recsys.utils import save_pickle, ensure_path_exists

def main():
    # Setup paths
    data_dir = Path(__file__).parent.parent / "data"
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    ensure_path_exists(artifacts_dir)
    
    # Load and preprocess data
    df = load_movielens_data(data_dir)
    user_map, item_map = create_id_mappings(df)
    
    # Save mappings
    save_pickle(user_map, artifacts_dir / "user_map.pkl")
    save_pickle(item_map, artifacts_dir / "item_map.pkl")
    
    # Train model
    model = MovieRecommender(factors=100)
    # TODO: Create user-item matrix and train model
    
    # Save model
    model.save_model(artifacts_dir / "model.pkl")
    
    print("Model training complete!")

if __name__ == "__main__":
    main()