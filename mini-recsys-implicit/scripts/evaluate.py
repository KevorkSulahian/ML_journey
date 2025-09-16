"""
Evaluate model performance.
"""
from pathlib import Path
import pandas as pd
from recsys.data import load_movielens_data, train_test_split
from recsys.model_als import MovieRecommender
from recsys.eval import precision_at_k, recall_at_k, ndcg_at_k

def main():
    # Load model and data
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load data and split into train/test
    df = load_movielens_data(data_dir)
    train_df, test_df = train_test_split(df)
    
    # Load model
    model = MovieRecommender()
    model.load_model(artifacts_dir / "model.pkl")
    
    # Generate predictions and evaluate
    metrics = {
        'precision@10': [],
        'recall@10': [],
        'ndcg@10': []
    }
    
    # TODO: Generate predictions and calculate metrics
    
    # Print results
    for metric, values in metrics.items():
        print(f"{metric}: {sum(values)/len(values):.4f}")

if __name__ == "__main__":
    main()