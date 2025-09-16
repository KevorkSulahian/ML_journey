"""
Data loading, preprocessing, and ID mapping utilities for the recommender system.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Iterable
from sklearn.model_selection import train_test_split as sk_train_test_split

def load_movielens_data(data_path: Path) -> pd.DataFrame:
    """Load MovieLens dataset."""

    data_file = data_path / "ml-100k" / "u1.base"
    column_names = ['userId', 'movieId', 'rating', 'timestamp']
    df = pd.read_csv(data_file, sep='\t', names=column_names)
    return df

def create_id_mappings(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Create user and item ID mappings."""
    user_map = {id: i for i, id in enumerate(df['userId'].unique())}
    item_map = {id: i for i, id in enumerate(df['movieId'].unique())}
    return user_map, item_map

def train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    train, test = sk_train_test_split(df, test_size=test_size, random_state=42)
    return train, test

def popular_items(
    df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Return a (movieId, count) frame for the top-N most interacted items."""
    counts = df["movieId"].value_counts()
    top = counts.head(top_n).rename_axis("movieId").reset_index(name="count")
    return top


def interactions_for_items(df: pd.DataFrame, items: Iterable[int]) -> pd.DataFrame:
    """Filter the interaction rows for the given item IDs."""
    items = set(items)
    return df[df["movieId"].isin(items)].copy()


def high_rating_items(
    df: pd.DataFrame,
    min_rating: float = 4.0,
    min_support: int = 5,
    sort_by: str = "mean_rating",
    ascending: bool = False,
) -> pd.DataFrame:
    """Aggregate to items and keep those with mean rating >= threshold and min_support.

    Returns columns: [movieId, mean_rating, n_ratings]
    """
    agg = (
        df.groupby("movieId")["rating"]
        .agg(mean_rating="mean", n_ratings="count")
        .reset_index()
    )
    out = agg[(agg["mean_rating"] >= min_rating) & (agg["n_ratings"] >= min_support)]
    return out.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)


def main():
    data_dir = Path(__file__).parent.parent / "data"
    df = load_movielens_data(data_dir)
    train, test = train_test_split(df)
    
    user_map, item_map = create_id_mappings(train)


    # Popularity baseline
    top_pop = popular_items(train)
    # If you want the interaction rows for these top items:
    pop_rows = interactions_for_items(train, top_pop["movieId"])

    # High-rated items with support
    top_rated = high_rating_items(train, min_rating=4.0, min_support=10)

    print(top_pop.head())
    print(top_rated.head())
    print(len(user_map), len(item_map))

if __name__ == "__main__":
    main()