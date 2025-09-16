"""
Run the FastAPI service.
"""
import uvicorn
from pathlib import Path
from recsys.service import app
from recsys.model_als import MovieRecommender
from recsys.utils import load_pickle

def load_artifacts():
    """Load model and mappings."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    
    model = MovieRecommender()
    model.load_model(artifacts_dir / "model.pkl")
    
    user_map = load_pickle(artifacts_dir / "user_map.pkl")
    item_map = load_pickle(artifacts_dir / "item_map.pkl")
    
    return model, user_map, item_map

def main():
    # Load model and mappings
    model, user_map, item_map = load_artifacts()
    
    # Add to app state
    app.state.model = model
    app.state.user_map = user_map
    app.state.item_map = item_map
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()