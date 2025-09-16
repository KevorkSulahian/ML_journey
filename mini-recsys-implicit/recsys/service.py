"""
FastAPI service for serving recommendations.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Movie Recommender API")

class RecommendationRequest(BaseModel):
    user_id: int
    n_items: int = 10

class RecommendationResponse(BaseModel):
    items: List[int]
    scores: List[float]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user."""
    pass