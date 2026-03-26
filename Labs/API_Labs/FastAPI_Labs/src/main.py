from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from src.predict import predict_recommendations
from typing import List

app = FastAPI(title="Movie Recommender API")

class MovieRequest(BaseModel):
    movie_title: str

class RecommendationResponse(BaseModel):
    recommendations: List[str]

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=RecommendationResponse)
async def get_recommendations(request: MovieRequest):
    try:
        recommendations = predict_recommendations(request.movie_title)
        return RecommendationResponse(recommendations=recommendations)
    
    except IndexError:
        raise HTTPException(
            status_code=404, 
            detail=f"Movie '{request.movie_title}' not found in database."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))