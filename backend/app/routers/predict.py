from fastapi import APIRouter, HTTPException
from app.schemas.predict import Predict_Request, Predict_Response
from app.services.model import predict_match, get_teams, get_feature_importance

router = APIRouter(prefix="/predict", tags=["predict"])

@router.get("/teams")
def teams():
    return get_teams()

@router.post("/predict")
def predict(request: Predict_Request):
    teams = get_teams()
    
    if request.team1 not in teams:
        raise HTTPException(status_code=400, detail=f"{request.team1} can't find the team")
    if request.team2 not in teams:
        raise HTTPException(status_code=400, detail=f"{request.team2} can't find the team")
    if request.team1 == request.team2:
        raise HTTPException(status_code=400, detail="can't chose the same team to predict")

    return predict_match(request.team1, request.team2)

@router.get("/features")
def features():
    return get_feature_importance()