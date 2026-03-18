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
    
    if request.blue_team not in teams:
        raise HTTPException(status_code=400, detail=f"{request.blue_team} can't find the team")
    if request.red_team not in teams:
        raise HTTPException(status_code=400, detail=f"{request.red_team} can't find the team")
    if request.blue_team == request.red_team:
        raise HTTPException(status_code=400, detail="can't chose the same team to predict")
    
    return predict_match(request.blue_team, request.red_team)

@router.get("/features")
def features():
    return get_feature_importance()