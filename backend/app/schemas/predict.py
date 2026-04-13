from pydantic import BaseModel, ConfigDict
from typing import Optional


class Predict_Request(BaseModel):
    team1: str
    team2: str

class Predict_Response(BaseModel):
    team1: str
    team1_win_rate: float
    team2: str
    team2_win_rate: float
    predicted_winner: str
    features: dict