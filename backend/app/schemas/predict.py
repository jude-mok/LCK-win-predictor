from pydantic import BaseModel, ConfigDict
from typing import Optional, Literal


class Predict_Request(BaseModel):
    team1: str
    team2: str
    best_of: Literal[3, 5] = 3

class Predict_Response(BaseModel):
    team1: str
    team1_win_rate: float
    team2: str
    team2_win_rate: float
    predicted_winner: str
    features: dict