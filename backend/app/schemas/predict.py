from pydantic import BaseModel, ConfigDict
from typing import Optional


class Predict_Request(BaseModel):
    blue_team: str
    red_team: str
    
class Predict_Response(BaseModel):
    blue_team: str
    blue_team_winrate: str
    red_team: str
    red_team_winrate: str
    predicted_winner: str
    features: dict