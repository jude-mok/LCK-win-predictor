from typing import Literal
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class Roster(BaseModel):
    model_config = ConfigDict(extra='forbid')
    top: str = Field(min_length=1, max_length=120)
    jng: str = Field(min_length=1, max_length=120)
    mid: str = Field(min_length=1, max_length=120)
    bot: str = Field(min_length=1, max_length=120)
    sup: str = Field(min_length=1, max_length=120)


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    team1: str = Field(min_length=1, max_length=100)
    team2: str = Field(min_length=1, max_length=100)
    best_of: Literal[3, 5]
    team1_roster: Roster
    team2_roster: Roster

    @model_validator(mode='after')
    def distinct(self):
        players = list(self.team1_roster.model_dump().values()) + list(self.team2_roster.model_dump().values())
        if self.team1.casefold() == self.team2.casefold() or len(set(players)) != 10:
            raise ValueError('Use distinct teams and ten distinct players')
        return self


class PublishRequest(SimulationRequest):
    series_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,100}$')
    scheduled_start: AwareDatetime
    roster_confirmed_at: AwareDatetime


class TeamPredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    team1: str = Field(min_length=1, max_length=100)
    team2: str = Field(min_length=1, max_length=100)
    best_of: Literal[3, 5] = 3
    team1_roster: dict[Literal['top', 'jng', 'mid', 'bot', 'sup'], str] | None = None
    team2_roster: dict[Literal['top', 'jng', 'mid', 'bot', 'sup'], str] | None = None
