from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "roll_winrate",
    "roll_golddiff15",
    "roll_firstdragon",
    "roll_firstherald",
    "roll_firsttower",
    "patch_winrate",
]
DIFF_COLS = [f"diff_{column}" for column in FEATURE_COLS]

RAW_FEATURES = {
    "roll_winrate": "result",
    "roll_golddiff15": "golddiffat15",
    "roll_firstdragon": "firstdragon",
    "roll_firstherald": "firstherald",
    "roll_firsttower": "firsttower",
}

NEUTRAL_VALUES = {
    "roll_winrate": 0.5,
    "roll_golddiff15": 0.0,
    "roll_firstdragon": 0.5,
    "roll_firstherald": 0.5,
    "roll_firsttower": 0.5,
    "patch_winrate": 0.5,
}

TEAM_NAME_ALIASES = {
    "DRX": "Kiwoom DRX",
}


def canonicalize_team_names(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["teamname"] = result["teamname"].replace(TEAM_NAME_ALIASES)
    return result


def add_series_keys(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    team_pair = result.groupby("gameid")["teamname"].agg(
        lambda values: "|".join(sorted(set(values)))
    )
    if team_pair.str.count(r"\|").ne(1).any():
        raise ValueError("Every game must contain exactly two distinct teams")

    result["team_pair"] = result["gameid"].map(team_pair)
    result["match_day"] = result["date"].dt.strftime("%Y-%m-%d")
    result["series_key"] = (
        result["year"].astype(str)
        + "|"
        + result["split"].fillna("unknown").astype(str)
        + "|"
        + result["match_day"]
        + "|"
        + result["team_pair"]
    )
    return result


def build_featured_games(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    featured = canonicalize_team_names(data)
    featured["date"] = pd.to_datetime(featured["date"], errors="raise")
    featured = featured.sort_values(["teamname", "date", "gameid"]).reset_index(
        drop=True
    )

    for output_column, raw_column in RAW_FEATURES.items():
        featured[output_column] = featured.groupby("teamname")[raw_column].transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )

    featured["patch_winrate"] = featured.groupby(
        ["teamname", "patch"]
    )["result"].transform(lambda values: values.shift(1).expanding().mean())

    for column, neutral_value in NEUTRAL_VALUES.items():
        featured[column] = featured[column].fillna(neutral_value)

    featured = add_series_keys(featured)
    return featured.sort_values(["date", "gameid", "side"]).reset_index(drop=True)


def _series_is_complete(game_count: int, winner_wins: int) -> bool:
    return (
        (game_count == 2 and winner_wins == 2)
        or (game_count == 3 and winner_wins in {2, 3})
        or (game_count in {4, 5} and winner_wins == 3)
    )


def build_match_dataset(featured_games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for series_key, series in featured_games.groupby("series_key", sort=False):
        series = series.sort_values(["date", "gameid"])
        teams = sorted(series["teamname"].unique())
        if len(teams) != 2:
            continue

        game_count = int(series["gameid"].nunique())
        wins = series.groupby("teamname")["result"].sum()
        winner = str(wins.idxmax())
        winner_wins = int(wins.max())
        if wins.nunique() == 1 or not _series_is_complete(game_count, winner_wins):
            continue

        first_game_id = series.iloc[0]["gameid"]
        first_game = series[series["gameid"].eq(first_game_id)]
        if len(first_game) != 2:
            continue

        team1, team2 = teams
        team1_row = first_game[first_game["teamname"].eq(team1)].iloc[0]
        team2_row = first_game[first_game["teamname"].eq(team2)].iloc[0]

        row = {
            "series_key": series_key,
            "date": first_game["date"].min(),
            "team1": team1,
            "team2": team2,
            "result": int(winner == team1),
            "game_count": game_count,
            "best_of": 3 if winner_wins == 2 else 5,
        }
        for feature in FEATURE_COLS:
            row[f"diff_{feature}"] = float(team1_row[feature] - team2_row[feature])
        rows.append(row)

    matches = pd.DataFrame(rows)
    if matches.empty:
        raise ValueError("No completed LCK match series could be constructed")

    if not np.isfinite(matches[DIFF_COLS].to_numpy(dtype=float)).all():
        raise ValueError("Match features contain non-finite values")

    return matches.sort_values(["date", "series_key"]).reset_index(drop=True)


def build_team_state(featured_games: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    latest_year = int(featured_games["year"].max())
    active_teams = sorted(
        featured_games.loc[featured_games["year"].eq(latest_year), "teamname"].unique()
    )
    states: list[dict] = []

    for team in active_teams:
        history = featured_games[featured_games["teamname"].eq(team)].sort_values("date")
        latest_patch = history.iloc[-1]["patch"]
        state = {
            "teamname": team,
            "last_game_date": history["date"].max(),
            "roll_winrate": float(history["result"].tail(window).mean()),
            "roll_golddiff15": float(history["golddiffat15"].tail(window).mean()),
            "roll_firstdragon": float(history["firstdragon"].tail(window).mean()),
            "roll_firstherald": float(history["firstherald"].tail(window).mean()),
            "roll_firsttower": float(history["firsttower"].tail(window).mean()),
            "patch_winrate": float(
                history.loc[history["patch"].eq(latest_patch), "result"].mean()
            ),
        }
        states.append(state)

    return pd.DataFrame(states).set_index("teamname").sort_index()
