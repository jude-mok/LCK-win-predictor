import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lck_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "lck_scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "lck_featured.pkl"
TEAM_STATE_PATH = BASE_DIR / "data" / "lck_team_state.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
lck_featured = joblib.load(DATA_PATH)

FEATURE_COLS = [
    'roll_winrate', 'roll_golddiff15', 'roll_firstdragon',
    'roll_firstherald', 'roll_firsttower', 'patch_winrate'
]

DIFF_COLS = [f'diff_{col}' for col in FEATURE_COLS]


def _build_team_state_from_legacy_data(data: pd.DataFrame) -> pd.DataFrame:
    """Build post-game state when the new state artifact is not published yet."""
    latest_date = data['date'].max()
    active_cutoff = latest_date - pd.Timedelta(days=180)
    active_teams = data.groupby('teamname')['date'].max()
    active_teams = active_teams[active_teams.ge(active_cutoff)].index

    states = []
    for team_name in active_teams:
        history = data[data['teamname'] == team_name].sort_values('date')
        latest_patch = history.iloc[-1]['patch']
        states.append({
            'teamname': team_name,
            'last_game_date': history['date'].max(),
            'roll_winrate': history['result'].tail(5).mean(),
            'roll_golddiff15': history['golddiffat15'].tail(5).mean(),
            'roll_firstdragon': history['firstdragon'].tail(5).mean(),
            'roll_firstherald': history['firstherald'].tail(5).mean(),
            'roll_firsttower': history['firsttower'].tail(5).mean(),
            'patch_winrate': history.loc[
                history['patch'].eq(latest_patch), 'result'
            ].mean(),
        })

    return pd.DataFrame(states).set_index('teamname').sort_index()


team_state = (
    joblib.load(TEAM_STATE_PATH)
    if TEAM_STATE_PATH.exists()
    else _build_team_state_from_legacy_data(lck_featured)
)


def get_teams() -> list[str]:
    return sorted(team_state.index.tolist())


def get_team_features(team_name: str) -> dict:
    latest = team_state.loc[team_name]
    return {column: float(latest[column]) for column in FEATURE_COLS}


def predict_match(team1: str, team2: str) -> dict:
    t1 = get_team_features(team1)
    t2 = get_team_features(team2)

    diff_1 = {f'diff_{k}': t1[k] - t2[k] for k in t1.keys()}
    diff_2 = {f'diff_{k}': t2[k] - t1[k] for k in t1.keys()}

    X1 = pd.DataFrame([diff_1])[DIFF_COLS]
    X2 = pd.DataFrame([diff_2])[DIFF_COLS]

    X1_sc = scaler.transform(X1)
    X2_sc = scaler.transform(X2)

    proba_1 = model.predict_proba(X1_sc)[0]
    proba_2 = model.predict_proba(X2_sc)[0]

    team1_win = round(float((proba_1[1] + proba_2[0]) / 2), 4)
    team2_win = round(float(1 - team1_win), 4)

    return {
        "team1": team1,
        "team1_win_rate": team1_win,
        "team2": team2,
        "team2_win_rate": team2_win,
        "predicted_winner": team1 if team1_win > 0.5 else team2,
        "features": diff_1,
        "team1_stats": {k: round(float(v), 4) for k, v in t1.items()},
        "team2_stats": {k: round(float(v), 4) for k, v in t2.items()},
    }


def get_feature_importance() -> dict:
    coef = model.coef_[0]
    label_map = {
        'diff_roll_winrate':     'Recent Win Rate',
        'diff_roll_golddiff15':  'Gold Diff @15',
        'diff_roll_firstdragon': 'First Dragon',
        'diff_roll_firstherald': 'First Herald',
        'diff_roll_firsttower':  'First Tower',
        'diff_patch_winrate':    'Patch Win Rate'
    }
    return {
        label_map[col]: round(float(val), 4)
        for col, val in zip(DIFF_COLS, coef)
    }
