import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lck_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "lck_scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "lck_featured.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
lck_featured = joblib.load(DATA_PATH)

FEATURE_COLS = [
    'roll_winrate', 'roll_golddiff15', 'roll_firstdragon',
    'roll_firstherald', 'roll_firsttower', 'patch_winrate'
]

DIFF_COLS = [f'diff_{col}' for col in FEATURE_COLS]


def get_teams() -> list[str]:
    return sorted(lck_featured['teamname'].unique().tolist())


def get_team_features(team_name: str) -> dict:
    team_df = lck_featured[lck_featured['teamname'] == team_name].sort_values('date')
    latest = team_df.iloc[-1]
    return {
        'roll_winrate':     latest['roll_winrate'],
        'roll_golddiff15':  latest['roll_golddiff15'],
        'roll_firstdragon': latest['roll_firstdragon'],
        'roll_firstherald': latest['roll_firstherald'],
        'roll_firsttower':  latest['roll_firsttower'],
        'patch_winrate':    latest['patch_winrate'] if pd.notna(latest['patch_winrate']) else 0.5
    }


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
    return {label_map[col]: round(float(val), 4) 
            for col, val in zip(DIFF_COLS, coef)}