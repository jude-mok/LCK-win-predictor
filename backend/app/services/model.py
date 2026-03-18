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


def predict_match(blue_team: str, red_team: str) -> dict:
    blue = get_team_features(blue_team)
    red = get_team_features(red_team)

    diff = {f'diff_{k}': blue[k] - red[k] for k in blue.keys()}

    X_pred = pd.DataFrame([diff])[DIFF_COLS]
    X_pred_sc = scaler.transform(X_pred)

    proba = model.predict_proba(X_pred_sc)[0]

    return {
        "blue_team": blue_team,
        "blue_win_rate": round(float(proba[1]), 4),
        "red_team": red_team,
        "red_win_rate": round(float(proba[0]), 4),
        "predicted_winner": blue_team if proba[1] > 0.5 else red_team,
        "features": diff
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