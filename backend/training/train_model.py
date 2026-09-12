from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from backend.training.feature_pipeline import (
    DIFF_COLS,
    FEATURE_COLS,
    build_featured_games,
    build_match_dataset,
    build_team_state,
)


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = BACKEND_DIR / "data" / "processed" / "lck_team_games.csv.gz"
MODEL_PATH = BACKEND_DIR / "models" / "lck_model.pkl"
SCALER_PATH = BACKEND_DIR / "models" / "lck_scaler.pkl"
MODEL_METADATA_PATH = BACKEND_DIR / "models" / "lck_model_metadata.json"
FEATURED_DATA_PATH = BACKEND_DIR / "data" / "lck_featured.pkl"
TEAM_STATE_PATH = BACKEND_DIR / "data" / "lck_team_state.pkl"
REPORT_PATH = BACKEND_DIR / "training" / "reports" / "latest_training_report.json"

MIN_MATCHES = int(os.getenv("MODEL_MIN_MATCHES", "120"))
MIN_HOLDOUT_MATCHES = int(os.getenv("MODEL_MIN_HOLDOUT_MATCHES", "40"))
MIN_LOG_LOSS_IMPROVEMENT = float(
    os.getenv("MODEL_MIN_LOG_LOSS_IMPROVEMENT", "0.002")
)
MIN_ACCEPTABLE_AUC = float(os.getenv("MODEL_MIN_ACCEPTABLE_AUC", "0.50"))


def fit_symmetric_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[LogisticRegression, StandardScaler]:
    augmented_features = pd.concat(
        [features, -features],
        ignore_index=True,
    )
    augmented_target = pd.concat(
        [target.reset_index(drop=True), 1 - target.reset_index(drop=True)],
        ignore_index=True,
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(augmented_features)
    model = LogisticRegression(
        fit_intercept=False,
        max_iter=2_000,
        random_state=42,
    )
    model.fit(scaled_features, augmented_target)
    return model, scaler


def predict_side_neutral(
    model: LogisticRegression,
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> np.ndarray:
    forward = model.predict_proba(scaler.transform(features))[:, 1]
    reverse = model.predict_proba(scaler.transform(-features))[:, 1]
    return (forward + (1.0 - reverse)) / 2.0


def evaluate(target: pd.Series, probabilities: np.ndarray) -> dict:
    predicted = probabilities >= 0.5
    auc = None
    if target.nunique() == 2:
        auc = float(roc_auc_score(target, probabilities))

    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "roc_auc": auc,
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
    }


def load_champion() -> tuple[LogisticRegression, StandardScaler] | None:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def decide_acceptance(
    candidate_metrics: dict,
    champion_metrics: dict | None,
) -> tuple[bool, str]:
    candidate_auc = candidate_metrics["roc_auc"]
    if candidate_auc is None or candidate_auc < MIN_ACCEPTABLE_AUC:
        return False, "candidate AUC is below the minimum acceptance threshold"

    baseline_log_loss = math.log(2.0)
    if candidate_metrics["log_loss"] >= baseline_log_loss:
        return False, "candidate log loss does not beat a 50% probability baseline"

    if champion_metrics is None:
        return True, "no champion model exists and the candidate beats the baseline"

    improvement = champion_metrics["log_loss"] - candidate_metrics["log_loss"]
    if improvement < MIN_LOG_LOSS_IMPROVEMENT:
        return (
            False,
            "candidate log-loss improvement is too small "
            f"({improvement:.6f} < {MIN_LOG_LOSS_IMPROVEMENT:.6f})",
        )

    return True, f"candidate improves champion log loss by {improvement:.6f}"


def atomic_joblib_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        joblib.dump(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json_dump(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def publish_artifacts(
    model: LogisticRegression,
    scaler: StandardScaler,
    featured_games: pd.DataFrame,
    team_state: pd.DataFrame,
    metadata: dict,
) -> None:
    atomic_joblib_dump(model, MODEL_PATH)
    atomic_joblib_dump(scaler, SCALER_PATH)
    atomic_joblib_dump(featured_games, FEATURED_DATA_PATH)
    atomic_joblib_dump(team_state, TEAM_STATE_PATH)
    atomic_json_dump(metadata, MODEL_METADATA_PATH)


def existing_source_is_current(latest_game_date: pd.Timestamp) -> bool:
    if not MODEL_METADATA_PATH.exists():
        return False
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    previous_latest = metadata.get("latest_game_date")
    if not previous_latest:
        return False
    return pd.Timestamp(previous_latest) >= latest_game_date


def run_training(input_path: Path, force_publish: bool = False) -> dict:
    team_games = pd.read_csv(input_path, parse_dates=["date"], low_memory=False)
    featured_games = build_featured_games(team_games)
    matches = build_match_dataset(featured_games)
    team_state = build_team_state(featured_games)
    latest_game_date = featured_games["date"].max()

    report: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "latest_game_date": latest_game_date.isoformat(),
        "game_count": int(featured_games["gameid"].nunique()),
        "match_count": int(len(matches)),
        "active_team_count": int(len(team_state)),
        "features": DIFF_COLS,
        "training_target": "pre-series Bo3/Bo5 match winner",
        "published": False,
    }

    if len(matches) < MIN_MATCHES:
        report["status"] = "rejected"
        report["reason"] = f"not enough completed matches ({len(matches)} < {MIN_MATCHES})"
        atomic_json_dump(report, REPORT_PATH)
        return report

    if existing_source_is_current(latest_game_date) and not force_publish:
        report["status"] = "no_new_data"
        report["reason"] = "published model already includes this latest game date"
        atomic_json_dump(report, REPORT_PATH)
        return report

    holdout_size = max(MIN_HOLDOUT_MATCHES, int(math.ceil(len(matches) * 0.2)))
    if len(matches) - holdout_size < 50:
        raise ValueError("Not enough training matches remain after holdout selection")

    train = matches.iloc[:-holdout_size]
    holdout = matches.iloc[-holdout_size:]
    candidate_model, candidate_scaler = fit_symmetric_model(
        train[DIFF_COLS],
        train["result"],
    )
    candidate_probabilities = predict_side_neutral(
        candidate_model,
        candidate_scaler,
        holdout[DIFF_COLS],
    )
    candidate_metrics = evaluate(holdout["result"], candidate_probabilities)

    champion = load_champion()
    champion_metrics = None
    if champion is not None:
        champion_probabilities = predict_side_neutral(
            champion[0],
            champion[1],
            holdout[DIFF_COLS],
        )
        champion_metrics = evaluate(holdout["result"], champion_probabilities)

    accepted, reason = decide_acceptance(candidate_metrics, champion_metrics)
    if force_publish:
        accepted = True
        reason = "manual force-publish requested"

    report.update(
        {
            "status": "accepted" if accepted else "rejected",
            "reason": reason,
            "holdout": {
                "match_count": int(len(holdout)),
                "first_date": holdout["date"].min().isoformat(),
                "last_date": holdout["date"].max().isoformat(),
            },
            "candidate_metrics": candidate_metrics,
            "champion_metrics": champion_metrics,
            "thresholds": {
                "minimum_matches": MIN_MATCHES,
                "minimum_holdout_matches": MIN_HOLDOUT_MATCHES,
                "minimum_log_loss_improvement": MIN_LOG_LOSS_IMPROVEMENT,
                "minimum_auc": MIN_ACCEPTABLE_AUC,
            },
        }
    )

    if accepted:
        final_model, final_scaler = fit_symmetric_model(
            matches[DIFF_COLS],
            matches["result"],
        )
        metadata = {
            **report,
            "published": True,
            "model_type": "LogisticRegression",
            "model_parameters": final_model.get_params(),
            "feature_columns": FEATURE_COLS,
            "diff_columns": DIFF_COLS,
        }
        publish_artifacts(
            final_model,
            final_scaler,
            featured_games,
            team_state,
            metadata,
        )
        report["published"] = True

    atomic_json_dump(report, REPORT_PATH)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and conditionally publish the weekly LCK match model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--force-publish",
        action="store_true",
        help="Publish without the acceptance gate. Never use this in the schedule.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_training(args.input, force_publish=args.force_publish)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
