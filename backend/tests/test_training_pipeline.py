from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.training.feature_pipeline import (
    DIFF_COLS,
    build_featured_games,
    build_match_dataset,
    build_team_state,
)
from backend.training.process_lck_data import (
    OUTPUT_COLUMNS,
    load_lck_team_games,
    write_processed_data,
)
from backend.training.train_model import (
    decide_acceptance,
    fit_symmetric_model,
    predict_side_neutral,
)


def team_row(
    gameid: str,
    side: str,
    teamname: str,
    result: int,
    *,
    league: str = "LCK",
    position: str = "team",
    completeness: str = "complete",
) -> dict:
    return {
        "gameid": gameid,
        "date": "2026-08-24 12:00:00",
        "league": league,
        "year": 2026,
        "split": "Season",
        "playoffs": 0,
        "game": 1,
        "patch": 16.16,
        "side": side,
        "position": position,
        "teamname": teamname,
        "teamid": teamname.lower(),
        "result": result,
        "golddiffat15": 500 if result else -500,
        "firstdragon": result,
        "firstherald": result,
        "firsttower": result,
        "datacompleteness": completeness,
    }


class ProcessLckDataTests(unittest.TestCase):
    def write_csv(self, rows: list[dict]) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "oracle.csv"
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(path, index=False)
        return path

    def test_keeps_only_complete_lck_team_rows(self) -> None:
        rows = [
            team_row("game-1", "Blue", "T1", 1),
            team_row("game-1", "Red", "Gen.G", 0),
            team_row("game-2", "Blue", "HLE", 0),
            team_row("game-2", "Red", "DK", 1),
            team_row("other-league", "Blue", "A", 1, league="LPL"),
            team_row("incomplete", "Blue", "T1", 1, completeness="partial"),
            team_row("player-row", "Blue", "Player", 1, position="top"),
        ]

        result = load_lck_team_games(self.write_csv(rows))

        self.assertEqual(len(result), 4)
        self.assertEqual(result["gameid"].nunique(), 2)
        self.assertEqual(set(result["league"]), {"LCK"})

    def test_rejects_duplicate_game_team_rows(self) -> None:
        duplicate = team_row("game-1", "Blue", "T1", 1)
        rows = [
            duplicate,
            duplicate.copy(),
            team_row("game-1", "Red", "Gen.G", 0),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate game/team"):
            load_lck_team_games(self.write_csv(rows))

    def test_writes_compressed_data_and_manifest(self) -> None:
        rows = [
            team_row("game-1", "Blue", "T1", 1),
            team_row("game-1", "Red", "Gen.G", 0),
        ]
        source_path = self.write_csv(rows)
        lck = load_lck_team_games(source_path)

        output_directory = tempfile.TemporaryDirectory()
        self.addCleanup(output_directory.cleanup)
        output_path = Path(output_directory.name) / "lck.csv.gz"

        manifest_path = write_processed_data(lck, source_path, output_path)

        self.assertTrue(output_path.exists())
        self.assertTrue(manifest_path.exists())
        restored = pd.read_csv(output_path)
        self.assertEqual(len(restored), 2)


class FeaturePipelineTests(unittest.TestCase):
    def make_series_rows(
        self,
        prefix: str,
        day: str,
        winner: str,
        scores: list[tuple[str, str]],
    ) -> list[dict]:
        rows = []
        for game_number, (blue, red) in enumerate(scores, start=1):
            blue_won = int(blue == winner)
            for side, team, result in [
                ("Blue", blue, blue_won),
                ("Red", red, 1 - blue_won),
            ]:
                row = team_row(f"{prefix}-{game_number}", side, team, result)
                row["date"] = f"{day} {10 + game_number:02d}:00:00"
                row["game"] = game_number
                rows.append(row)
        return rows

    def test_builds_completed_match_targets_and_post_game_state(self) -> None:
        rows = []
        rows += self.make_series_rows(
            "s1",
            "2026-01-01",
            "T1",
            [("T1", "Gen.G"), ("Gen.G", "T1")],
        )
        rows += self.make_series_rows(
            "s2",
            "2026-01-08",
            "Gen.G",
            [("T1", "Gen.G"), ("Gen.G", "T1")],
        )
        data = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

        featured = build_featured_games(data)
        matches = build_match_dataset(featured)
        state = build_team_state(featured)

        self.assertEqual(len(matches), 2)
        self.assertEqual(set(matches["best_of"]), {3})
        self.assertEqual(set(matches["result"]), {0, 1})
        self.assertEqual(state.loc["T1", "roll_winrate"], 0.5)

    def test_symmetric_model_is_input_order_invariant(self) -> None:
        features = pd.DataFrame(
            [
                [0.5, 1000, 0.3, 0.2, 0.4, 0.1],
                [-0.4, -800, -0.2, -0.1, -0.3, -0.1],
                [0.2, 300, 0.1, 0.0, 0.2, 0.2],
                [-0.3, -500, -0.1, -0.2, -0.1, -0.2],
            ],
            columns=DIFF_COLS,
        )
        target = pd.Series([1, 0, 1, 0])
        model, scaler = fit_symmetric_model(features, target)

        forward = predict_side_neutral(model, scaler, features.iloc[[0]])[0]
        reverse = predict_side_neutral(model, scaler, -features.iloc[[0]])[0]

        self.assertAlmostEqual(forward + reverse, 1.0, places=12)

    def test_acceptance_gate_requires_real_log_loss_improvement(self) -> None:
        candidate = {"roc_auc": 0.60, "log_loss": 0.64}
        champion = {"roc_auc": 0.58, "log_loss": 0.66}

        accepted, _ = decide_acceptance(candidate, champion)

        self.assertTrue(accepted)


if __name__ == "__main__":
    unittest.main()
