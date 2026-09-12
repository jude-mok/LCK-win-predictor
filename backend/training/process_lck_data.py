from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


YEAR = os.getenv("ORACLES_ELIXIR_YEAR", "2026")

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    BACKEND_DIR
    / "data"
    / "raw"
    / f"{YEAR}_LoL_esports_match_data_from_OraclesElixir.csv"
)
DEFAULT_OUTPUT_PATH = (
    BACKEND_DIR
    / "data"
    / "processed"
    / f"lck_team_games_{YEAR}.csv.gz"
)

OUTPUT_COLUMNS = [
    "gameid",
    "date",
    "league",
    "year",
    "split",
    "playoffs",
    "game",
    "patch",
    "side",
    "position",
    "teamname",
    "teamid",
    "result",
    "golddiffat15",
    "firstdragon",
    "firstherald",
    "firsttower",
    "datacompleteness",
]


def load_lck_team_games(source_path: Path) -> pd.DataFrame:
    source_columns = set(pd.read_csv(source_path, nrows=0).columns)
    missing_columns = set(OUTPUT_COLUMNS) - source_columns

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    data = pd.read_csv(
        source_path,
        usecols=OUTPUT_COLUMNS,
        low_memory=False,
    )

    lck = data[
        data["league"].eq("LCK")
        & data["position"].eq("team")
        & data["datacompleteness"].eq("complete")
    ].copy()

    if lck.empty:
        raise ValueError("No complete LCK team rows were found")

    lck["date"] = pd.to_datetime(lck["date"], errors="raise")
    lck = lck.sort_values(["date", "gameid", "side"]).reset_index(drop=True)

    validate_lck_team_games(lck)
    return lck


def load_many_lck_team_games(source_paths: list[Path]) -> pd.DataFrame:
    frames = [load_lck_team_games(path) for path in source_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "gameid", "side"]).reset_index(drop=True)
    validate_lck_team_games(combined)
    return combined


def validate_lck_team_games(data: pd.DataFrame) -> None:
    duplicate_rows = data.duplicated(["gameid", "teamname"], keep=False)
    if duplicate_rows.any():
        duplicate_games = sorted(
            data.loc[duplicate_rows, "gameid"].astype(str).unique()
        )
        raise ValueError(
            "Duplicate game/team rows found: "
            f"{duplicate_games[:5]}"
        )

    rows_per_game = data.groupby("gameid").size()
    invalid_row_counts = rows_per_game[rows_per_game.ne(2)]
    if not invalid_row_counts.empty:
        raise ValueError(
            "Every complete game must contain exactly two team rows: "
            f"{invalid_row_counts.head().to_dict()}"
        )

    side_count = data.groupby("gameid")["side"].nunique()
    invalid_sides = side_count[side_count.ne(2)]
    if not invalid_sides.empty:
        raise ValueError(
            "Every game must contain Blue and Red sides: "
            f"{invalid_sides.head().to_dict()}"
        )

    side_values = set(data["side"].dropna().unique())
    if side_values != {"Blue", "Red"}:
        raise ValueError(f"Unexpected side values: {sorted(side_values)}")

    result_sum = data.groupby("gameid")["result"].sum()
    invalid_results = result_sum[result_sum.ne(1)]
    if not invalid_results.empty:
        raise ValueError(
            "Every game must contain exactly one winning team: "
            f"{invalid_results.head().to_dict()}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_processed_data(
    data: pd.DataFrame,
    source_paths: Path | list[Path],
    destination_path: Path,
) -> Path:
    if isinstance(source_paths, Path):
        source_paths = [source_paths]

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(destination_path.name + ".part")
    manifest_path = destination_path.with_suffix(".manifest.json")
    temporary_manifest_path = manifest_path.with_suffix(".json.part")

    temporary_path.unlink(missing_ok=True)
    temporary_manifest_path.unlink(missing_ok=True)

    try:
        data.to_csv(
            temporary_path,
            index=False,
            compression="gzip",
            date_format="%Y-%m-%d %H:%M:%S",
        )

        manifest = {
            "source_files": [
                {
                    "name": source_path.name,
                    "sha256": sha256(source_path),
                }
                for source_path in source_paths
            ],
            "processed_file": destination_path.name,
            "processed_sha256": sha256(temporary_path),
            "row_count": int(len(data)),
            "game_count": int(data["gameid"].nunique()),
            "first_game_date": data["date"].min().isoformat(),
            "latest_game_date": data["date"].max().isoformat(),
        }
        temporary_manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        os.replace(temporary_path, destination_path)
        os.replace(temporary_manifest_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        temporary_manifest_path.unlink(missing_ok=True)

    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a validated LCK-only dataset from Oracle's Elixir data."
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        type=Path,
        default=None,
        help="Path to the downloaded Oracle's Elixir CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the compressed LCK team dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = args.inputs or [DEFAULT_INPUT_PATH]
    lck = load_many_lck_team_games(input_paths)
    manifest_path = write_processed_data(lck, input_paths, args.output)

    print("LCK processing complete")
    print(f"File: {args.output}")
    print(f"Rows: {len(lck)}")
    print(f"Games: {lck['gameid'].nunique()}")
    print(f"Latest LCK date: {lck['date'].max()}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
