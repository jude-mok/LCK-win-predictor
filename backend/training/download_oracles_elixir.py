from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_YEAR = "2026"
DEFAULT_FILE_ID = "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"

YEAR = os.getenv("ORACLES_ELIXIR_YEAR", DEFAULT_YEAR)
FILE_ID = os.getenv("ORACLES_ELIXIR_FILE_ID", DEFAULT_FILE_ID)

BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BACKEND_DIR / "data" / "raw"

EXPECTED_COLUMNS = {
    "gameid",
    "date",
    "league",
    "position",
    "teamname",
    "result",
    "golddiffat15",
    "firstdragon",
    "firstherald",
    "firsttower",
}


class DownloadUnavailableError(RuntimeError):
    """Raised when Google Drive returns a download error page."""


def build_download_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


def download_file(destination: Path, file_id: str = FILE_ID) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    temporary_path.unlink(missing_ok=True)

    request = Request(
        build_download_url(file_id),
        headers={
            "User-Agent": "LOL-ML weekly data pipeline/1.0",
        },
    )

    print(f"Downloading Oracle's Elixir data to {destination.name}...", flush=True)

    try:
        with urlopen(request, timeout=180) as response:
            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(
                    response,
                    output_file,
                    length=1024 * 1024,
                )

        validate_file(temporary_path)

        os.replace(temporary_path, destination)

    finally:
        temporary_path.unlink(missing_ok=True)


def validate_file(path: Path) -> None:
    file_size = path.stat().st_size

    if file_size < 1_000_000:
        response_preview = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )[:4_000]

        if "Quota exceeded" in response_preview:
            raise DownloadUnavailableError(
                "Google Drive download quota is currently exceeded. "
                "The existing dataset was not changed; try again later."
            )

        raise DownloadUnavailableError(
            "Google Drive returned a non-CSV response "
            f"({file_size} bytes). The existing dataset was not changed."
        )

    columns = set(pd.read_csv(path, nrows=0).columns)
    missing_columns = EXPECTED_COLUMNS - columns

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )


def print_summary(path: Path) -> None:
    df = pd.read_csv(
        path,
        usecols=["gameid", "date", "league", "position"],
        parse_dates=["date"],
        low_memory=False,
    )

    lck_team_rows = df[
        (df["league"] == "LCK")
        & (df["position"] == "team")
    ]

    if lck_team_rows.empty:
        raise ValueError("Downloaded data contains no LCK team rows")

    file_size_mb = path.stat().st_size / 1024 / 1024

    print("Download complete")
    print(f"File: {path}")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"LCK games: {lck_team_rows['gameid'].nunique()}")
    print(f"Latest LCK date: {lck_team_rows['date'].max()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a public Oracle's Elixir yearly CSV."
    )
    parser.add_argument("--year", default=YEAR)
    parser.add_argument("--file-id", default=FILE_ID)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or (
        RAW_DATA_DIR
        / f"{args.year}_LoL_esports_match_data_from_OraclesElixir.csv"
    )

    try:
        download_file(output_path, args.file_id)
    except DownloadUnavailableError as error:
        print(f"Download unavailable: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print_summary(output_path)


if __name__ == "__main__":
    main()
