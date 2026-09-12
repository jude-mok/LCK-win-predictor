# Weekly LCK data pipeline

This pipeline downloads the public Oracle's Elixir yearly CSVs, keeps only
complete LCK team rows, validates game integrity, constructs pre-series
Bo3/Bo5 features, and trains a candidate model. The candidate replaces the
deployed model only when it improves recent chronological holdout log loss.

## Local run

From the repository root:

```bash
uv sync --frozen
uv run python -m unittest backend.tests.test_training_pipeline
uv run python backend/training/download_oracles_elixir.py \
  --year 2025 --file-id 1v6LRphp2kYciU4SXp0PCjEMuev1bDejc
uv run python backend/training/download_oracles_elixir.py \
  --year 2026 --file-id 1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm
uv run python backend/training/process_lck_data.py \
  --input backend/data/raw/2025_LoL_esports_match_data_from_OraclesElixir.csv \
  --input backend/data/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv \
  --output backend/data/processed/lck_team_games.csv.gz
uv run python backend/training/train_model.py \
  --input backend/data/processed/lck_team_games.csv.gz
```

Raw and processed files are intentionally excluded from Git.

## Schedule

`.github/workflows/update-lck-data.yml` runs every Monday at 12:00 KST
(03:00 UTC). It can also be started manually with **Run workflow** in the
GitHub Actions page. A successful run stores the processed dataset and its
manifest as a GitHub Actions artifact for 30 days.

## Configuration

The current 2026 public file is the default. To change the year or Drive file,
set these environment variables:

```bash
ORACLES_ELIXIR_YEAR=2026
ORACLES_ELIXIR_FILE_ID=google_drive_file_id
```

The 2025 file is immutable and cached by GitHub Actions. Only the current-year
file is refreshed every Monday.

Google Drive may temporarily return a file-level quota error. The downloader
detects that HTML response, preserves the previous local file, and exits with
a clear error. Do not bypass Drive access controls; retry the workflow later.

## Model acceptance gate

The last 20% of completed match series (at least 40) is held out in time order.
The candidate is published only if:

- its ROC-AUC is at least 0.50;
- its log loss beats a constant 50% probability baseline; and
- its log loss improves the deployed champion by at least 0.002.

Rejected candidates create a report but never overwrite model artifacts. The
scheduled workflow never uses `--force-publish`.
