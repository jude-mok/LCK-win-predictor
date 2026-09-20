# LCK Win Predictor

Next.js + FastAPI service backed by Supabase. Select two teams to load their current Main lineups, then swap players by position to recalculate the match probability.

## Current model

Player A: 13 features per player, shared 13→8→4 encoder and five position-specific 4→2→1 heads. Team scores are compared using a sigmoid. Three seeds are averaged, then converted to BO3/BO5 probabilities. Trained on 555 LCK sets from 2025; retrospective 2026 evaluation: 118/186 series (63.44%). The 2026 period has been repeatedly inspected, so it is not an untouched test set.

Portable JSON weights, scaling and imputation values are served using NumPy; PyTorch is required for research only. The exported runtime reproduces the research predictions within 2.4e-7.

## Roster flow

`public.players` contains `team_code`, `team_name`, `position`, `player_name`, `player_id`, `league`, `roster_status`, and `roster_as_of` (timestamptz). Exactly one `Main` player is required per team and position (`top`, `jng`, `mid`, `bot`, `sup`). Others are `sub`.

- `GET /predict/rosters`: DB roster catalog and history availability.
- `POST /predict/predict`: `{ "team1": "T1", "team2": "GEN", "best_of": 3 }` automatically resolves Main players.
- Optional `team1_roster` / `team2_roster` maps replace individual positions by player ID. The server checks team and position; these requests do not change DB defaults.
- The match list assumes BO3; the detail view supports BO3/BO5. Finished matches also show current-roster calculations, not purported historical forecasts.

LCK histories retain the champion's original feature policy. Players without LCK history use their last five complete CL games; CL use is disclosed and its accuracy is not separately validated. Roster changes in DB are live; performance statistics require refreshing the exported player-state artifact. Current source cutoff: 2026-09-12.

## Run

```sh
uv sync --frozen
uv run uvicorn backend.app.main:app --reload --port 8000
```

Configure `SUPABASE_URL` and `SUPABASE_KEY` on the backend only. For the frontend:

```sh
cd frontend
npm ci
npm run dev
```

Frontend requests use a same-origin `/api` proxy. Development defaults to port 8000. Production defaults to the existing Railway API; override `API_BACKEND_URL` before building to change the destination. `NEXT_PUBLIC_API_URL` can explicitly bypass the proxy when needed.

The root Dockerfile serves the backend. A backend service rooted at `backend/` may instead install `requirements.txt` and run `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. A frontend service uses root `frontend/`, `npm run build`, then `npm run start -- --hostname 0.0.0.0 --port $PORT`.

`/health` is liveness and `/ready` checks local model loading. Verify `/predict/rosters` and a prediction as well when deploying because DB access is separate. Optional historical snapshot endpoints are not part of the current UI flow; production snapshot storage requires an explicit persistent `PREDICTION_DB_PATH`.

## Verify and research

```sh
uv run python -m unittest discover -s backend/tests
cd frontend
npm run build
npm run lint
```

Research scripts and evidence live under `backend/training/` and `notebook/experiments/`. Experiment 10 is the deployed-model reference; experiments 11–14 did not replace it. Raw Oracle's Elixir data and secrets are excluded from Git. The weekly workflow audits legacy team-model candidates and does not automatically promote a new player model.
