# LCK Win Rate Predictor

A web service that predicts LCK (League of Legends Champions Korea) match win rates using machine learning, and lets users earn LP points by predicting match outcomes.

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS 4
- **Backend**: FastAPI, Uvicorn
- **ML**: scikit-learn, pandas, numpy, joblib

## Project Structure

```
LOL_ML/
├── frontend/                    # Next.js 16 web app
│   └── app/
│       ├── page.tsx             # Main page (match list + AI predictions)
│       └── match/
│           └── [id]/
│               └── page.tsx     # Match detail page
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── routers/
│   │   │   ├── predict.py       # Win rate prediction API
│   │   │   └── schedule.py      # Match schedule API
│   │   ├── schemas/
│   │   │   └── predict.py       # Request/response schemas
│   │   └── services/
│   │       └── model.py         # Model loading and prediction logic
│   ├── models/
│   │   ├── lck_model.pkl        # Trained classification model
│   │   └── lck_scaler.pkl       # Feature scaler
│   ├── data/
│   │   └── lck_featured.pkl     # Feature-engineered team data
│   └── requirements.txt
└── notebook/
    └── lol_predict.ipynb        # Data analysis and model training notebook
```

## Getting Started

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger UI available at http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App available at http://localhost:3000

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/schedule/` | Current week's match schedule |
| `GET` | `/schedule/entire/week/{week}` | All matches for a given week |
| `GET` | `/predict/teams` | Get list of available teams |
| `POST` | `/predict/predict` | Predict win rate between two teams |
| `GET` | `/predict/features` | Get model feature importances |

### Prediction Request

```json
POST /predict/predict
{
  "blue_team": "T1",
  "red_team": "Gen.G"
}
```

### Prediction Response

```json
{
  "blue_team": "T1",
  "blue_win_rate": 0.6312,
  "red_team": "Gen.G",
  "red_win_rate": 0.3688,
  "predicted_winner": "T1",
  "features": {
    "diff_roll_winrate": 0.12,
    "diff_roll_golddiff15": 350.5,
    ...
  }
}
```

## ML Features

The model uses the difference (blue − red) of the following rolling team statistics as input features.

| Feature | Description |
|---------|-------------|
| `roll_winrate` | Recent rolling win rate |
| `roll_golddiff15` | Gold difference at 15 minutes |
| `roll_firstdragon` | First dragon rate |
| `roll_firstherald` | First herald rate |
| `roll_firsttower` | First tower rate |
| `patch_winrate` | Win rate on the current patch |

## Supported Teams (2026 Season)

| Code | Team |
|------|------|
| T1 | T1 |
| GEN | Gen.G |
| HLE | Hanwha Life Esports |
| DK | Dplus KIA |
| KT | KT Rolster |
| BFX | BNK FEARX |
| NS | Nongshim RedForce |
| KRX | DRX |
| DNS | DN SOOPers |
| BRO | HANJIN BRION |
