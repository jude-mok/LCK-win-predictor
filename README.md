# LCK Win Rate Predictor

A web service that predicts LCK (League of Legends Champions Korea) match win rates using machine learning. The model is trained on 650+ matches from the 2025–2026 LCK seasons and predicts win probability based on each team's recent performance statistics.

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS 4
- **Backend**: FastAPI, Uvicorn
- **Database**: Supabase (PostgreSQL)
- **ML**: scikit-learn, pandas, numpy, joblib
- **Deployment**: Railway

## Project Structure

```
LOL_ML/
├── frontend/                    # Next.js 16 web app
│   └── app/
│       ├── page.tsx             # Main page (match schedule + AI predictions)
│       └── match/
│           └── [id]/
│               └── page.tsx     # Match detail page (stat breakdown)
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
│   │   ├── lck_model.pkl        # Trained Logistic Regression model
│   │   └── lck_scaler.pkl       # StandardScaler for feature normalization
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

Swagger UI available at `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App available at `http://localhost:3000`

### Environment Variables

```bash
# Backend
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
```

Set via Railway Shared Variables in production. Use `os.environ.get()` (not `load_dotenv()`) for Railway compatibility.

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
  "team1": "T1",
  "team2": "Gen.G"
}
```

### Prediction Response

```json
{
  "team1": "T1",
  "team1_win_rate": 0.6312,
  "team2": "Gen.G",
  "team2_win_rate": 0.3688,
  "predicted_winner": "T1",
  "features": {
    "diff_roll_winrate": 0.12,
    "diff_roll_golddiff15": 350.5
  },
  "team1_stats": {
    "roll_winrate": 0.8,
    "roll_golddiff15": 420.3
  },
  "team2_stats": {
    "roll_winrate": 0.6,
    "roll_golddiff15": 120.1
  }
}
```

## ML Model

### Architecture

- **Model**: Logistic Regression
- **Scaler**: StandardScaler
- **Cross-validation**: TimeSeriesSplit (n_splits=5)
- **Performance**: Accuracy 57.0% / ROC-AUC 0.632

### Why Logistic Regression

The dataset contains ~692 training samples. Complex models (e.g. Gradient Boosting) underperform with this data volume due to overfitting. Logistic Regression is better suited for small datasets and produces interpretable coefficients — directly explainable as feature weights.

### Side-Agnostic Prediction

Predictions are independent of Blue/Red side assignment. Win probability is computed by averaging two directional predictions (team1−team2 and team2−team1), ensuring identical results regardless of input order.

```python
# Both calls return the same team1 win probability
predict_match("T1", "GEN.G")
predict_match("GEN.G", "T1")  # team2_win_rate == above team1_win_rate
```

### Features

The model uses the difference (team1 − team2) of the following rolling team statistics:

| Feature | Description |
|---------|-------------|
| `roll_winrate` | Rolling win rate over last 5 games |
| `roll_golddiff15` | Average gold difference at 15 minutes |
| `roll_firstdragon` | First dragon rate |
| `roll_firstherald` | First herald rate |
| `roll_firsttower` | First tower rate |
| `patch_winrate` | Win rate on the current patch |

### Feature Importance (Logistic Regression Coefficients)

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| Recent Win Rate | +0.358 | Strongest positive signal |
| Gold Diff @15 | +0.135 | Second strongest |
| First Tower | +0.099 | Moderate positive |
| First Dragon | +0.012 | Weak positive |
| Patch Win Rate | +0.012 | Weak positive |
| First Herald | -0.067 | Negative (likely meta-dependent) |

### Data Pipeline

1. Load Oracle's Elixir CSV data (LCK 2025 + 2026)
2. Filter `position == 'team'` and `league == 'LCK'`
3. Compute rolling features with `shift(1)` to prevent data leakage
4. Merge Blue/Red rows per game to create diff features
5. Train Logistic Regression with StandardScaler
6. Save model + scaler + featured data as `.pkl`

## Schedule Data

Match schedule (90 matches, 2026 Spring Split) is stored in Supabase and updated manually per round. Web scraping is intentionally avoided due to legal risk.

**Supabase `schedule` table schema:**

| Column | Type | Description |
|--------|------|-------------|
| `team_1` | text | Team abbreviation |
| `team_2` | text | Team abbreviation |
| `date` | date | Match date |
| `time` | time | Match time |
| `round` | int | Round number |
| `split` | text | Spring / Summer |
| `year` | int | Season year |
| `winner` | text | Winner abbreviation (null if upcoming) |
| `week` | int | Week number |

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

## Versioning Roadmap

| Version | Features | Status |
|---------|----------|--------|
| v1 | Match schedule + ML win prediction | ✅ Complete |
| v2 | RAG + LangChain Agent for team/player Q&A | 🔨 In Progress |
| v3 | Betting point system + user login | 📋 Planned |
| v4 | Model Update | 📋 Planned |