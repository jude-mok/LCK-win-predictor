import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers.player import router as player_router
from .routers.schedule import router as schedule_router
from .services.player_runtime import PlayerService, load_document
from .services.snapshots import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]


def create_app(artifact_dir=None, database_path=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.error = None
        try:
            artifacts = Path(artifact_dir or os.getenv('PLAYER_ARTIFACT_DIR', ROOT / 'models/player_a'))
            path = database_path or os.getenv('PREDICTION_DB_PATH')
            app.state.player = PlayerService(load_document(artifacts / 'model.json'), load_document(artifacts / 'players.json'))
            app.state.snapshots = (None if not path and os.getenv('APP_ENV') == 'production'
                                   else SnapshotStore(path or ROOT / 'data/runtime/predictions.sqlite3'))
        except Exception:
            logging.exception('Prediction service initialization failed')
            app.state.error = 'Prediction service initialization failed'
        yield

    app = FastAPI(title='LCK Player A', version='2.0.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware,
                       allow_origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(','),
                       allow_credentials=False, allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
    app.include_router(player_router)
    app.include_router(schedule_router)

    @app.get('/health')
    def health():
        return {'status': 'alive'}

    @app.get('/ready')
    def ready():
        try:
            if app.state.error or (app.state.snapshots is not None and not app.state.snapshots.ready()):
                raise ValueError('Not ready')
            return {'status': 'ready', 'model_version': app.state.player.runtime.version,
                    'player_state_version': app.state.player.state_version,
                    'last_game_at': app.state.player.state['last_game_at']}
        except Exception:
            return JSONResponse({'status': 'not_ready'}, status_code=503)

    return app


app = create_app()
