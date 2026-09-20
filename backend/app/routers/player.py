"""Simulations are transient; only authenticated, pre-match publication is stored."""
import hmac
import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Query, Request

from ..schemas.player import PublishRequest, SimulationRequest, TeamPredictionRequest
from ..services.player_runtime import PlayerRuntime, digest, utc
from ..services.snapshots import Conflict
from ..services.rosters import catalog, resolve

router = APIRouter(prefix='/predict', tags=['player A'])


def components(request):
    if request.app.state.error:
        raise HTTPException(503, 'Prediction service is not ready')
    return request.app.state.player, request.app.state.snapshots


def public(document):
    return {k: v for k, v in document.items() if k not in ('model', 'raw_features', 'request_hash')}


@router.get('/model')
def model(request: Request):
    service, _ = components(request)
    return {'model_version': service.runtime.version, 'feature_schema': service.state['schema'],
            'architecture': 'position_additive', 'members': 3, 'parameters_per_member': 213,
            'data_observed_at': service.state['observed_at'], 'last_game_at': service.state['last_game_at'],
            'evaluation': service.runtime.document['payload']['evaluation']}


@router.get('/players')
def players(request: Request):
    service, _ = components(request)
    return {'teams': service.state['teams'], 'players': {key: {k: v for k, v in p.items() if k != 'features'}
            for key, p in service.state['players'].items()}, 'last_game_at': service.state['last_game_at']}


@router.post('/simulate')
def simulate(body: SimulationRequest, request: Request):
    service, _ = components(request)
    try:
        result = service.predict(body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {**public(result), 'kind': 'simulation', 'saved': False}


@router.get('/rosters')
def rosters(request: Request):
    service, _ = components(request)
    return {'teams': catalog(service), 'last_game_at': service.state['last_game_at']}


@router.post('/predict')
def predict_teams(body: TeamPredictionRequest, request: Request):
    service, _ = components(request)
    teams = catalog(service)
    resolved = resolve(body, teams)
    try:
        result = service.predict(resolved)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    for roster in result['rosters']:
        for player in roster:
            record = next(p for t in teams.values() for p in t['players'] if p['player_id'] == player['player_id'])
            player.update(name=record['player_name'], roster_status=record['roster_status'], history_league=record['history_league'])
    return {**public(result), 'kind': 'current_roster', 'saved': False,
            'uses_cl_history': any(p['history_league'] == 'LCKC' for r in result['rosters'] for p in r)}


@router.get('/snapshots')
def snapshots(request: Request, series_ids: str = Query('', max_length=10000)):
    _, store = components(request)
    if store is None:
        raise HTTPException(503, 'Snapshot storage is not configured')
    ids = list(dict.fromkeys(filter(None, series_ids.split(','))))
    if len(ids) > 100:
        raise HTTPException(422, 'At most 100 series per request')
    return {key: public(value) for key, value in store.many(ids).items()}


@router.get('/snapshots/{series_id}')
def snapshot(series_id: str, request: Request):
    _, store = components(request)
    if store is None:
        raise HTTPException(503, 'Snapshot storage is not configured')
    document = store.get(series_id)
    if document is None:
        raise HTTPException(404, 'No pre-match prediction was recorded')
    return public(document)


@router.get('/snapshots/{series_id}/replay')
def replay(series_id: str, request: Request):
    _, store = components(request)
    if store is None:
        raise HTTPException(503, 'Snapshot storage is not configured')
    document = store.get(series_id)
    if document is None:
        raise HTTPException(404, 'No pre-match prediction was recorded')
    result = PlayerRuntime(document['model']).probabilities([document['raw_features']], document['best_of'])
    q = float(result['series_probability'][0])
    return {'series_id': series_id, 'model_version': document['model_version'], 'team1_win_rate': q,
            'matches_saved_prediction': abs(q - document['team1_win_rate']) < 1e-6}


@router.post('/snapshots')
def publish(body: PublishRequest, request: Request, x_publish_key: str = Header('')):
    expected = os.getenv('PREDICTION_PUBLISH_KEY', '')
    if not expected or not hmac.compare_digest(x_publish_key.encode(), expected.encode()):
        raise HTTPException(401, 'Publisher credentials required')
    service, store = components(request)
    if store is None:
        raise HTTPException(503, 'Snapshot storage is not configured')
    request_hash = digest(body.model_dump(mode='json'))
    existing = store.get(body.series_id)
    if existing:
        if existing['request_hash'] != request_hash:
            raise HTTPException(409, 'A different prediction already exists for this series')
        return public(existing)
    now = datetime.now(timezone.utc)
    if utc(body.scheduled_start) <= now or utc(body.roster_confirmed_at) > now:
        raise HTTPException(422, 'Publish before the declared match start, after roster confirmation')
    try:
        result = service.predict(body, now)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    document = {**result, 'series_id': body.series_id, 'prediction_id': str(uuid4()),
                'kind': 'prematch', 'predicted_at': now.isoformat(),
                'scheduled_start': utc(body.scheduled_start).isoformat(),
                'roster_confirmed_at': utc(body.roster_confirmed_at).isoformat(),
                'fixture_source': 'trusted_publisher_declaration',
                'request_hash': request_hash, 'model': service.runtime.document}
    try:
        return public(store.insert(document))
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
