"""Export the existing experiment10 champion without retraining any weights."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.services.player_runtime import FEATURES, ROLES, SCHEMA, PlayerRuntime, envelope
from backend.training.player_two_track import ROOT, STATS, source, prepare
from backend.training.feature_pipeline import canonicalize_team_names
from backend.training.process_lck_data import sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def export(directory):
    research = ROOT/'notebook/experiments/10_player_grouped'
    frozen = joblib.load(research/'frozen_2025.joblib')
    scaler = frozen['preprocessor']['scaler']
    model = envelope({'schema': SCHEMA, 'architecture': 'position_additive', 'features': FEATURES,
                      'roles': ROLES, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                      'median': frozen['preprocessor']['median'].tolist(),
                      'members': [{k: v.numpy().tolist() for k,v in member.items()}
                                  for member in frozen['weights']['position_additive']],
                      'training_years': [2025], 'research_checkpoint_sha256': sha256(research/'frozen_2025.joblib'),
                      'evaluation': {'kind': 'retrospective', 'year': 2026, 'series': 186, 'correct': 118}})
    runtime = PlayerRuntime(model)
    frame, x, series, _ = prepare([2025,2026])
    test = series[series.date.dt.year.eq(2026)]
    q = np.array([runtime.probabilities(x[[row.game_index]], row.best_of)['series_probability'][0]
                  for row in test.itertuples()])
    original = pd.read_csv(research/'predictions_2026.csv')
    original = original[original.track.eq('position_additive')].set_index('series_key').loc[test.series_key]
    np.testing.assert_allclose(q, original.series_probability, atol=1e-6, rtol=0)
    assert int(((q >= .5) == test.result.to_numpy()).sum()) == 118
    columns = ['gameid','date','league','position','datacompleteness','playerid','playername','teamname',
               'result','golddiffat15','xpdiffat15','csdiffat15','kills','deaths','assists','dpm',
               'damageshare','earnedgoldshare','cspm','visionscore','gamelength']
    records = []
    for year in [2025,2026]:
        d = pd.read_csv(source(year), usecols=columns, low_memory=False)
        records.append(d[d.league.isin(['LCK', 'LCKC']) & d.position.isin(ROLES) & d.datacompleteness.eq('complete')])
    d = canonicalize_team_names(pd.concat(records, ignore_index=True))
    d['date'] = pd.to_datetime(d.date, utc=True)
    assert d.playerid.notna().all() and not d.duplicated(['gameid','playerid']).any()
    assert d.groupby('gameid').size().eq(10).all() and d.gamelength.gt(0).all()
    for stat in ['kills','deaths','assists']:
        d[stat+'_per_min'] = d[stat] / (d.gamelength/60)
    d['vision_per_min'] = d.visionscore / (d.gamelength/60)
    assert np.isfinite(d[STATS]).all().all()
    d = d.sort_values(['date','gameid'])
    players = {}
    for player_id, g in d.groupby('playerid'):
        # Preserve the champion's LCK history; CL-only substitutes use CL history.
        g = g[g.league.eq('LCK')] if g.league.eq('LCK').any() else g
        last = g.iloc[-1]
        players[player_id] = {'name': last.playername, 'team': last.teamname, 'role': last.position,
                              'last_game_at': last.date.isoformat(), 'history_league': str(last.league),
                              'features': g[STATS].tail(5).mean().tolist() + [min(len(g),5)/5]}
    teams = {}
    active = d.date.max() - pd.Timedelta(days=180)
    for team, g in d[d.league.eq('LCK')].groupby('teamname'):
        if g.date.max() < active:
            continue
        latest = g[g.gameid.eq(g.iloc[-1].gameid)].set_index('position')
        assert set(latest.index) == set(ROLES)
        teams[team] = {'roster': {r: latest.loc[r,'playerid'] for r in ROLES},
                       'last_seen_at': g.date.max().isoformat(), 'status': 'last_observed_not_confirmed'}
    state = envelope({'schema': SCHEMA, 'observed_at': datetime.now(timezone.utc).isoformat(),
                      'last_game_at': d.date.max().isoformat(), 'players': players, 'teams': teams,
                      'source_sha256': {str(y): sha256(source(y)) for y in [2025,2026]}})
    write(directory/'model.json', model)
    write(directory/'players.json', state)
    print(json.dumps({'model_version': model['sha256'], 'players': len(players), 'teams': len(teams),
                      'parity_series': len(q), 'correct': 118, 'max_error': float(np.abs(q-original.series_probability).max())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'backend/models/player_a')
    export(parser.parse_args().output)
