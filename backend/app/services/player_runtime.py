"""Validated, portable inference for the frozen player A ensemble (no torch)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROLES = ['top', 'jng', 'mid', 'bot', 'sup']
FEATURES = ['past_result', 'past_golddiffat15', 'past_xpdiffat15', 'past_csdiffat15',
            'past_kills_per_min', 'past_deaths_per_min', 'past_assists_per_min',
            'past_dpm', 'past_damageshare', 'past_earnedgoldshare', 'past_cspm',
            'past_vision_per_min', 'history_fraction']
SCHEMA = 'player-a-13-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def envelope(payload):
    return {'sha256': digest(payload), 'payload': payload}


def load_document(path):
    document = json.loads(Path(path).read_text())
    if digest(document['payload']) != document['sha256']:
        raise ValueError('Artifact checksum mismatch')
    return document


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError('Timezone is required')
    return parsed.astimezone(timezone.utc)


class PlayerRuntime:
    def __init__(self, document):
        if digest(document['payload']) != document['sha256']:
            raise ValueError('Model checksum mismatch')
        p = document['payload']
        if p['schema'] != SCHEMA or p['features'] != FEATURES or p['roles'] != ROLES:
            raise ValueError('Unsupported player feature contract')
        if p['architecture'] != 'position_additive' or len(p['members']) != 3:
            raise ValueError('Expected frozen three-member player A ensemble')
        self.document, self.version = document, document['sha256']
        self.mean = self.array(p['mean'], (13,))
        self.scale = self.array(p['scale'], (13,))
        self.median = self.array(p['median'], (5, 13))
        if (self.scale <= 0).any():
            raise ValueError('Invalid scaler')
        expected = {'player.0.weight': (8,13), 'player.0.bias': (8,),
                    'player.2.weight': (4,8), 'player.2.bias': (4,)}
        for role in range(5):
            expected.update({f'heads.{role}.0.weight': (2,4), f'heads.{role}.0.bias': (2,),
                             f'heads.{role}.2.weight': (1,2), f'heads.{role}.2.bias': (1,)})
        self.members = []
        for member in p['members']:
            if set(member) != set(expected):
                raise ValueError('Invalid weight keys')
            self.members.append({k: self.array(member[k], shape).astype(np.float32) for k, shape in expected.items()})
        # Exercise every layer during readiness, including cold-start imputation.
        probe = self.probabilities(np.zeros((1,2,5,13)), 3)
        if not np.allclose(probe['series_probability'], .5, atol=1e-6):
            raise ValueError('Model symmetry smoke check failed')

    @staticmethod
    def array(value, shape):
        a = np.asarray(value, dtype=np.float64)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'Invalid array, expected {shape}')
        return a

    def probabilities(self, raw, best_of):
        if best_of not in (3,5):
            raise ValueError('best_of must be 3 or 5')
        a = np.asarray(raw, dtype=np.float64)
        if a.ndim != 4 or a.shape[1:] != (2,5,13) or np.isinf(a).any():
            raise ValueError('Expected [batch,2,5,13] finite or missing features')
        x = ((np.where(np.isnan(a), self.median, a) - self.mean) / self.scale).astype(np.float32)
        members = []
        for w in self.members:
            z = np.maximum(x @ w['player.0.weight'].T + w['player.0.bias'], 0)
            z = np.maximum(z @ w['player.2.weight'].T + w['player.2.bias'], 0)
            score = np.zeros((len(x),2), dtype=np.float32)
            for i in range(5):
                h = np.maximum(z[:,:,i,:] @ w[f'heads.{i}.0.weight'].T + w[f'heads.{i}.0.bias'], 0)
                score += (h @ w[f'heads.{i}.2.weight'].T + w[f'heads.{i}.2.bias'])[...,0]
            logit = score[:,0] - score[:,1]
            members.append(np.exp(-np.logaddexp(0, -logit)))
        p = np.mean(members, axis=0)
        q = 3*p**2-2*p**3 if best_of == 3 else 10*p**3-15*p**4+6*p**5
        return {'set_probability': p, 'series_probability': q}


class PlayerService:
    def __init__(self, model, state):
        self.runtime = PlayerRuntime(model)
        if digest(state['payload']) != state['sha256']:
            raise ValueError('Player state checksum mismatch')
        p = state['payload']
        if p['schema'] != SCHEMA:
            raise ValueError('Player state schema mismatch')
        self.state, self.state_version = p, state['sha256']
        if utc(p['last_game_at']) > utc(p['observed_at']):
            raise ValueError('Player data from the future')
        for player in p['players'].values():
            self.runtime.array(player['features'], (13,))
            if utc(player['last_game_at']) > utc(p['last_game_at']):
                raise ValueError('Invalid player history cutoff')

    def predict(self, request, now=None):
        now = now or datetime.now(timezone.utc)
        if utc(self.state['observed_at']) > now or utc(self.state['last_game_at']) >= now:
            raise ValueError('Player state is not available at prediction time')
        raw, rosters = [], []
        for roster in [request.team1_roster, request.team2_roster]:
            players = []
            for role in ROLES:
                player_id = getattr(roster, role)
                if player_id not in self.state['players']:
                    raise ValueError(f'Unknown player: {player_id}')
                players.append(self.state['players'][player_id])
            raw.append([p['features'] for p in players])
            rosters.append([{'role': role, 'player_id': getattr(roster, role), 'name': p['name']}
                            for role, p in zip(ROLES, players)])
        prediction = self.runtime.probabilities([raw], request.best_of)
        q = float(prediction['series_probability'][0])
        return {'team1': request.team1, 'team2': request.team2, 'best_of': request.best_of,
                'team1_win_rate': q, 'team2_win_rate': 1-q,
                'set_probability': float(prediction['set_probability'][0]),
                'predicted_winner': request.team1 if q >= .5 else request.team2,
                'rosters': rosters, 'model_version': self.runtime.version,
                'feature_schema': SCHEMA, 'player_state_version': self.state_version,
                'data_observed_at': self.state['observed_at'], 'last_game_at': self.state['last_game_at'],
                'raw_features': raw}
