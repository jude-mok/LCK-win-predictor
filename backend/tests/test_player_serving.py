from contextlib import closing
import copy
import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import ROOT, create_app
from backend.app.services.player_runtime import PlayerRuntime, digest, envelope, load_document
from backend.app.services.snapshots import SnapshotStore


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.model = load_document(ROOT / 'models/player_a/model.json')
        state = load_document(ROOT / 'models/player_a/players.json')['payload']
        # Stable synthetic availability dates; no dependence on test execution date.
        state['observed_at'] = state['last_game_at'] = '2020-01-01T00:00:00+00:00'
        for player in state['players'].values():
            player['last_game_at'] = state['last_game_at']
        (self.path / 'model.json').write_text(json.dumps(self.model))
        (self.path / 'players.json').write_text(json.dumps(envelope(state)))
        self.db = self.path / 'snapshots.sqlite3'
        self.app = create_app(self.path, self.db)
        self.client = self.enterContext(TestClient(self.app))
        self.enterContext(patch.dict(os.environ, {'PREDICTION_PUBLISH_KEY': 'test-secret'}))
        self.auth = {'X-Publish-Key': 'test-secret'}
        now = datetime.now(timezone.utc)
        self.body = {'team1': 'T1', 'team2': 'Gen.G', 'best_of': 3,
                     'team1_roster': state['teams']['T1']['roster'], 'team2_roster': state['teams']['Gen.G']['roster'],
                     'series_id': '123', 'scheduled_start': (now + timedelta(days=1)).isoformat(),
                     'roster_confirmed_at': (now - timedelta(minutes=1)).isoformat()}

    def publish(self, body=None):
        return self.client.post('/predict/snapshots', json=body or self.body, headers=self.auth)

    def test_prediction_is_immutable_idempotent_and_survives_restart(self):
        first = self.publish()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json(), self.publish().json())
        self.assertEqual(self.publish({**self.body, 'best_of': 5}).status_code, 409)
        self.assertNotIn('model', first.json())
        self.assertNotIn('raw_features', first.json())
        with TestClient(create_app(self.path, self.db)) as second:
            self.assertEqual(second.get('/predict/snapshots/123').json(), first.json())
        with closing(sqlite3.connect(self.db)) as db:
            for statement in ['DELETE FROM predictions', "UPDATE predictions SET request_hash = 'bad'"]:
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute(statement)

    def test_authorization_and_temporal_rules(self):
        self.assertEqual(self.client.post('/predict/snapshots', json=self.body).status_code, 401)
        self.assertEqual(self.publish({**self.body, 'scheduled_start': '2020-01-01T00:00:00Z'}).status_code, 422)
        self.assertEqual(self.publish({**self.body, 'roster_confirmed_at': '2099-01-01T00:00:00Z'}).status_code, 422)
        self.assertEqual(self.publish({**self.body, 'scheduled_start': '2099-01-01T00:00:00'}).status_code, 422)
        self.app.state.player.state['observed_at'] = '2099-01-01T00:00:00Z'
        self.assertEqual(self.publish().status_code, 422)

    def test_retry_after_start_returns_original_without_inference(self):
        first = self.publish().json()
        with patch('backend.app.routers.player.datetime') as clock:
            clock.now.return_value = datetime(2099, 1, 1, tzinfo=timezone.utc)
            with patch.object(self.app.state.player, 'predict', side_effect=AssertionError('Must not recompute')):
                self.assertEqual(first, self.publish().json())

    def test_simulation_does_not_store_and_contract_rejects_invalid_rosters(self):
        body = {k: v for k, v in self.body.items() if k not in ('series_id', 'scheduled_start', 'roster_confirmed_at')}
        result = self.client.post('/predict/simulate', json=body)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertFalse(result.json()['saved'])
        self.assertEqual(self.client.get('/predict/snapshots?series_ids=123').json(), {})
        body['team2_roster'] = body['team1_roster']
        self.assertEqual(self.client.post('/predict/simulate', json=body).status_code, 422)
        self.assertEqual(self.client.post('/predict/predict', json={}).status_code, 422)

    def test_replay_uses_saved_model_and_read_never_runs_live_inference(self):
        original = self.publish().json()
        with patch.object(self.app.state.player, 'predict', side_effect=AssertionError('Must not recompute')):
            self.assertEqual(self.client.get('/predict/snapshots/123').json(), original)
            self.assertEqual(self.client.get('/predict/snapshots?series_ids=123,missing').json(), {'123': original})
        self.app.state.player.runtime = None  # Simulate replacing/removing the active model.
        replay = self.client.get('/predict/snapshots/123/replay').json()
        self.assertTrue(replay['matches_saved_prediction'])
        self.assertEqual(replay['team1_win_rate'], original['team1_win_rate'])

    def test_concurrent_publish_commits_one_record(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.publish(), range(6)))
        self.assertTrue(all(r.status_code == 200 for r in results))
        self.assertEqual(len({r.json()['prediction_id'] for r in results}), 1)

    def test_corrupt_artifact_fails_readiness_not_liveness(self):
        broken = copy.deepcopy(self.model)
        broken['payload']['scale'][0] = -1
        (self.path / 'model.json').write_text(json.dumps(broken))
        with self.assertLogs(level='ERROR'):
            with TestClient(create_app(self.path, self.db)) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                self.assertEqual(client.get('/ready').status_code, 503)
                self.assertEqual(client.get('/predict/players').status_code, 503)

    def test_model_schema_and_swap_symmetry(self):
        runtime = PlayerRuntime(self.model)
        raw = np.random.default_rng(12).normal(size=(9, 2, 5, 13))
        raw[0, 0, 0, 0] = np.nan
        for best_of in (3, 5):
            a = runtime.probabilities(raw, best_of)['series_probability']
            b = runtime.probabilities(raw[:, ::-1], best_of)['series_probability']
            np.testing.assert_allclose(a + b, 1, atol=1e-6)
        broken = copy.deepcopy(self.model['payload'])
        broken['features'].reverse()
        with self.assertRaises(ValueError):
            PlayerRuntime(envelope(broken))

    def test_runtime_matches_independent_scalar_forward_pass(self):
        # Scalar Python implementation is independent of vectorization/broadcasting.
        import math
        runtime = PlayerRuntime(self.model)
        payload = self.model['payload']
        raw = np.random.default_rng(42).normal(size=(2,5,13)) * 100
        def linear(x, w, b):
            return [sum(a * b for a, b in zip(row, x)) + bias for row, bias in zip(w, b)]
        votes = []
        for member in payload['members']:
            scores = []
            for team in raw:
                score = 0
                for role, player in enumerate(team):
                    x = [(value - mean) / scale for value, mean, scale in zip(player, payload['mean'], payload['scale'])]
                    for prefix in ('player.0', 'player.2', f'heads.{role}.0'):
                        x = [max(0, v) for v in linear(x, member[prefix+'.weight'], member[prefix+'.bias'])]
                    prefix = f'heads.{role}.2'
                    score += linear(x, member[prefix+'.weight'], member[prefix+'.bias'])[0]
                scores.append(score)
            votes.append(1 / (1 + math.exp(scores[1] - scores[0])))
        p = sum(votes) / 3
        result = runtime.probabilities([raw], 3)
        self.assertAlmostEqual(float(result['set_probability'][0]), p, places=6)
        self.assertAlmostEqual(float(result['series_probability'][0]), 3*p*p-2*p*p*p, places=6)


if __name__ == '__main__':
    unittest.main()
