import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from backend.app.main import ROOT, create_app


class DBRosterTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.client = self.enterContext(TestClient(create_app(database_path=Path(self.temp)/'test.db')))
        with (ROOT/'tests/fixtures/db_rosters.json').open() as source:
            self.rows = json.load(source)
        db = MagicMock()
        db.table.return_value.select.return_value.execute.return_value.data = self.rows
        self.enterContext(patch('backend.app.services.rosters.get_supabase', return_value=db))

    def predict(self, **extra):
        return self.client.post('/predict/predict', json={'team1':'T1','team2':'GEN',**extra})

    def test_main_default_and_substitution(self):
        initial = self.predict()
        self.assertEqual(initial.status_code,200,initial.text)
        expected = {p['player_id'] for p in self.rows if p['team_code']=='T1' and p['roster_status']=='Main'}
        self.assertEqual({p['player_id'] for p in initial.json()['rosters'][0]},expected)
        sub = next(p for p in self.rows if p['team_code']=='T1' and p['position']=='mid' and p['roster_status']=='sub')
        result = self.predict(team1_roster={'mid':sub['player_id']})
        self.assertEqual(result.status_code,200,result.text)
        self.assertEqual(result.json()['rosters'][0][2]['player_id'],sub['player_id'])
        self.assertNotEqual(result.json()['team1_win_rate'],initial.json()['team1_win_rate'])
        self.assertEqual(self.predict().json()['team1_win_rate'],initial.json()['team1_win_rate'])

    def test_live_db_main_change_and_validation(self):
        main = next(p for p in self.rows if p['team_code']=='T1' and p['position']=='mid' and p['roster_status']=='Main')
        sub = next(p for p in self.rows if p['team_code']=='T1' and p['position']=='mid' and p['roster_status']=='sub')
        main['roster_status'],sub['roster_status']='sub','Main'
        self.assertEqual(self.predict().json()['rosters'][0][2]['player_id'],sub['player_id'])
        main['roster_status']='Main'
        self.assertEqual(self.predict().status_code,503)
        main['roster_status']='sub'
        foreign = next(p for p in self.rows if p['team_code']=='GEN')
        self.assertEqual(self.predict(team1_roster={'top':foreign['player_id']}).status_code,422)
        self.assertEqual(self.predict(team1_roster={'top':sub['player_id']}).status_code,422)

    def test_every_registered_player_has_history(self):
        response = self.client.get('/predict/rosters')
        self.assertEqual(response.status_code,200,response.text)
        players = [p for t in response.json()['teams'].values() for p in t['players']]
        self.assertEqual(len(players),116)
        self.assertTrue(all(p['available'] for p in players))

    def test_no_silent_fallback_when_db_unavailable(self):
        with patch('backend.app.services.rosters.get_supabase', side_effect=RuntimeError('offline')):
            self.assertEqual(self.predict().status_code,503)
