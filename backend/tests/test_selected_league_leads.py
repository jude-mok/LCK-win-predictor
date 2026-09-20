import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from backend.training.selected_league_leads import build_data,LeadNet,Scale
from backend.training.player_two_track import ROLES

class SelectedLeadTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for i,(gold,win) in enumerate([(1000,1),(200,0),(-100,1),(5000,0)]):
            for j,team in enumerate(['A','B']):
                for role in ROLES+['team']:
                    rows.append({'gameid':f'g{i}','date':f'2025-01-0{i+1} 10:00:00','league':'LCK',
                        'position':role,'datacompleteness':'complete','playerid':f'{team}-{role}' if role!='team' else None,
                        'teamname':team,'side':'Blue' if j==0 else 'Red','result':win if j==0 else 1-win,
                        'golddiffat15':gold*(1 if j==0 else -1)/(1 if role=='team' else 5),
                        'xpdiffat15':10 if j==0 else -10,'csdiffat15':5 if j==0 else -5,
                        'kills':3,'deaths':2,'assists':4,'dpm':300,'damageshare':.2,'earnedgoldshare':.2,
                        'cspm':5,'visionscore':30,'gamelength':1800})
        return pd.DataFrame(rows)

    def test_past_lead_conversion_and_no_current_game_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.csv';raw=self.fixture();raw.to_csv(path,index=False)
            with patch('backend.training.selected_league_leads.source',return_value=path):
                frame,x,t,_=build_data([2025],['LCK'])
                idx=int(frame.index[frame.gameid.eq('g3')][0])
                np.testing.assert_allclose(t[idx,0],[1100/3,2/3,1/3,3/5,.5])
                np.testing.assert_allclose(x[idx,0,:,13],np.full(5,2/3))
                raw.loc[raw.gameid.eq('g3'),'result']=1-raw.loc[raw.gameid.eq('g3'),'result']
                raw.loc[raw.gameid.eq('g3'),['golddiffat15','xpdiffat15','csdiffat15']]=-99999
                raw.to_csv(path,index=False)
                frame2,x2,t2,_=build_data([2025],['LCK'])
                np.testing.assert_allclose(x[idx],x2[idx],equal_nan=True)
                np.testing.assert_allclose(t[idx],t2[idx],equal_nan=True)

    def test_team_swap_and_shared_team_imputation(self):
        torch.manual_seed(42);net=LeadNet(16,5)
        x=torch.randn(4,2,5,16);t=torch.randn(4,2,5)
        torch.testing.assert_close(net(x,t),-net(x.flip(1),t.flip(1)))
        rng=np.random.default_rng(42);train=rng.normal(size=(4,2,5));scale=Scale().fit(train)
        self.assertEqual(scale.median.shape,(5,))
        missing=np.full((1,2,5),np.nan);z=scale.transform(missing)
        torch.testing.assert_close(z[:,0],z[:,1])

if __name__=='__main__':unittest.main()
