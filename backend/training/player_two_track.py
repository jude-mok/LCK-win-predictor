"""Small hierarchical player models; 2025-only fitting, frozen 2026 replay."""
from pathlib import Path
from copy import deepcopy
import json,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from backend.training.feature_pipeline import canonicalize_team_names,add_series_keys,build_featured_games,build_match_dataset
from backend.training.process_lck_data import load_many_lck_team_games
from backend.training.train_model import evaluate

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'notebook/experiments/09_player_two_track'
ROLES=['top','jng','mid','bot','sup']
STATS=['result','golddiffat15','xpdiffat15','csdiffat15','kills_per_min','deaths_per_min','assists_per_min','dpm','damageshare','earnedgoldshare','cspm','vision_per_min']
INPUTS=['past_'+c for c in STATS]+['history_fraction']
SEEDS=[42,43,44]


def source(year):
    return ROOT/'backend/data/raw'/f'{year}_LoL_esports_match_data_from_OraclesElixir.csv'


def prepare(years):
    raw=[]
    cols=['gameid','date','league','year','split','position','datacompleteness','playerid','playername','teamname','side','result','golddiffat15','xpdiffat15','csdiffat15','kills','deaths','assists','dpm','damageshare','earnedgoldshare','cspm','visionscore','gamelength']
    for year in years:
        d=pd.read_csv(source(year),usecols=cols,low_memory=False)
        raw.append(d[d.league.eq('LCK')&d.position.isin(ROLES)&d.datacompleteness.eq('complete')].copy())
    d=canonicalize_team_names(pd.concat(raw,ignore_index=True))
    d['date']=pd.to_datetime(d.date)
    assert d.playerid.notna().all() and not d.duplicated(['gameid','playerid']).any()
    assert d.groupby('gameid').size().eq(10).all()
    assert (d.gamelength>0).all()
    for stat in ['kills','deaths','assists']:
        d[stat+'_per_min']=d[stat]/(d.gamelength/60)
    d['vision_per_min']=d.visionscore/(d.gamelength/60)
    assert np.isfinite(d[STATS]).all().all()
    d=d.sort_values(['playerid','date','gameid']).reset_index(drop=True)
    for stat in STATS:
        d['past_'+stat]=d.groupby('playerid')[stat].transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
    d['history_fraction']=d.groupby('playerid').cumcount().clip(upper=5)/5
    teams=load_many_lck_team_games([source(y) for y in years])
    games=build_featured_games(teams)
    matches=build_match_dataset(games)
    keys=games[['gameid','series_key']].drop_duplicates().set_index('gameid').series_key
    d['series_key']=d.gameid.map(keys)
    metadata=[];arrays=[];rosters=[]
    for gameid,g in d.groupby('gameid',sort=False):
        names=sorted(g.teamname.unique());assert len(names)==2
        arr=[];players=[]
        for name in names:
            t=g[g.teamname.eq(name)].set_index('position')
            assert len(t)==5 and set(t.index)==set(ROLES)
            arr.append(t.loc[ROLES,INPUTS].to_numpy(float))
            players.append(t.loc[ROLES,'playerid'].tolist())
        arrays.append(np.stack(arr));rosters.append(players)
        metadata.append({'gameid':gameid,'series_key':g.series_key.iloc[0],'date':g.date.min(),
            'team1':names[0],'team2':names[1],'result':int(g[g.teamname.eq(names[0])].result.iloc[0])})
    frame=pd.DataFrame(metadata);order=frame.sort_values(['date','gameid']).index.to_numpy()
    frame=frame.iloc[order].reset_index(drop=True)
    x=np.stack(arrays)[order];rosters=np.array(rosters)[order]
    # Only complete series are eligible, for both training games and series evaluation.
    mask=frame.series_key.isin(matches.series_key).to_numpy()
    frame=frame[mask].reset_index(drop=True);x=x[mask];rosters=rosters[mask]
    first_indices=frame.reset_index().sort_values(['date','gameid']).drop_duplicates('series_key').set_index('series_key')['index']
    matches['game_index']=matches.series_key.map(first_indices).astype(int)
    assert frame.iloc[matches.game_index].team1.tolist()==matches.team1.tolist()
    return frame,x,matches,rosters


class PlayerNetwork(nn.Module):
    def __init__(self,track):
        super().__init__();self.track=track
        self.player=nn.Sequential(nn.Linear(len(INPUTS),8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
        if track=='position_additive':
            self.heads=nn.ModuleList([nn.Sequential(nn.Linear(4,2),nn.ReLU(),nn.Linear(2,1)) for _ in ROLES])
        elif track=='cross_position':
            self.team=nn.Sequential(nn.Linear(20,4),nn.ReLU(),nn.Linear(4,1))
        elif track=='upper_lower':
            self.upper=nn.Sequential(nn.Linear(12,2),nn.ReLU())
            self.lower=nn.Sequential(nn.Linear(12,2),nn.ReLU())
            self.team=nn.Sequential(nn.Linear(4,4),nn.ReLU(),nn.Linear(4,1))
        else:
            raise ValueError(f'Unknown track: {track}')
    def forward(self,x):
        # x: [games, teams=2, roles=5, statistics=13]. Shared encoder for every player.
        z=self.player(x)
        if self.track=='position_additive':
            score=sum(head(z[:,:,i,:]).squeeze(-1) for i,head in enumerate(self.heads))
        elif self.track=='upper_lower':
            upper=self.upper(z[:,:,[0,1,2],:].flatten(start_dim=2))
            lower=self.lower(z[:,:,[3,4,1],:].flatten(start_dim=2))
            score=self.team(torch.cat([upper,lower],dim=-1)).squeeze(-1)
        else:
            score=self.team(z.flatten(start_dim=2)).squeeze(-1)
        return score[:,0]-score[:,1]


class Preprocessor:
    def fit(self,x):
        # Role-specific cold-start values estimated only from fitting-period history.
        self.median=np.nanmedian(x,axis=(0,1))
        self.median=np.nan_to_num(self.median,nan=0.)
        filled=np.where(np.isnan(x),self.median,x)
        self.scaler=StandardScaler().fit(filled.reshape(-1,len(INPUTS)))
        return self
    def transform(self,x):
        filled=np.where(np.isnan(x),self.median,x)
        a=self.scaler.transform(filled.reshape(-1,len(INPUTS))).reshape(x.shape)
        assert np.isfinite(a).all()
        return torch.tensor(a,dtype=torch.float32)


def series_probability(p,best_of):
    # Independent, identically distributed sets; first to 2 or 3 wins.
    return np.where(np.asarray(best_of)==3,3*p**2-2*p**3,10*p**3-15*p**4+6*p**5)


def train_one(track,seed,x,y,epochs=500,validation=None):
    torch.manual_seed(seed);net=PlayerNetwork(track)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    history=[];best=float('inf');saved=None;chosen=epochs;stale=0
    for epoch in range(1,epochs+1):
        net.train();opt.zero_grad()
        loss=nn.functional.binary_cross_entropy_with_logits(net(x),y)
        loss.backward();opt.step()
        if validation is not None:
            vx,vy,bo=validation;net.eval()
            with torch.no_grad():p=torch.sigmoid(net(vx)).numpy()
            q=series_probability(p,bo)
            vl=evaluate(pd.Series(vy),q)['log_loss']
            history.append({'epoch':epoch,'train_loss':loss.item(),'validation_series_log_loss':vl})
            if vl<best:
                best=vl;saved=deepcopy(net.state_dict());chosen=epoch;stale=0
            else:stale+=1
            if stale>=30:break
    if saved is not None:net.load_state_dict(saved)
    net.eval()
    return net,chosen,history


def run(tracks=None,output_dir=None):
    OUT=Path(output_dir) if output_dir is not None else ROOT/'notebook/experiments/09_player_two_track'
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'train_year':2025,'evaluation_year':2026,'tracks':tracks or ['position_additive','cross_position'],
        'inputs':INPUTS,'roles':ROLES,'window':5,'seed_ensemble':SEEDS,'architecture':'shared 13→8→4 player encoder; A role-wise 4→2→1 sum; B joint 20→4→1; C upper(top,jng,mid) 12→2 and lower(bot,sup,jng) 12→2 concatenated into 4→4→1' ,
        'training_target':'individual set win, one sample per set; series grouped in train/validation',
        'selection':'last 20% of 2025 series dates for epoch selection by series log loss; max500 patience30',
        'refit':'all 2025 sets for selected per-seed epochs; scaler fit only 2025',
        'series_assumption':'first-set announced roster available; identical independent set win probabilities; no later drafts/roster substitutions',
        'warning':'2026 previously inspected. Retrospective replay, not untouched test. best_of inferred from completed series scores.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    frame,x,matches,rosters=prepare([2025]);days=np.sort(matches.date.dt.normalize().unique());cut=int(len(days)*.8)
    fit_series=matches[matches.date.dt.normalize().isin(days[:cut])]
    val_series=matches[matches.date.dt.normalize().isin(days[cut:])]
    mask=frame.series_key.isin(fit_series.series_key).to_numpy()
    assert frame[mask].date.max()<val_series.date.min()
    prep=Preprocessor().fit(x[mask]);xf=prep.transform(x[mask]);yf=torch.tensor(frame.loc[mask,'result'].to_numpy(),dtype=torch.float32)
    xv=prep.transform(x[val_series.game_index]);validation=(xv,val_series.result.to_numpy(),val_series.best_of.to_numpy())
    epochs={};history=[];models={};validation_results=[]
    for track in protocol['tracks']:
        epochs[track]={};vp=[]
        for seed in SEEDS:
            net,epoch,h=train_one(track,seed,xf,yf,validation=validation)
            epochs[track][seed]=epoch
            history.extend({'track':track,'seed':seed,**r} for r in h)
            with torch.no_grad():vp.append(torch.sigmoid(net(xv)).numpy())
            print(track,seed,'2025 epoch:',epoch,flush=True)
        validation_results.append({'track':track,'n':len(val_series),**evaluate(val_series.result,series_probability(np.mean(vp,axis=0),val_series.best_of))})
    pd.DataFrame(history).to_csv(OUT/'history_2025.csv',index=False)
    pd.DataFrame(validation_results).to_csv(OUT/'validation_2025.csv',index=False)
    prep_all=Preprocessor().fit(x);xa=prep_all.transform(x);ya=torch.tensor(frame.result.to_numpy(),dtype=torch.float32)
    for track in protocol['tracks']:
        models[track]=[train_one(track,s,xa,ya,epochs=epochs[track][s])[0] for s in SEEDS]
    lock={'epochs':epochs,'train_sets':len(frame),'train_series':len(matches),'fit_sets':int(mask.sum()),'validation_series':len(val_series),
          'parameter_counts':{t:sum(p.numel() for p in models[t][0].parameters()) for t in models},
          'primary_comparison':'2026 series accuracy; log loss secondary. All tracks locked before reading 2026.'}
    (OUT/'lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    import joblib
    joblib.dump({'preprocessor':{'median':prep_all.median,'scaler':prep_all.scaler},'weights':{t:[m.state_dict() for m in ns] for t,ns in models.items()},'lock':lock},OUT/'frozen_2025.joblib')
    frozen_hash=hashlib.sha256((OUT/'frozen_2025.joblib').read_bytes()).hexdigest()
    full,xx,ss,rr=prepare([2025,2026])
    year25=full.date.dt.year.eq(2025).to_numpy()
    np.testing.assert_allclose(x,xx[year25],equal_nan=True,rtol=0,atol=0)
    pd.testing.assert_frame_equal(frame,full[year25].reset_index(drop=True))
    game_mask=full.date.dt.year.eq(2026).to_numpy();test_games=full[game_mask]
    test_series=ss[ss.date.dt.year.eq(2026)]
    xt=prep_all.transform(xx[game_mask]);xs=prep_all.transform(xx[test_series.game_index])
    summaries=[];predictions=[];monthly=[];seed_scores=[]
    for track,nets in models.items():
        gp=[];sp=[]
        with torch.no_grad():
            for seed,net in zip(SEEDS,nets):
                g=torch.sigmoid(net(xt)).numpy();p=torch.sigmoid(net(xs)).numpy()
                np.testing.assert_allclose(torch.sigmoid(net(xs.flip(1))).numpy()+p,1,atol=1e-6)
                gp.append(g);sp.append(p)
                q=series_probability(p,test_series.best_of.to_numpy())
                seed_scores.append({'track':track,'seed':seed,**evaluate(test_series.result,q)})
        g=np.mean(gp,axis=0);p=np.mean(sp,axis=0);q=series_probability(p,test_series.best_of.to_numpy())
        for level,truth,prob in [('set',test_games.result,g),('series',test_series.result,q)]:
            summaries.append({'track':track,'level':level,'n':len(truth),'correct':int(((prob>=.5)==truth.to_numpy()).sum()),**evaluate(truth,prob)})
        for row,ps,pseries in zip(test_series.itertuples(),p,q):
            predictions.append({'track':track,'series_key':row.series_key,'date':row.date,'team1':row.team1,'team2':row.team2,'best_of':row.best_of,'y':row.result,'set_probability':float(ps),'series_probability':float(pseries)})
    pred=pd.DataFrame(predictions)
    for (track,month),g in pred.groupby(['track',pred.date.dt.to_period('M')]):
        monthly.append({'track':track,'month':str(month),'n':len(g),**evaluate(g.y,g.series_probability)})
    # Baselines fixed from 2025 only. Always50 accuracy depends on tie convention.
    for name,p in [('always_50',.5),('2025_set_winrate',float(frame.result.mean()))]:
        q=series_probability(np.full(len(test_series),p),test_series.best_of.to_numpy())
        summaries.append({'track':name,'level':'series','n':len(q),'correct':int(((q>=.5)==test_series.result.to_numpy()).sum()),**evaluate(test_series.result,q)})
    # Some rosters may be new in 2026: count players with zero preceding history.
    first_x=xx[test_series.game_index]
    audit={'cold_start_player_slots':int((first_x[:,:,:,-1]==0).sum()),'series_player_slots':int(len(first_x)*10),
           'checks':['2025 inputs identical when 2026 appended','team swap probabilities sum to one','frozen checkpoint unchanged'],
           'first_set_roster_assumption':True,'checkpoint_sha256':frozen_hash,
           'source_sha256':{str(y):hashlib.sha256(source(y).read_bytes()).hexdigest() for y in [2025,2026]}}
    assert hashlib.sha256((OUT/'frozen_2025.joblib').read_bytes()).hexdigest()==frozen_hash
    for name,data in [('evaluation_2026',summaries),('monthly_2026',monthly),('seed_scores_2026',seed_scores)]:pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False)
    pred.to_csv(OUT/'predictions_2026.csv',index=False)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(lock),flush=True)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    return pd.DataFrame(summaries)

if __name__=='__main__':run()
