"""Selected-league ablation: player lead frequency and team lead conversion.
The experiment keeps the additive architecture and frozen-2026 LCK replay.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from backend.training.player_two_track import ROOT,ROLES,STATS,INPUTS,SEEDS,source,prepare,series_probability
from backend.training.feature_pipeline import canonicalize_team_names
from backend.training.train_model import evaluate

OUT=ROOT/'notebook/experiments/12_selected_league_leads'
ROLE_EXTRA=['past_gold_lead15_rate','past_xp_lead15_rate','past_cs_lead15_rate']
TEAM_LEADS=['team_mean_gold15','team_lead15_rate','team_strong_lead15_rate','team_history_fraction']
TEAM_EXTRA=['team_conversion15']


def build_data(years,leagues):
    frames=[]
    columns=['gameid','date','league','position','datacompleteness','playerid','teamname','side','result','golddiffat15','xpdiffat15','csdiffat15','kills','deaths','assists','dpm','damageshare','earnedgoldshare','cspm','visionscore','gamelength']
    for year in years:
        d=pd.read_csv(source(year),usecols=columns,low_memory=False)
        allowed=leagues if year==2025 else ['LCK']
        frames.append(d[d.league.isin(allowed)].copy())
    raw=canonicalize_team_names(pd.concat(frames,ignore_index=True))
    raw['date']=pd.to_datetime(raw.date)
    raw['_key']=raw.date.dt.year.astype(str)+'|'+raw.league+'|'+raw.gameid.astype(str)
    players=raw[raw.position.isin(ROLES)].copy();teams=raw[raw.position.eq('team')].copy()
    audit=raw.drop_duplicates('_key').set_index('_key')[['league','date']].copy();audit['reason']='eligible'
    bad=players.loc[~players.datacompleteness.eq('complete'),'_key'].unique();audit.loc[bad,'reason']='not_complete'
    bad=players.loc[players.playerid.isna(),'_key'].unique();audit.loc[audit.index.isin(bad)&audit.reason.eq('eligible'),'reason']='missing_player_id'
    nums=['result','golddiffat15','xpdiffat15','csdiffat15','kills','deaths','assists','dpm','damageshare','earnedgoldshare','cspm','visionscore','gamelength']
    bad=players.loc[~np.isfinite(players[nums]).all(axis=1)|players.gamelength.le(0),'_key'].unique()
    audit.loc[audit.index.isin(bad)&audit.reason.eq('eligible'),'reason']='invalid_player_statistics'
    valid=[]
    teamgroups={k:g for k,g in teams.groupby('_key',sort=False)}
    for key,g in players[players._key.isin(audit[audit.reason.eq('eligible')].index)].groupby('_key',sort=False):
        t=teamgroups.get(key,pd.DataFrame());tg=list(g.groupby('teamname'))
        ok=len(g)==10 and g.playerid.nunique()==10 and len(tg)==2
        ok=ok and all(len(v)==5 and set(v.position)==set(ROLES) and v.result.nunique()==1 for _,v in tg)
        ok=ok and g.result.isin([0,1]).all() and g.groupby('teamname').result.first().sum()==1
        ok=ok and len(t)==2 and set(t.teamname)==set(g.teamname)
        if ok:
            ok=np.isfinite(t[['golddiffat15','result']]).all().all() and t.datacompleteness.eq('complete').all()
        if ok:valid.append(key)
        else:audit.loc[key,'reason']='invalid_roster_or_team_data'
    players=players[players._key.isin(valid)].copy();teams=teams[teams._key.isin(valid)].copy()
    # Keep the target LCK input history identical to previous experiments.
    # Non-LCK player histories can cross the selected non-LCK leagues, not excluded leagues.
    players['_pool']=np.where(players.league.eq('LCK'),'LCK','selected_other')
    players=players.sort_values(['_pool','playerid','date','gameid'])
    assert not players.duplicated(['_pool','playerid','date']).any()
    for c in ['kills','deaths','assists']:players[c+'_per_min']=players[c]/(players.gamelength/60)
    players['vision_per_min']=players.visionscore/(players.gamelength/60)
    group=players.groupby(['_pool','playerid'])
    for c in STATS:players['past_'+c]=group[c].transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
    players['history_fraction']=group.cumcount().clip(upper=5)/5
    for name,c in zip(ROLE_EXTRA,['golddiffat15','xpdiffat15','csdiffat15']):
        players['_lead']=players[c].gt(0).astype(float)
        players[name]=players.groupby(['_pool','playerid'])._lead.transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
    teams['_pool']=np.where(teams.league.eq('LCK'),'LCK',teams.league)
    teams=teams.sort_values(['_pool','teamname','date','gameid'])
    teams['_lead']=teams.golddiffat15.gt(0).astype(float)
    teams['_strong']=teams.golddiffat15.ge(1000).astype(float)
    teams['_lead_win']=teams['_lead']*teams.result
    tg=teams.groupby(['_pool','teamname'])
    for name,col in [('team_mean_gold15','golddiffat15'),('team_lead15_rate','_lead'),('team_strong_lead15_rate','_strong')]:
        teams[name]=tg[col].transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
    lead_count=tg['_lead'].transform(lambda s:s.shift(1).rolling(5,min_periods=1).sum()).fillna(0)
    lead_wins=tg['_lead_win'].transform(lambda s:s.shift(1).rolling(5,min_periods=1).sum()).fillna(0)
    # Beta(1,1) smoothing: avoid interpreting 1/1 leads as certain conversion.
    teams['team_conversion15']=(lead_wins+1)/(lead_count+2)
    teams['team_history_fraction']=tg.cumcount().clip(upper=5)/5
    players['_team']=players.groupby('_key').teamname.rank(method='dense').astype(int)-1
    players['_role']=players.position.map({r:i for i,r in enumerate(ROLES)})
    players=players.sort_values(['date','_key','_team','_role'])
    frame=players.drop_duplicates('_key')[['_key','gameid','date','league','teamname','result']].reset_index(drop=True)
    xp=players[INPUTS+ROLE_EXTRA].to_numpy(float).reshape(-1,2,5,len(INPUTS)+3)
    assert (players._key.to_numpy().reshape(-1,10)==frame._key.to_numpy()[:,None]).all()
    tt=teams.set_index(['_key','teamname'])
    teamnames=players.drop_duplicates(['_key','teamname'])[['teamname','_key']]
    xt=tt.reindex(pd.MultiIndex.from_frame(teamnames[['_key','teamname']]))[TEAM_LEADS+TEAM_EXTRA].to_numpy(float).reshape(-1,2,5)
    assert len(xp)==len(xt)==len(frame)
    return frame,xp,xt,audit.reset_index()


class LeadNet(nn.Module):
    def __init__(self,player_dim,team_dim):
        super().__init__()
        self.player=nn.Sequential(nn.Linear(player_dim,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(4,2),nn.ReLU(),nn.Linear(2,1)) for _ in ROLES])
        self.team_dim=team_dim
        if team_dim:self.team=nn.Linear(team_dim,1,bias=False)
    def forward(self,x,t):
        z=self.player(x);score=sum(h(z[:,:,i,:]).squeeze(-1) for i,h in enumerate(self.heads))
        if self.team_dim:score=score+self.team(t).squeeze(-1)
        return score[:,0]-score[:,1]


class Scale:
    def fit(self,x):
        self.median=np.nan_to_num(np.nanmedian(x,axis=(0,1)),nan=0.)
        a=np.where(np.isnan(x),self.median,x)
        self.scaler=StandardScaler().fit(a.reshape(-1,a.shape[-1]))
        return self
    def transform(self,x):
        a=np.where(np.isnan(x),self.median,x)
        return torch.tensor(self.scaler.transform(a.reshape(-1,a.shape[-1])).reshape(a.shape),dtype=torch.float32)


def fit_model(seed,px,tx,y,epoch_limit=500,validation=None):
    torch.manual_seed(seed);net=LeadNet(px.shape[-1],tx.shape[-1]);opt=torch.optim.Adam(net.parameters(),lr=.001)
    best=float('inf');chosen=epoch_limit;saved=None;stale=0
    for epoch in range(1,epoch_limit+1):
        net.train();opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(px,tx),y);loss.backward();opt.step()
        if validation is not None:
            vx,vt,vy,bo=validation;net.eval()
            with torch.no_grad():p=torch.sigmoid(net(vx,vt)).numpy()
            vl=evaluate(pd.Series(vy),series_probability(p,bo))['log_loss']
            if vl<best:best=vl;chosen=epoch;saved=deepcopy(net.state_dict());stale=0
            else:stale+=1
            if stale>=30:break
    if saved is not None:net.load_state_dict(saved)
    net.eval();return net,chosen


def run(leagues):
    OUT.mkdir(parents=True,exist_ok=True)
    groups={'selected_baseline':(13,0),'player_leads':(16,0),'team_lead_frequency':(16,4),'team_lead_conversion':(16,5)}
    protocol={'leagues_requested':leagues,'variants':groups,'train_year':2025,'test':'2026 LCK same186series',
        'player_extras':ROLE_EXTRA,'team_leads':TEAM_LEADS,'conversion':'(past5 lead wins +1)/(past5 lead count+2), lead=gold15>0',
        'controls':'A additive architecture; original13 LCK inputs unchanged; same2025 LCK val33; seeds42/43/44; no manual feature weight boosts',
        'history':'LCK-only histories for LCK; selected non-LCK player histories pooled; non-LCK teams grouped by league+name',
        'warning':'2026 reused retrospective test; first-set roster known; iid series conversion; LPL mandatory lead fields missing.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    base,bx,series,_=prepare([2025]);data,px,tx,audit=build_data([2025],leagues)
    audit.to_csv(OUT/'audit_2025.csv',index=False);audit.groupby(['league','reason']).size().unstack(fill_value=0).to_csv(OUT/'league_audit.csv')
    mapping=data[data.league.eq('LCK')].reset_index().set_index('gameid')['index']
    np.testing.assert_allclose(px[mapping.loc[base.gameid].to_numpy(),:,:,:13],bx,equal_nan=True,atol=0,rtol=0)
    first_gameids=base.iloc[series.game_index].gameid.to_numpy();si=mapping.loc[first_gameids].to_numpy()
    days=np.sort(series.date.dt.normalize().unique());cut=int(len(days)*.8);cutoff=pd.Timestamp(days[cut])
    val_mask=series.date.dt.normalize().isin(days[cut:]).to_numpy();val=series[val_mask];vi=si[val_mask]
    fit_mask=data.date.lt(cutoff).to_numpy();assert data[fit_mask].date.max()<val.date.min()
    y=torch.tensor(data.result.to_numpy(),dtype=torch.float32)
    models={};locks={}
    for variant,(pdims,tdims) in groups.items():
        pscale=Scale().fit(px[fit_mask,:,:,:pdims]);tscale=Scale().fit(tx[fit_mask,:,:tdims]) if tdims else None
        pt=pscale.transform(px[fit_mask,:,:,:pdims]);tt=tscale.transform(tx[fit_mask,:,:tdims]) if tdims else torch.empty((fit_mask.sum(),2,0))
        pv=pscale.transform(px[vi,:,:,:pdims]);tv=tscale.transform(tx[vi,:,:tdims]) if tdims else torch.empty((len(vi),2,0))
        epochs={}
        for seed in SEEDS:
            _,epoch=fit_model(seed,pt,tt,y[fit_mask],validation=(pv,tv,val.result.to_numpy(),val.best_of.to_numpy()))
            epochs[seed]=epoch;print(variant,seed,'epoch',epoch,flush=True)
        pscale=Scale().fit(px[:,:,:,:pdims]);tscale=Scale().fit(tx[:,:,:tdims]) if tdims else None
        pa=pscale.transform(px[:,:,:,:pdims]);ta=tscale.transform(tx[:,:,:tdims]) if tdims else torch.empty((len(px),2,0))
        nets=[fit_model(seed,pa,ta,y,epoch_limit=epochs[seed])[0] for seed in SEEDS]
        models[variant]=(nets,pscale,tscale)
        locks[variant]={'epochs':epochs,'player_dim':pdims,'team_dim':tdims,'parameters':sum(p.numel() for p in nets[0].parameters())}
    lock={'train_sets':len(data),'fit_sets':int(fit_mask.sum()),'validation_series':len(val),'leagues':data.league.value_counts().to_dict(),'variants':locks}
    (OUT/'lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    joblib.dump({'lock':lock,'variants':{k:{'weights':[n.state_dict() for n in ns],'player_scaler':{'median':ps.median,'scaler':ps.scaler},'team_scaler':{'median':ts.median,'scaler':ts.scaler} if ts else None} for k,(ns,ps,ts) in models.items()}},OUT/'frozen_2025.joblib')
    frozen_hash=hashlib.sha256((OUT/'frozen_2025.joblib').read_bytes()).hexdigest()
    full,fpx,ftx,_=build_data([2025,2026],leagues)
    year25=full.date.dt.year.eq(2025);np.testing.assert_allclose(px,fpx[year25],equal_nan=True,atol=0,rtol=0);np.testing.assert_allclose(tx,ftx[year25],equal_nan=True,atol=0,rtol=0)
    bf,bxx,ss,_=prepare([2025,2026]);test=ss[ss.date.dt.year.eq(2026)]
    lookup=full[full.league.eq('LCK')].reset_index().set_index('gameid')['index']
    ids=bf.iloc[test.game_index].gameid.to_numpy();indices=lookup.loc[ids].to_numpy()
    np.testing.assert_allclose(fpx[indices,:,:,:13],bxx[test.game_index],equal_nan=True,atol=0,rtol=0)
    results=[];predictions=[];monthly=[]
    for variant,(nets,ps,ts) in models.items():
        pdim,tdim=groups[variant];x=ps.transform(fpx[indices,:,:,:pdim]);t=ts.transform(ftx[indices,:,:tdim]) if tdim else torch.empty((len(indices),2,0))
        probs=[]
        with torch.no_grad():
            for net in nets:
                p=torch.sigmoid(net(x,t)).numpy();np.testing.assert_allclose(p+torch.sigmoid(net(x.flip(1),t.flip(1))).numpy(),1,atol=1e-6);probs.append(p)
        p=np.mean(probs,axis=0);q=series_probability(p,test.best_of)
        results.append({'variant':variant,'n':len(test),'correct':int(((q>=.5)==test.result.to_numpy()).sum()),**evaluate(test.result,q)})
        for row,prob in zip(test.itertuples(),q):predictions.append({'variant':variant,'series_key':row.series_key,'date':row.date,'y':row.result,'p':float(prob)})
    pred=pd.DataFrame(predictions)
    for (variant,month),g in pred.groupby(['variant',pred.date.dt.to_period('M')]):monthly.append({'variant':variant,'month':str(month),'n':len(g),**evaluate(g.y,g.p)})
    pd.DataFrame(results).to_csv(OUT/'evaluation_2026.csv',index=False);pred.to_csv(OUT/'predictions_2026.csv',index=False);pd.DataFrame(monthly).to_csv(OUT/'monthly_2026.csv',index=False)
    assert hashlib.sha256((OUT/'frozen_2025.joblib').read_bytes()).hexdigest()==frozen_hash
    (OUT/'manifest.json').write_text(json.dumps({'checkpoint_sha256':frozen_hash,'source_sha256':{str(z):hashlib.sha256(source(z).read_bytes()).hexdigest() for z in [2025,2026]},'checks':['original LCK13 inputs identical','appending2026 leaves2025 features unchanged','team swap symmetry','checkpoint frozen']},indent=2)+'\n')
    print(json.dumps(lock),flush=True);print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--leagues',nargs='+',required=True);args=parser.parse_args();run(args.leagues)
