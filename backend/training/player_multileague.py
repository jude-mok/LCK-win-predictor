"""Expand 2025 training leagues while keeping LCK validation/test inputs fixed.
No production artifact is changed. LCK-only history is deliberately retained for
LCK rows to isolate extra training examples from changes in feature construction.
"""
from pathlib import Path
import hashlib,json
import joblib
import numpy as np
import pandas as pd
import torch
from backend.training.player_two_track import (
    ROOT,ROLES,STATS,INPUTS,SEEDS,source,prepare,Preprocessor,
    PlayerNetwork,train_one,series_probability,
)
from backend.training.feature_pipeline import canonicalize_team_names
from backend.training.train_model import evaluate

OUT=ROOT/'notebook/experiments/11_multileague'
TRACKS=['position_additive','cross_position','upper_lower']


def extra_training_data():
    columns=['gameid','date','league','position','datacompleteness','playerid','teamname','side','result','golddiffat15','xpdiffat15','csdiffat15','kills','deaths','assists','dpm','damageshare','earnedgoldshare','cspm','visionscore','gamelength']
    raw=pd.read_csv(source(2025),usecols=columns,low_memory=False)
    raw['_key']=raw.league.astype(str)+'|'+raw.gameid.astype(str)
    universe=raw.drop_duplicates('_key')[['_key','league','datacompleteness']]
    d=raw[raw.position.isin(ROLES)].copy()
    assert set(d._key)==set(universe._key)
    audit=universe.copy().set_index('_key');audit['reason']='eligible'
    counts=d.groupby('_key').size()
    audit.loc[~audit.index.isin(counts[counts.eq(10)].index),'reason']='not_ten_players'
    incomplete=d.loc[~d.datacompleteness.eq('complete'),'_key'].unique()
    audit.loc[incomplete,'reason']='not_complete'
    missing_id=d.loc[d.playerid.isna(),'_key'].unique()
    audit.loc[audit.index.isin(missing_id)&audit.reason.eq('eligible'),'reason']='missing_player_id'
    numeric=['result','golddiffat15','xpdiffat15','csdiffat15','kills','deaths','assists','dpm','damageshare','earnedgoldshare','cspm','visionscore','gamelength']
    bad=d.loc[~np.isfinite(d[numeric]).all(axis=1)|d.gamelength.le(0),'_key'].unique()
    audit.loc[audit.index.isin(bad)&audit.reason.eq('eligible'),'reason']='missing_or_invalid_statistics'
    d=d[d._key.isin(audit[audit.reason.eq('eligible')].index)].copy()
    d=canonicalize_team_names(d);d['date']=pd.to_datetime(d.date,errors='raise')
    assert d.date.dt.year.eq(2025).all()
    dup=d[d.duplicated(['_key','playerid'],keep=False)]._key.unique()
    audit.loc[dup,'reason']='duplicate_player_id';d=d[~d._key.isin(dup)].copy()
    # Avoid using another game with the same player timestamp as past information.
    conflict=d[d.duplicated(['playerid','date'],keep=False)]._key.unique()
    audit.loc[conflict,'reason']='ambiguous_player_time';d=d[~d._key.isin(conflict)].copy()
    valid=[]
    for key,g in d.groupby('_key',sort=False):
        teams=list(g.groupby('teamname'))
        good=len(teams)==2 and set(g.side)=={'Blue','Red'}
        good=good and all(len(t)==5 and set(t.position)==set(ROLES) and t.result.nunique()==1 and t.side.nunique()==1 for _,t in teams)
        good=good and g.result.isin([0,1]).all() and g.groupby('teamname').result.first().sum()==1
        if good:valid.append(key)
        else:audit.loc[key,'reason']='invalid_roster_or_target'
    d=d[d._key.isin(valid)].copy()
    for c in ['kills','deaths','assists']:d[c+'_per_min']=d[c]/(d.gamelength/60)
    d['vision_per_min']=d.visionscore/(d.gamelength/60)
    d=d.sort_values(['playerid','date','_key'])
    for c in STATS:
        d['past_'+c]=d.groupby('playerid')[c].transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
    d['history_fraction']=d.groupby('playerid').cumcount().clip(upper=5)/5
    # Alphabetical teams match the existing target orientation.
    d['_team']=d.groupby('_key').teamname.rank(method='dense').astype(int)-1
    d['_role']=d.position.map({r:i for i,r in enumerate(ROLES)})
    d=d.sort_values(['date','_key','_team','_role'])
    x=d[INPUTS].to_numpy(float).reshape(-1,2,5,len(INPUTS))
    frame=d.drop_duplicates('_key')[['_key','gameid','league','date','result','teamname']].reset_index(drop=True)
    assert len(x)==len(frame)
    # Verify each reshaped block belongs to exactly one game with the expected order.
    assert (d._key.to_numpy().reshape(-1,10)==frame._key.to_numpy()[:,None]).all()
    assert (d._role.to_numpy().reshape(-1,10)==np.tile(np.arange(5),2)).all()
    return frame,x,audit.reset_index()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'hypothesis':'More 2025 league examples improve the learned player/team representations on LCK.',
        'train_year':2025,'test':'same 2026 LCK 186 series / 497 sets',
        'tracks':TRACKS,'seeds':SEEDS,'training_weighting':'one equal-weight sample per set; no league balancing',
        'controls':'same architectures, 13 inputs, max500/patience30/Adam.001; same LCK 2025 validation33; exact original LCK input arrays',
        'history_policy':'LCK rows retain original LCK-only histories; extra non-LCK rows use preceding eligible games across leagues. LCK history expansion is NOT part of this experiment.',
        'quality_policy':'complete game, 10 identified players, two valid teams, one of each role, finite required stats; no league-name exclusions',
        'series_policy':'Non-LCK Bo1/Bo2 games can train a set classifier; no inference of their series format required.',
        'warnings':['2026 has been repeatedly inspected: retrospective comparison, not untouched test.','League strength is not explicitly modeled.','Same first-set announced roster and iid set-to-series assumptions as experiment10.']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    base,bx,series,_=prepare([2025])
    other,ox,audit=extra_training_data()
    audit.to_csv(OUT/'game_audit_2025.csv',index=False)
    audit.groupby(['league','reason']).size().unstack(fill_value=0).to_csv(OUT/'league_audit_2025.csv')
    # Replace all LCK rows with the exact baseline frame and tensors, including cold starts.
    extra_mask=other.league.ne('LCK').to_numpy()
    extras=other[extra_mask].copy();extra_x=ox[extra_mask]
    base=base.copy();base['league']='LCK';base['_key']='LCK|'+base.gameid.astype(str)
    combined=pd.concat([base,extras],ignore_index=True)
    allx=np.concatenate([bx,extra_x])
    order=combined.sort_values(['date','_key']).index.to_numpy()
    combined=combined.iloc[order].reset_index(drop=True);allx=allx[order]
    assert combined.date.dt.year.eq(2025).all()
    pd.DataFrame(combined.groupby('league').size(),columns=['train_sets']).to_csv(OUT/'training_counts.csv')
    days=np.sort(series.date.dt.normalize().unique());cut=int(len(days)*.8)
    fit_series=series[series.date.dt.normalize().isin(days[:cut])]
    val_series=series[series.date.dt.normalize().isin(days[cut:])]
    cutoff=pd.Timestamp(days[cut])
    fit_mask=(combined.league.eq('LCK')&combined.series_key.isin(fit_series.series_key)) | (combined.league.ne('LCK')&combined.date.lt(cutoff))
    assert combined[fit_mask].date.max()<val_series.date.min()
    prep=Preprocessor().fit(allx[fit_mask.to_numpy()])
    xf=prep.transform(allx[fit_mask.to_numpy()]);yf=torch.tensor(combined.loc[fit_mask,'result'].to_numpy(),dtype=torch.float32)
    xv=prep.transform(bx[val_series.game_index]);validation=(xv,val_series.result.to_numpy(),val_series.best_of.to_numpy())
    epochs={};validation_scores=[];hist=[]
    print('TRAIN:',len(combined),'sets,',combined.league.nunique(),'leagues; fitting:',int(fit_mask.sum()),'validation:',len(val_series),flush=True)
    for track in TRACKS:
        epochs[track]={};probs=[]
        for seed in SEEDS:
            net,epoch,h=train_one(track,seed,xf,yf,validation=validation)
            epochs[track][seed]=epoch
            hist.extend({'track':track,'seed':seed,**r} for r in h)
            with torch.no_grad():probs.append(torch.sigmoid(net(xv)).numpy())
            print(track,seed,'selected epoch',epoch,flush=True)
        validation_scores.append({'track':track,**evaluate(val_series.result,series_probability(np.mean(probs,axis=0),val_series.best_of))})
    pd.DataFrame(hist).to_csv(OUT/'history_2025.csv',index=False)
    pd.DataFrame(validation_scores).to_csv(OUT/'validation_2025.csv',index=False)
    prep_all=Preprocessor().fit(allx);xa=prep_all.transform(allx);ya=torch.tensor(combined.result.to_numpy(),dtype=torch.float32)
    models={}
    for track in TRACKS:
        models[track]=[]
        for seed in SEEDS:
            models[track].append(train_one(track,seed,xa,ya,epochs=epochs[track][seed])[0])
        print('refit',track,'done',flush=True)
    lock={'epochs':epochs,'train_sets':len(combined),'training_leagues':int(combined.league.nunique()),
        'extra_sets':len(extras),'lck_sets':len(base),'fit_sets':int(fit_mask.sum()),'validation_series':len(val_series),
        'cutoff':str(cutoff),'train_end':str(combined.date.max()),'parameter_counts':{t:sum(p.numel() for p in models[t][0].parameters()) for t in TRACKS}}
    (OUT/'lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    joblib.dump({'preprocessor':{'median':prep_all.median,'scaler':prep_all.scaler},'weights':{t:[n.state_dict() for n in nets] for t,nets in models.items()},'lock':lock},OUT/'frozen_2025.joblib')
    checkpoint_hash=hashlib.sha256((OUT/'frozen_2025.joblib').read_bytes()).hexdigest()
    full,xx,ss,_=prepare([2025,2026])
    np.testing.assert_allclose(xx[full.date.dt.year.eq(2025)],bx,equal_nan=True,atol=0,rtol=0)
    test_games=full[full.date.dt.year.eq(2026)];test_series=ss[ss.date.dt.year.eq(2026)]
    xt=prep_all.transform(xx[full.date.dt.year.eq(2026)]);xs=prep_all.transform(xx[test_series.game_index])
    results=[];predictions=[];seed_scores=[]
    for track,nets in models.items():
        gp=[];sp=[]
        with torch.no_grad():
            for seed,net in zip(SEEDS,nets):
                g=torch.sigmoid(net(xt)).numpy();p=torch.sigmoid(net(xs)).numpy()
                np.testing.assert_allclose(p+torch.sigmoid(net(xs.flip(1))).numpy(),1,atol=1e-6)
                gp.append(g);sp.append(p)
                seed_scores.append({'track':track,'seed':seed,**evaluate(test_series.result,series_probability(p,test_series.best_of))})
        g=np.mean(gp,axis=0);p=np.mean(sp,axis=0);q=series_probability(p,test_series.best_of)
        for level,y,prob in [('set',test_games.result,g),('series',test_series.result,q)]:
            results.append({'regime':'multi_league_2025','track':track,'level':level,'n':len(y),'correct':int(((prob>=.5)==y.to_numpy()).sum()),**evaluate(y,prob)})
        for row,prob,set_p in zip(test_series.itertuples(),q,p):
            predictions.append({'track':track,'series_key':row.series_key,'date':row.date,'team1':row.team1,'team2':row.team2,'best_of':row.best_of,'y':row.result,'set_probability':float(set_p),'series_probability':float(prob)})
    baseline=pd.read_csv(ROOT/'notebook/experiments/10_player_grouped/evaluation_2026.csv')
    baseline=baseline[baseline.track.isin(TRACKS)].copy();baseline.insert(0,'regime','lck_only_2025')
    comparison=pd.concat([baseline,pd.DataFrame(results)],ignore_index=True)
    comparison.to_csv(OUT/'comparison_2026.csv',index=False)
    predictions=pd.DataFrame(predictions);predictions.to_csv(OUT/'predictions_2026.csv',index=False)
    old_predictions=pd.read_csv(ROOT/'notebook/experiments/10_player_grouped/predictions_2026.csv')
    paired=[]
    for t,g in predictions.groupby('track'):
        old=old_predictions[old_predictions.track.eq(t)]
        pair=g.merge(old,on='series_key',suffixes=('_new','_old'),validate='one_to_one')
        assert len(pair)==186 and pair.y_new.eq(pair.y_old).all()
        a=pair.series_probability_new.ge(.5).eq(pair.y_new);b=pair.series_probability_old.ge(.5).eq(pair.y_old)
        paired.append({'track':t,'new_only_correct':int((a&~b).sum()),'old_only_correct':int((b&~a).sum()),'both_correct':int((a&b).sum()),'both_wrong':int((~a&~b).sum())})
    pd.DataFrame(paired).to_csv(OUT/'paired_comparison.csv',index=False)
    pd.DataFrame(seed_scores).to_csv(OUT/'seed_scores_2026.csv',index=False)
    monthly=[]
    for (track,month),g in predictions.groupby(['track',predictions.date.dt.to_period('M')]):
        monthly.append({'track':track,'month':str(month),'n':len(g),**evaluate(g.y,g.series_probability)})
    pd.DataFrame(monthly).to_csv(OUT/'monthly_2026.csv',index=False)
    assert hashlib.sha256((OUT/'frozen_2025.joblib').read_bytes()).hexdigest()==checkpoint_hash
    manifest={'source_sha256':{str(y):hashlib.sha256(source(y).read_bytes()).hexdigest() for y in [2025,2026]},'checkpoint_sha256':checkpoint_hash,
              'checks':['same LCK training inputs','same 186 test series and labels','no 2026 fitting','team swap symmetry','checkpoint unchanged'],'raw_leagues':int(audit.league.nunique()),'raw_games':len(audit),'excluded_games':int(audit.reason.ne('eligible').sum())}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(comparison[comparison.level.eq('series')].to_string(index=False),flush=True)
    print('AUDIT',json.dumps(audit.reason.value_counts().to_dict()),flush=True)
    return comparison

if __name__=='__main__':run()
