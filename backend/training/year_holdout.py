"""2025-only model selection/refit followed by frozen-model 2026 replay.
Run: python -m backend.training.year_holdout
This experiment writes only notebook/experiments/08_year_holdout artifacts.
"""
from pathlib import Path
from itertools import combinations
from contextlib import redirect_stdout
import io
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from backend.training.process_lck_data import OUTPUT_COLUMNS, validate_lck_team_games
from backend.training.feature_pipeline import DIFF_COLS, build_featured_games, build_match_dataset
from backend.training.mlp_experiment import ts_run_experiment, ts_arrays, ts_probs, TS_SEEDS
from backend.training.train_model import evaluate

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'notebook/experiments/08_year_holdout'
CANDIDATES=list(DIFF_COLS)+[f'diff_strong_lead{m}_rate' for m in [10,15,20]]+['diff_firstbaron_rate']


def load_year(year):
    path=ROOT/'backend/data/raw'/f'{year}_LoL_esports_match_data_from_OraclesElixir.csv'
    cols=list(dict.fromkeys(OUTPUT_COLUMNS+['golddiffat20']))
    raw=pd.read_csv(path,usecols=cols,low_memory=False)
    raw=raw[raw.league.eq('LCK') & raw.position.eq('team') & raw.datacompleteness.eq('complete')].copy()
    raw['date']=pd.to_datetime(raw.date)
    assert raw.date.dt.year.eq(year).all()
    validate_lck_team_games(raw)
    return raw


def features(raw):
    raw=raw.copy()
    missing=raw.firstbaron.isna()
    assert not (missing & ~(raw.barons.eq(0)&raw.opp_barons.eq(0))).any()
    raw.loc[missing,'firstbaron']=0
    assert raw.firstbaron.isin([0,1]).all()
    assert np.isfinite(raw[['golddiffat10','golddiffat15','golddiffat20']]).all().all()
    games=build_featured_games(raw).sort_values(['teamname','date','gameid']).copy()
    for minute in [10,15,20]:
        games[f'strong_lead{minute}_rate_indicator']=games[f'golddiffat{minute}'].ge(1000).astype(float)
    sources={f'strong_lead{m}_rate':f'strong_lead{m}_rate_indicator' for m in [10,15,20]}
    sources['firstbaron_rate']='firstbaron'
    for name,source in sources.items():
        games[name]=games.groupby('teamname')[source].transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean()).fillna(.5)
    matches=build_match_dataset(games)
    first=games.sort_values(['date','gameid']).drop_duplicates(['series_key','teamname']).set_index(['series_key','teamname'])
    for name in sources:
        matches['diff_'+name]=[float(first.loc[(r.series_key,r.team1),name]-first.loc[(r.series_key,r.team2),name]) for r in matches.itertuples()]
    matches['is_bo5']=matches.best_of.eq(5).astype(float)
    matches['day']=matches.date.dt.normalize()
    return matches


def groups():
    # A systematic menu, not combinations ranked using the 2026 results.
    result={}
    for size in [1,2]:
        for cols in combinations(CANDIDATES,size):
            result[f'candidate_{len(result)+1:02d}']=list(cols)
    result['base_7']=list(DIFF_COLS)
    for m in [10,15,20]:
        result[f'lead{m}_baron_9']=list(DIFF_COLS)+[f'diff_strong_lead{m}_rate','diff_firstbaron_rate']
    result['all_11']=CANDIDATES
    return result


def frozen_fit(train,cols,seed,epochs):
    scaler=StandardScaler().fit(train[cols].to_numpy(float))
    x=ts_arrays(train,cols,scaler)
    y=torch.tensor(train.result.to_numpy(),dtype=torch.float32)
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(len(cols)+1,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    for _ in range(epochs):
        opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(-1),y)
        loss.backward();opt.step()
    net.eval()
    return net,scaler


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    train=features(load_year(2025))
    assert train.date.max()<pd.Timestamp('2026-01-01')
    options=groups()
    protocol={'training_year':2025,'evaluation_year':2026,'selection':'2025 pooled walk-forward ensemble accuracy; ties: log loss, then input count',
        'seeds':TS_SEEDS,'epochs':'per-seed median of selected model best epochs in 2025 folds; refit all 2025',
        'window':5,'ensemble':'mean of 3 symmetrized probabilities','groups':options,
        'caveat':'2026 has previously been inspected; retrospective replay, not an untouched test.',
        'best_of':'Reconstructed from completed-series win counts; assumes format known before series.',
        'inputs':'2026 prior completed games update rolling inputs only; weights and scaler stay frozen.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    ranking=[];epoch_map={};cv_records=[]
    for index,(name,cols) in enumerate(options.items(),1):
        with redirect_stdout(io.StringIO()):
            scores,preds,folds=ts_run_experiment(train,{name:cols})
        pooled=preds.groupby('series_key').agg(y=('y','first'),p=('p','mean'))
        ranking.append({'variant':name,'input_count':len(cols)+1,'n':len(pooled),**evaluate(pooled.y,pooled.p)})
        epoch_map[name]={seed:max(1,int(np.median(scores[scores.seed.eq(seed)].epoch))) for seed in TS_SEEDS}
        cv_records.append(preds)
        print(f'2025 selection {index}/{len(options)}: {name}',flush=True)
    ranking=pd.DataFrame(ranking).sort_values(['accuracy','log_loss','input_count'],ascending=[False,True,True])
    ranking.to_csv(OUT/'selection_2025.csv',index=False)
    pd.concat(cv_records).to_csv(OUT/'selection_predictions_2025.csv',index=False)
    folds.to_csv(OUT/'folds_2025.csv',index=False)
    selected=ranking.iloc[0].variant;cols=options[selected];epochs=epoch_map[selected]
    frozen=[frozen_fit(train,cols,seed,epochs[seed]) for seed in TS_SEEDS]
    lock={'variant':selected,'columns':cols+['is_bo5'],'epochs':epochs,'train_series':len(train),
          'train_start':str(train.date.min()),'train_end':str(train.date.max()),'selection_metrics':ranking.iloc[0].to_dict()}
    # Freeze and persist before loading 2026 for this run.
    (OUT/'selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    checkpoint={'selection':lock,'members':[{'seed':seed,'state_dict':net.state_dict(),'scaler':scale} for seed,(net,scale) in zip(TS_SEEDS,frozen)]}
    joblib.dump(checkpoint,OUT/'frozen_2025.joblib')
    checkpoint_hash=hashlib.sha256((OUT/'frozen_2025.joblib').read_bytes()).hexdigest()
    allmatches=features(pd.concat([load_year(2025),load_year(2026)],ignore_index=True))
    pd.testing.assert_frame_equal(train.reset_index(drop=True),allmatches[allmatches.date.dt.year.eq(2025)].reset_index(drop=True))
    test=allmatches[allmatches.date.dt.year.eq(2026)].copy()
    # Confirm features at a chosen 2026 series equal a prefix-only reconstruction.
    probe=test.iloc[len(test)//2]
    prefix_raw=pd.concat([load_year(2025),load_year(2026)],ignore_index=True)
    # Including its first completed game is sufficient to construct the same pregame features;
    # shift(1) excludes that game and every later game from the rolling input.
    prefix=features(prefix_raw[prefix_raw.date.le(probe.date)])
    # Incomplete target series is not emitted, so verify all completed prefix series instead.
    overlap=allmatches.set_index('series_key').loc[prefix.series_key,CANDIDATES]
    np.testing.assert_allclose(overlap.to_numpy(),prefix[CANDIDATES].to_numpy(),atol=0,rtol=0)
    member_probs=[]
    with torch.no_grad():
        for net,scale in frozen:
            member_probs.append(ts_probs(net,ts_arrays(test,cols,scale),ts_arrays(test,cols,scale,True)).numpy())
    probs=np.mean(member_probs,axis=0)
    # Fixed linear comparator on identical selected inputs, trained only on 2025.
    scale=StandardScaler().fit(train[cols].to_numpy(float))
    def linear_x(frame,reverse=False):
        values=frame[cols].to_numpy(float)*(-1 if reverse else 1)
        return np.column_stack([scale.transform(values),frame.is_bo5])
    linear=LogisticRegression(max_iter=2000).fit(linear_x(train),train.result)
    lp=(linear.predict_proba(linear_x(test))[:,1]+1-linear.predict_proba(linear_x(test,True))[:,1])/2
    variants={'selected_mlp':probs,'linear_same_features':lp,'always_50':np.full(len(test),.5),'train_winrate':np.full(len(test),train.result.mean())}
    summary=[]
    for name,p in variants.items():
        summary.append({'model':name,'n':len(test),'correct':int(((p>=.5)==test.result.to_numpy()).sum()),**evaluate(test.result,p)})
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'evaluation_2026.csv',index=False)
    predictions=test[['series_key','date','team1','team2','best_of','result']].copy()
    predictions['probability']=probs
    for seed,p in zip(TS_SEEDS,member_probs):predictions[f'p_seed_{seed}']=p
    predictions['correct']=(probs>=.5)==test.result.to_numpy()
    predictions.to_csv(OUT/'predictions_2026.csv',index=False)
    monthly=[]
    for month,g in predictions.groupby(predictions.date.dt.to_period('M')):
        monthly.append({'month':str(month),'n':len(g),'correct':int(g.correct.sum()),**evaluate(g.result,g.probability)})
    monthly=pd.DataFrame(monthly);monthly.to_csv(OUT/'monthly_2026.csv',index=False)
    assert hashlib.sha256((OUT/'frozen_2025.joblib').read_bytes()).hexdigest()==checkpoint_hash
    manifest={'checkpoint_sha256':checkpoint_hash,'source_sha256':{str(y):hashlib.sha256((ROOT/'backend/data/raw'/f'{y}_LoL_esports_match_data_from_OraclesElixir.csv').read_bytes()).hexdigest() for y in [2025,2026]},'checks':['2025 feature prefix equality','completed prefix invariance','frozen checkpoint unchanged'],'test_start':str(test.date.min()),'test_end':str(test.date.max())}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('LOCK',json.dumps(lock),flush=True)
    print(summary.to_string(index=False),flush=True)
    print(monthly.to_string(index=False),flush=True)
    return lock,summary,monthly

if __name__=='__main__':run()
