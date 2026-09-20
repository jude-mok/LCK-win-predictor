"""Reproduce the nine-input experiment, audit, then export a three-seed MLP."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from backend.training.mlp_experiment import ts_run_experiment, ts_fit_predict, ts_arrays, TS_SEEDS
from backend.training.mlp_features import prepare_mlp_data, MLP_FEATURES, MLP_DIFFS
from backend.training.process_lck_data import load_many_lck_team_games
from backend.training.train_model import evaluate, atomic_joblib_dump, atomic_json_dump
from backend.app.services.mlp_runtime import predict_bundle

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'backend/models/lck_mlp.joblib'


def refit(frame, seed, epochs):
    scale = StandardScaler().fit(frame[MLP_DIFFS].to_numpy(float))
    x = ts_arrays(frame, MLP_DIFFS, scale)
    y = torch.tensor(frame.result.to_numpy(), dtype=torch.float32)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(9,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,1))
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(-1), y)
        loss.backward()
        opt.step()
    net.eval()
    member = {'seed':seed,'epochs':epochs,'mean':scale.mean_,'scale':scale.scale_,
              'layers':[(net[i].weight.detach().numpy(),net[i].bias.detach().numpy()) for i in [0,2,4]]}
    # Export parity on both series formats, including swapped teams.
    for best_of in [3,5]:
        probe = frame.head(20).copy(); probe['is_bo5'] = float(best_of==5)
        from backend.training.mlp_experiment import ts_probs
        with torch.no_grad():
            expected = ts_probs(net,ts_arrays(probe,MLP_DIFFS,scale),ts_arrays(probe,MLP_DIFFS,scale,True)).numpy()
        actual = predict_bundle({'members':[member]},probe[MLP_DIFFS].to_numpy(),best_of)
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=1e-6)
    return member


def run(paths, publish=False):
    raw = load_many_lck_team_games(paths)
    games, matches, state = prepare_mlp_data(raw)
    dev = matches.iloc[:int(len(matches)*.8)].copy()
    scores, predictions, folds = ts_run_experiment(dev, {'lead10_baron_9':MLP_DIFFS})
    seed_metrics = [evaluate(g.y,g.p) for _,g in predictions.groupby('seed')]
    pooled = predictions.groupby('series_key').agg(y=('y','first'),p=('p','mean'))
    days = np.sort(dev.day.unique()); cut = int(len(days)*.8)
    fit, stop = dev[dev.day.isin(days[:cut])], dev[dev.day.isin(days[cut:])]
    recent = matches.iloc[len(dev):]
    recent_probs=[]
    for seed in TS_SEEDS:
        p,_ = ts_fit_predict(fit,stop,recent,MLP_DIFFS,seed)
        recent_probs.append(p)
    report = {'model_type':'MLP 9-16-8-1, three-seed probability ensemble',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'game_count':int(games.gameid.nunique()),'series_count':len(matches),
        'latest_game_date':games.date.max().isoformat(),'feature_columns':MLP_FEATURES,
        'input_columns':MLP_DIFFS+['is_bo5'],'seeds':TS_SEEDS,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'walk_forward':{'series_count':len(pooled),'folds':folds.to_dict('records'),
            'seed_mean':{k:float(np.mean([v[k] for v in seed_metrics])) for k in ['accuracy','log_loss','brier_score','roc_auc']},
            'ensemble':evaluate(pooled.y,pooled.p),'fold_seed_scores':scores.to_dict('records')},
        'recent_audit':{'series_count':len(recent),'first_date':recent.date.min().isoformat(),
            'last_date':recent.date.max().isoformat(),'metrics':evaluate(recent.result,np.mean(recent_probs,axis=0)),
            'note':'Previously inspected test period; descriptive audit, not untouched test.'},
        'evaluation_note':'Feature choices informed by development results. 70.51% is a seed mean on one 78-series fold, not overall or production accuracy.',
        'published':False}
    # JSON timestamp conversion for fold boundaries.
    report=json.loads(json.dumps(report,default=str))
    if publish:
        all_days=np.sort(matches.day.unique());cut=int(len(all_days)*.8)
        fit=matches[matches.day.isin(all_days[:cut])];stop=matches[matches.day.isin(all_days[cut:])]
        members=[]
        for seed in TS_SEEDS:
            _,epoch=ts_fit_predict(fit,stop,stop,MLP_DIFFS,seed)
            members.append(refit(matches,seed,epoch))
        version=hashlib.sha256(b''.join(m['layers'][0][0].tobytes() for m in members)).hexdigest()[:12]
        report.update(published=True,version=version,final_epochs=[m['epochs'] for m in members])
        bundle={'members':members,'team_state':state,'metadata':report,'feature_columns':MLP_FEATURES,'diff_columns':MLP_DIFFS}
        atomic_joblib_dump(bundle,BUNDLE)
    atomic_json_dump(report,ROOT/'backend/models/lck_mlp_metadata.json')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true',help='Explicitly publish audited model; no automatic promotion.')
    parser.add_argument('--input',action='append',type=Path)
    args=parser.parse_args()
    paths=args.input or [ROOT/'backend/data/raw'/f'{year}_LoL_esports_match_data_from_OraclesElixir.csv' for year in [2025,2026]]
    run(paths,args.publish)
