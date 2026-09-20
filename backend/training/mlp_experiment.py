from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

ts_root = Path(__file__).resolve().parents[2]
if str(ts_root) not in sys.path:
    sys.path.insert(0, str(ts_root))
from backend.training.process_lck_data import load_many_lck_team_games
from backend.training.feature_pipeline import build_featured_games, build_match_dataset, DIFF_COLS


def ts_arrays(frame, columns, scale, reverse=False):
    values = frame[columns].to_numpy(dtype=float)
    if reverse:
        values = -values
    x = np.column_stack([scale.transform(values), frame['is_bo5'].to_numpy()])
    return torch.tensor(x, dtype=torch.float32)

def ts_probs(net, forward, reverse):
    return (torch.sigmoid(net(forward).squeeze(-1)) + 1 - torch.sigmoid(net(reverse).squeeze(-1))) / 2


# 학습 함수: 외부 평가 구간을 보지 않고 inner validation으로 epoch 선택
TS_SEEDS = [42, 43, 44]
TS_MAX_EPOCHS = 500
TS_PATIENCE = 30
TS_LR = 0.001

def ts_fit_predict(fit, stop, future, columns, seed):
    scale = StandardScaler().fit(fit[columns].to_numpy(dtype=float))
    xf = ts_arrays(fit, columns, scale)
    xs, xsr = ts_arrays(stop, columns, scale), ts_arrays(stop, columns, scale, True)
    xt, xtr = ts_arrays(future, columns, scale), ts_arrays(future, columns, scale, True)
    yf = torch.tensor(fit['result'].to_numpy(), dtype=torch.float32)
    ys = torch.tensor(stop['result'].to_numpy(), dtype=torch.float32)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(len(columns)+1,16), nn.ReLU(), nn.Linear(16,8), nn.ReLU(), nn.Linear(8,1))
    opt = torch.optim.Adam(net.parameters(), lr=TS_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    best, saved, best_epoch, stale = float('inf'), None, 0, 0
    for epoch in range(1, TS_MAX_EPOCHS+1):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(xf).squeeze(-1), yf)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pstop = ts_probs(net, xs, xsr).clamp(1e-7,1-1e-7)
            stop_loss = nn.functional.binary_cross_entropy(pstop, ys).item()
        if stop_loss < best:
            best, saved, best_epoch, stale = stop_loss, deepcopy(net.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= TS_PATIENCE:
            break
    net.load_state_dict(saved)
    net.eval()
    with torch.no_grad():
        pred = ts_probs(net, xt, xtr).numpy()
    return pred, best_epoch


def ts_run_experiment(dev, groups):
    days = np.sort(dev['day'].unique())
    scores, predictions, fold_info = [], [], []
    for fold, (past_idx, future_idx) in enumerate(TimeSeriesSplit(n_splits=3).split(days), 1):
        past_days, future_days = days[past_idx], days[future_idx]
        inner_cut = int(len(past_days)*0.8)
        fit = dev[dev['day'].isin(past_days[:inner_cut])]
        stop = dev[dev['day'].isin(past_days[inner_cut:])]
        future = dev[dev['day'].isin(future_days)]
        assert fit['date'].max() < stop['date'].min() < future['date'].min()
        fold_info.append({'fold':fold,'fit':len(fit),'early_stop':len(stop),'future':len(future),
                          'future_start':future['date'].min(),'future_end':future['date'].max()})
        for variant, columns in groups.items():
            for seed in TS_SEEDS:
                pred, epoch = ts_fit_predict(fit, stop, future, columns, seed)
                truth = future['result'].to_numpy()
                scores.append({'fold':fold,'variant':variant,'seed':seed,'n':len(future),'epoch':epoch,
                               'log_loss':log_loss(truth,pred,labels=[0,1]),'brier':brier_score_loss(truth,pred),
                               'accuracy':accuracy_score(truth,pred>=0.5)})
                for row, probability in zip(future.itertuples(), pred):
                    predictions.append({'fold':fold,'variant':variant,'seed':seed,'series_key':row.series_key,
                                        'date':row.date,'y':int(row.result),'p':float(probability)})
        print(f'fold {fold} 완료')
    return pd.DataFrame(scores), pd.DataFrame(predictions), pd.DataFrame(fold_info)
