"""Player feature removal and historical LCK training, selected using 2025 only.

Run with: python -m backend.training.player_history_ablation
Outputs are research artifacts; production models are never replaced.
"""
from copy import deepcopy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch import nn

from backend.training.player_two_track import (
    ROOT, INPUTS, SEEDS, PlayerNetwork, prepare, series_probability, source,
)
from backend.training.selected_league_leads import Scale, build_data
from backend.training.train_model import evaluate

OUT = ROOT / 'notebook/experiments/13_player_history_ablation'


def make_net(seed, columns):
    """Keep retained weights and all downstream initial weights paired across ablations."""
    torch.manual_seed(seed)
    net = PlayerNetwork('position_additive')
    if columns != list(range(len(INPUTS))):
        old = net.player[0]
        layer = nn.Linear(len(columns), 8)
        with torch.no_grad():
            layer.weight.copy_(old.weight[:, columns])
            layer.bias.copy_(old.bias)
        net.player[0] = layer
    return net


def train(seed, columns, x, y, epochs=500, validation=None):
    net = make_net(seed, columns)
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    best, stale, chosen, saved = float('inf'), 0, epochs, None
    for epoch in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(net(x), y)
        loss.backward()
        opt.step()
        if validation is not None:
            vx, vy, bo = validation
            net.eval()
            with torch.no_grad():
                q = series_probability(torch.sigmoid(net(vx)).numpy(), bo)
            score = log_loss(vy, q, labels=[0, 1])
            if score < best:
                best, stale, chosen, saved = score, 0, epoch, deepcopy(net.state_dict())
            else:
                stale += 1
            if stale >= 30:
                break
    if saved is not None:
        net.load_state_dict(saved)
    net.eval()
    return net, chosen


def fit_ensemble(frame, x, series, columns):
    """Select epochs on the last 20% of available 2025 dates, then refit prefix."""
    dates = np.sort(series.date.dt.normalize().unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * .8)])
    validation_series = series[series.date.ge(cutoff)]
    fitting = frame.date.lt(cutoff).to_numpy()
    assert frame.date.dt.year.le(2025).all()
    assert frame.loc[fitting, 'date'].max() < validation_series.date.min()
    selected = x[..., columns]
    prep = Scale().fit(selected[fitting])
    xf = prep.transform(selected[fitting])
    yf = torch.tensor(frame.loc[fitting, 'result'].to_numpy(), dtype=torch.float32)
    validation = (prep.transform(selected[validation_series.game_index]),
                  validation_series.result.to_numpy(), validation_series.best_of.to_numpy())
    epochs = [train(s, columns, xf, yf, validation=validation)[1] for s in SEEDS]
    final_prep = Scale().fit(selected)
    xa = final_prep.transform(selected)
    ya = torch.tensor(frame.result.to_numpy(), dtype=torch.float32)
    nets = [train(s, columns, xa, ya, epochs=e)[0] for s, e in zip(SEEDS, epochs)]
    return nets, final_prep, epochs


def predict(nets, prep, x, columns):
    xt = prep.transform(x[..., columns])
    with torch.no_grad():
        members = np.stack([torch.sigmoid(n(xt)).numpy() for n in nets])
        reverse = np.stack([torch.sigmoid(n(xt.flip(1))).numpy() for n in nets])
    np.testing.assert_allclose(members + reverse, 1, atol=1e-6)
    return members.mean(axis=0), members


def combine(base, bx, series, historical, years, cutoff=None):
    """Keep base feature values fixed. Historical inputs use within-year histories."""
    keep = np.ones(len(base), dtype=bool) if cutoff is None else base.date.lt(cutoff).to_numpy()
    frames = [base[keep].copy()]
    arrays = [bx[keep]]
    for year in years:
        if year == 2025:
            continue
        frame, x = historical[year]
        assert frame.date.dt.year.eq(year).all()
        frames.append(frame)
        arrays.append(x)
    frame = pd.concat(frames, ignore_index=True)
    x = np.concatenate(arrays)
    order = frame.sort_values(['date', 'gameid']).index.to_numpy()
    frame = frame.iloc[order].reset_index(drop=True)
    x = x[order]
    assert not frame.gameid.duplicated().any()
    selected_series = series if cutoff is None else series[series.date.lt(cutoff)]
    selected_series = selected_series.copy()
    lookup = pd.Series(np.arange(len(frame)), index=frame.gameid)
    selected_series['game_index'] = lookup.loc[base.iloc[selected_series.game_index].gameid].to_numpy()
    np.testing.assert_allclose(x[frame.date.dt.year.eq(2025)], bx[keep], equal_nan=True, atol=0, rtol=0)
    return frame, x, selected_series


def cv(name, columns, years, base, bx, series, historical, folds, rows, fold_rows, epoch_rows):
    predictions = []
    for fold in folds:
        cutoff = pd.Timestamp(fold['start'])
        future = series[series.date.ge(cutoff) & series.date.lt(pd.Timestamp(fold['end']))]
        frame, x, past = combine(base, bx, series, historical, years, cutoff)
        assert frame.date.max() < future.date.min()
        nets, prep, epochs = fit_ensemble(frame, x, past, columns)
        p, _ = predict(nets, prep, bx[future.game_index], columns)
        q = series_probability(p, future.best_of.to_numpy())
        metrics = evaluate(future.result, q)
        fold_rows.append({'candidate': name, 'fold': fold['fold'], 'n': len(future),
                          'correct': int(((q >= .5) == future.result.to_numpy()).sum()), **metrics})
        epoch_rows.append({'candidate': name, 'fold': fold['fold'], 'epochs': epochs,
                           'train_sets': len(frame), 'train_end': str(frame.date.max())})
        for row, prob in zip(future.itertuples(), q):
            predictions.append({'candidate': name, 'fold': fold['fold'], 'series_key': row.series_key,
                                'date': str(row.date), 'y': row.result, 'p': float(prob)})
    d = pd.DataFrame(predictions)
    result = {'candidate': name, 'inputs': len(columns), 'n': len(d),
              'correct': int((d.p.ge(.5) == d.y).sum()), **evaluate(d.y, d.p.to_numpy())}
    rows.extend(predictions)
    print('2025 CV', name, result['correct'], '/', len(d), 'logloss', round(result['log_loss'], 5), flush=True)
    return result


def run():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'architecture': 'A: shared d→8→4, five role-specific 4→2→1 heads, sum and team difference',
        'primary_metric': '2025 chronological out-of-fold ensemble series accuracy; log loss tie-break',
        'feature_search': '13 leave-one-feature-out candidates; optional joint removal of up to 3 individually helpful features; all13 remains eligible',
        'joint_rule': 'Remove up to 3 features with higher CV accuracy AND lower log loss than all13, ranked by accuracy then log loss',
        'reduction_rule': 'Use best reduced candidate only if CV correct count exceeds all13; else retain all13',
        'outer_folds': '2025 series dates: first40% train, next20% test; expand to60%,80% train',
        'epoch_selection': 'inner last20% of available2025 series dates, series log loss, max500 patience30; then refit entire prefix',
        'final_fit': 'epoch selection on same2025 last33series as baseline; refit selected years',
        'history_regimes': [[2025], [2024, 2025], [2023, 2024, 2025]],
        'league': 'LCK only; equal weight per training set',
        'history_control': '2025 and2026 player inputs EXACTLY baseline; older years processed separately with own within-year rolling histories',
        'initialization': 'same original13-input initial weights; remove columns only, preserve downstream weights across feature ablations',
        'seeds': SEEDS, 'optimizer': 'Adam .001, full batch, BCEWithLogitsLoss',
        'warnings': ['2025 CV used for selection; winner score is optimistically selected, not independent test.',
                     '2026 repeatedly inspected in project; retrospective audit only, no fitting/selection in this run.',
                     'First-set roster known; iid set-to-series conversion; no draft or later substitutions.',
                     'Feature removal changes capacity and optimization too; not causal evidence a statistic is harmful.',
                     'Extra historical records do not warm-start2025 input history in this controlled comparison.'],
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    base, bx, series, _ = prepare([2025])
    historical, audits = {}, []
    for year in [2023, 2024]:
        frame, x, _, audit = build_data([year], ['LCK'])
        historical[year] = (frame, x[..., :13])
        audit['year'] = year
        audits.append(audit)
        print('historical', year, len(frame), 'eligible sets', flush=True)
    pd.concat(audits).to_csv(OUT / 'historical_audit.csv', index=False)
    dates = np.sort(series.date.dt.normalize().unique())
    boundaries = [pd.Timestamp(dates[int(len(dates) * f)]) for f in [.4, .6, .8]]
    boundaries.append(series.date.max().normalize() + pd.Timedelta(days=1))
    folds = [{'fold': i + 1, 'start': str(boundaries[i]), 'end': str(boundaries[i + 1])} for i in range(3)]
    (OUT / 'folds.json').write_text(json.dumps(folds, indent=2) + '\n')
    all_cols = list(range(13))
    candidates = {'all13': all_cols}
    candidates.update({'without_' + c: [j for j in all_cols if j != i] for i, c in enumerate(INPUTS)})
    cv_predictions, fold_scores, cv_epochs, scores = [], [], [], []
    for name, columns in candidates.items():
        scores.append(cv(name, columns, [2025], base, bx, series, historical, folds,
                         cv_predictions, fold_scores, cv_epochs))
    baseline = scores[0]
    helpful = sorted([s for s in scores[1:] if s['correct'] > baseline['correct'] and s['log_loss'] < baseline['log_loss']],
                     key=lambda s: (-s['correct'], s['log_loss']))[:3]
    if len(helpful) >= 2:
        dropped = {INPUTS.index(s['candidate'].removeprefix('without_')) for s in helpful}
        candidates['joint_removal'] = [c for c in all_cols if c not in dropped]
        scores.append(cv('joint_removal', candidates['joint_removal'], [2025], base, bx, series, historical,
                         folds, cv_predictions, fold_scores, cv_epochs))
    best = sorted(scores, key=lambda s: (-s['correct'], s['log_loss']))[0]
    selected_name = best['candidate'] if best['correct'] > baseline['correct'] else 'all13'
    selected_columns = candidates[selected_name]
    pd.DataFrame(scores).to_csv(OUT / 'feature_selection_2025.csv', index=False)
    print('FEATURE LOCK', selected_name, [INPUTS[i] for i in selected_columns], flush=True)
    feature_sets = {'all13': all_cols}
    if selected_name != 'all13':
        feature_sets['reduced'] = selected_columns
    regime_scores, configs = [], {}
    for feature_name, columns in feature_sets.items():
        for years in protocol['history_regimes']:
            name = feature_name + '_' + '_'.join(map(str, years))
            configs[name] = {'columns': columns, 'years': years}
            if years == [2025]:
                original = 'all13' if feature_name == 'all13' else selected_name
                score = dict(next(s for s in scores if s['candidate'] == original))
                score['candidate'] = name
            else:
                score = cv(name, columns, years, base, bx, series, historical, folds,
                           cv_predictions, fold_scores, cv_epochs)
            regime_scores.append(score)
    chosen = sorted(regime_scores, key=lambda s: (-s['correct'], s['log_loss']))[0]['candidate']
    pd.DataFrame(regime_scores).to_csv(OUT / 'history_selection_2025.csv', index=False)
    pd.DataFrame(cv_predictions).to_csv(OUT / 'cv_predictions_2025.csv', index=False)
    pd.DataFrame(fold_scores).to_csv(OUT / 'cv_folds_2025.csv', index=False)
    (OUT / 'cv_epochs.json').write_text(json.dumps(cv_epochs, indent=2) + '\n')
    lock = {'selected_feature_candidate': selected_name, 'selected_features': [INPUTS[i] for i in selected_columns],
            'chosen_using_2025': chosen, 'configs': configs,
            'training_counts': {'2023': len(historical[2023][0]), '2024': len(historical[2024][0]), '2025': len(base)},
            'ensemble': 'mean set probabilities of three seeds, then series polynomial'}
    bundles = {}
    for name, config in configs.items():
        frame, x, ss = combine(base, bx, series, historical, config['years'])
        nets, prep, epochs = fit_ensemble(frame, x, ss, config['columns'])
        bundles[name] = {'weights': [n.state_dict() for n in nets], 'median': prep.median,
                         'scaler': prep.scaler, 'epochs': epochs, **config}
        config['epochs'] = epochs
        config['train_sets'] = len(frame)
        config['parameters_per_seed'] = sum(p.numel() for p in nets[0].parameters())
        print('FINAL FIT', name, len(frame), epochs, flush=True)
    (OUT / 'selection_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    joblib.dump(bundles, OUT / 'frozen_2025.joblib')
    checkpoint_hash = hashlib.sha256((OUT / 'frozen_2025.joblib').read_bytes()).hexdigest()
    # Only now load2026, after all feature/year/epoch choices are locked.
    full, xx, ss, _ = prepare([2025, 2026])
    np.testing.assert_allclose(bx, xx[full.date.dt.year.eq(2025)], equal_nan=True, rtol=0, atol=0)
    test = ss[ss.date.dt.year.eq(2026)]
    results, predictions, individual = [], [], []
    for name, bundle in bundles.items():
        columns = bundle['columns']
        nets = [make_net(s, columns) for s in SEEDS]
        for net, weights in zip(nets, bundle['weights']):
            net.load_state_dict(weights)
        prep = Scale()
        prep.median, prep.scaler = bundle['median'], bundle['scaler']
        p, members = predict(nets, prep, xx[test.game_index], columns)
        q = series_probability(p, test.best_of.to_numpy())
        results.append({'candidate': name, 'selected_on_2025': name == chosen, 'n': len(test),
                        'correct': int(((q >= .5) == test.result.to_numpy()).sum()), **evaluate(test.result, q)})
        for seed, probs in zip(SEEDS, members):
            individual.append({'candidate': name, 'seed': seed,
                               **evaluate(test.result, series_probability(probs, test.best_of.to_numpy()))})
        for row, prob, set_p in zip(test.itertuples(), q, p):
            predictions.append({'candidate': name, 'series_key': row.series_key, 'date': str(row.date),
                                'team1': row.team1, 'team2': row.team2, 'best_of': row.best_of,
                                'y': row.result, 'p': float(prob), 'set_probability': float(set_p)})
    pred = pd.DataFrame(predictions)
    old = pd.read_csv(ROOT / 'notebook/experiments/10_player_grouped/predictions_2026.csv')
    old = old[old.track.eq('position_additive')]
    baseline_pred = pred[pred.candidate.eq('all13_2025')].merge(old, on='series_key', validate='one_to_one')
    assert len(baseline_pred) == len(test) == 186
    np.testing.assert_array_equal(baseline_pred.y_x, baseline_pred.y_y)
    np.testing.assert_allclose(baseline_pred.p, baseline_pred.series_probability, atol=1e-6, rtol=0)
    pd.DataFrame(results).to_csv(OUT / 'evaluation_2026.csv', index=False)
    pred.to_csv(OUT / 'predictions_2026.csv', index=False)
    pd.DataFrame(individual).to_csv(OUT / 'seed_scores_2026.csv', index=False)
    monthly = []
    for (candidate, month), g in pred.groupby(['candidate', pd.to_datetime(pred.date).dt.to_period('M')]):
        monthly.append({'candidate': candidate, 'month': str(month), 'n': len(g), **evaluate(g.y, g.p.to_numpy())})
    pd.DataFrame(monthly).to_csv(OUT / 'monthly_2026.csv', index=False)
    paired = []
    original = pred[pred.candidate.eq('all13_2025')]
    for name, g in pred.groupby('candidate'):
        pair = g.merge(original, on='series_key', suffixes=('_new', '_base'), validate='one_to_one')
        a = pair.p_new.ge(.5).eq(pair.y_new)
        b = pair.p_base.ge(.5).eq(pair.y_base)
        paired.append({'candidate': name, 'new_only_correct': int((a & ~b).sum()),
                       'baseline_only_correct': int((~a & b).sum()), 'both_correct': int((a & b).sum()),
                       'both_wrong': int((~a & ~b).sum())})
    pd.DataFrame(paired).to_csv(OUT / 'paired_comparison.csv', index=False)
    assert hashlib.sha256((OUT / 'frozen_2025.joblib').read_bytes()).hexdigest() == checkpoint_hash
    manifest = {'checkpoint_sha256': checkpoint_hash,
                'source_sha256': {str(y): hashlib.sha256(source(y).read_bytes()).hexdigest() for y in [2023,2024,2025,2026]},
                'checks': ['all13_2025 predictions reproduce experiment10 within1e-6',
                           '2025 inputs unchanged when2026 appended', 'team swap symmetry',
                           'scalers fitted only on fitting data', '2026 loaded after selection lock', 'checkpoint unchanged']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('2025 SELECTED:', chosen, flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
