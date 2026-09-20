"""Coarse-to-fine player feature search and controlled historical-data diagnostics.

All decisions use2025. The2026 replay is loaded only after model locking.
"""
from copy import deepcopy
from itertools import combinations
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch import nn

from backend.training.player_history_ablation import combine, make_net, predict
from backend.training.player_two_track import ROOT, INPUTS, ROLES, SEEDS, prepare, source, series_probability
from backend.training.selected_league_leads import Scale, build_data
from backend.training.train_model import evaluate

OUT = ROOT / 'notebook/experiments/14_player_subset_search'
GROUPS = {'form_history': [0, 12], 'lane15': [1, 2, 3], 'kda': [4, 5, 6],
          'damage': [7, 8], 'economy': [9, 10], 'vision': [11]}
ALL = list(range(13))


def key(columns):
    return 'f' + format(sum(1 << i for i in columns), '04x')


def coarse_candidates():
    groups = list(GROUPS.values())
    candidates = set()
    for count in range(1, len(groups) + 1):
        for subset in combinations(groups, count):
            candidates.add(tuple(sorted(i for group in subset for i in group)))
    candidates.update((i,) for i in ALL)
    candidates.update(tuple(j for j in ALL if j != i) for i in ALL)
    return sorted(candidates, key=lambda c: (len(c), c))


def sample_weights(frame, recent=False):
    values = frame.date.dt.year.map({2023: .25, 2024: .5, 2025: 1.}).to_numpy() if recent else np.ones(len(frame))
    assert np.isfinite(values).all() and (values > 0).all()
    return torch.tensor(values, dtype=torch.float32)


def scaler_mask(frame, policy):
    if policy == '2025_only':
        mask = frame.date.dt.year.eq(2025).to_numpy()
    elif policy == 'pooled':
        mask = np.ones(len(frame), dtype=bool)
    else:
        raise ValueError(policy)
    assert mask.any() and frame.date.dt.year.le(2025).all()
    return mask


def train(seed, columns, x, y, weights, epochs=500, validation=None, record=False):
    net = make_net(seed, columns)
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    best, stale, chosen, saved = float('inf'), 0, epochs, None
    history = []
    equal_weights = bool(torch.all(weights == weights[0]))
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        logits = net(x)
        if equal_weights:
            # Preserve the original mean-reduction backward path exactly. An
            # algebraically equivalent weighted sum changes float32 rounding,
            # which can affect later early-stopping choices on small datasets.
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        else:
            per_example = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
            loss = (per_example * weights).sum() / weights.sum()
        loss.backward()
        opt.step()
        row = {'epoch': epoch, 'train_loss': float(loss.detach())}
        if validation is not None:
            vx, vy, bo = validation
            with torch.no_grad():
                q = series_probability(torch.sigmoid(net(vx)).numpy(), bo)
            score = log_loss(vy, q, labels=[0, 1])
            row['validation_series_log_loss'] = float(score)
            if score < best:
                best, stale, chosen, saved = score, 0, epoch, deepcopy(net.state_dict())
            else:
                stale += 1
        if record:
            history.append(row)
        if validation is not None and stale >= 30:
            break
    if saved is not None:
        net.load_state_dict(saved)
    net.eval()
    return net, chosen, history


def fit(frame, x, series, columns, seeds, policy='pooled', recent=False, fixed_epochs=None, record=False):
    dates = np.sort(series.date.dt.normalize().unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * .8)])
    val = series[series.date.ge(cutoff)]
    fm = frame.date.lt(cutoff).to_numpy()
    assert frame.date.dt.year.le(2025).all() and frame.loc[fm, 'date'].max() < val.date.min()
    xx = x[..., columns]
    prep = Scale().fit(xx[fm & scaler_mask(frame, policy)])
    xf, yf = prep.transform(xx[fm]), torch.tensor(frame.loc[fm, 'result'].to_numpy(), dtype=torch.float32)
    vx = prep.transform(xx[val.game_index])
    validation = (vx, val.result.to_numpy(), val.best_of.to_numpy())
    epochs, curves = [], []
    for i, seed in enumerate(seeds):
        if fixed_epochs is None:
            _, epoch, h = train(seed, columns, xf, yf, sample_weights(frame[fm], recent), validation=validation, record=record)
            curves.extend({'seed': seed, **r} for r in h)
        else:
            epoch = fixed_epochs[i]
        epochs.append(epoch)
    final_prep = Scale().fit(xx[scaler_mask(frame, policy)])
    xa, ya = final_prep.transform(xx), torch.tensor(frame.result.to_numpy(), dtype=torch.float32)
    weights = sample_weights(frame, recent)
    nets = [train(s, columns, xa, ya, weights, epochs=e)[0] for s, e in zip(seeds, epochs)]
    return nets, final_prep, epochs, curves


class Search:
    def __init__(self, base, bx, series, folds):
        self.base, self.bx, self.series, self.folds = base, bx, series, folds
        self.cache = OUT / 'search_cache'
        self.cache.mkdir(exist_ok=True)
        self.prefixes = []
        for fold in folds:
            start, end = pd.Timestamp(fold['start']), pd.Timestamp(fold['end'])
            frame, x, past = combine(base, bx, series, {}, [2025], start)
            future = series[series.date.ge(start) & series.date.lt(end)]
            assert frame.date.max() < future.date.min()
            self.prefixes.append((frame, x, past, future))

    def seed_result(self, columns, seed):
        path = self.cache / f'{key(columns)}_seed{seed}.json'
        if path.exists():
            return json.loads(path.read_text())
        rows, epochs = [], []
        for i, (frame, x, past, future) in enumerate(self.prefixes):
            nets, prep, ep, _ = fit(frame, x, past, columns, [seed])
            p, _ = predict(nets, prep, self.bx[future.game_index], columns)
            epochs.append({'fold': i + 1, 'epoch': ep[0], 'train_sets': len(frame)})
            for row, pp in zip(future.itertuples(), p):
                rows.append({'fold': i + 1, 'series_key': row.series_key, 'date': str(row.date),
                             'y': int(row.result), 'best_of': int(row.best_of), 'set_p': float(pp)})
        result = {'columns': columns, 'seed': seed, 'predictions': rows, 'epochs': epochs}
        path.write_text(json.dumps(result) + '\n')
        return result

    def evaluate(self, columns, seeds):
        members = [self.seed_result(columns, s) for s in seeds]
        first = pd.DataFrame(members[0]['predictions'])
        for member in members[1:]:
            assert [r['series_key'] for r in member['predictions']] == first.series_key.tolist()
        p = np.stack([np.array([r['set_p'] for r in m['predictions']], dtype=np.float32) for m in members]).mean(axis=0)
        q = series_probability(p, first.best_of.to_numpy())
        metrics = evaluate(first.y, q)
        row = {'candidate': key(columns), 'columns': list(columns), 'features': [INPUTS[i] for i in columns],
               'inputs': len(columns), 'seeds': seeds, 'n': len(first),
               'correct': int(((q >= .5) == first.y.to_numpy()).sum()), **metrics}
        first['p'] = q
        row['folds'] = [{'fold': int(fold), 'n': len(g), 'correct': int((g.p.ge(.5) == g.y).sum()),
                         **evaluate(g.y, g.p.to_numpy())} for fold, g in first.groupby('fold')]
        return row


def rank(rows):
    return sorted(rows, key=lambda r: (-r['correct'], r['log_loss'], r['inputs'], r['candidate']))


def distributions(base, bx, historical):
    frames = {2025: (base, bx), **historical}
    ref_mean, ref_std = np.nanmean(bx, axis=(0, 1)), np.nanstd(bx, axis=(0, 1))
    rows = []
    for year, (frame, x) in sorted(frames.items()):
        means, stds = np.nanmean(x, axis=(0, 1)), np.nanstd(x, axis=(0, 1))
        for role in range(5):
            for col, name in enumerate(INPUTS):
                diff = x[:, 0, role, col] - x[:, 1, role, col]
                good = np.isfinite(diff)
                corr = np.corrcoef(diff[good], frame.result.to_numpy()[good])[0, 1] if np.std(diff[good]) > 0 else 0.
                rows.append({'year': year, 'role': ROLES[role], 'feature': name, 'mean': means[role, col],
                             'std': stds[role, col], 'mean_shift_in_2025_std': (means[role, col] - ref_mean[role, col]) / max(ref_std[role, col], 1e-8),
                             'std_ratio_to_2025': stds[role, col] / max(ref_std[role, col], 1e-8),
                             'diff_win_correlation': corr, 'valid_sets': int(good.sum())})
    pd.DataFrame(rows).to_csv(OUT / 'year_feature_diagnostics.csv', index=False)


def gradient_diagnostics(base, bx, historical, bundle):
    prep = Scale()
    prep.median, prep.scaler = bundle['median'], bundle['scaler']
    rows = []
    for seed, weights in zip(SEEDS, bundle['weights']):
        for stage in ['initial', 'after_2025_fit']:
            net = make_net(seed, ALL)
            if stage == 'after_2025_fit':
                net.load_state_dict(weights)
            gradients, losses = {}, {}
            for year, (frame, x) in {2025: (base, bx), **historical}.items():
                net.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(net(prep.transform(x)), torch.tensor(frame.result.to_numpy(), dtype=torch.float32))
                loss.backward()
                gradients[year] = torch.cat([p.grad.flatten() for p in net.parameters()]).detach().numpy()
                losses[year] = float(loss.detach())
            for year in [2023, 2024]:
                a, b = gradients[year], gradients[2025]
                rows.append({'seed': seed, 'stage': stage, 'historical_year': year,
                             'gradient_cosine_with_2025': float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
                             'historical_loss': losses[year], 'loss_2025': losses[2025]})
    pd.DataFrame(rows).to_csv(OUT / 'gradient_diagnostics.csv', index=False)


def run():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'search': 'All63 nonempty combinations of6 semantic groups, plus13 singleton and13 leave-one-out anchors; refine best3 screening candidates by toggling each individual feature once. Deduplicate.',
        'groups': GROUPS, 'screen_seeds': [42], 'confirmation_seeds': SEEDS,
        'confirmation': 'top12 screening candidates plus all13 and previous XP15-removal anchor; same3 chronological2025 folds',
        'ranking': 'ensemble series correct count descending, log loss ascending, feature count ascending',
        'search_scope': '13 existing player inputs only; not exhaustive8191 subsets; all candidates use A d→8→4 + role heads',
        'folds': 'same123series as experiment13; inner chronological early stopping; max500 patience30 Adam.001 fullbatch',
        'historical_controls': 'same baseline2025/2026 inputs; separate within-year historical input histories; LCK only',
        'history_selection': 'freeze best confirmed feature subset, compare2025/2024-25/2023-25 on2025 CV; accuracy first',
        'diagnostics': 'all13 pooled2023-25 compared with2025-only scaler, recency weights(.25,.5,1), or matched baseline2025-only epoch counts; one change at a time',
        'diagnostic_selection': 'diagnostic variants are prespecified mechanistic comparisons, not candidates for the selected feature/year model',
        'warnings': ['Adaptive search winner is not a global optimum and2025 search scores are optimistic.',
                     '2026 reused retrospective audit, loaded only after selection/weights locked.',
                     'Shared initialization and scaler fit use training data only.',
                     'First-set roster known, iid series conversion; no player IDs or champions as input.'],
    }
    signature = {'protocol': protocol, 'source2025': hashlib.sha256(source(2025).read_bytes()).hexdigest(),
                 'code': hashlib.sha256(__file__.encode() + open(__file__, 'rb').read()).hexdigest()}
    signature_path = OUT / 'cache_signature.json'
    if signature_path.exists():
        assert json.loads(signature_path.read_text()) == signature, 'Search cache belongs to different inputs or code'
    else:
        signature_path.write_text(json.dumps(signature, indent=2) + '\n')
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    base, bx, series, _ = prepare([2025])
    dates = np.sort(series.date.dt.normalize().unique())
    boundaries = [pd.Timestamp(dates[int(len(dates) * f)]) for f in [.4, .6, .8]] + [series.date.max().normalize() + pd.Timedelta(days=1)]
    folds = [{'fold': i + 1, 'start': str(boundaries[i]), 'end': str(boundaries[i + 1])} for i in range(3)]
    (OUT / 'folds.json').write_text(json.dumps(folds, indent=2) + '\n')
    search = Search(base, bx, series, folds)
    # Verify the full chronological control before spending time on the search.
    control = search.evaluate(ALL, SEEDS)
    previous = pd.read_csv(ROOT / 'notebook/experiments/13_player_history_ablation/feature_selection_2025.csv')
    previous = previous[previous.candidate.eq('all13')].iloc[0]
    assert control['correct'] == int(previous.correct)
    np.testing.assert_allclose(control['log_loss'], previous.log_loss, atol=1e-7, rtol=0)
    print('BASELINE CV VERIFIED', control['correct'], '/', control['n'], flush=True)
    screening, seen = [], set()
    def screen(candidates, stage):
        for cols in candidates:
            columns = tuple(sorted(cols))
            if not columns or columns in seen:
                continue
            seen.add(columns)
            row = search.evaluate(list(columns), [42])
            row['stage'] = stage
            screening.append(row)
            (OUT / 'screening_2025.json').write_text(json.dumps(rank(screening), indent=2) + '\n')
            print('SCREEN', len(screening), key(columns), row['correct'], '/', row['n'], 'inputs', len(columns), flush=True)
    screen(coarse_candidates(), 'coarse')
    neighbors = set()
    centers = rank(screening)[:3]
    for center in centers:
        columns = set(center['columns'])
        for i in ALL:
            changed = tuple(sorted(columns ^ {i}))
            if changed:
                neighbors.add(changed)
    screen(sorted(neighbors, key=lambda c: (len(c), c)), 'individual_refinement')
    shortlist = {tuple(r['columns']) for r in rank(screening)[:12]}
    shortlist.update([tuple(ALL), tuple(i for i in ALL if i != 2)])
    confirmation = []
    for cols in sorted(shortlist, key=lambda c: (len(c), c)):
        row = search.evaluate(list(cols), SEEDS)
        confirmation.append(row)
        (OUT / 'confirmation_2025.json').write_text(json.dumps(rank(confirmation), indent=2) + '\n')
        print('CONFIRM', row['candidate'], row['correct'], '/', row['n'], 'inputs', row['inputs'], flush=True)
    selected = rank(confirmation)[0]
    columns = selected['columns']
    print('FEATURE LOCK', selected['candidate'], selected['features'], flush=True)
    historical = {}
    for year in [2023, 2024]:
        frame, x, _, audit = build_data([year], ['LCK'])
        historical[year] = (frame, x[..., :13])
        audit.to_csv(OUT / f'historical_audit_{year}.csv', index=False)
    distributions(base, bx, historical)
    configs = {
        'reference_2025': {'columns': ALL, 'years': [2025]},
        'pooled_2023_2025': {'columns': ALL, 'years': [2023, 2024, 2025]},
        'pooled_2025_scaler': {'columns': ALL, 'years': [2023, 2024, 2025], 'policy': '2025_only'},
        'pooled_recent_weights': {'columns': ALL, 'years': [2023, 2024, 2025], 'recent': True},
        'pooled_matched_epochs': {'columns': ALL, 'years': [2023, 2024, 2025], 'matched': True},
        'selected_2025': {'columns': columns, 'years': [2025]},
        'selected_2024_2025': {'columns': columns, 'years': [2024, 2025]},
        'selected_2023_2025': {'columns': columns, 'years': [2023, 2024, 2025]},
    }
    cv_results, cv_rows, cv_epochs = [], [], []
    for name, config in configs.items():
        all_rows = []
        for i, fold in enumerate(folds):
            cutoff = pd.Timestamp(fold['start'])
            future = series[series.date.ge(cutoff) & series.date.lt(pd.Timestamp(fold['end']))]
            frame, x, past = combine(base, bx, series, historical, config['years'], cutoff)
            fixed = [search.seed_result(ALL, s)['epochs'][i]['epoch'] for s in SEEDS] if config.get('matched') else None
            nets, prep, epochs, _ = fit(frame, x, past, config['columns'], SEEDS,
                                        policy=config.get('policy', 'pooled'), recent=config.get('recent', False), fixed_epochs=fixed)
            p, _ = predict(nets, prep, bx[future.game_index], config['columns'])
            q = series_probability(p, future.best_of.to_numpy())
            cv_epochs.append({'candidate': name, 'fold': i + 1, 'epochs': epochs})
            for row, pp in zip(future.itertuples(), q):
                all_rows.append({'candidate': name, 'fold': i + 1, 'series_key': row.series_key, 'date': str(row.date), 'y': row.result, 'p': float(pp)})
        d = pd.DataFrame(all_rows)
        cv_results.append({'candidate': name, 'n': len(d), 'correct': int(d.p.ge(.5).eq(d.y).sum()), **evaluate(d.y, d.p.to_numpy())})
        cv_rows.extend(all_rows)
        print('HISTORY CV', name, cv_results[-1]['correct'], 'logloss', cv_results[-1]['log_loss'], flush=True)
    chosen = sorted([r for r in cv_results if r['candidate'].startswith('selected_')], key=lambda r: (-r['correct'], r['log_loss']))[0]['candidate']
    pd.DataFrame(cv_results).to_csv(OUT / 'history_diagnostics_cv_2025.csv', index=False)
    pd.DataFrame(cv_rows).to_csv(OUT / 'history_cv_predictions_2025.csv', index=False)
    (OUT / 'history_cv_epochs.json').write_text(json.dumps(cv_epochs, indent=2) + '\n')
    bundles, all_curves = {}, []
    for name, config in configs.items():
        frame, x, past = combine(base, bx, series, historical, config['years'])
        fixed = bundles['reference_2025']['epochs'] if config.get('matched') else None
        nets, prep, epochs, curves = fit(frame, x, past, config['columns'], SEEDS,
                                        policy=config.get('policy', 'pooled'), recent=config.get('recent', False), fixed_epochs=fixed, record=True)
        bundles[name] = {**config, 'epochs': epochs, 'weights': [n.state_dict() for n in nets],
                         'median': prep.median, 'scaler': prep.scaler, 'train_sets': len(frame),
                         'parameters': sum(p.numel() for p in nets[0].parameters())}
        all_curves.extend({'candidate': name, **r} for r in curves)
        print('FINAL FIT', name, epochs, flush=True)
    gradient_diagnostics(base, bx, historical, bundles['reference_2025'])
    pd.DataFrame(all_curves).to_csv(OUT / 'early_stopping_curves_2025.csv', index=False)
    lock = {'screened_subsets': len(screening), 'confirmed_subsets': len(confirmation),
            'selected_subset': selected, 'selected_feature_year_candidate': chosen,
            'configs': {k: {f: v for f, v in b.items() if f not in ['weights', 'median', 'scaler']} for k, b in bundles.items()}}
    (OUT / 'selection_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    joblib.dump(bundles, OUT / 'frozen_2025.joblib')
    checkpoint_hash = hashlib.sha256((OUT / 'frozen_2025.joblib').read_bytes()).hexdigest()
    # No2026 data or labels loaded anywhere above this line.
    full, xx, ss, _ = prepare([2025, 2026])
    np.testing.assert_allclose(xx[full.date.dt.year.eq(2025)], bx, equal_nan=True, atol=0, rtol=0)
    test = ss[ss.date.dt.year.eq(2026)]
    results, predictions, seed_scores = [], [], []
    for name, bundle in bundles.items():
        nets = [make_net(s, bundle['columns']) for s in SEEDS]
        for net, weights in zip(nets, bundle['weights']):
            net.load_state_dict(weights)
        prep = Scale()
        prep.median, prep.scaler = bundle['median'], bundle['scaler']
        p, members = predict(nets, prep, xx[test.game_index], bundle['columns'])
        q = series_probability(p, test.best_of.to_numpy())
        results.append({'candidate': name, 'selected_on_2025': name == chosen, 'n': len(test),
                        'correct': int(((q >= .5) == test.result.to_numpy()).sum()),
                        'mean_abs_distance_from_50': float(np.abs(q - .5).mean()), **evaluate(test.result, q)})
        for row, pp in zip(test.itertuples(), q):
            predictions.append({'candidate': name, 'series_key': row.series_key, 'date': str(row.date),
                                'team1': row.team1, 'team2': row.team2, 'y': row.result, 'p': float(pp)})
        for seed, pp in zip(SEEDS, members):
            seed_scores.append({'candidate': name, 'seed': seed, **evaluate(test.result, series_probability(pp, test.best_of.to_numpy()))})
    pred = pd.DataFrame(predictions)
    old = pd.read_csv(ROOT / 'notebook/experiments/13_player_history_ablation/predictions_2026.csv')
    for new, previous in [('reference_2025', 'all13_2025'), ('pooled_2023_2025', 'all13_2023_2024_2025')]:
        paired = pred[pred.candidate.eq(new)].merge(old[old.candidate.eq(previous)], on='series_key', validate='one_to_one')
        assert len(paired) == 186
        np.testing.assert_array_equal(paired.y_x, paired.y_y)
        np.testing.assert_allclose(paired.p_x, paired.p_y, atol=1e-6, rtol=0)
    pd.DataFrame(results).to_csv(OUT / 'evaluation_2026.csv', index=False)
    pred.to_csv(OUT / 'predictions_2026.csv', index=False)
    pd.DataFrame(seed_scores).to_csv(OUT / 'seed_scores_2026.csv', index=False)
    monthly = []
    for (candidate, month), g in pred.groupby(['candidate', pd.to_datetime(pred.date).dt.to_period('M')]):
        monthly.append({'candidate': candidate, 'month': str(month), 'n': len(g), **evaluate(g.y, g.p.to_numpy())})
    pd.DataFrame(monthly).to_csv(OUT / 'monthly_2026.csv', index=False)
    assert hashlib.sha256((OUT / 'frozen_2025.joblib').read_bytes()).hexdigest() == checkpoint_hash
    manifest = {'checkpoint_sha256': checkpoint_hash,
                'source_sha256': {str(y): hashlib.sha256(source(y).read_bytes()).hexdigest() for y in [2023, 2024, 2025, 2026]},
                'checks': ['baseline and pooled baseline reproduce experiment13 within1e-6', 'same2025 inputs',
                           'team-swap probability complement', 'checkpoint unchanged', 'all choices locked before2026 load']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('CHOSEN ON2025:', chosen, 'features:', selected['features'], flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
