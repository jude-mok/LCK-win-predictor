import numpy as np
from backend.training.feature_pipeline import (
    FEATURE_COLS, DIFF_COLS, build_featured_games, build_match_dataset, build_team_state,
)

MLP_FEATURES = FEATURE_COLS + ['strong_lead10_rate', 'firstbaron_rate']
MLP_DIFFS = ['diff_' + c for c in MLP_FEATURES]


def prepare_mlp_data(raw):
    raw = raw.copy()
    missing = raw.firstbaron.isna()
    no_baron = raw.barons.eq(0) & raw.opp_barons.eq(0)
    if (missing & ~no_baron).any():
        raise ValueError('Unexplained firstbaron missing values')
    raw.loc[missing & no_baron, 'firstbaron'] = 0
    if not raw.firstbaron.isin([0, 1]).all() or not np.isfinite(raw.golddiffat10).all():
        raise ValueError('Invalid MLP source features')
    games = build_featured_games(raw).sort_values(['teamname', 'date', 'gameid']).copy()
    games['lead10_indicator'] = games.golddiffat10.ge(1000).astype(float)
    for name, source in [('strong_lead10_rate', 'lead10_indicator'), ('firstbaron_rate', 'firstbaron')]:
        games[name] = games.groupby('teamname')[source].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0.5)
    matches = build_match_dataset(games)
    first = games.sort_values(['date', 'gameid']).drop_duplicates(['series_key','teamname']).set_index(['series_key','teamname'])
    for name in MLP_FEATURES[-2:]:
        matches['diff_' + name] = [float(first.loc[(x.series_key,x.team1),name] - first.loc[(x.series_key,x.team2),name]) for x in matches.itertuples()]
    matches['is_bo5'] = matches.best_of.eq(5).astype(float)
    matches['day'] = matches.date.dt.normalize()
    state = build_team_state(games)
    for team in state.index:
        history = games[games.teamname.eq(team)].sort_values(['date','gameid']).tail(5)
        state.loc[team, 'strong_lead10_rate'] = history.lead10_indicator.mean()
        state.loc[team, 'firstbaron_rate'] = history.firstbaron.mean()
    return games, matches, state
