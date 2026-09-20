"""Build a readable report and executed notebook from experiment13 artifacts."""
import json

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from backend.training.player_history_ablation import OUT, ROOT

LABELS = {
    'past_result': '최근 승률', 'past_golddiffat15': '15분 골드 차이',
    'past_xpdiffat15': '15분 경험치 차이', 'past_csdiffat15': '15분 CS 차이',
    'past_kills_per_min': '분당 킬', 'past_deaths_per_min': '분당 데스',
    'past_assists_per_min': '분당 어시스트', 'past_dpm': 'DPM',
    'past_damageshare': '피해 점유율', 'past_earnedgoldshare': '획득 골드 점유율',
    'past_cspm': 'CSPM', 'past_vision_per_min': '분당 시야 점수',
    'history_fraction': '기록량',
}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def run():
    features = pd.read_csv(OUT / 'feature_selection_2025.csv')
    folds = pd.read_csv(OUT / 'cv_folds_2025.csv')
    regimes = pd.read_csv(OUT / 'history_selection_2025.csv')
    results = pd.read_csv(OUT / 'evaluation_2026.csv')
    pred = pd.read_csv(OUT / 'predictions_2026.csv')
    cv_pred = pd.read_csv(OUT / 'cv_predictions_2025.csv')
    lock = json.loads((OUT / 'selection_lock.json').read_text())
    base = features.iloc[0]
    base_folds = folds[folds.candidate.eq('all13')].set_index('fold').correct
    feature_rows = []
    for row in features.sort_values(['correct', 'log_loss'], ascending=[False, True]).itertuples():
        name = LABELS.get(row.candidate.removeprefix('without_'), row.candidate)
        if row.candidate == 'all13':
            name = '삭제 없음 (13개)'
        elif row.candidate == 'joint_removal':
            name = '유리했던 피처 동시 제거'
        own = folds[folds.candidate.eq(row.candidate)].set_index('fold').correct
        positive = int((own > base_folds).sum())
        feature_rows.append([name, f'{row.correct}/{row.n}', f'{row.accuracy:.2%}',
                             f'{row.correct - base.correct:+d}', f'{row.log_loss:.4f}', f'{positive}/3'])
    feature_table = table(['제거한 피처', '정답', '정확도', '기준 대비 정답', 'Log loss', '개선 구간 수'], feature_rows)
    def result_table(frame):
        return table(['구성', '정답', '정확도', 'Log loss', 'AUC'],
                     [[r.candidate, f'{r.correct}/{r.n}', f'{r.accuracy:.2%}', f'{r.log_loss:.4f}', f'{r.roc_auc:.4f}']
                      for r in frame.itertuples()])
    original = pred[pred.candidate.eq('all13_2025')]
    intervals = []
    rng = np.random.default_rng(42)
    for name, g in pred.groupby('candidate'):
        pair = g.merge(original, on='series_key', suffixes=('_new', '_base'), validate='one_to_one')
        delta = (pair.p_new.ge(.5).eq(pair.y_new).astype(int) - pair.p_base.ge(.5).eq(pair.y_base).astype(int)).to_numpy()
        boot = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
        low, high = np.quantile(boot, [.025, .975])
        intervals.append({'candidate': name, 'accuracy_delta': float(delta.mean()),
                          'ci95_low': float(low), 'ci95_high': float(high)})
    pd.DataFrame(intervals).to_csv(OUT / 'paired_bootstrap_2026.csv', index=False)
    selected = results[results.candidate.eq(lock['chosen_using_2025'])].iloc[0]
    removed = [v for k, v in LABELS.items() if k not in lock['selected_features']]
    report = f'''# 실험 13 — 선수 피처 제거와 과거 연도 학습

## 결정과 요약

- 2025년 검증으로 고정한 피처 후보: `{lock['selected_feature_candidate']}`. 제거: {', '.join(removed) or '없음'}.
- 2025년 검증으로 선택한 전체 구성: **{lock['chosen_using_2025']}**.
- 선택 구성의 2026년 회고 평가: **{selected.correct}/{selected.n} = {selected.accuracy:.2%}**, log loss {selected.log_loss:.4f}.
- 기존 A 기준은 118/186 = 63.44%. 아래 2026 결과에서 최고값을 골라 선택 결정을 바꾸지 않았다.
- 현재 연구 기준은 기존 LCK-only 13피처 A로 유지한다. 2025에서 정한 새 후보가 이를 대체할 근거를 얻지 못했다.
- 온라인 배포와 기존 모델 파일은 변경하지 않았다.

## 비교 설계

1. LCK-only A 구조, 3 seed 확률 앙상블, Adam .001, 최대500/patience30 유지.
2. 2025년 날짜의 앞40%로 시작하는 세 개 expanding-window fold: 40→60%, 60→80%, 80→100%. 총123시리즈. 각 fold의 학습 구간 내부 마지막20% 날짜에서 epoch를 정한 뒤 해당 학습 구간 전체로 재학습했다. 외부 평가 구간은 epoch 선택에 사용하지 않았다.
3. 13개 피처를 하나씩 제거해 실제 재학습했다. 나머지 초기 가중치는 동일하게 유지했다. 정확도와 log loss가 모두 좋아진 피처가 둘 이상이면 상위 최대3개를 동시에 제거하는 후보도 시험했다. 전체13개도 후보로 남겼다.
4. 피처를 먼저 고정하고, 전체13개와 축소 입력 각각에서 2025 / 2024~2025 / 2023~2025를 비교했다. 정확도를 우선하고 동점에서 log loss로 선택했다.
5. 최종 epoch는 기존과 같은2025 마지막33시리즈로 정하고, 해당 연도 전체로 재학습했다. 피처·연도·epoch·가중치를 저장한 뒤2026을 로드했다.

검증 기간: {cv_pred.date.min()} ~ {cv_pred.date.max()}.
회고 평가 기간: {pred.date.min()} ~ {pred.date.max()}.

## 선수 피처를 하나씩 제거한 결과 — 2025년 검증

{feature_table}

양수는 해당 피처를 빼고 다시 학습했을 때 더 맞힌 시리즈 수다. 이것은 해당 통계가 본질적으로 해롭다는 뜻이 아니다. 피처 중복, 작은 표본, 최적화와 모델 용량 변화가 함께 작용한다. 세 구간의 일관성도 함께 봐야 한다. 이 표를 보고 후보를 선택했으므로 선택된 검증 점수에는 선택 편향이 있다.

이번에는 XP 차이 제거(+2)와 CSPM 제거(+1)가 각각 세 구간 중 한 구간에서만 개선됐다. 둘을 함께 제거하면 오히려78/123으로 낮아졌다. 따라서 두 피처를 확실히 해로운 피처로 분류할 수 없다. 시야 점수와 기록량 제거는 각각6개, 승률 제거는5개를 덜 맞혔다.

## 과거 데이터 품질과 통제

- 2023 LCK 488세트 중15분 골드/XP/CS 통계가 없는1세트를 제외해487세트 사용.
- 2024 LCK 482세트 사용. 2025 LCK 555세트 유지.
- 학습 표본 수: 555 → 1,037 → 1,524세트. 모두 동일한 세트 가중치.
- 2025·2026 입력은 기존 A와 완전히 동일하다. 2023·2024는 각각 해당 연도 내부의 과거 기록으로 피처를 생성했다. 따라서 이번 실험은 오래된 학습 표본 추가 효과를 본다. 2024 기록으로2025 시즌 초 입력을 보강하는 효과는 시험하지 않았다.
- 리그 확대, 선수 ID embedding, 밴픽, 시간 가중치, fine-tuning은 추가하지 않았다.

## 학습 연도 비교 — 2025년 검증 및 선택

`all13`은 기존13피처, `reduced`는 위에서 고정한 축소 피처다.

{result_table(regimes)}

## 고정된 구성의 2026년 회고 평가

{result_table(results)}

2026년은 이전 연구에서 여러 번 살펴본 자료다. 독립된 새 테스트가 아니다. 2025년 검증으로 선택한 구성은 `{lock['chosen_using_2025']}`이며, 2026년 최고 점수와 구분한다. 정확도와 log loss가 다른 방향으로 움직이면 어느 지표가 개선됐는지 구분해야 한다.

13피처에서는 과거 연도를 더할수록118→112→110개로 정확도가 낮아졌다. XP를 뺀12피처에서는106→109→115개로 개선됐지만 기존 A보다3개 적었다. 반면12피처/2025의 log loss는0.6688로 기존0.6754보다 낮았다. 과거 데이터가 항상 해롭다거나 피처 축소가 모든 지표에서 실패했다고 일반화할 수 없다. 이번의 동일 가중치 혼합 방식으로는 기준 정확도를 넘지 못했다.

`paired_comparison.csv`는 기존 A와 비교해 새로 맞히거나 놓친 시리즈 수를 기록한다. `paired_bootstrap_2026.csv`는 시리즈별 정답 차이를 10,000회 재표집한 기술적95% 구간이다. 경기 사이의 의존성이나 반복적인 모델 선택은 반영하지 않으므로 확정적인 유의성 검정으로 해석하지 않는다.

## 검증과 재현

- 기존 all13/2025 예측을 실험10과 확률 오차1e-6 이내로 재현.
- 2026 추가 시2025 입력 불변, 팀 순서 반전 대칭, 평가 전 선택 고정, 체크포인트 불변 확인.
- `python -m unittest backend.tests.test_player_history_ablation backend.tests.test_player_two_track`
- `python -m backend.training.player_history_ablation`
- `python -m backend.training.report_player_history_ablation`

원본 선택 결과·fold별 결과·예측·scaler·가중치·source hash는 이 폴더에 저장했다.
'''
    (OUT / 'RESULTS.md').write_text(report)
    cells = [nbformat.v4.new_markdown_cell(report), nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
        f"RESULTS = Path({str(OUT)!r})\n"
        "features = pd.read_csv(RESULTS/'feature_selection_2025.csv')\n"
        "evaluation = pd.read_csv(RESULTS/'evaluation_2026.csv')\n"
        "display(features.sort_values(['correct','log_loss'], ascending=[False,True]))\n"
        "display(evaluation)"), nbformat.v4.new_code_cell(
        "base = features.loc[features.candidate.eq('all13'),'correct'].iloc[0]\n"
        "f = features[features.candidate.ne('all13')].sort_values('correct').copy()\n"
        "fig, ax = plt.subplots(figsize=(10,6))\n"
        "d = f.correct-base\n"
        "ax.barh(f.candidate.str.replace('without_','',regex=False), d, color=['#27836d' if v>0 else '#b85c54' for v in d])\n"
        "ax.axvline(0,color='black',linewidth=.8)\n"
        "ax.set_xlabel('Change in correct series after removing feature (2025 CV, n=123)')\n"
        "ax.set_title('Feature removal: retrained models, paired initialization')\n"
        "fig.tight_layout()\nfig.savefig(RESULTS/'feature_removal.png',dpi=160)\nplt.show()"),
        nbformat.v4.new_code_cell(
        "fig, ax = plt.subplots(figsize=(10,4))\n"
        "ax.barh(evaluation.candidate, evaluation.accuracy*100,color='#367aa6')\n"
        "ax.axvline(118/186*100,color='#b85c54',linestyle='--',label='Original A: 118/186')\n"
        "for i,r in enumerate(evaluation.itertuples()): ax.text(r.accuracy*100+.4,i,f'{r.correct}/186',va='center')\n"
        "ax.set_xlim(0,85)\nax.set_xlabel('Series accuracy (%)')\n"
        "ax.set_title('Frozen models: 2026 retrospective comparison')\nax.legend()\n"
        "fig.tight_layout()\nfig.savefig(RESULTS/'history_comparison.png',dpi=160)\nplt.show()")]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    NotebookClient(notebook, timeout=120, resources={'metadata': {'path': str(ROOT)}}).execute()
    path = ROOT / 'notebook/06_player_history_ablation.ipynb'
    nbformat.write(notebook, path)
    print('Wrote', OUT / 'RESULTS.md', 'and', path)


if __name__ == '__main__':
    run()
