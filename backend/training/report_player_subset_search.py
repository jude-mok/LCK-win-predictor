"""Render experiment14 tables, diagnostics and an executed explanation notebook."""
import json

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from backend.training.player_subset_search import OUT, ROOT, INPUTS
from backend.training.report_player_history_ablation import LABELS, table

NAMES = {
    'reference_2025': '기존13 / 2025',
    'pooled_2023_2025': '기존13 / 2023~25 동일 가중치',
    'pooled_2025_scaler': '기존13 / 과거 추가 + 2025 표준화',
    'pooled_recent_weights': '기존13 / 과거 추가 + 최근 가중치',
    'pooled_matched_epochs': '기존13 / 과거 추가 + 학습 횟수 맞춤',
    'selected_2025': '선택 조합 / 2025',
    'selected_2024_2025': '선택 조합 / 2024~25',
    'selected_2023_2025': '선택 조합 / 2023~25',
}


def run():
    lock = json.loads((OUT / 'selection_lock.json').read_text())
    confirmation = json.loads((OUT / 'confirmation_2025.json').read_text())
    screening = json.loads((OUT / 'screening_2025.json').read_text())
    selected = lock['selected_subset']
    results = pd.read_csv(OUT / 'evaluation_2026.csv')
    cv = pd.read_csv(OUT / 'history_diagnostics_cv_2025.csv')
    diagnostics = pd.read_csv(OUT / 'year_feature_diagnostics.csv')
    gradients = pd.read_csv(OUT / 'gradient_diagnostics.csv')
    curves = pd.read_csv(OUT / 'early_stopping_curves_2025.csv')
    predictions = pd.read_csv(OUT / 'predictions_2026.csv')
    chosen_name = lock['selected_feature_year_candidate']
    chosen = results[results.candidate.eq(chosen_name)].iloc[0]
    baseline = results[results.candidate.eq('reference_2025')].iloc[0]
    subset_names = [LABELS[f] for f in selected['features']]
    removed = [LABELS[f] for f in INPUTS if f not in selected['features']]
    def metric_table(frame):
        return table(['구성', '정답', '정확도', 'Log loss', 'AUC'],
                     [[NAMES[r.candidate], f'{r.correct}/{r.n}', f'{r.accuracy:.2%}', f'{r.log_loss:.4f}', f'{r.roc_auc:.4f}'] for r in frame.itertuples()])
    top_table = table(['피처 수', '사용 피처', '2025 정답', '정확도', 'Log loss'],
                      [[r['inputs'], ', '.join(LABELS[f] for f in r['features']), f"{r['correct']}/{r['n']}",
                        f"{r['accuracy']:.2%}", f"{r['log_loss']:.4f}"] for r in confirmation[:8]])
    summary_rows = []
    for year, group in diagnostics[diagnostics.year.ne(2025)].groupby('year'):
        agg = group.groupby('feature').mean_shift_in_2025_std.apply(lambda x: x.abs().mean()).sort_values(ascending=False)
        for feature, shift in agg.head(5).items():
            summary_rows.append([year, LABELS[feature], f'{shift:.3f}'])
    shifts_table = table(['연도', '피처', '포지션 평균 절대 이동 / 2025 표준편차'], summary_rows)
    epochs_table = table(['구성', 'seed42', 'seed43', 'seed44'],
                         [[NAMES[name], *bundle['epochs']] for name, bundle in lock['configs'].items()])
    matched = results[results.candidate.eq('pooled_matched_epochs')].iloc[0]
    pooled = results[results.candidate.eq('pooled_2023_2025')].iloc[0]
    weighted = results[results.candidate.eq('pooled_recent_weights')].iloc[0]
    rescaled = results[results.candidate.eq('pooled_2025_scaler')].iloc[0]
    paired_rows = []
    original = predictions[predictions.candidate.eq('reference_2025')]
    for name, group in predictions.groupby('candidate'):
        pair = group.merge(original, on='series_key', suffixes=('_new', '_old'), validate='one_to_one')
        a, b = pair.p_new.ge(.5).eq(pair.y_new), pair.p_old.ge(.5).eq(pair.y_old)
        paired_rows.append({'candidate': name, 'new_only_correct': int((a & ~b).sum()),
                            'reference_only_correct': int((~a & b).sum()), 'both_correct': int((a & b).sum()),
                            'both_wrong': int((~a & ~b).sum())})
    pd.DataFrame(paired_rows).to_csv(OUT / 'paired_comparison.csv', index=False)
    early_summary = []
    for (name, seed), group in curves.groupby(['candidate', 'seed']):
        best_epoch = lock['configs'][name]['epochs'][[42, 43, 44].index(seed)]
        early_summary.append({'candidate': name, 'seed': int(seed), 'chosen_epoch': best_epoch,
                              'first_validation_loss': float(group.iloc[0].validation_series_log_loss),
                              'best_validation_loss': float(group.validation_series_log_loss.min()),
                              'last_validation_loss': float(group.iloc[-1].validation_series_log_loss),
                              'last_epoch': int(group.iloc[-1].epoch)})
    pd.DataFrame(early_summary).to_csv(OUT / 'early_stopping_summary.csv', index=False)
    report = f'''# 실험14 — 선수 피처 조합 탐색과 과거 데이터 진단

## 선택 결과

선수 피처 **{lock['screened_subsets']}개 조합**을1개 seed로 탐색하고, 상위 후보와 비교 기준 **{lock['confirmed_subsets']}개 조합**을3개 seed 확률 앙상블로 재확인했다. 모두2025년 시간순 검증123시리즈만 사용했다.

선택한 **{selected['inputs']}피처**: {', '.join(subset_names)}.
제외한 피처: {', '.join(removed) or '없음'}.
구조: `{selected['inputs']}→8→4`, 포지션별 `4→2→1` 합산. 입력 수만 달라지고 나머지 A 구조는 같다.

조합 검증 점수는 **{selected['correct']}/{selected['n']} = {selected['accuracy']:.2%}**, log loss {selected['log_loss']:.4f}. 이것은 조합 선택에 사용한 점수이므로 독립 성능이 아니다.
2025에서 추가로 선택한 학습 연도 구성은 **{NAMES[chosen_name]}**. 이 고정 후보의2026 회고 결과는 **{chosen.correct}/{chosen.n} = {chosen.accuracy:.2%}**, 기존 A 대비 **{int(chosen.correct - baseline.correct):+d}시리즈**다.

## 탐색 범위와 평가 규칙

- 기존13개 선수 피처를 대상으로 승률·기록량 / 라인전15분 / KDA / 피해량 / 경제력 / 시야의6개 그룹을 만들고, 비어 있지 않은 그룹 조합63개를 모두 평가했다.
- 개별 피처1개만 쓰는13개 조합과1개씩 제거하는13개 조합을 추가했다. 중복 제거 후 첫 단계87개.
- seed42로 평가한 상위3개 조합에서 각 피처를 하나씩 추가/제거하는 이웃 조합을 탐색했다.
- 최종 screening 상위12개와 전체13개·기존 XP제거 기준을3개 seed로 재확인했다. 세트 확률을 평균한 후 시리즈 확률로 바꿨다.
- 정확도 우선, 동점 시 log loss, 다시 동점이면 피처 수로 정렬했다.
- **13개 피처의8,191개 전체 조합을 전수 탐색한 것은 아니다. 이번 탐색에서 가장 좋은 조합이다.** 한 seed의 screening에서 제외된 안정적인 조합이 있을 수 있다.
-2025 날짜 기준40→60%,60→80%,80→100%의3개 외부 구간. 각 구간보다 앞선 학습 데이터 내부에서만 epoch를 선택하고 재학습했다. 초기 가중치는 유지한 입력끼리 일치시켰다.
-2025 검증 자료를 여러 후보 선택에 반복 사용했으므로 검증 최고값은 낙관적이다.2026도 프로젝트에서 반복 열람한 회고 평가이며 untouched test가 아니다.

## 상위 조합 — 3 seed 확인 결과

{top_table}

## 학습 연도와 진단 조건 — 2025 검증

{metric_table(cv)}

학습 연도 선택은 `selected_*` 세 구성 안에서만 수행했다. 진단 조건은 원인 후보를 비교하기 위해 미리 정했으며,2026 점수를 보고 최종 모델을 고르는 데 사용하지 않았다.

## 고정한 후보의2026 회고 결과

{metric_table(results)}

평가 기간: {predictions.date.min()} ~ {predictions.date.max()},186시리즈. 첫 세트 명단과 그 시점 이전 통계만 사용하고, 독립·동일 세트 승률을 가정해 Bo3/Bo5로 변환한다.

## 데이터가 늘면 왜 떨어질 수 있나 — 이번에 확인한 내용

### 1. 학습 표본 수와 목표 시즌의 비중은 다르다

2025만555세트를 학습할 때 목표 시즌 비중은100%다.2023의487세트와2024의482세트를 동일하게 섞으면2025 비중은 **36.42%**로 줄어든다. 표본 수는1,524개로 늘지만 최적화하는 손실은 과거 두 시즌의 영향을 더 많이 받는다.

선수 입력은 기존2025/2026과 동일하게 유지했다.2023·2024 입력은 각 연도 내부 rolling 기록이다. 이번 진단은 추가 표본을 섞는 효과이며,2024 기록으로2025 시즌 초 선수 이력을 보강하는 효과는 포함하지 않는다.

최근 가중치 진단은2023=.25,2024=.5,2025=1로 고정했고, scaler는 원래 pooled 상태를 유지했다.2025의 손실 가중치 비중은60.47%가 된다.2026 정답은 동일 가중치 **{pooled.correct}개 → {weighted.correct}개**였다. 이 한 가지 가중치 설정의 결과로 모든 시간 가중치 방식을 일반화할 수 없다.

### 2. 표준화 기준과 선수 통계 분포가 달라진다

포지션별로2023/2024 평균이2025 평균에서 얼마나 떨어져 있는지2025 표준편차 단위로 계산했다. 다음은 각 연도에서 평균 절대 이동이 큰5개 피처다.

{shifts_table}

예를 들어 정글 선수의 과거 평균 DPM은2023년 약299,2024년342,2025년434였다. 전체 파일에서 같은 포지션의 피처 평균을 비교한 값이다. 입력 분포가 달라진 채 학습 표본을 섞는다는 구체적인 예다.

이는 입력 분포 차이의 관측 증거다. 패치, 로스터, 리그 운영 중 무엇이 차이를 만들었는지 분리 검증한 것은 아니다. 나머지를 유지하고 결측 보충·표준화 기준만2025로 제한한 결과는 **{pooled.correct}개 → {rescaled.correct}개**였다. 이 진단은 전처리의 영향을 비교하며 원본 게임 환경 변화 전체를 제거하는 실험은 아니다.

### 3. 학습 횟수 선택도 크게 바뀐다

{epochs_table}

기존13/2025는8·7·77 epoch, 기존13/2023~25는1·2·67 epoch가 선택됐다. 후자는 두 seed가 매우 이른 시점에서 멈춘 모델로 최종 재학습됐다. 최대 epoch를500으로 지정했다고500번 학습된 것은 아니다. validation33시리즈에서 확률 손실로 epoch를 고르기 때문이다.

과거 데이터를 섞되 학습 횟수를2025-only 기준과 동일하게 맞춘 진단은 **{pooled.correct}개 → {matched.correct}개**였다.2025 외부fold에서도 해당fold의 과거2025 자료로 선택한 epoch만 가져와 사용했다.2026 점수로 학습 횟수를 고른 것은 아니다.

이번 세 통제 실험은 입력·가중치·학습 종료 정책을 각각 바꾼 관측 비교다. 여러 요인이 상호작용할 수 있어 전체 정확도 하락을 원인별 퍼센트로 분해하거나 하나의 확정 원인으로 선언하지 않는다.

관측상 표준화 변경은1개 회복, 최근 가중치는1개 하락, 학습 횟수 맞춤은 정답 수 동일이었다. 따라서 조기 종료나 표준화 하나가 정확도 하락의 전부라는 근거는 얻지 못했다. 학습 횟수를 맞추면 log loss는0.6811→0.6741로 개선됐다. 확률의 품질과0.5 기준 승자 정확도가 서로 다르게 움직이는 실제 사례다.

### 4. 과거와 최근 손실의 학습 방향도 점검했다

`gradient_diagnostics.csv`에는 같은 모델·2025 scaler에서 각 과거 연도와2025의 손실 gradient 사이 cosine을 저장했다. 초기 상태와2025-only 학습 후를 seed별로 비교한다. 음수는 그 지점에서 한쪽 손실을 줄이는 작은 이동이 다른 쪽 손실과 충돌할 수 있다는 뜻이다. 국소적인 진단이며2026 일반화 저하의 인과 증명은 아니다.

실제 측정한12개 cosine은 모두 양수({gradients.gradient_cosine_with_2025.min():.3f}~{gradients.gradient_cosine_with_2025.max():.3f})였다. 따라서 이번 측정에서 과거와2025의 학습 gradient가 반대 방향이라는 증거는 나오지 않았다. 데이터 분포 차이는 확인했지만 성능 하락의 단일 원인은 아직 특정하지 못했다.

## 재현 및 산출물

- 테스트10개 통과. 동일 가중치 학습을 기존 구현과 비교했고, baseline/pooled 예측을 실험13과1e-6 이내로 재현했다.
- 예비 실행에서 동일 가중치에도 weighted-sum BCE를 사용하자 float32 누적 차이로 기준 CV가79→80개로 달라져 폐기했다. 수정본은 원래 mean BCE 경로와 파라미터가 정확히 일치하는지 테스트하고, 전체 기준 CV79/123 재현 후 탐색했다. 예비 기록은 `14_player_subset_search_initial_numeric_audit/`에 분리했으며 선택 근거로 사용하지 않는다.
- 평가 전 선택·epoch·가중치 고정,2025 입력 불변, 팀 순서 대칭, 체크포인트 불변 확인.
- `python -m backend.training.player_subset_search`
- `python -m backend.training.report_player_subset_search`
- 검색 결과는 `screening_2025.json`, `confirmation_2025.json`, `search_cache/`에 저장. 선택 결과는 `selection_lock.json`, 가중치/scaler는 `frozen_2025.joblib`.
- 온라인 모델이나 기존 연구 결과를 덮어쓰지 않았다.
'''
    (OUT / 'RESULTS.md').write_text(report)
    cells = [nbformat.v4.new_markdown_cell(report), nbformat.v4.new_code_cell(
        "from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
        f"RESULTS = Path({str(OUT)!r})\n"
        "evaluation = pd.read_csv(RESULTS/'evaluation_2026.csv')\n"
        "confirmation = pd.DataFrame(json.loads((RESULTS/'confirmation_2025.json').read_text()))\n"
        "display(confirmation[['inputs','features','correct','accuracy','log_loss']])\n"
        "display(evaluation)"), nbformat.v4.new_code_cell(
        "curves = pd.read_csv(RESULTS/'early_stopping_curves_2025.csv')\n"
        "fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)\n"
        "for ax,seed in zip(axes,[42,43,44]):\n"
        "    for name,label in [('reference_2025','2025 only'),('pooled_2023_2025','2023-2025 pooled')]:\n"
        "        g=curves[(curves.candidate==name)&(curves.seed==seed)]\n"
        "        ax.plot(g.epoch,g.validation_series_log_loss,label=label)\n"
        "        best=g.loc[g.validation_series_log_loss.idxmin()]\n"
        "        ax.scatter(best.epoch,best.validation_series_log_loss,s=35)\n"
        "    ax.set_title(f'Seed {seed}'); ax.set_xlabel('Epoch'); ax.legend(fontsize=8)\n"
        "axes[0].set_ylabel('2025 validation series log loss')\n"
        "fig.suptitle('More historical data changes the selected stopping point')\n"
        "fig.tight_layout(); fig.savefig(RESULTS/'early_stopping.png',dpi=160); plt.show()"),
        nbformat.v4.new_code_cell(
        "fig,ax=plt.subplots(figsize=(11,5))\n"
        "colors=['#367aa6' if not r.selected_on_2025 else '#27836d' for r in evaluation.itertuples()]\n"
        "ax.barh(evaluation.candidate,evaluation.accuracy*100,color=colors)\n"
        "ax.axvline(118/186*100,color='#b85c54',linestyle='--',label='Original A: 118/186')\n"
        "for i,r in enumerate(evaluation.itertuples()): ax.text(r.accuracy*100+.4,i,f'{r.correct}/186',va='center')\n"
        "ax.set_xlim(0,85); ax.set_xlabel('Series accuracy (%)'); ax.legend(loc='lower right')\n"
        "ax.set_title('Frozen subset search and historical-data controls: 2026 replay')\n"
        "fig.tight_layout(); fig.savefig(RESULTS/'comparison_2026.png',dpi=160); plt.show()")]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    NotebookClient(notebook, timeout=120, resources={'metadata': {'path': str(ROOT)}}).execute()
    path = ROOT / 'notebook/07_player_subset_search.ipynb'
    nbformat.write(notebook, path)
    print('Wrote', path, 'and', OUT / 'RESULTS.md')


if __name__ == '__main__':
    run()
