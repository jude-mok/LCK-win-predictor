from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,nbformat
root=Path('/Users/seungyunmok/Developer/LOL_ML');sys.path.insert(0,str(root))
from backend.training.mlp_experiment import *
from sklearn.linear_model import LogisticRegression
from IPython.display import display
from backend.training.train_model import evaluate
out=root/'notebook/experiments/07_full_comparison';out.mkdir(exist_ok=True,parents=True)
n=nbformat.read(root/'notebook/01_mlp.ipynb',4)
# Replay the recorded feature preparation, retaining exact definitions.
exec(n.cells[27].source.replace("e5_raw = load_many_lck_team_games(e5_paths)", "e5_raw = load_many_lck_team_games(e5_paths).drop(columns=['golddiffat10','firstbaron','barons','opp_barons'], errors='ignore')"))
# Add the earlier positive-lead feature for the original nine-input experiment.
e5_games['positive15']=e5_games.golddiffat15.gt(0).astype(float)
e5_games['lead15_rate']=e5_games.groupby('teamname').positive15.transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean()).fillna(.5)
first=e5_games.sort_values(['date','gameid']).drop_duplicates(['series_key','teamname']).set_index(['series_key','teamname'])
e5_matches['diff_lead15_rate']=[float(first.loc[(r.series_key,r.team1),'lead15_rate']-first.loc[(r.series_key,r.team2),'lead15_rate']) for r in e5_matches.itertuples()]
dev=e5_matches.iloc[:int(len(e5_matches)*.8)].copy();dev['day']=dev.date.dt.normalize()
recent=e5_matches.iloc[len(dev):].copy();recent['day']=recent.date.dt.normalize()
rank=json.loads((root/'notebook/experiments/06_feature_search/search_ranking.json').read_text())
groups={}
for label,cols in [('base_7',DIFF_COLS),('lead10_baron_9',e5_groups['lead10_baron_9']),('lead15_baron_9',e5_groups['lead15_baron_9']),('lead20_baron_9',e5_groups['lead20_baron_9']),('strong_lead15_8',list(DIFF_COLS)+['diff_strong_lead15_rate']),('original_lead15_9',list(DIFF_COLS)+['diff_lead15_rate','diff_strong_lead15_rate']),('minimal_3',['diff_roll_winrate','diff_roll_golddiff15'])]:
 groups[label]=list(cols)
for i,r in enumerate(rank):
 if not any(set(r['columns'])==set(c) for c in groups.values()):groups[f'search_{i+1:02d}']=r['columns']
json.dump(groups,open(out/'feature_groups.json','w'),indent=2)
days=np.sort(dev.day.unique());folds=[]
for fold,(a,b) in enumerate(TimeSeriesSplit(n_splits=3).split(days),1):
 past=dev[dev.day.isin(days[a])];future=dev[dev.day.isin(days[b])];pdys=np.sort(past.day.unique());cut=int(len(pdys)*.8)
 folds.append((fold,past[past.day.isin(pdys[:cut])],past[past.day.isin(pdys[cut:])],future))
cut=int(len(days)*.8);folds.append((4,dev[dev.day.isin(days[:cut])],dev[dev.day.isin(days[cut:])],recent))
foldinfo=[]
for fold,fit,stop,future in folds:
 assert fit.date.max()<stop.date.min()<future.date.min()
 foldinfo.append({'fold':fold,'fit':len(fit),'stop':len(stop),'evaluation':len(future),'start':str(future.date.min()),'end':str(future.date.max())})
json.dump(foldinfo,open(out/'folds.json','w'),indent=2)
records=[]
for index,(name,cols) in enumerate(groups.items(),1):
 for fold,fit,stop,future in folds:
  for seed in TS_SEEDS:
   p,epoch=ts_fit_predict(fit,stop,future,cols,seed)
   records.extend({'variant':name,'fold':fold,'seed':seed,'epoch':epoch,'series_key':r.series_key,'y':int(r.result),'p':float(prob)} for r,prob in zip(future.itertuples(),p))
 pd.DataFrame(records).to_csv(out/'predictions.csv',index=False)
 print(f'{index}/{len(groups)} done: {name}',flush=True)
# All models get identical rows; constant and linear references use fit data only.
for fold,fit,stop,future in folds:
 scale=StandardScaler().fit(fit[DIFF_COLS]);lr=LogisticRegression(max_iter=2000).fit(scale.transform(fit[DIFF_COLS]),fit.result)
 p=(lr.predict_proba(scale.transform(future[DIFF_COLS]))[:,1]+1-lr.predict_proba(scale.transform(-future[DIFF_COLS]))[:,1])/2
 for name,probs in [('logistic_6',p),('always_50',np.full(len(future),.5)),('fit_winrate',np.full(len(future),fit.result.mean()))]:
  records.extend({'variant':name,'fold':fold,'seed':0,'epoch':0,'series_key':r.series_key,'y':int(r.result),'p':float(prob)} for r,prob in zip(future.itertuples(),probs))
preds=pd.DataFrame(records);preds.to_csv(out/'predictions.csv',index=False)
ens=preds.groupby(['variant','fold','series_key'],as_index=False).agg(y=('y','first'),p=('p','mean'))
summary=[]
for name,g in ens.groupby('variant'):
 for label,rows in [('all_311',g),('development_232',g[g.fold<4]),('recent_79',g[g.fold==4])]:
  assert not rows.series_key.duplicated().any()
  summary.append({'variant':name,'period':label,'n':len(rows),'correct':int(((rows.p>=.5)==rows.y).sum()),**evaluate(rows.y,rows.p)})
summary=pd.DataFrame(summary);summary.to_csv(out/'ensemble_scores.csv',index=False)
seedrows=[]
for (name,seed),g in preds.groupby(['variant','seed']):seedrows.append({'variant':name,'seed':seed,**evaluate(g.y,g.p)})
pd.DataFrame(seedrows).to_csv(out/'individual_seed_scores.csv',index=False)
board=summary[summary.period.eq('all_311')].sort_values(['accuracy','log_loss'],ascending=[False,True])
print(board[['variant','n','correct','accuracy','log_loss','brier_score']].head(15).to_string(index=False),flush=True)
print('WINNER FEATURES',groups.get(board.iloc[0].variant),flush=True)
print(summary[summary.variant.isin(board.head(5).variant)].to_string(index=False),flush=True)
# Attach executable reproduction instructions and saved outputs to the notebook.
text='''## 실험 07 — 이전 후보 전체 동일 조건 재비교

- 이전 49조합 및 초기 실험의 추가 조합을 중복 제거해 비교. 모든 MLP는 16→8 은닉층, seed 42/43/44, 동일 early stopping 및 입력 순서 대칭 처리.
- 예측 확률을 세 seed에서 평균한 ensemble로 정확도 순위를 비교. 동률은 log loss로 정렬. 개별 seed 점수도 별도 저장.
- 개발 3개 fold 232시리즈 + 뒤 79시리즈 = 서로 겹치지 않는 311시리즈. 모든 학습/early stopping 구간은 해당 평가 구간보다 과거다.
- 여러 조합 선택에 재사용한 데이터다. 미관측 test나 확정 미래 성능이 아니다. 초기 84시리즈는 학습에 필요하므로 평가에서 제외.
- 전체 정확도로 선택하며 최근 구간은 별도로 보고한다. 기존 70.51%는 특정 fold의 seed 평균으로, 이 표의 전체 ensemble 정확도와 다르다.
'''
cell=nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\ncomparison_dir = Path('"+str(out)+"')\ncomparison = pd.read_csv(comparison_dir / 'ensemble_scores.csv')\ndisplay(comparison[comparison.period.eq('all_311')].sort_values(['accuracy','log_loss'], ascending=[False,True]))")
cell.outputs=[nbformat.v4.new_output('display_data',data={'text/plain':board.to_string(index=False),'text/html':board.to_html(index=False)})]
# Read latest notebook to preserve unrelated changes made while this runs.
notebook=nbformat.read(root/'notebook/01_mlp.ipynb',4);notebook.cells.extend([nbformat.v4.new_markdown_cell(text),cell]);nbformat.write(notebook,root/'notebook/01_mlp.ipynb')
