# 동일 조건 모델 재비교

51개 MLP 조합 및 3개 기준 모델. 모든 MLP는 seed 42/43/44 확률 평균. 전체 311개는 재사용한 평가 데이터이며 독립 최종 test가 아니다.

| 순위 | 모델 | 피처 (is_bo5 추가) | 전체 정답 | 전체 정확도 | 최근 79 정확도 | 전체 log loss |
|---:|---|---|---:|---:|---:|---:|
| 1 | search_20 | diff_roll_golddiff15 | 194/311 | 62.38% | 51.90% | 0.6715 |
| 2 | search_23 | diff_roll_winrate, diff_roll_firstherald | 194/311 | 62.38% | 60.76% | 0.6738 |
| 3 | search_06 | diff_roll_winrate, diff_strong_lead15_rate | 192/311 | 61.74% | 53.16% | 0.6758 |
| 4 | minimal_3 | diff_roll_winrate, diff_roll_golddiff15 | 191/311 | 61.41% | 54.43% | 0.6716 |
| 5 | search_21 | diff_strong_lead20_rate | 190/311 | 61.09% | 49.37% | 0.6902 |
| 6 | search_07 | diff_roll_winrate | 188/311 | 60.45% | 58.23% | 0.6778 |
| 7 | search_04 | diff_roll_winrate, diff_roll_golddiff15, diff_patch_winrate | 187/311 | 60.13% | 50.63% | 0.6712 |
| 8 | search_05 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon | 186/311 | 59.81% | 51.90% | 0.6703 |
| 9 | search_02 | diff_roll_winrate, diff_strong_lead20_rate | 185/311 | 59.49% | 48.10% | 0.6744 |
| 10 | search_11 | diff_roll_winrate, diff_firstbaron_rate | 185/311 | 59.49% | 51.90% | 0.6746 |
| 11 | search_09 | diff_roll_winrate, diff_roll_golddiff15, diff_firstbaron_rate | 185/311 | 59.49% | 55.70% | 0.6757 |
| 12 | search_13 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstherald | 185/311 | 59.49% | 53.16% | 0.6771 |
| 13 | search_30 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 185/311 | 59.49% | 54.43% | 0.6795 |
| 14 | search_47 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 185/311 | 59.49% | 50.63% | 0.6830 |
| 15 | search_08 | diff_roll_winrate, diff_roll_golddiff15, diff_strong_lead15_rate | 184/311 | 59.16% | 48.10% | 0.6777 |
| 16 | search_12 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 184/311 | 59.16% | 51.90% | 0.6788 |
| 17 | search_40 | diff_roll_firstherald | 184/311 | 59.16% | 56.96% | 0.6846 |
| 18 | search_43 | diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 184/311 | 59.16% | 55.70% | 0.6889 |
| 19 | search_37 | diff_strong_lead10_rate | 184/311 | 59.16% | 60.76% | 0.6895 |
| 20 | search_10 | diff_roll_winrate, diff_roll_golddiff15, diff_strong_lead20_rate | 183/311 | 58.84% | 53.16% | 0.6722 |
| 21 | search_34 | diff_roll_winrate, diff_roll_golddiff15, diff_strong_lead10_rate | 183/311 | 58.84% | 54.43% | 0.6790 |
| 22 | search_46 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 183/311 | 58.84% | 50.63% | 0.6851 |
| 23 | search_49 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 183/311 | 58.84% | 51.90% | 0.6871 |
| 24 | search_25 | diff_strong_lead15_rate | 183/311 | 58.84% | 55.70% | 0.6882 |
| 25 | search_42 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 182/311 | 58.52% | 54.43% | 0.6760 |
| 26 | search_48 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_firstbaron_rate | 182/311 | 58.52% | 55.70% | 0.6845 |
| 27 | search_45 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_strong_lead20_rate | 182/311 | 58.52% | 50.63% | 0.6862 |
| 28 | search_29 | diff_roll_firsttower | 182/311 | 58.52% | 54.43% | 0.6883 |
| 29 | search_19 | diff_roll_winrate, diff_roll_firstdragon | 182/311 | 58.52% | 55.70% | 0.6889 |
| 30 | search_28 | diff_roll_winrate, diff_strong_lead10_rate | 180/311 | 57.88% | 49.37% | 0.6763 |
| 31 | search_17 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 180/311 | 57.88% | 51.90% | 0.6767 |
| 32 | search_03 | diff_roll_winrate, diff_patch_winrate | 180/311 | 57.88% | 50.63% | 0.6782 |
| 33 | search_14 | diff_roll_winrate, diff_roll_firsttower | 180/311 | 57.88% | 48.10% | 0.6801 |
| 34 | search_44 | diff_roll_winrate, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 180/311 | 57.88% | 53.16% | 0.6864 |
| 35 | search_36 | diff_patch_winrate | 179/311 | 57.56% | 51.90% | 0.6754 |
| 36 | search_16 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_patch_winrate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 179/311 | 57.56% | 53.16% | 0.6829 |
| 37 | search_41 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 179/311 | 57.56% | 51.90% | 0.6832 |
| 38 | base_7 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate | 178/311 | 57.23% | 53.16% | 0.6945 |
| 39 | lead10_baron_9 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead10_rate, diff_firstbaron_rate | 177/311 | 56.91% | 54.43% | 0.6762 |
| 40 | lead20_baron_9 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead20_rate, diff_firstbaron_rate | 177/311 | 56.91% | 54.43% | 0.6769 |
| 41 | search_15 | diff_roll_winrate, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 177/311 | 56.91% | 53.16% | 0.6826 |
| 42 | search_18 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firsttower | 176/311 | 56.59% | 50.63% | 0.6759 |
| 43 | lead15_baron_9 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead15_rate, diff_firstbaron_rate | 176/311 | 56.59% | 54.43% | 0.6774 |
| 44 | search_32 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 176/311 | 56.59% | 49.37% | 0.6806 |
| 45 | search_22 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 176/311 | 56.59% | 53.16% | 0.6828 |
| 46 | search_24 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead15_rate, diff_strong_lead20_rate | 175/311 | 56.27% | 51.90% | 0.6948 |
| 47 | search_27 | diff_firstbaron_rate | 174/311 | 55.95% | 49.37% | 0.6854 |
| 48 | original_lead15_9 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_lead15_rate, diff_strong_lead15_rate | 172/311 | 55.31% | 50.63% | 0.6865 |
| 49 | logistic_6 | baseline | 171/311 | 54.98% | 50.63% | 0.6630 |
| 50 | strong_lead15_8 | diff_roll_winrate, diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead15_rate | 166/311 | 53.38% | 46.84% | 0.6896 |
| 51 | search_39 | diff_roll_golddiff15, diff_roll_firstdragon, diff_roll_firstherald, diff_roll_firsttower, diff_patch_winrate, diff_strong_lead15_rate, diff_strong_lead20_rate, diff_firstbaron_rate | 163/311 | 52.41% | 56.96% | 0.6899 |
| 52 | always_50 | baseline | 159/311 | 51.13% | 55.70% | 0.6931 |
| 53 | fit_winrate | baseline | 152/311 | 48.87% | 44.30% | 0.7111 |
| 54 | search_33 | diff_roll_firstdragon | 144/311 | 46.30% | 48.10% | 0.6941 |

선택 규칙: 전체 정확도 최대, 동률이면 log loss 최소. 선택 결과 search_20: diff_roll_golddiff15 + is_bo5. 최근 구간 성능 저하로 미래 우위는 확인되지 않았다. 배포 모델 변경은 이 비교 스크립트에서 수행하지 않는다.
