import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. 데이터 불러오기 및 전처리
# 전체 데이터셋과 2단계 필터링된 데이터셋을 모두 불러옵니다.
df_full = pd.read_csv("cicids2017_cleaned_for_model.csv")
df_2nd_stage = pd.read_csv("2nd_stage_dataset.csv")

# 1단계 XGBoost 모델로 걸러진 데이터만 담은 filtered_df를 재구성합니다.
X_full = df_full.drop('Label', axis=1)
y_full = df_full['Label']

model_xgb = joblib.load('model_xgb_1st_stage.pkl')
le = joblib.load('le_xgb.pkl')

y_pred_encoded_full = model_xgb.predict(X_full)
y_pred_label_full = le.inverse_transform(y_pred_encoded_full)

# XGBoost가 오탐(False Positive)하거나, 미탐(False Negative)한 데이터를 추출
# 즉, XGBoost가 잘못 분류한 모든 데이터입니다.
filtered_df = df_full[y_full!= y_pred_label_full]

# 2. BENIGN 트래픽 일부를 샘플링하여 2단계 데이터셋에 추가
# 전체 데이터셋에서 BENIGN 트래픽만 추출합니다.
benign_data = df_full[df_full['Label'] == 'BENIGN']

# '정상' 트래픽 10,000개를 무작위로 샘플링하여 2단계 학습에 포함시킵니다.
# 이는 1단계 모델의 오탐(false positive)을 보완하고, 2단계 모델이 정상 트래픽의
# 복잡한 패턴도 학습하도록 유도합니다.
benign_sample = benign_data.sample(n=10000, random_state=42)

# 3. 2단계 모델 학습용 최종 데이터셋 결합 및 저장
# XGBoost가 놓친 데이터(filtered_df)와 정상 트래픽 샘플을 합칩니다.
df_final_2nd_stage = pd.concat([filtered_df, benign_sample], ignore_index=True)

# 결합된 데이터셋을 무작위로 섞어줍니다.
df_final_2nd_stage = df_final_2nd_stage.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n1단계 XGBoost 모델로 걸러진 오탐/미탐 데이터 크기: {len(filtered_df)}개")
print(f"2단계 모델 학습용 데이터셋에 추가된 정상 트래픽 샘플: {len(benign_sample)}개")
print(f"2단계 모델 학습용 최종 데이터셋 크기: {len(df_final_2nd_stage)}개")

# 2단계 모델 학습용 데이터셋 저장
# 이 데이터셋이 다음 단계인 딥러닝 모델 학습에 사용됩니다.
df_final_2nd_stage.to_csv("2nd_stage_dataset_enriched.csv", index=False)
print("2단계 모델 학습용 데이터셋이 '2nd_stage_dataset_enriched.csv' 파일로 저장되었습니다.")