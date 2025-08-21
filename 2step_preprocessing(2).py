import pandas as pd
import numpy as np
import joblib

# 1단계에서 저장한 전처리된 전체 데이터 불러오기
file_path = "cicids2017_cleaned_for_model.csv"
df_full = pd.read_csv(file_path)

# 샘플링할 목표 샘플 수 설정
TARGET_SAMPLES = 10000

# 각 레이블별로 샘플링을 진행할 리스트
sampled_dataframes = []

# 전체 데이터셋에 존재하는 고유한 레이블 목록을 가져옵니다.
unique_labels = df_full['Label'].unique()
print("데이터셋에 존재하는 총 공격 유형:", list(unique_labels))

print("\n각 공격 유형별로 데이터 샘플링 중...")
for label in unique_labels:
    # 현재 레이블에 해당하는 데이터만 추출
    label_data = df_full[df_full['Label'] == label]
    num_samples = len(label_data)
    
    # 만약 현재 데이터가 TARGET_SAMPLES보다 적으면, 모든 데이터를 사용
    if num_samples < TARGET_SAMPLES:
        sampled_dataframes.append(label_data)
        print(f"- '{label}': {num_samples}개 (전체 포함)")
    # TARGET_SAMPLES 이상이면, 무작위로 10,000개만 샘플링
    else:
        sampled_dataframes.append(label_data.sample(n=TARGET_SAMPLES, random_state=42))
        print(f"- '{label}': {TARGET_SAMPLES}개 (무작위 샘플링)")

# 샘플링된 데이터프레임들을 하나로 합칩니다.
df_final_2nd_stage = pd.concat(sampled_dataframes, ignore_index=True)

# 결합된 데이터셋을 무작위로 섞어줍니다.
df_final_2nd_stage = df_final_2nd_stage.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n2단계 모델 학습용 최종 데이터셋 크기: {len(df_final_2nd_stage)}개")
print("\n최종 데이터셋에 존재하는 공격 종류 및 개수:")
print(df_final_2nd_stage['Label'].value_counts())

# 2단계 모델 학습용 데이터셋 저장
df_final_2nd_stage.to_csv("2nd_stage_dataset_balanced.csv", index=False)
print("\n2단계 모델 학습용 데이터셋이 '2nd_stage_dataset_balanced.csv' 파일로 저장되었습니다.")