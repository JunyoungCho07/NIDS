import numpy as np
import pandas as pd

# 'cicids2017_combined.csv' 파일을 다시 불러옵니다.
df = pd.read_csv("cicids2017_combined.csv", engine='python')

# 불필요한 특징 열을 제거합니다.
# 이 열들은 모델이 공격을 '외우게' 만들어 일반화 성능을 떨어뜨립니다.
irrelevant_features = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df = df.drop(columns=irrelevant_features, errors='ignore')

# 결측값(NaN)과 무한대(Infinity) 값을 가진 행을 제거합니다.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("\n특징 제거 및 결측값 처리 후 데이터 정보:")
df.info()

print("\n데이터셋에 존재하는 공격 종류 및 개수:")
print(df['Label'].value_counts())
df.to_csv("cicids2017_cleaned_for_model.csv", index=False)