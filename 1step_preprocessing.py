import pandas as pd
import glob
import os
import numpy as np

# 압축 해제한 파일들이 있는 폴더 경로를 입력하세요.
# 예: 'C:\\Users\\YourName\\Downloads\\MachineLearningCSV'
# 사용자님의 경로: 'C:\\Users\\cho-j\\OneDrive\\바탕 화면\\NIDS\\MachineLearningCSV'
path = 'C:\\Users\\cho-j\\Downloads\\MachineLearningCSV\\MachineLearningCVE'

all_files = glob.glob(os.path.join(path, "*.csv"))

li = []

for filename in all_files:
    # 'engine='python'' 옵션을 추가하여 오류를 무시하고 파일을 읽음
    df = pd.read_csv(filename, index_col=None, header=0, on_bad_lines='skip', engine='python')
    li.append(df)

# 모든 CSV 파일을 하나의 데이터프레임으로 합치기
combined_df = pd.concat(li, axis=0, ignore_index=True)

# 컬럼(열) 이름의 앞뒤 공백 제거
combined_df.columns = combined_df.columns.str.strip()

# 불필요한 특징 열을 제거합니다.
# 이 열들은 모델이 공격을 '외우게' 만들어 일반화 성능을 떨어뜨립니다.
irrelevant_features = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df = combined_df.drop(columns=irrelevant_features, errors='ignore')

# 결측값(NaN)과 무한대(Infinity) 값을 가진 행을 제거합니다.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("데이터프레임 정보:")
df.info()

print("\n상위 5개 행:")
print(df.head())

print("\n데이터셋에 존재하는 공격 종류 및 개수:")
print(df['Label'].value_counts())

# 전처리한 데이터를 CSV 파일로 저장 (다음 단계에서 사용)
df.to_csv("cicids2017_cleaned_for_model.csv", index=False)

print("\n\n'cicids2017_cleaned_for_model.csv' 파일이 성공적으로 저장되었습니다.")