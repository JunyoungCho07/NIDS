import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import numpy as np
import joblib

# 1단계에서 저장한 전처리된 데이터 불러오기
file_path = "cicids2017_cleaned_for_model.csv"
df = pd.read_csv(file_path)

# 특성(X)과 타겟(y) 분리
X = df.drop('Label', axis=1)
y = df['Label']

# 문자열 레이블을 숫자로 변환
le = LabelEncoder()
y_encoded = le.fit_transform(y)
labels_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("인코딩된 레이블:", labels_mapping)

# 저장
joblib.dump(le, 'le_xgb.pkl')

# 학습용 데이터와 테스트용 데이터로 분리
# shuffle=True (기본값)를 사용하여 데이터를 무작위로 섞고,
# stratify=y_encoded 옵션으로 각 클래스의 비율을 유지합니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 데이터 불균형 해결을 위한 가중치 계산
class_counts = y.value_counts()
class_weights = class_counts.max() / class_counts
sample_weights = np.array([class_weights[label] for label in y])
sample_weights_train = np.array([class_weights[le.inverse_transform([y_enc])] for y_enc in y_train])

# 1단계 모델: XGBoost 분류기
model_xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    n_estimators=100,
    random_state=42,
    use_label_encoder=False
)

# 모델 학습
print("\n1단계 XGBoost 모델 학습 중...")
model_xgb.fit(X_train, y_train, sample_weight=sample_weights_train)
print("학습 완료.")

# 모델 평가
print("\n모델 예측 및 평가 중...")
y_pred_xgb = model_xgb.predict(X_test)
report = classification_report(y_test, y_pred_xgb, target_names=le.classes_)
print("XGBoost 모델 성능 보고서:")
print(report)

# 모델 저장 (다음 단계에서 사용)
import joblib
joblib.dump(model_xgb, 'model_xgb_1st_stage.pkl')

print("XGBoost 모델이 'model_xgb_1st_stage.pkl' 파일로 저장되었습니다.")