import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# GPU 사용을 설정합니다.
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("GPU를 찾을 수 없습니다. CPU를 사용합니다.")

# 1. 1단계 XGBoost 모델 및 LabelEncoder 불러오기
model_xgb = joblib.load('model_xgb_1st_stage.pkl')
le_original = joblib.load('le_xgb.pkl')

# 2. 2단계 Bi-LSTM 모델 및 LabelEncoder 불러오기
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional=True, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=self.bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if self.bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        output = self.fc(hidden)
        return output

# 모델 인스턴스화 및 가중치 로드
input_dim = 78 # 전체 특징 수 (전처리 후)
hidden_dim = 128
output_dim_2nd = len(joblib.load('le_bilstm.pkl').classes_)
n_layers = 2
model_bilstm = BiLSTMClassifier(input_dim, hidden_dim, output_dim_2nd, n_layers).to(device)

model_bilstm_state_dict = torch.load('model_bilstm_2nd_stage.pth', map_location=device)
model_bilstm.load_state_dict(model_bilstm_state_dict, strict=True)
model_bilstm.eval()

# 3. 전체 테스트 데이터셋 불러오기
df = pd.read_csv("cicids2017_cleaned_for_model.csv")
X = df.drop('Label', axis=1)
y = df['Label']

# 학습용/테스트용 데이터로 분리 (이전과 동일하게)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. 전체 테스트 데이터에 대한 2단계 파이프라인 시뮬레이션
print("\n전체 테스트 데이터에 대해 2단계 파이프라인 시뮬레이션 시작...")

# 1단계: XGBoost 모델로 1차 예측
print("1단계 XGBoost 예측 중...")
# XGBoost 예측을 전체 테스트 데이터에 대해 수행
xgb_preds = model_xgb.predict(X_test)
xgb_preds_labels = le_original.inverse_transform(xgb_preds)

# 2단계: XGBoost가 'BENIGN'이 아니라고 판단한 데이터만 필터링
print("2단계 Bi-LSTM용 데이터 필터링 중...")
is_not_benign_mask = (xgb_preds_labels!= 'BENIGN')
X_test_2nd_stage = X_test[is_not_benign_mask]
y_test_2nd_stage = y_test[is_not_benign_mask]
le_bilstm = joblib.load('le_bilstm.pkl')

print(f"2단계 모델로 전달될 데이터 크기: {len(X_test_2nd_stage)}개")

# 2단계 모델용 데이터 스케일링
scaler = StandardScaler()
X_test_2nd_stage_scaled = scaler.fit_transform(X_test_2nd_stage)
X_test_2nd_stage_tensor = torch.FloatTensor(X_test_2nd_stage_scaled).to(device)

# 2단계 모델 예측을 위해 데이터 로더 사용 (메모리 절약)
test_dataset_2nd_stage = TensorDataset(X_test_2nd_stage_tensor)
test_loader_2nd_stage = DataLoader(dataset=test_dataset_2nd_stage, batch_size=64, shuffle=False)

bilstm_preds_encoded = []
with torch.no_grad():
    for inputs in tqdm(test_loader_2nd_stage, desc="2단계 모델 예측"):
        inputs = inputs[0].to(device)
        outputs = model_bilstm(inputs)
        _, predicted = torch.max(outputs.data, 1)
        bilstm_preds_encoded.extend(predicted.cpu().numpy())

# Bi-LSTM 예측 결과를 원래 레이블로 변환
bilstm_preds_labels = le_bilstm.inverse_transform(bilstm_preds_encoded)

# 5. 최종 결과 병합
# 최종 예측 결과를 저장할 배열을 생성하고 XGBoost 예측으로 초기화
y_pred_final = xgb_preds_labels.copy()

# 2단계 모델이 예측한 결과로 특정 부분만 업데이트
y_pred_final[is_not_benign_mask] = bilstm_preds_labels

# 6. 최종 성능 보고서 출력
print("\n최종 2단계 시스템 성능 보고서 (전체 테스트 데이터 기준):")
final_report = classification_report(y_test, y_pred_final, target_names=le_original.classes_)
print(final_report)