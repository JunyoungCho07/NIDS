import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

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

# 1. 데이터 불러오기 및 전처리
file_path = "2nd_stage_dataset_balanced.csv"
df = pd.read_csv(file_path)

# 문제 해결: 단 1개만 존재하는 클래스 제거
class_counts = df['Label'].value_counts()
single_member_classes = class_counts[class_counts == 1].index
df_filtered = df[~df['Label'].isin(single_member_classes)]

print(f"제거된 클래스: {list(single_member_classes)}")
print(f"필터링 후 데이터셋 크기: {len(df_filtered)}개")
print("\n필터링 후 데이터셋에 존재하는 공격 종류 및 개수:")
print(df_filtered['Label'].value_counts())

# 특성(X)과 타겟(y) 분리
X = df_filtered.drop('Label', axis=1)
y = df_filtered['Label']

# 필터링된 레이블을 새로운, 연속된 숫자로 다시 인코딩합니다.
le_bilstm = LabelEncoder()
y_encoded = le_bilstm.fit_transform(y)

# 2단계 모델용 LabelEncoder 객체 저장
joblib.dump(le_bilstm, 'le_bilstm.pkl')
print("\nBi-LSTM 모델용 LabelEncoder가 'le_bilstm.pkl' 파일로 저장되었습니다.")

# 학습용 데이터와 테스트용 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 넘파이 배열을 PyTorch 텐서로 변환
X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# 데이터 로더 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. Bi-LSTM 모델 정의
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional=True, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_dim * 2 if self.bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        output = self.fc(hidden)
        return output

# 모델 인스턴스화
input_dim = X_train_scaled.shape[1]
hidden_dim = 128
output_dim = len(le_bilstm.classes_)
n_layers = 2
model_bilstm = BiLSTMClassifier(input_dim, hidden_dim, output_dim, n_layers).to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_bilstm.parameters(), lr=0.001)

# 3. 모델 학습
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"에포크 [{epoch+1}/{num_epochs}], 손실: {loss.item():.4f}")

print("\n2단계 Bi-LSTM 모델 학습 중...")
train_model(model_bilstm, train_loader, criterion, optimizer, num_epochs=20) # 에포크를 20으로 증가시켰습니다.
print("학습 완료.")

# 4. 모델 평가 및 결과 보고서
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

print("\n2단계 모델 예측 및 평가 중...")
y_true, y_pred = evaluate_model(model_bilstm, test_loader)
report = classification_report(y_true, y_pred, target_names=le_bilstm.classes_)

print("\nBi-LSTM 모델 성능 보고서:")
print(report)

# 모델 저장 (다음 단계에서 사용)
torch.save(model_bilstm.state_dict(), 'model_bilstm_2nd_stage.pth')
print("Bi-LSTM 모델이 'model_bilstm_2nd_stage.pth' 파일로 저장되었습니다.")