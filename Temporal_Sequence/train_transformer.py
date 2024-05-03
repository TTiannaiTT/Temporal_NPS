import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score

# 0.5513

# 读取数据
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, 3:].values  # 特征
    y = df.iloc[:, 2].values   # 标签
    return X, y
X_train, y_train = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train_0.6.csv')
X_test, y_test = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv')
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
# 数据预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(type(X_train))
X_train = np.nan_to_num(X_train, nan=0)
X_test = np.nan_to_num(X_test,nan=0)

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(TransformerClassifier, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=2, dim_feedforward=hidden_dim*8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.fc1 = nn.Linear(hidden_dim, 1000)
        self.fc2 = nn.Linear(1000,output_dim)
        self.relu = nn.ReLU()
    def forward(self, src):
        src = src.unsqueeze(1)  # 为Transformer输入添加序列维度
        output = self.transformer_encoder(src)
        output = self.fc1(output.squeeze(1))
        output = self.relu(output)
        output = self.fc2(output)
        return output

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# 初始化模型
model = TransformerClassifier(input_dim=X_train.shape[1], hidden_dim=X_train.shape[1], output_dim=1, n_layers=2).to(device)
# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
# 将数据转换为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float).view(-1, 1).to(device)

best_auc = 0
epochs = 200
# 训练模型
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
# 评估模型
    model.eval()
    with torch.no_grad():
        y_test_pred = torch.sigmoid(model(X_test_tensor))
        y_train_pred = torch.sigmoid(model(X_train_tensor))
        testauc = roc_auc_score(y_test, y_test_pred.cpu().numpy())
        trainauc = roc_auc_score(y_train, y_train_pred.cpu().numpy())

        if testauc > best_auc:
            best_auc = testauc
        print('#'*30)
        print('Epochs: ',epoch)
        print('Train_loss: ',loss)
        print(f'Train_AUC: {trainauc}')
        print(f'Test_AUC: {testauc}')
print(best_auc)
# X_train, y_train = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train.csv')
# X_test, y_test = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_test.csv')
# # print(X_train)
# # print(X_test)
# # print(y_train)
# # print(y_test)
# # 数据预处理
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# # print(type(X_train))
# X_train = np.nan_to_num(X_train, nan=0)
# X_test = np.nan_to_num(X_test,nan=0)

# class CustomDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.int64)
#     def __len__(self):
#         return len(self.X)
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# # 创建训练集和测试集的DataLoader
# train_dataset = CustomDataset(X_train, y_train)
# test_dataset = CustomDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
