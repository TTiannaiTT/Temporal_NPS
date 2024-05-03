import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
from model.model import MLP
from model.model import Deep_MLP, simple_MLP
from model.evalAUC import evalAUC, checkAUC, acc_cal
from utils.feature import pca
import sys
import random
sys.path.append('/home/tyk/EventAug/Temporal_Sequence')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机种子
setup_seed(713)

def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, 3:].values  # 特征
    y = df.iloc[:, 2].values   # 标签
    return X, y
# 加载训练集和测试集
X_train, y_train = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train.csv')
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

# PCA
# X_train, X_test = pca(X_train,X_test,500)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建训练集和测试集的DataLoader
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 初始化模型
input_dim = X_train.shape[1]
hidden_dim = 3200 
output_dim = 2
model = torch.load('/home/tyk/EventAug/Temporal_Sequence/model/checkpoint/model_max.pth')
# model = simple_MLP(input_dim, hidden_dim, output_dim)
# model = MLP(input_dim, hidden_dim, output_dim)
# model = Deep_MLP(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
epochs = 100
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

def train_eval_model(model, train_loader, criterion, optimizer, epochs=100):
    print('Start training and testing.')
    model.to(device)
    max_auc = 0.0
    bestepoch = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            label_onehot = F.one_hot(labels, 2).float() 
            # print(label_onehot)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            # print(label_onehot)
            # print(outputs)
            loss = criterion(outputs, label_onehot)
            loss.backward()
            # total_norm = utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        print('##################### Train #######################')
        print(f"Epoch {epoch+1}")
        print(f"train loss: {running_loss/len(train_loader)}")

        # evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for input, label in test_loader:
                input, label = input.to(device), label.to(device)
                label_onehot = F.one_hot(label, 2).float() 
                outputs = model(input) 
                # print(label_onehot)
                # print(outputs)
                loss = criterion(outputs, label_onehot)
                test_loss += loss.item()
            print('##################### Test #######################')
            print(f'Epoch {epoch+1}')
            print(f'test loss: {test_loss/len(test_loader)}.')
            train_auc = evalAUC(model,device,X_train,y_train)
            test_auc = evalAUC(model,device,X_test,y_test)
            if test_auc > max_auc:
                torch.save(model, '/home/tyk/EventAug/Temporal_Sequence/model/checkpoint/model_max.pth')
                max_auc = test_auc
                bestepoch = epoch+1
            print(f'train auc: {train_auc}')
            print('test auc: ' , test_auc)
    print('The best auc: ',max_auc)
    print('The best epoch',bestepoch)

# train & evaluation
# train_eval_model(model, train_loader, criterion, optimizer,epochs)

model = torch.load('/home/tyk/EventAug/Temporal_Sequence/model/checkpoint/model_max.pth')
# model = torch.load('/home/tyk/EventAug/Temporal_Sequence/model/checkpoint/0.5672.pth')
model.to(device)
checkAUC(model,device,X_test,y_test)
# checkAUC(model,device,X_train,y_train)

count = 0
for label in y_test:
    if label == 1:
        count+=1
print(count/len(y_test))

print(acc_cal(model(torch.tensor(X_test, dtype=torch.float32).to(device)),y_test))
print(acc_cal(model(torch.tensor(X_train, dtype=torch.float32).to(device)),y_train))
print(model)
# 保存模型权重
# torch.save(model.state_dict(), 'model_weights.pth')

