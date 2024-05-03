import torch
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# 假设数据已经被加载和预处理
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, 3:].values  # 特征
    y = df.iloc[:, 2].values   # 标签
    return X, y
X_train, y_train = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train.csv')
X_test, y_test = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(type(X_train))
X_train = np.nan_to_num(X_train, nan=0)
X_test = np.nan_to_num(X_test,nan=0)
train_features, train_labels = torch.tensor(X_train), y_train
test_features, test_labels = torch.tensor(X_test), y_test

# basic cluster-like method
# 计算类中心 0.5415
print('#'*30)
print('Basic cluster-like method:')
class_centers = {}
for class_label in set(train_labels.tolist()):
    class_features = train_features[train_labels == class_label]
    class_centers[class_label] = torch.mean(class_features, dim=0)
# 预测过程
predictions = []
for sample in test_features:
    distances = {label: torch.dist(sample, center) for label, center in class_centers.items()}
    predicted_label = min(distances, key=distances.get)
    predictions.append(predicted_label)
# 将预测结果转换为概率（1表示预测为1类的概率，0表示预测为0类的概率）
probabilities = torch.tensor(predictions, dtype=torch.float32)
torch.set_printoptions(threshold=np.inf)
# print(probabilities)
# 计算AUC
auc = roc_auc_score(test_labels, probabilities.numpy())
print(f"AUC: {auc}")

# knn method
# K=138 0.5659 (the best in 1000)
print('#'*30)
print('Knn method:')
bestauc = 0
bestn = 0
for i in range(138,139,1):
    knn = KNeighborsClassifier(n_neighbors=i+1)  
    knn.fit(train_features.numpy(), train_labels)
    # 对测试集进行预测
    predictions = knn.predict_proba(test_features.numpy())[:, 1]  # 选择第二列，即正类的预测概率
    # 计算AUC
    auc = roc_auc_score(test_labels, predictions)
    print(f"AUC: {auc}")
    # if auc>bestauc:
    #     bestauc = auc
    #     bestn = i
# print(bestn)
# print(bestauc)

# svm method
print('#'*30)
print('SVM method:')
clf = svm.SVC(kernel='linear', probability=True)  # 使用线性核，并启用概率估计
clf.fit(X_train, y_train)
# 在测试集上进行预测，获取概率估计
y_score = clf.predict_proba(X_test)[:, 1]
# 直接计算AUC
roc_auc = roc_auc_score(y_test, y_score)
print("AUC: %.2f" % roc_auc)