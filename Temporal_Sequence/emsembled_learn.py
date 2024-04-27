import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
sys.path.append('/home/tyk/EventAug/Temporal_Sequence')
from utils.feature import detect_outliers_iqr

# 初始化分类器
clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = LogisticRegression(random_state=42)

# 加载训练集和测试集
train_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/train.csv')
test_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv')

# 假设特征列名为'features'，目标列名为'target'
X_train = train_data.iloc[:, 3:].values
y_train = train_data.iloc[:, 2].values
X_test = test_data.iloc[:, 3:].values
y_test = test_data.iloc[:, 2].values

X_train_iqr, y_train_iqr = detect_outliers_iqr(train_data.iloc[:, 3:],train_data.iloc[:, 2])
# 数据预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(type(X_train))
X_train = np.nan_to_num(X_train, nan=0)
X_test = np.nan_to_num(X_test,nan=0)

# 训练模型
print('Start training.')
clf1.fit(X_train, y_train)
print('Finish clf1')
clf2.fit(X_train_iqr, y_train_iqr)
print('Finish clf2')
clf3.fit(X_train, y_train)
print('Finish clf3')

# 预测概率
print('Start evaluation.')
y_pred_prob1 = clf1.predict_proba(X_test)[:, 1]
y_pred_prob2 = clf2.predict_proba(X_test)[:, 1]
y_pred_prob3 = clf3.predict_proba(X_test)[:, 1]

# 集成预测概率
y_pred_prob = (y_pred_prob1 * 0.8 +  y_pred_prob3 * 0.2) 

auc_test1 = roc_auc_score(y_test, y_pred_prob1)
auc_test2 = roc_auc_score(y_test, y_pred_prob2)
auc_test3 = roc_auc_score(y_test, y_pred_prob3)
print(f'auc1: {auc_test1}')
print(f'auc2: {auc_test2}')
print(f'auc3: {auc_test3}')

# 计算AUC
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"emsembled AUC: {auc_score:.4f}")