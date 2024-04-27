from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/train.csv')
test_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv')

# 假设特征列名为'features'，目标列名为'target'
X_train = train_data.iloc[:, 3:].values
y_train = train_data.iloc[:, 2].values
X_test = test_data.iloc[:, 3:].values
y_test = test_data.iloc[:, 2].values

# 数据预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(type(X_train))
X_train = np.nan_to_num(X_train, nan=0)
X_test = np.nan_to_num(X_test,nan=0)

# 初始化梯度提升分类器
gb_clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=3, random_state=42)
# 训练模型
gb_clf.fit(X_train, y_train)
# 预测概率
y_pred_prob_gb = gb_clf.predict_proba(X_test)[:, 1]
# 计算AUC
gb_auc_score = roc_auc_score(y_test, y_pred_prob_gb)
print(f"梯度提升模型的AUC: {gb_auc_score:.4f}")
