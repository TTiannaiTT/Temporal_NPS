import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


# 加载数据集
train_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train.csv')
test_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv')

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
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
# 初始化LightGBM分类器
params = {
    'num_leaves': 60,
    'min_data_in_leaf': 30,
    'objective': 'binary',
    'max_depth': -1,
    'learning_rate': 0.03,
    'min_sum_hessian_in_leaf': 6,
    'boosting': 'gbdt',
    'feature_fraction': 0.9,
    'bagging_freq': 1,
    'bagging_fraction': 0.8,
}
model = LGBMClassifier(**params)
# 训练模型
model.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = model.predict_proba(X_test)[:, 1]
# 计算AUC
auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc_score}") # 0.55

param_grid = {
    'num_leaves': [31, 62, 127],  # 叶子节点的数量
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'n_estimators': [100, 200, 300],  # 迭代次数
    'min_child_samples': [20, 30, 50],  # 最小数据量
    'subsample': [0.6, 0.8, 1.0],  # 样本采样比例
    'colsample_bytree': [0.6, 0.8, 1.0],  # 特征采样比例
}

from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
# 初始化模型
model = LGBMClassifier()
# 初始化Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
# 使用Grid Search进行参数搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)
# 使用最佳参数训练模型
best_model = LGBMClassifier(**best_params)
best_model.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = best_model.predict_proba(X_test)[:, 1]
# 计算AUC
auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC on test set with best parameters: {auc_score}")
