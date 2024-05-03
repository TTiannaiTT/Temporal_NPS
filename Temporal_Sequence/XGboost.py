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

#0.5554 0.5train
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc',
                          max_depth = 6, min_child_weight = 1, gamma = 0, sub_sample = 1, n_estimator = 100,
                          reg_alpha = 0,reg_lambda = 1,
                          learning_rate = 0.3) 
# 训练模型
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc')
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
# 进行预测，注意使用predict_proba来获取概率
y_pred_proba = model.predict_proba(X_test)[:, 1]
# 计算AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score}")
print(model)

# 定义 XGBoost 模型
# model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc')
# # 定义超参数网格
# param_grid = {
#     'learning_rate': [0.01, 0.005],
#     'max_depth': [25,35,45],
#     'n_estimators': [500,600,700],
#     'gamma': [0.3,0.4,0.5],
#     'subsample': [0.5],
#     'colsample_bytree': [1.0],
# }

# # 使用 GridSearchCV 进行超参数搜索
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # 输出最优参数
# print("Best parameters found: ", grid_search.best_params_)

# # 使用最优模型进行预测
# best_model = grid_search.best_estimator_
# y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# # 计算 AUC
# auc = roc_auc_score(y_test, y_pred_proba)
# print(f"AUC: {auc}")
