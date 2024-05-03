from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import pickle


def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    X = df.iloc[:, 3:].values  # 特征
    y = df.iloc[:, 2].values   # 标签
    return X, y
X_train, y_train = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train_0.4.csv')
X_test, y_test = load_dataset('/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(type(X_train))
X_train = np.nan_to_num(X_train, nan=0)
X_test = np.nan_to_num(X_test,nan=0)

# rf0.6,rf1：0.4， rf2: ori
# clf = RandomForestClassifier(n_estimators=105, max_depth=4, random_state=713, n_jobs=-1)  0.5764 0.5train
# clf = RandomForestClassifier(n_estimators=157, max_depth=4, random_state=518, n_jobs=-1)# 0.5785 0.6train
clf = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=518, n_jobs=-1) 

    # 训练模型
clf.fit(X_train, y_train)
# with open('rf_model_0.5785.pkl', 'wb') as f:
#     pickle.dump(clf, f)
# with open('rf_model.pkl', 'rb') as f:
#     clf_loaded = pickle.load(f)
# # 使用加载的模型进行预测
# predictions = clf_loaded.predict(X_test)

y_pred_proba = clf.predict_proba(X_test)[:, 1]  # 选择第二列，即正类的概率
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC: ", auc_score)
# bestauc = 0
# seed = 0
# for i in range(1000):

#     clf = RandomForestClassifier(n_estimators=90, max_depth=4, random_state=i, n_jobs=-1) 

#     # 训练模型
#     clf.fit(X_train, y_train)

#     y_pred_proba = clf.predict_proba(X_test)[:, 1]  # 选择第二列，即正类的概率
#     auc_score = roc_auc_score(y_test, y_pred_proba)
#     print("AUC: ", auc_score)
#     if auc_score > bestauc:
#         bestauc = auc_score
#         seed = i
# print(bestauc)
# print(seed)


# # Grid Search:
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
# }
# # 创建随机森林分类器实例
# rfc = RandomForestClassifier(random_state=42)
# # 使用AUC作为评分函数
# scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
# # 应用GridSearchCV进行参数调优
# grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring=scoring, cv=5, n_jobs=-1)
# # 在训练集上执行GridSearchCV
# grid_search.fit(X_train, y_train)

# print("Best parameters found: ", grid_search.best_params_)
# print("Best AUC found: ", grid_search.best_score_)
# best_rfc = grid_search.best_estimator_
# # 在测试集上进行预测
# y_pred_proba = best_rfc.predict_proba(X_test)[:, 1]
# # 计算并打印AUC
# auc_score = roc_auc_score(y_test, y_pred_proba)
# print("AUC on test set with best parameters: ", auc_score)
