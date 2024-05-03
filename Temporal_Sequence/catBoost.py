from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from catboost import CatBoostClassifier, Pool


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

# 将数据转换为CatBoost的Pool格式
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)
# 初始化CatBoost分类器
model = CatBoostClassifier(
    iterations=1000,  # 迭代次数
    learning_rate=0.1,  # 学习率
    depth=10,  # 树的最大深度
    loss_function='Logloss',  # 损失函数，适用于二分类
    eval_metric='AUC',  # 评价指标
    task_type='GPU',  # 使用GPU
    random_seed=42,  # 随机种子
    logging_level='Silent'  # 日志级别
)
# 训练模型
model.fit(train_pool, eval_set=test_pool, verbose=False)

# 在测试集上进行预测
y_pred = model.predict(X_test)
# 计算AUC
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_test, y_pred)
print(f"AUC on test set: {auc_score}")