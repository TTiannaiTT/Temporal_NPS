from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train.csv')
test_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv')
#堆叠集成模型的AUC: 0.5541

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

# 定义基础分类器
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=1000, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]
# 定义元分类器
meta_classifier = LogisticRegression(random_state=42)
# 初始化堆叠分类器
stack_clf = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=meta_classifier,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)
# 训练模型
stack_clf.fit(X_train, y_train)
# 预测概率
y_pred_prob_stack = stack_clf.predict_proba(X_test)[:, 1]
# 计算AUC
stack_auc_score = roc_auc_score(y_test, y_pred_prob_stack)
print(f"堆叠集成模型的AUC: {stack_auc_score:.4f}")