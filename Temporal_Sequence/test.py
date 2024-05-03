import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

# 步骤1: 读取CSV文件并分离特征和标签
train_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train.csv')
test_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/selected_test.csv')

X_train = train_data.iloc[:, 3:].values
y_train = train_data.iloc[:, 2].values
X_test = test_data.iloc[:, 3:].values
y_test = test_data.iloc[:, 2].values
X = train_data.iloc[:, 3:]  # 特征
y_true = train_data.iloc[:, 2]  # 标签
# 步骤2: 根据行缺失值比例是否大于0.6来预测标签
missing_percent = X.isnull().sum(axis=1) / X.shape[1]
y_predicted = np.where(missing_percent >= 0.6, 1, 0)
# 步骤3: 计算AUC
# 将标签转换为二进制格式（如果标签不是0和1）
binarized_true = label_binarize(y_true, classes=[0, 1])
binarized_predicted = label_binarize(y_predicted, classes=[0, 1])
try:
    auc = roc_auc_score(binarized_true, binarized_predicted, multi_class='ovr')
    print(f'The AUC of the model is: {auc}')
except ValueError as e:
    print(f"Error occurred when calculating AUC: {e}")