import pandas as pd
from scipy import stats
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
# 读取训练集
train_data = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/train.csv')

# 假设特征列名为'features'，目标列名为'target'
X_train = train_data.iloc[:, 3:]
y_train = train_data.iloc[:, 2]

# 使用Z分数检测异常值
def detect_outliers_zscore(data, label, threshold=3):
    return data[(np.abs(stats.zscore(data)) > threshold).any(axis=1)]

# 使用IQR方法检测异常值
def detect_outliers_iqr(data,label):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)],label[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

def under_sample(X,y):
    # 计算少数类的样本数量
    minority_class_count = sum(y)  # 假设y是标签向量
    print(minority_class_count)
    # 初始化RandomUnderSampler，设置采样策略为少数类的样本数
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=0)
    # 进行欠采样
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

# 检测并删除异常值
print(X_train.shape)
train_data_cleaned1,_ = detect_outliers_iqr(X_train,y_train)
print(train_data_cleaned1.shape)
print(_.shape)
x_,y_ = under_sample(X_train.values,y_train.values)
print(x_.shape)
print(y_.shape)

