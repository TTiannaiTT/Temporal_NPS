import pandas as pd
from scipy import stats
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取训练集
train_path = '/home/tyk/EventAug/Temporal_Sequence/dataset/train.csv'
test_path = '/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X_train = train_data.iloc[:, 3:]
y_train = train_data.iloc[:, 2]

# 数据预处理
X_train_ = train_data.iloc[:,2:]

# 计算每列的均值
mean_values = X_train_.mean()
# 使用每列的均值填充NaN
X_train_ = X_train_.fillna(mean_values)
# print(X_train_['LABEL'])

# # 使用Z分数检测异常值
# def detect_outliers_zscore(data, threshold=3):
#     return data[(np.abs(stats.zscore(data)) > threshold).any(axis=1)]

# # 使用IQR方法检测异常值
# def detect_outliers_iqr(data,label):
#     Q1 = data.quantile(0.25)
#     Q3 = data.quantile(0.75)
#     IQR = Q3 - Q1
#     return data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)],label[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

# def under_sample(X,y):
#     # 计算少数类的样本数量
#     minority_class_count = sum(y)  # 假设y是标签向量
#     print(minority_class_count)
#     # 初始化RandomUnderSampler，设置采样策略为少数类的样本数
#     rus = RandomUnderSampler(sampling_strategy='majority', random_state=0)
#     # 进行欠采样
#     X_resampled, y_resampled = rus.fit_resample(X, y)
#     return X_resampled, y_resampled

def dataClean(train_data,test_data):
    print('original row: ',train_data.shape[0])
    print('original column: ',train_data.shape[1])
    row_missing = train_data.isnull().sum(axis=1) / len(train_data.columns)
    col_missing = train_data.isnull().sum(axis=0) / len(train_data.index)

    # 删除缺失值比例大于0.5的行
    train_data = train_data[row_missing <= 0.5]

    # 删除缺失值比例大于0.5的列
    train_data = train_data.loc[:, col_missing <= 0.5]
    test_data = test_data.loc[:, col_missing <= 0.5]

    print('new row: ',train_data.shape[0])
    print('new column: ',train_data.shape[1])
    # 将处理后的数据存入新的CSV文件
    new_file_path = '/home/tyk/EventAug/Temporal_Sequence/dataset/cleaned_train.csv'  # 新文件的路径
    train_data.to_csv(new_file_path, index=False)

    new_file_path = '/home/tyk/EventAug/Temporal_Sequence/dataset/cleaned_test.csv'  # 新文件的路径
    test_data.to_csv(new_file_path, index=False)

    print(f"Cleaned data has been saved to {new_file_path}")

def dataSelect(train_data,test_data):
    print('original row: ',train_data.shape[0])
    print('original column: ',train_data.shape[1])
    row_missing = train_data.isnull().sum(axis=1) / len(train_data.columns)

    # 删除缺失值比例大于0.5的行
    train_data = train_data[row_missing <= 0.35]

    # 删除缺失值比例大于0.5的列
    # train_data = train_data.loc[:, col_missing <= 0.2]
    # test_data = test_data.loc[:, col_missing <= 0.2]

    print('new row: ',train_data.shape[0])
    print('new column: ',train_data.shape[1])
    # 将处理后的数据存入新的CSV文件
    new_file_path = '/home/tyk/EventAug/Temporal_Sequence/dataset/selected_train_0.65.csv'  # 新文件的路径
    train_data.to_csv(new_file_path, index=False)

    print(f"Cleaned data has been saved to {new_file_path}")

def get_main_feature():
    # 方差阈值筛选
    selector = VarianceThreshold(threshold=0)  # 移除方差为0的特征
    df_variance_filtered = selector.fit_transform(X_train_)
    # 相关性筛选示例（以卡方检验为例，适用于分类特征）
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    print(X_train_['LABEL'])
    fit = bestfeatures.fit(df_variance_filtered, X_train_['LABEL'])
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train_.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Feature', 'Score']
    print(featureScores.nlargest(10, 'Score'))  # 输出得分最高的10个特征
    print(type(featureScores)) #DataFrame
    print(featureScores['Feature'])
    # 注意这里是移除了前两列之后的列数，和真实的比要前2

    # # 计算相关系数矩阵
    # corr_matrix = X_train_.corr()
    # # 选择与目标变量相关性高于某一阈值的特征
    # print(corr_matrix["LABEL"].nlargest(10, "LABEL").index)
    index = [1113,659,854,984,1761,672,1775,19,1774,1312]
    print('main feature index: ',index)
    return index

def pca(X_train,X_test,dimensions):
    pca = PCA(n_components=dimensions)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # 输出降维前后特征的方差解释率，帮助理解降维效果
    explained_variance_ratio = pca.explained_variance_ratio_
    print("解释方差比率（按降序排列）:", np.sum(explained_variance_ratio))
    return X_train_pca,X_test_pca
# 检测并删除异常值
# print(X_train.shape)
# train_data_cleaned1,_ = detect_outliers_iqr(X_train,y_train)
# print(train_data_cleaned1.shape)
# print(_.shape)
# x_,y_ = under_sample(X_train.values,y_train.values)
# print(x_.shape)
# print(y_.shape)
# get_main_feature()
# dataClean(train_data,test_data)
# dataSelect(train_data,test_data)