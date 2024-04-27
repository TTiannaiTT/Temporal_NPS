import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
# load the data
df = pd.read_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/train.csv')
print(df)

numeric_cols = df.select_dtypes(include=[np.number]).columns
# 创建一个函数，该函数将为每个空值找到最近的10个样本，并计算它们对应位置的平均值
def impute_with_nearest_values(df, numeric_cols, k=10):
    X = df[numeric_cols].values  # 转换为numpy数组进行处理
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)  # 初始化最近邻模型，+1是为了排除自身
    
    # 找到每个样本最接近的k个样本的索引
    _, indices = nbrs.kneighbors(X)
    
    # 遍历DataFrame中的每个空值，并用找到的最近邻的平均值填充
    for col in numeric_cols:
        for i, row in df.iterrows():
            if np.isnan(row[col]):  # 检查是否为空值
                nearest_values = df.iloc[indices[i, 1:], col]  # 获取最近的k个样本的对应值
                row[col] = nearest_values.mean()  # 用平均值替换空值
    return df

def replace_na_with_mean(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    df = df.iloc[1:,2:]
    # 遍历DataFrame的每一列
    for col in df.columns:
        # 计算当前列的非空值均值
        mean = df[col].mean(skipna=True)
        
        # 将当前列的空值替换为计算出的均值
        df[col].fillna(mean, inplace=True)
    
    # 可选：将处理后的数据写回到CSV文件，这里不覆盖原文件，而是生成一个新文件
    df.to_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/meanfilled_test.csv', index=False)

# 假设你的CSV文件名为'data.csv'
replace_na_with_mean('/home/tyk/EventAug/Temporal_Sequence/dataset/test.csv')
# df_filled = impute_with_nearest_values(df, numeric_cols)
# df_filled.to_csv('/home/tyk/EventAug/Temporal_Sequence/dataset/filled_data.csv', index=False)
