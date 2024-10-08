import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(train_path, test_path):
    """
    加载并预处理数据，包括替换缺失值和提取目标变量。

    参数：
    - train_path: 训练数据文件路径
    - test_path: 测试数据文件路径

    返回：
    - X_train: 训练集特征数组
    - y_train: 训练集目标变量（经度和纬度）
    - X_val: 验证集特征数组
    - y_val: 验证集目标变量（经度和纬度）
    - X_test: 测试集特征数组
    - y_test: 测试集目标变量（经度和纬度）
    """
    # 加载训练数据和测试数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 替换缺失值（100）为 -105，表示极弱的信号
    train_features = train_data.iloc[:, 0:520].replace(100, -105)
    test_features = test_data.iloc[:, 0:520].replace(100, -105)

    # 提取目标变量（经度和纬度）
    y_train = train_data[['LONGITUDE', 'LATITUDE']].values
    y_test = test_data[['LONGITUDE', 'LATITUDE']].values

    # 划分训练集和验证集（从训练数据中划分）
    X_train, X_val, y_train, y_val = train_test_split(train_features.values, y_train, test_size=0.1, random_state=42)

    return X_train, y_train, X_val, y_val, test_features.values, y_test
