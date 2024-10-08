# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_and_preprocess_data(train_path, test_path):
    # 加载训练数据和测试数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 替换缺失值（100）为 -105，表示极弱的信号
    train_features = train_data.iloc[:, 0:520].replace(100, -105)
    test_features = test_data.iloc[:, 0:520].replace(100, -105)

    # 处理其他可能的缺失值（如NaN）
    train_features = train_features.fillna(-105)
    test_features = test_features.fillna(-105)

    # 提取目标变量（经度、纬度和楼层）
    y_train_full = train_data[['LONGITUDE', 'LATITUDE', 'FLOOR']].values
    y_test = test_data[['LONGITUDE', 'LATITUDE', 'FLOOR']].values

    # 划分训练集和验证集（从训练数据中划分）
    X_train, X_val, y_train, y_val = train_test_split(
        train_features.values, y_train_full, test_size=0.1, random_state=42
    )

    # 对特征进行 Min-Max 缩放
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(test_features.values)

    # 对目标变量进行标准化（如果需要）
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # 检查数据中是否存在 NaN 或无穷值
    for name, data in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                      ('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError(f"{name} 包含 NaN 或无穷值。请检查数据预处理步骤。")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y
