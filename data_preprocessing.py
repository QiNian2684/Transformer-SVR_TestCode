# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

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

    # 提取目标变量（经度、纬度）和楼层标签
    y_train_coords = train_data[['LONGITUDE', 'LATITUDE']].values
    y_test_coords = test_data[['LONGITUDE', 'LATITUDE']].values

    y_train_floor = train_data['FLOOR'].values
    y_test_floor = test_data['FLOOR'].values

    # 对楼层标签进行编码
    label_encoder = LabelEncoder()
    y_train_floor_encoded = label_encoder.fit_transform(y_train_floor)
    y_test_floor_encoded = label_encoder.transform(y_test_floor)

    # 划分训练集和验证集（从训练数据中划分）
    X_train, X_val, y_train_coords_split, y_val_coords_split, y_train_floor_split, y_val_floor_split = train_test_split(
        train_features.values, y_train_coords, y_train_floor_encoded, test_size=0.1, random_state=42
    )

    # 对特征进行 Min-Max 缩放
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(test_features.values)

    # 分别对经度和纬度目标变量进行标准化
    scaler_y_longitude = StandardScaler()
    scaler_y_latitude = StandardScaler()

    y_train_longitude = scaler_y_longitude.fit_transform(y_train_coords_split[:, 0].reshape(-1, 1))
    y_val_longitude = scaler_y_longitude.transform(y_val_coords_split[:, 0].reshape(-1, 1))
    y_test_longitude = scaler_y_longitude.transform(y_test_coords[:, 0].reshape(-1, 1))

    y_train_latitude = scaler_y_latitude.fit_transform(y_train_coords_split[:, 1].reshape(-1, 1))
    y_val_latitude = scaler_y_latitude.transform(y_val_coords_split[:, 1].reshape(-1, 1))
    y_test_latitude = scaler_y_latitude.transform(y_test_coords[:, 1].reshape(-1, 1))

    # 将标准化后的经度和纬度目标变量合并
    y_train = np.hstack((y_train_longitude, y_train_latitude))
    y_val = np.hstack((y_val_longitude, y_val_latitude))
    y_test = np.hstack((y_test_longitude, y_test_latitude))

    # 检查数据中是否存在 NaN 或无穷值
    for name, data in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                      ('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError(f"{name} 包含 NaN 或无穷值。请检查数据预处理步骤。")

    # 返回处理后的数据和缩放器
    scaler_y = {
        'scaler_y_longitude': scaler_y_longitude,
        'scaler_y_latitude': scaler_y_latitude
    }

    return X_train, y_train, y_train_floor_split, X_val, y_val, y_val_floor_split, X_test, y_test, y_test_floor_encoded, scaler_X, scaler_y, label_encoder

if __name__ == "__main__":
    # 示例用法
    data = load_and_preprocess_data('train.csv', 'test.csv')
    print("预处理完成，数据已加载。")
