# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA  # 使用非线性降维方法KernelPCA替代原PCA


def load_and_preprocess_data(train_path, test_path, max_missing_ratio=0.96, feature_missing_threshold=0.96,
                             pca_components=64):
    """
    加载并预处理训练集和测试集数据。

    参数说明：
    - train_path:
        类型：字符串（str）
        含义：训练数据集的CSV文件路径。
        该文件应包含WAP信号特征列（通常有520列）以及对应的位置信息（LONGITUDE、LATITUDE）和楼层信息（FLOOR）。

    - test_path:
        类型：字符串（str）
        含义：测试数据集的CSV文件路径。
        同样包含WAP信号特征列和目标位置信息，以用于在模型训练完成后进行验证或最终评估。

    - max_missing_ratio:(数据集中的行)
        类型：浮点数（float）
        含义：测试集数据清洗时的过滤阈值。
        测试集中每条样本会统计有多少比例的WAP特征是-105（表示极弱或缺失的信号）。
        当某条测试样本中，WAP信号为-105的比例超过max_missing_ratio时，该样本将被剔除。
        举例：max_missing_ratio=0.99表示当一条测试样本中，超过99%的特征都是-105，就会过滤掉这条数据。
        调大该值可以减少过滤掉的测试样本数量（更宽松），调小则更严格。

    - feature_missing_threshold:(数据集中的列)
        类型：浮点数（float）
        含义：特征级别的过滤阈值。
        我们会统计训练集中每个WAP特征中有多少比例的样本对应值为-105。
        当某特征在训练集中超过feature_missing_threshold比例的样本为-105时，则该特征被认为无信息价值并被剔除。
        举例：feature_missing_threshold=0.95表示如果有95%的训练样本在某特征上都是-105，则删除该特征。
        调高该值会更宽松（需要更高比例的空缺才剔除特征），调低则更严格。

    - pca_components:
        类型：整数（int）
        含义：PCA降维目标维度数。（在这里我们用KernelPCA代替）
        在特征清洗完成后，我们对特征数据应用非线性降维（KernelPCA）进行降维，以减少特征维度、降低噪声和冗余。
        pca_components定义了降维后的特征数目上限。
        如果最终清洗后特征数量小于该值，则以实际特征数为准。
        数值越大，保留的特征维度越多，数据维度高但特征可能更丰富；数值越小，特征维度更精简但可能损失部分信息。
    """

    # 加载训练数据和测试数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 特征列（WAP等信号强度列）的索引范围，这里为前520列
    feature_cols = list(range(0, 520))

    # 替换训练集和测试集特征中的 100 为 -105（表示极弱的信号）
    train_features = train_data.iloc[:, feature_cols].replace(100, -105)
    test_features = test_data.iloc[:, feature_cols].replace(100, -105)

    # 处理其他可能的缺失值（如NaN），填充为 -105
    train_features = train_features.fillna(-105)
    test_features = test_features.fillna(-105)

    # 对测试集数据质量进行筛选
    total_features = train_features.shape[1]  # 这里是520
    test_missing_ratio_series = (test_features == -105).sum(axis=1) / total_features
    # 保留那些缺失比例低于max_missing_ratio的测试样本
    high_quality_test_mask = test_missing_ratio_series <= max_missing_ratio

    # 在这里保存筛选后的test_data的原始行索引，以便后续对应
    filtered_test_indices = test_data.index[high_quality_test_mask]

    test_data = test_data.loc[high_quality_test_mask].reset_index(drop=True)
    test_features = test_features.loc[high_quality_test_mask].reset_index(drop=True)

    # 若清洗后测试集为空则抛出异常
    if test_data.shape[0] == 0:
        raise ValueError("清洗后测试集为空，没有可用于测试的数据样本。请放宽max_missing_ratio或检查数据质量。")

    # 特征筛选：根据训练集特征情况剔除无用特征
    # 当某特征在训练集中超过feature_missing_threshold比例的样本为-105时，剔除该特征
    train_missing_ratio_per_feature = (train_features == -105).mean(axis=0)
    feature_mask = train_missing_ratio_per_feature < feature_missing_threshold

    train_features = train_features.loc[:, feature_mask]
    test_features = test_features.loc[:, feature_mask]

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
    X_train_raw, X_val_raw, y_train_coords_split, y_val_coords_split, y_train_floor_split, y_val_floor_split = train_test_split(
        train_features.values, y_train_coords, y_train_floor_encoded, test_size=0.1, random_state=42
    )

    # 对特征进行 Min-Max 缩放
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_val_scaled = scaler_X.transform(X_val_raw)
    X_test_scaled = scaler_X.transform(test_features.values)

    # 使用KernelPCA对特征降维（非线性降维）
    actual_pca_components = min(pca_components, X_train_scaled.shape[1])
    kpca = KernelPCA(n_components=actual_pca_components, kernel='rbf', random_state=42)
    X_train = kpca.fit_transform(X_train_scaled)
    X_val = kpca.transform(X_val_scaled)
    X_test = kpca.transform(X_test_scaled)

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

    # 输出处理后的数据集信息
    print("\n=== 数据集处理完成后的概况 ===")
    print("-----------------------------------------------------")
    print("| Dataset   | Samples | Features(X) | Target(Y Dim) |")
    print("-----------------------------------------------------")
    print(f"| X_train   | {X_train.shape[0]:7d} | {X_train.shape[1]:11d} | {y_train.shape[1]:11d} |")
    print(f"| X_val     | {X_val.shape[0]:7d} | {X_val.shape[1]:11d} | {y_val.shape[1]:11d} |")
    print(f"| X_test    | {X_test.shape[0]:7d} | {X_test.shape[1]:11d} | {y_test.shape[1]:11d} |")
    print("-----------------------------------------------------\n")

    print("训练集样本数：", X_train.shape[0])
    print("验证集样本数：", X_val.shape[0])
    print("测试集样本数：", X_test.shape[0])
    print("每条样本特征数（降维后）：", X_train.shape[1])
    print("目标维度（经度、纬度）：", y_train.shape[1])
    print("楼层类别数：", len(np.unique(y_train_floor_encoded)))

    # 返回处理后的数据和缩放器，以及经过筛选后的测试集索引
    scaler_y = {
        'scaler_y_longitude': scaler_y_longitude,
        'scaler_y_latitude': scaler_y_latitude
    }

    return X_train, y_train, y_train_floor_split, X_val, y_val, y_val_floor_split, X_test, y_test, y_test_floor_encoded, scaler_X, scaler_y, label_encoder, filtered_test_indices


if __name__ == "__main__":
    # 示例用法
    data = load_and_preprocess_data('train.csv', 'test.csv')
    print("预处理完成，数据已加载。")
