# data_filtering.py
import numpy as np

import pandas as pd

def filter_building_zero(train_input_path, test_input_path, train_output_path, test_output_path):
    """
    从原始数据集中筛选出 BUILDINGID 为 0 的数据，并保存为新的 CSV 文件。

    参数：
    - train_input_path: 原始训练数据文件路径
    - test_input_path: 原始测试数据文件路径
    - train_output_path: 筛选后的训练数据保存路径
    - test_output_path: 筛选后的测试数据保存路径
    """
    # 1. 加载原始训练数据和测试数据
    print("加载原始训练数据...")
    train_data = pd.read_csv(train_input_path)
    print("加载原始测试数据...")
    test_data = pd.read_csv(test_input_path)

    # 2. 筛选出 BUILDINGID 为 0 的数据
    print("筛选 BUILDINGID 为 0 的训练数据...")
    train_data_building0 = train_data[train_data['BUILDINGID'] == 0].reset_index(drop=True)
    print(f"筛选后的训练数据形状: {train_data_building0.shape}")

    print("筛选 BUILDINGID 为 0 的测试数据...")
    test_data_building0 = test_data[test_data['BUILDINGID'] == 0].reset_index(drop=True)
    print(f"筛选后的测试数据形状: {test_data_building0.shape}")

    # 3. 检查是否存在 NaN 或 Inf 值
    def check_for_nan_inf(df, name):
        num_nan = df.isnull().sum().sum()
        num_inf = df.isin([np.inf, -np.inf]).sum().sum()
        print(f"{name} 中 NaN 值的总数：{num_nan}")
        print(f"{name} 中 Inf 值的总数：{num_inf}")

    check_for_nan_inf(train_data_building0, "筛选后的训练数据")
    check_for_nan_inf(test_data_building0, "筛选后的测试数据")

    # 4. 保存筛选后的数据到新的 CSV 文件
    print("保存筛选后的训练数据...")
    train_data_building0.to_csv(train_output_path, index=False)
    print("保存筛选后的测试数据...")
    test_data_building0.to_csv(test_output_path, index=False)

    print("数据过滤和保存完成。")

if __name__ == "__main__":
    # 指定原始数据文件路径
    train_input_path = 'UJIndoorLoc/trainingData.csv'
    test_input_path = 'UJIndoorLoc/validationData.csv'

    # 指定筛选后的数据保存路径
    train_output_path = 'UJIndoorLoc/trainingData_building0.csv'
    test_output_path = 'UJIndoorLoc/validationData_building0.csv'

    filter_building_zero(train_input_path, test_input_path, train_output_path, test_output_path)
