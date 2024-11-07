import pandas as pd


def load_data(file_path):
    # 加载CSV文件
    return pd.read_csv(file_path)


def analyze_data(df):
    # 选择信号强度列，这些列以"WAP"开头
    signal_columns = [col for col in df.columns if 'WAP' in col]
    building_floors = df[['BUILDINGID', 'FLOOR']]
    signal_features = df[signal_columns]

    # 计算每栋建筑物及其楼层的坐标点数量
    counts = building_floors.groupby(['BUILDINGID', 'FLOOR']).size().unstack(fill_value=0)
    building_counts = counts.sum(axis=1)
    print("每栋建筑的坐标点数量:")
    print(building_counts)
    print("每栋建筑下每层的坐标点数量:")
    print(counts)

    # 计算非零信号强度的平均比率
    non_zero_signals = (signal_features != 100).astype(int)  # 假设无信号的值是100
    non_zero_ratios = non_zero_signals.mean(axis=1)
    df['NON_ZERO_RATIO'] = non_zero_ratios

    # 计算每栋建筑及每层的平均非零信号比率
    avg_non_zero_ratios = df.groupby(['BUILDINGID', 'FLOOR'])['NON_ZERO_RATIO'].mean()
    print("每栋建筑及每层的非零信号比率平均值:")
    print(avg_non_zero_ratios)


# 加载数据
training_data = load_data('UJIndoorLoc/trainingData.csv')
validation_data = load_data('UJIndoorLoc/validationData.csv')

# 分析数据
print("训练数据分析结果:")
analyze_data(training_data)
print("\n验证数据分析结果:")
analyze_data(validation_data)
