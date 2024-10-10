import pandas as pd

# 读取CSV文件
data = pd.read_csv('UJIndoorLoc/trainingData_building2.csv')

# 遍历所有列
columns_to_drop = []  # 用于存储需要删除的列名
for column in data.columns:
    # 检查除了第一行外的所有值是否都为100
    if (data[column][1:] == 100).all():
        columns_to_drop.append(column)

# 删除满足条件的列
data.drop(columns=columns_to_drop, inplace=True)

# 将处理后的数据写回到CSV
data.to_csv('UJIndoorLoc/trainingData_building2_af.csv', index=False)
