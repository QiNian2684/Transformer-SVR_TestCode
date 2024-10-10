import pandas as pd
import numpy as np
import os


def check_csv(file_path):
    print(f"\n检查文件: {file_path}")
    if not os.path.exists(file_path):
        print("文件不存在!")
        return

    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    report = {}

    # 1. 缺失值检查
    missing_values = data.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    if not missing_columns.empty:
        report['缺失值'] = missing_columns
    else:
        report['缺失值'] = "没有缺失值"

    # 2. 无限值检查
    infinite_values = (data.isin([np.inf, -np.inf]).sum())
    infinite_columns = infinite_values[infinite_values > 0]
    if not infinite_columns.empty:
        report['无限值'] = infinite_columns
    else:
        report['无限值'] = "没有无限值"

    # 3. 常数列检查（除第一行外）
    constant_columns = []
    for column in data.columns:
        if data[column].nunique() == 1:
            constant_columns.append(column)
    if constant_columns:
        report['常数列'] = constant_columns
    else:
        report['常数列'] = "没有常数列"

    # 4. 数据类型检查
    data_types = data.dtypes
    non_numeric_columns = data_types[data_types == 'object'].index.tolist()
    if non_numeric_columns:
        report['非数值型列'] = non_numeric_columns
    else:
        report['非数值型列'] = "所有列都是数值型"

    # 5. 重复行检查
    duplicate_rows = data.duplicated().sum()
    if duplicate_rows > 0:
        report['重复行'] = f"存在 {duplicate_rows} 行重复数据"
    else:
        report['重复行'] = "没有重复行"

    # 打印报告
    print("\n=== 检查报告 ===")
    for key, value in report.items():
        print(f"\n{key}:")
        print(value)

    # 额外建议
    if report['缺失值'] != "没有缺失值" or report['无限值'] != "没有无限值" or report[
        '非数值型列'] != "所有列都是数值型":
        print("\n建议:")
        if report['缺失值'] != "没有缺失值":
            print("- 处理缺失值，例如填充或删除含有缺失值的行/列。")
        if report['无限值'] != "没有无限值":
            print("- 处理无限值，例如替换为合理的数值或删除相关行。")
        if report['非数值型列'] != "所有列都是数值型":
            print("- 确认非数值型列是否需要转换为数值型，或者在模型训练时排除这些列。")
    else:
        print("\n数据看起来很好，可以继续模型训练。")


if __name__ == "__main__":
    # 替换为你的CSV文件路径
    training_csv = 'UJIndoorLoc/trainingData_building2.csv'
    validation_csv = 'UJIndoorLoc/validationData_building2.csv'

    check_csv(training_csv)
    check_csv(validation_csv)
