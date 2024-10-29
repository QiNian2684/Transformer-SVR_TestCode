import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
import sys

# 尝试导入cuML的SVR以支持CUDA加速
try:
    from cuml.svm import SVR as cuSVR
    import cudf

    cuda_available = True
    print("检测到CUDA支持，使用cuML的SVR进行GPU加速。\n")
except ImportError:
    from sklearn.svm import SVR

    cuda_available = False
    print("未检测到CUDA支持，使用scikit-learn的SVR进行CPU计算。\n")


def main():
    print("=== 使用SVR和Optuna进行室内定位预测 ===\n")

    # 加载数据
    print("加载训练和验证数据...")
    try:
        train_data = pd.read_csv('UJIndoorLoc/trainingData.csv')  # 更新文件路径以包含所有建筑物
        validation_data = pd.read_csv('UJIndoorLoc/validationData.csv')  # 更新文件路径以包含所有建筑物
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        sys.exit(1)
    print("数据加载成功。\n")

    # 可选：显示数据集的基本信息
    print("训练数据形状:", train_data.shape)
    print("验证数据形状:", validation_data.shape, "\n")

    # 动态选择以'WAP'开头的特征列
    print("动态选择以'WAP'开头的特征列...")
    feature_columns = [col for col in train_data.columns if col.startswith('WAP')]
    print(f"选择的特征列数量: {len(feature_columns)}\n")

    # 分离特征和目标
    print("分离特征和目标...")
    if cuda_available:
        # 使用cuDF将数据转换为GPU DataFrame
        features_train = cudf.DataFrame.from_pandas(train_data[feature_columns])
        target_train = cudf.DataFrame.from_pandas(train_data[['LONGITUDE', 'LATITUDE']])

        features_val = cudf.DataFrame.from_pandas(validation_data[feature_columns])
        target_val = cudf.DataFrame.from_pandas(validation_data[['LONGITUDE', 'LATITUDE']])
    else:
        # 使用pandas DataFrame
        features_train = train_data[feature_columns]
        target_train = train_data[['LONGITUDE', 'LATITUDE']]

        features_val = validation_data[feature_columns]
        target_val = validation_data[['LONGITUDE', 'LATITUDE']]
    print("特征和目标分离完成。\n")

    # 定义Optuna的目标函数
    def objective(trial):
        print(f"--- 开始第 {trial.number + 1} 次试验 ---")

        # 超参数建议
        svr_kernel = trial.suggest_categorical('svr_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        svr_C = trial.suggest_float('svr_C', 1e-1, 1e2, log=True)
        svr_epsilon = trial.suggest_float('svr_epsilon', 0.01, 1.0)
        svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])

        # 如果核函数是多项式（poly），则选择多项式的度数和coef0参数
        if svr_kernel == 'poly':
            svr_degree = trial.suggest_int('svr_degree', 2, 5)  # 多项式的度数
            svr_coef0 = trial.suggest_float('svr_coef0', 0.0, 1.0)  # 多项式核函数中的独立项系数
        else:
            svr_degree = 3  # 默认值为3
            svr_coef0 = 0.0  # 默认coef0为0.0

        print(
            f"第 {trial.number + 1} 次试验参数: kernel={svr_kernel}, C={svr_C:.4f}, epsilon={svr_epsilon:.4f}, gamma={svr_gamma}, degree={svr_degree}, coef0={svr_coef0:.4f}")

        # 根据核函数类型初始化SVR模型的参数字典
        svr_kwargs = {
            'C': svr_C,
            'epsilon': svr_epsilon,
            'kernel': svr_kernel,
            'gamma': svr_gamma
        }

        if svr_kernel == 'poly':
            svr_kwargs['degree'] = svr_degree
            svr_kwargs['coef0'] = svr_coef0

        # 选择使用cuML或scikit-learn的SVR
        if cuda_available:
            svr_lon = cuSVR(**svr_kwargs)
            svr_lat = cuSVR(**svr_kwargs)
        else:
            svr_lon = SVR(**svr_kwargs)
            svr_lat = SVR(**svr_kwargs)

        print("训练SVR模型...")
        svr_lon.fit(features_train, target_train['LONGITUDE'])
        svr_lat.fit(features_train, target_train['LATITUDE'])
        print("模型训练完成。")

        # 在验证集上进行预测
        print("在验证集上进行预测...")
        pred_lon = svr_lon.predict(features_val)
        pred_lat = svr_lat.predict(features_val)
        print("预测完成。")

        # 计算误差
        if cuda_available:
            # 将预测结果从GPU转换回CPU以进行计算
            pred_lon_cpu = pred_lon.to_pandas().values
            pred_lat_cpu = pred_lat.to_pandas().values
            target_lon_cpu = target_val['LONGITUDE'].to_pandas().values
            target_lat_cpu = target_val['LATITUDE'].to_pandas().values
        else:
            pred_lon_cpu = pred_lon
            pred_lat_cpu = pred_lat
            target_lon_cpu = target_val['LONGITUDE'].values
            target_lat_cpu = target_val['LATITUDE'].values

        errors = np.sqrt((pred_lon_cpu - target_lon_cpu) ** 2 + (pred_lat_cpu - target_lat_cpu) ** 2)
        mean_error = np.mean(errors)
        median_error = np.median(errors)

        print(f"第 {trial.number + 1} 次试验结果: 平均误差 = {mean_error:.2f} 米, 中位误差 = {median_error:.2f} 米\n")

        # 将平均误差作为目标函数值进行最小化
        return mean_error

    # 开始超参数优化
    print("开始使用Optuna进行超参数优化...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    print("超参数优化完成。\n")

    # 获取最佳超参数
    best_params = study.best_params
    print("找到的最佳超参数:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print()

    # 提取最佳超参数
    best_svr_kernel = best_params['svr_kernel']
    best_svr_C = best_params['svr_C']
    best_svr_epsilon = best_params['svr_epsilon']
    best_svr_gamma = best_params['svr_gamma']

    if best_svr_kernel == 'poly':
        best_svr_degree = best_params['svr_degree']
        best_svr_coef0 = best_params['svr_coef0']
    else:
        best_svr_degree = 3  # 默认值
        best_svr_coef0 = 0.0  # 默认值

    # 根据最佳超参数初始化SVR模型的参数字典
    svr_kwargs = {
        'C': best_svr_C,
        'epsilon': best_svr_epsilon,
        'kernel': best_svr_kernel,
        'gamma': best_svr_gamma
    }

    if best_svr_kernel == 'poly':
        svr_kwargs['degree'] = best_svr_degree
        svr_kwargs['coef0'] = best_svr_coef0

    print("使用最佳超参数训练最终模型...")
    if cuda_available:
        svr_lon = cuSVR(**svr_kwargs)
        svr_lat = cuSVR(**svr_kwargs)
    else:
        svr_lon = SVR(**svr_kwargs)
        svr_lat = SVR(**svr_kwargs)

    svr_lon.fit(features_train, target_train['LONGITUDE'])
    svr_lat.fit(features_train, target_train['LATITUDE'])
    print("最终模型训练完成。\n")

    # 在验证集上进行最终预测
    print("在验证集上进行最终预测...")
    pred_lon = svr_lon.predict(features_val)
    pred_lat = svr_lat.predict(features_val)
    print("最终预测完成。\n")

    # 计算误差指标
    if cuda_available:
        # 将预测结果从GPU转换回CPU以进行计算
        pred_lon_cpu = pred_lon.to_pandas().values
        pred_lat_cpu = pred_lat.to_pandas().values
        target_lon_cpu = target_val['LONGITUDE'].to_pandas().values
        target_lat_cpu = target_val['LATITUDE'].to_pandas().values
    else:
        pred_lon_cpu = pred_lon
        pred_lat_cpu = pred_lat
        target_lon_cpu = target_val['LONGITUDE'].values
        target_lat_cpu = target_val['LATITUDE'].values

    print("计算误差指标...")
    errors = np.sqrt((pred_lon_cpu - target_lon_cpu) ** 2 + (pred_lat_cpu - target_lat_cpu) ** 2)
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    # 计算MSE
    mse_lon = mean_squared_error(target_lon_cpu, pred_lon_cpu)
    mse_lat = mean_squared_error(target_lat_cpu, pred_lat_cpu)

    # 计算平均2D误差
    predictions = np.vstack((pred_lon_cpu, pred_lat_cpu)).T
    actuals = np.vstack((target_lon_cpu, target_lat_cpu)).T
    differences = np.sqrt(np.sum((predictions - actuals) ** 2, axis=1))
    average_2d_error = np.mean(differences)

    print("误差指标计算完成。\n")

    # 打印评估结果
    print("=== 评估指标 ===")
    print(f"经度的均方误差 (MSE): {mse_lon:.2f}")
    print(f"纬度的均方误差 (MSE): {mse_lat:.2f}")
    print(f"平均误差: {mean_error:.2f} 米")
    print(f"中位误差: {median_error:.2f} 米")
    print(f"平均2D误差: {average_2d_error:.2f} 米\n")

    # 可选：显示一些样本预测结果
    print("=== 样本预测结果 ===")
    sample_size = 5
    for i in range(sample_size):
        print(f"样本 {i + 1}:")
        print(f"  预测经度: {pred_lon_cpu[i]:.2f}, 实际经度: {target_lon_cpu[i]:.2f}")
        print(f"  预测纬度: {pred_lat_cpu[i]:.2f}, 实际纬度: {target_lat_cpu[i]:.2f}")
        print(f"  误差: {errors[i]:.2f} 米\n")


if __name__ == "__main__":
    main()
