import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import optuna


def main():

    # 定义训练次数
    train_times = 200

    print("=== 使用SVR和Optuna进行室内定位预测 ===\n")

    # 加载数据
    print("加载训练和验证数据...")
    train_data = pd.read_csv('UJIndoorLoc/trainingData.csv')  # 更新文件路径以包含所有建筑物
    validation_data = pd.read_csv('UJIndoorLoc/validationData.csv')  # 更新文件路径以包含所有建筑物
    print("数据加载成功。\n")

    # 可选：显示数据集的基本信息
    print("训练数据形状:", train_data.shape)
    print("验证数据形状:", validation_data.shape, "\n")

    # 分离特征和目标
    print("分离特征和目标...")
    # 根据数据集的实际结构调整需要删除的列
    features_train = train_data.drop(columns=[
        'LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID',
        'SPACEID', 'RELATIVEPOSITION', 'USERID',
        'PHONEID', 'TIMESTAMP'
    ])
    target_train = train_data.loc[:, ['LONGITUDE', 'LATITUDE']]

    features_val = validation_data.drop(columns=[
        'LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID',
        'SPACEID', 'RELATIVEPOSITION', 'USERID',
        'PHONEID', 'TIMESTAMP'
    ])
    target_val = validation_data.loc[:, ['LONGITUDE', 'LATITUDE']]
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

        # 根据核函数类型初始化SVR模型
        svr_kwargs = {
            'C': svr_C,
            'epsilon': svr_epsilon,
            'kernel': svr_kernel,
            'gamma': svr_gamma
        }

        if svr_kernel == 'poly':
            svr_kwargs['degree'] = svr_degree
            svr_kwargs['coef0'] = svr_coef0

        # 训练SVR模型用于经度
        svr_lon = SVR(**svr_kwargs)
        # 训练SVR模型用于纬度
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
        errors = np.sqrt((pred_lon - target_val['LONGITUDE']) ** 2 + (pred_lat - target_val['LATITUDE']) ** 2)
        mean_error = np.mean(errors)
        median_error = np.median(errors)

        print(f"第 {trial.number + 1} 次试验结果: 平均误差 = {mean_error:.2f} 米, 中位误差 = {median_error:.2f} 米\n")

        # 将平均误差作为目标函数值进行最小化
        return mean_error

    # 开始超参数优化
    print("开始使用Optuna进行超参数优化...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=train_times, show_progress_bar=True)
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

    # 根据最佳超参数初始化SVR模型
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
    print("计算误差指标...")
    errors = np.sqrt((pred_lon - target_val['LONGITUDE']) ** 2 + (pred_lat - target_val['LATITUDE']) ** 2)
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    # 计算MSE
    mse_lon = mean_squared_error(target_val['LONGITUDE'], pred_lon)
    mse_lat = mean_squared_error(target_val['LATITUDE'], pred_lat)

    # 计算平均2D误差
    predictions = np.vstack((pred_lon, pred_lat)).T
    actuals = target_val[['LONGITUDE', 'LATITUDE']].values
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
        print(f"  预测经度: {pred_lon[i]:.2f}, 实际经度: {target_val['LONGITUDE'].iloc[i]:.2f}")
        print(f"  预测纬度: {pred_lat[i]:.2f}, 实际纬度: {target_val['LATITUDE'].iloc[i]:.2f}")
        print(f"  误差: {errors[i]:.2f} 米\n")


if __name__ == "__main__":
    main()
