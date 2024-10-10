# run_best_model.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
    train_autoencoder,
    extract_features,
    train_and_evaluate_svr,
    compute_error_distances,
    NaNLossError
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 引入3D绘图工具包

def main():
    try:
        # 固定训练轮数
        epochs = 200  # 根据超参数调优的最佳轮数进行调整

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 1. 数据加载与预处理
        train_path = 'UJIndoorLoc/trainingData_building0.csv'
        test_path = 'UJIndoorLoc/validationData_building0.csv'
        print("加载并预处理数据...")
        X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_and_preprocess_data(train_path, test_path)

        # 2. 加载最优超参数
        best_hyperparams_path = 'best_hyperparameters_optuna.json'
        if not os.path.exists(best_hyperparams_path):
            print(f"最优超参数文件 {best_hyperparams_path} 不存在，请先运行超参数调优。")
            return

        with open(best_hyperparams_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)

        print("\n加载到的最优超参数组合：")
        print(json.dumps(best_params, indent=4, ensure_ascii=False))

        # 提取超参数
        model_dim = best_params['model_dim']
        num_heads = best_params['num_heads']
        num_layers = best_params['num_layers']
        dropout = best_params['dropout']
        learning_rate = best_params['learning_rate']
        batch_size = best_params['batch_size']
        patience = best_params['early_stopping_patience']
        svr_kernel = best_params['svr_kernel']
        svr_C = best_params['svr_C']
        svr_epsilon = best_params['svr_epsilon']
        svr_gamma = best_params['svr_gamma']
        # 如果 kernel 是 'poly'，需要提取 degree 和 coef0
        if svr_kernel == 'poly':
            svr_degree = best_params['svr_degree']
            svr_coef0 = best_params['svr_coef0']
        else:
            svr_degree = 3  # 默认值
            svr_coef0 = 0.0  # 默认值

        # 3. 初始化 Transformer 自编码器模型
        print("\n初始化 Transformer 自编码器模型...")
        model = WiFiTransformerAutoencoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # 4. 训练 Transformer 自编码器模型
        print("\n训练 Transformer 自编码器模型...")
        model = train_autoencoder(
            model, X_train, X_val,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=patience
        )

        # 5. 提取特征
        print("\n提取训练和测试特征...")
        X_train_features = extract_features(model, X_train, device=device, batch_size=batch_size)
        X_val_features = extract_features(model, X_val, device=device, batch_size=batch_size)
        X_test_features = extract_features(model, X_test, device=device, batch_size=batch_size)

        # 检查提取的特征中是否存在 NaN
        if np.isnan(X_train_features).any() or np.isnan(X_test_features).any():
            print("提取的特征中包含 NaN，无法训练 SVR 模型。")
            return

        # 6. 逆标准化目标变量
        y_train_original = scaler_y.inverse_transform(y_train)
        y_val_original = scaler_y.inverse_transform(y_val)
        y_test_original = scaler_y.inverse_transform(y_test)

        # 7. 定义 SVR 参数
        svr_params = {
            'kernel': svr_kernel,
            'C': svr_C,
            'epsilon': svr_epsilon,
            'gamma': svr_gamma,
            'degree': svr_degree,
            'coef0': svr_coef0
        }

        # 8. 训练并评估 SVR 模型
        print("\n训练并评估 SVR 回归模型...")
        svr_model = train_and_evaluate_svr(
            X_train_features, y_train_original,
            X_test_features, y_test_original,
            svr_params=svr_params
        )

        # 9. 预测并评估
        print("\n预测并计算评估指标...")
        y_pred = svr_model.predict(X_test_features)

        # 轮廓预测：四舍五入楼层
        y_pred_rounded = y_pred.copy()
        y_pred_rounded[:, 2] = np.round(y_pred_rounded[:, 2])

        mse = mean_squared_error(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        error_distances = compute_error_distances(y_test_original, y_pred)
        mean_error_distance = np.mean(error_distances)
        median_error_distance = np.median(error_distances)

        print("\n评估结果：")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R^2 Score: {r2:.6f}")
        print(f"平均误差距离（米）: {mean_error_distance:.2f}")
        print(f"中位数误差距离（米）: {median_error_distance:.2f}")

        # 分别评估每个目标变量
        print("\n各目标变量的评估结果：")
        for i, target in enumerate(['LONGITUDE', 'LATITUDE', 'FLOOR']):
            mse_i = mean_squared_error(y_test_original[:, i], y_pred[:, i])
            mae_i = mean_absolute_error(y_test_original[:, i], y_pred[:, i])
            r2_i = r2_score(y_test_original[:, i], y_pred[:, i])
            print(f"{target} - MSE: {mse_i:.6f}, MAE: {mae_i:.6f}, R^2 Score: {r2_i:.6f}")

        # 计算误差距离并生成3D误差散点图
        error_x = y_pred[:, 0] - y_test_original[:, 0]
        error_y = y_pred[:, 1] - y_test_original[:, 1]
        error_floor = y_pred[:, 2] - y_test_original[:, 2]

        # 将楼层误差转换为米
        FLOOR_HEIGHT = 3  # 根据您的设定，每层楼的高度
        error_z = error_floor * FLOOR_HEIGHT

        # 配置 Matplotlib 使用支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']，根据您的系统字体选择
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

        # 创建3D图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 计算每个点的误差距离
        error_distance = np.sqrt(error_x ** 2 + error_y ** 2 + error_z ** 2)

        scatter = ax.scatter(error_x, error_y, error_z, c=error_distance, cmap='viridis', alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('误差距离 (米)')

        ax.set_title('3D 预测误差散点图')
        ax.set_xlabel('经度误差 (米)')
        ax.set_ylabel('纬度误差 (米)')
        ax.set_zlabel('高度误差 (米)')  # Z 代表垂直误差

        plt.show()

        print("\n模型训练和评估完成。")

    except Exception as e:
        print(f"程序遇到未处理的异常：{e}")

if __name__ == '__main__':
    main()

