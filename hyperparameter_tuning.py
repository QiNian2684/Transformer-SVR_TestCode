# hyperparameter_tuning.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import train_autoencoder, extract_features, train_and_evaluate_svr, compute_error_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import joblib
import os
import json

def hyperparameter_tuning():
    """
    调优项目中的所有参数，包括 Transformer 自编码器和 SVR 回归模型的参数。

    该函数将遍历预定义的参数网格，训练模型，评估性能，并记录最佳参数组合。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 数据加载与预处理
    train_path = 'UJIndoorLoc/trainingData.csv'
    test_path = 'UJIndoorLoc/validationData.csv'
    print("加载并预处理数据...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_and_preprocess_data(train_path, test_path)

    # 2. 定义参数网格（缩小参数范围以减少计算量）
    transformer_params = {
        'model_dim': [128],  # 模型的嵌入维度（隐藏层维度）
        'num_heads': [4],  # 多头注意力机制中的头数
        'num_layers': [4],  # Transformer模型的层数
        'dropout': [0.3],  # 用于防止过拟合的Dropout率
        'learning_rate': [0.0005],  # 优化器的学习率
        'batch_size': [32],  # 训练时使用的批大小
        'epochs': [50],  # 训练的轮数
        'early_stopping_patience': [5]  # 提前停止的耐心参数（若验证损失在5轮内未改善则停止训练）
    }

    svr_params_grid = {
        'kernel': ['rbf'],  # SVR中使用的核函数类型（'rbf'表示径向基函数）
        'C': [1, 10],  # 误差项的惩罚参数C
        'epsilon': [0.1, 0.2]  # epsilon-SVR模型中的epsilon值
    }

    # 创建所有参数组合的笛卡尔积
    param_combinations = list(itertools.product(
        transformer_params['model_dim'],
        transformer_params['num_heads'],
        transformer_params['num_layers'],
        transformer_params['dropout'],
        transformer_params['learning_rate'],
        transformer_params['batch_size'],
        transformer_params['epochs'],
        transformer_params['early_stopping_patience'],
        svr_params_grid['kernel'],
        svr_params_grid['C'],
        svr_params_grid['epsilon']
    ))

    print(f"总共有 {len(param_combinations)} 组参数组合需要评估。")

    # 初始化最佳性能记录
    best_mean_error_distance = float('inf')
    best_params = {}
    best_svr_model = None
    best_transformer_model_state = None
    best_transformer_params = {}

    # 遍历每一组参数组合
    for idx, (model_dim, num_heads, num_layers, dropout, learning_rate, batch_size, epochs, patience, svr_kernel, svr_C, svr_epsilon) in enumerate(param_combinations):
        print(f"\n正在评估第 {idx+1}/{len(param_combinations)} 组参数：")
        print(f"Transformer参数 - model_dim: {model_dim}, num_heads: {num_heads}, num_layers: {num_layers}, dropout: {dropout}")
        print(f"训练参数 - learning_rate: {learning_rate}, batch_size: {batch_size}, epochs: {epochs}, early_stopping_patience: {patience}")
        print(f"SVR参数 - kernel: {svr_kernel}, C: {svr_C}, epsilon: {svr_epsilon}")

        # 3. 初始化 Transformer 自编码器模型
        model = WiFiTransformerAutoencoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # 4. 训练 Transformer 自编码器模型
        model = train_autoencoder(
            model, X_train, X_val,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=patience
        )

        # 5. 提取特征
        X_train_features = extract_features(model, X_train, device=device, batch_size=batch_size)
        X_test_features = extract_features(model, X_test, device=device, batch_size=batch_size)

        # 6. 训练和评估 SVR 模型
        print("训练和评估 SVR 回归模型...")
        # 逆标准化目标变量进行训练和评估
        y_train_original = scaler_y.inverse_transform(y_train)
        y_test_original = scaler_y.inverse_transform(y_test)

        # 定义当前 SVR 参数
        current_svr_params = {
            'kernel': svr_kernel,
            'C': svr_C,
            'epsilon': svr_epsilon
        }

        # 训练 SVR 模型
        svr_model = train_and_evaluate_svr(
            X_train_features, y_train_original,
            X_test_features, y_test_original,
            svr_params=current_svr_params
        )

        # 预测并评估
        y_pred = svr_model.predict(X_test_features)

        mse = mean_squared_error(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        # 计算误差距离
        error_distances = compute_error_distances(y_test_original, y_pred)
        mean_error_distance = np.mean(error_distances)
        median_error_distance = np.median(error_distances)

        print(f"当前参数组合的评估结果：")
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R^2 Score: {r2:.6f}")
        print(f"平均误差距离（米）: {mean_error_distance:.2f}")
        print(f"中位数误差距离（米）: {median_error_distance:.2f}")

        # 更新最佳参数
        if mean_error_distance < best_mean_error_distance:
            best_mean_error_distance = mean_error_distance
            best_params = {
                'transformer_params': {
                    'model_dim': model_dim,
                    'num_heads': num_heads,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'early_stopping_patience': patience
                },
                'svr_params': current_svr_params
            }
            best_svr_model = svr_model
            best_transformer_model_state = model.state_dict()
            best_transformer_params = {
                'model_dim': model_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout
            }

    # 保存最佳模型和参数
    print("\n超参数调优完成。最佳参数组合如下：")
    print(json.dumps(best_params, indent=4, ensure_ascii=False))
    print(f"最佳平均误差距离（米）: {best_mean_error_distance:.2f}")

    # 保存最佳 Transformer 模型
    transformer_model = WiFiTransformerAutoencoder(
        model_dim=best_transformer_params['model_dim'],
        num_heads=best_transformer_params['num_heads'],
        num_layers=best_transformer_params['num_layers'],
        dropout=best_transformer_params['dropout']
    ).to(device)
    transformer_model.load_state_dict(best_transformer_model_state)
    torch.save(transformer_model.state_dict(), 'best_transformer_autoencoder_tuned.pth')

    # 保存最佳 SVR 模型
    joblib.dump(best_svr_model, 'best_svr_model_tuned.pkl')

    # 保存最佳参数
    with open('best_hyperparameters.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False)

    print("最佳模型和参数已保存。")

if __name__ == '__main__':
    hyperparameter_tuning()
