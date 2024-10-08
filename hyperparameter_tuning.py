# hyperparameter_tuning.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import train_autoencoder, extract_features, train_and_evaluate_svr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools
import joblib
import os

def hyperparameter_tuning():
    """
    调优项目中的所有参数，包括Transformer自编码器和SVR回归模型的参数。

    该函数将遍历预定义的参数网格，训练模型，评估性能，并记录最佳参数组合。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 数据加载与预处理
    train_path = 'UJIndoorLoc/trainingData.csv'
    test_path = 'UJIndoorLoc/validationData.csv'
    print("加载并预处理数据...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_and_preprocess_data(train_path, test_path)

    # 2. 定义参数网格
    transformer_params = {
        'model_dim': [128, 256],          # 模型维度
        'num_heads': [8, 16],             # 多头注意力头数
        'num_layers': [2, 4],             # Transformer编码器层数
        'dropout': [0.1, 0.2, 0.3],            # Dropout概率
        'learning_rate': [1e-4,1e-3,1e-2,2e-4],    # 学习率
        'batch_size': [256, 512],         # 批次大小
        'epochs': [40,50,70],                    # 训练轮数
        'early_stopping_patience': [5]    # 早停轮数
    }

    svr_params = {
        # 'estimator__kernel' 指定 SVR 模型的核函数类型
        'estimator__kernel': ['rbf', 'linear'],  # 'rbf' 是径向基函数，适用于非线性问题；'linear' 是线性核，适用于线性问题

        # 'estimator__C' 指定 SVR 模型的惩罚参数 C
        'estimator__C': [1, 10, 100],  # C 的值越大，模型越倾向于正确分类所有训练样本（可能导致过拟合）

        # 'estimator__epsilon' 指定 SVR 模型中 epsilon 参数的值
        'estimator__epsilon': [0.1, 0.2, 0.5]  # epsilon 控制模型对预测误差的容忍度，值越小，模型对误差的敏感度越高
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
        transformer_params['early_stopping_patience']
    ))

    print(f"总共有 {len(param_combinations)} 组 Transformer 参数组合需要评估。")

    # 初始化最佳性能记录
    best_r2 = -float('inf')
    best_params = {}
    best_svr_model = None
    best_transformer_params = {}

    # 遍历每一组Transformer参数
    for idx, (model_dim, num_heads, num_layers, dropout, learning_rate, batch_size, epochs, patience) in enumerate(param_combinations):
        print(f"\n正在评估第 {idx+1}/{len(param_combinations)} 组参数：")
        print(f"Transformer参数 - model_dim: {model_dim}, num_heads: {num_heads}, num_layers: {num_layers}, dropout: {dropout}")
        print(f"训练参数 - learning_rate: {learning_rate}, batch_size: {batch_size}, epochs: {epochs}, early_stopping_patience: {patience}")

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

        # 训练 MultiOutputRegressor SVR
        best_svr = train_and_evaluate_svr(X_train_features, y_train_original, X_test_features, y_test_original)

        # 预测并评估
        y_pred = best_svr.predict(X_test_features)

        mse = mean_squared_error(y_test_original, y_pred)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        print(f"当前参数组合的评估结果：")
        print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R^2 Score: {r2:.6f}")

        # 更新最佳参数
        if r2 > best_r2:
            best_r2 = r2
            best_params = {
                'model_dim': model_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'early_stopping_patience': patience,
                'svr_params': best_svr.estimator.get_params()
            }
            best_svr_model = best_svr
            best_transformer_params = {
                'model_dim': model_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'early_stopping_patience': patience
            }

    # 保存最佳模型和参数
    print("\n超参数调优完成。最佳参数组合如下：")
    print(best_params)
    print(f"最佳 R^2 Score: {best_r2:.6f}")

    # 保存最佳Transformer模型
    transformer_model = WiFiTransformerAutoencoder(
        model_dim=best_transformer_params['model_dim'],
        num_heads=best_transformer_params['num_heads'],
        num_layers=best_transformer_params['num_layers'],
        dropout=best_transformer_params['dropout']
    ).to(device)
    transformer_model.load_state_dict(torch.load('best_transformer_autoencoder.pth'))
    torch.save(transformer_model.state_dict(), 'best_transformer_autoencoder_tuned.pth')

    # 保存最佳SVR模型
    joblib.dump(best_svr_model, 'best_svr_model_tuned.pkl')

    # 保存最佳参数
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)

    print("最佳模型和参数已保存。")

if __name__ == '__main__':
    hyperparameter_tuning()
