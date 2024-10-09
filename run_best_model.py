# run_best_model.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
    train_autoencoder,
    extract_features,
    train_and_evaluate_svr,
    compute_error_distances
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json


def main():
    # 固定训练轮数
    epochs = 50  # 要根据超参数脚本中的最佳轮数进行调整

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

    print("加载到的最优超参数组合：")
    print(json.dumps(best_params, indent=4, ensure_ascii=False))

    # 提取超参数
    model_dim = best_params['model_dim']
    num_heads = best_params['num_heads']
    num_layers = best_params['num_layers']
    dropout = best_params['dropout']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    patience = best_params['early_stopping_patience']
    svr_C = best_params['svr_C']
    svr_epsilon = best_params['svr_epsilon']

    # 3. 初始化 Transformer 自编码器模型
    print("初始化 Transformer 自编码器模型...")
    model = WiFiTransformerAutoencoder(
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # 4. 训练 Transformer 自编码器模型
    print("训练 Transformer 自编码器模型...")
    model = train_autoencoder(
        model, X_train, X_val,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=patience
    )

    # 5. 提取特征
    print("提取训练和测试特征...")
    X_train_features = extract_features(model, X_train, device=device, batch_size=batch_size)
    X_test_features = extract_features(model, X_test, device=device, batch_size=batch_size)

    # 6. 逆标准化目标变量
    y_train_original = scaler_y.inverse_transform(y_train)
    y_test_original = scaler_y.inverse_transform(y_test)

    # 7. 定义 SVR 参数
    svr_params = {
        'kernel': 'rbf',
        'C': svr_C,
        'epsilon': svr_epsilon
    }

    # 8. 训练并评估 SVR 模型
    print("训练并评估 SVR 回归模型...")
    svr_model = train_and_evaluate_svr(
        X_train_features, y_train_original,
        X_test_features, y_test_original,
        svr_params=svr_params
    )

    # 9. 预测并评估
    print("预测并计算评估指标...")
    y_pred = svr_model.predict(X_test_features)
    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    error_distances = compute_error_distances(y_test_original, y_pred)
    mean_error_distance = np.mean(error_distances)
    median_error_distance = np.median(error_distances)

    print(f"评估结果：")
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, R^2 Score: {r2:.6f}")
    print(f"平均误差距离（米）: {mean_error_distance:.2f}")
    print(f"中位数误差距离（米）: {median_error_distance:.2f}")

    # 10. 保存模型和参数
    print("保存模型和参数...")
    # 保存 Transformer 自编码器模型
    transformer_model_path = 'final_transformer_autoencoder.pth'
    torch.save(model.state_dict(), transformer_model_path)
    print(f"Transformer 自编码器模型已保存到 {transformer_model_path}")

    # 保存 SVR 模型
    svr_model_path = 'final_svr_model.pkl'
    joblib.dump(svr_model, svr_model_path)
    print(f"SVR 模型已保存到 {svr_model_path}")

    # 保存评估结果
    evaluation_results = {
        'MSE': mse,
        'MAE': mae,
        'R2_Score': r2,
        'Mean_Error_Distance': mean_error_distance,
        'Median_Error_Distance': median_error_distance
    }
    evaluation_results_path = 'evaluation_results.json'
    with open(evaluation_results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    print(f"评估结果已保存到 {evaluation_results_path}")

    print("模型训练和评估完成。")


if __name__ == '__main__':
    main()
