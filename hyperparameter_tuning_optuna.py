# hyperparameter_tuning_optuna.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
    train_autoencoder,
    extract_features,
    train_and_evaluate_svr,
    compute_error_distances,
)
import optuna
from optuna.exceptions import TrialPruned
import joblib
import json

def main():
    # === 参数设置 ===
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    train_path = 'UJIndoorLoc/trainingData_building0.csv'
    test_path = 'UJIndoorLoc/validationData_building0.csv'

    # 固定训练参数
    epochs = 50  # 训练轮数
    n_trials = 100  # Optuna 试验次数，根据计算资源调整
    n_jobs = 1     # 并行进程数量

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_and_preprocess_data(train_path, test_path)

    # === 定义优化目标函数 ===
    def objective(trial):
        # Transformer 自编码器超参数
        model_dim = trial.suggest_categorical('model_dim', [16, 32, 64])
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        num_layers = trial.suggest_categorical('num_layers', [4, 8, 16])
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128])
        patience = trial.suggest_int('early_stopping_patience', 5, 15)

        # SVR 超参数
        svr_C = trial.suggest_float('svr_C', 1e-1, 1e2, log=True)
        svr_epsilon = trial.suggest_float('svr_epsilon', 0.0, 1.0)

        # 收集当前超参数组合
        current_params = {
            'model_dim': model_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'early_stopping_patience': patience,
            'svr_C': svr_C,
            'svr_epsilon': svr_epsilon
        }

        # 打印当前超参数组合
        print(f"\n当前超参数组合：\n{json.dumps(current_params, indent=4, ensure_ascii=False)}")

        # 初始化 Transformer 自编码器模型
        model = WiFiTransformerAutoencoder(
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        # 训练自编码器
        model = train_autoencoder(
            model, X_train, X_val,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=patience
        )

        # 提取特征
        X_train_features = extract_features(model, X_train, device=device, batch_size=batch_size)
        X_test_features = extract_features(model, X_test, device=device, batch_size=batch_size)

        # 逆标准化目标变量
        y_train_original = scaler_y.inverse_transform(y_train)
        y_test_original = scaler_y.inverse_transform(y_test)

        # 定义 SVR 参数
        svr_params = {
            'kernel': 'rbf',
            'C': svr_C,
            'epsilon': svr_epsilon
        }

        # 训练 SVR 模型
        svr_model = train_and_evaluate_svr(
            X_train_features, y_train_original,
            X_test_features, y_test_original,
            svr_params=svr_params
        )

        # 预测并评估
        y_pred = svr_model.predict(X_test_features)
        error_distances = compute_error_distances(y_test_original, y_pred)
        mean_error_distance = np.mean(error_distances)

        # 报告中间结果并检查是否应该剪枝
        trial.report(mean_error_distance, step=0)

        if trial.should_prune():
            raise TrialPruned()

        # 返回要最小化的目标值
        return mean_error_distance

    # === 创建和运行优化研究 ===
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # === 打印和保存最佳结果 ===
    print("最佳参数：")
    print(json.dumps(study.best_params, indent=4, ensure_ascii=False))
    print(f"最佳平均误差距离（米）：{study.best_value:.2f}")

    best_params = study.best_params

    # === 使用最佳超参数重新训练模型 ===
    print("\n使用最佳超参数重新训练模型...")
    # 提取最佳超参数
    model_dim = best_params['model_dim']
    num_heads = best_params['num_heads']
    num_layers = best_params['num_layers']
    dropout = best_params['dropout']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    patience = best_params['early_stopping_patience']
    svr_C = best_params['svr_C']
    svr_epsilon = best_params['svr_epsilon']
    epochs = 50  # 固定的训练轮数

    # 初始化并训练最佳模型
    best_model = WiFiTransformerAutoencoder(
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    best_model = train_autoencoder(
        best_model, X_train, X_val,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=patience
    )

    # 提取特征
    X_train_features = extract_features(best_model, X_train, device=device, batch_size=batch_size)
    X_test_features = extract_features(best_model, X_test, device=device, batch_size=batch_size)

    # 逆标准化目标变量
    y_train_original = scaler_y.inverse_transform(y_train)
    y_test_original = scaler_y.inverse_transform(y_test)

    # 定义最佳 SVR 参数
    svr_params = {
        'kernel': 'rbf',
        'C': svr_C,
        'epsilon': svr_epsilon
    }

    # 训练最佳 SVR 模型
    best_svr_model = train_and_evaluate_svr(
        X_train_features, y_train_original,
        X_test_features, y_test_original,
        svr_params=svr_params
    )

    # === 保存模型和参数 ===
    # 保存最佳 Transformer 模型
    torch.save(best_model.state_dict(), 'best_transformer_autoencoder_optuna.pth')
    print("Transformer 自编码器模型已保存到 best_transformer_autoencoder_optuna.pth")

    # 保存最佳 SVR 模型
    joblib.dump(best_svr_model, 'best_svr_model_optuna.pkl')
    print("SVR 模型已保存到 best_svr_model_optuna.pkl")

    # 保存最佳参数
    with open('best_hyperparameters_optuna.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False)
    print("最佳参数已保存到 best_hyperparameters_optuna.json")

    # 保存所有试验的详细信息
    all_trials = []
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'params': trial.params,
            'value': trial.value,
            'state': trial.state.name,
            'duration': trial.duration.total_seconds() if trial.duration else None
        }
        all_trials.append(trial_info)

    with open('all_trials_optuna.json', 'w', encoding='utf-8') as f:
        json.dump(all_trials, f, indent=4, ensure_ascii=False)
    print("所有试验的详细信息已保存到 all_trials_optuna.json。")

if __name__ == '__main__':
    main()
