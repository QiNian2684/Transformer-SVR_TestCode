# hyperparameter_tuning_optuna.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
    train_autoencoder,
    extract_features,
    train_and_evaluate_models,
    compute_error_distances,
    NaNLossError
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
    train_path = 'UJIndoorLoc/trainingData_building2.csv'
    test_path = 'UJIndoorLoc/validationData_building2.csv'

    # 固定训练参数
    epochs = 10  # 训练轮数
    n_trials = 300  # Optuna 试验次数，根据计算资源调整

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, y_train_coords, y_train_floor, X_val, y_val_coords, y_val_floor, X_test, y_test_coords, y_test_floor, scaler_X, scaler_y, label_encoder = load_and_preprocess_data(train_path, test_path)

    # === 定义优化目标函数 ===
    def objective(trial):
        try:
            # Transformer 自编码器超参数

            # 1. 选择 model_dim
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64])

            # 2. 选择 num_heads
            num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16])

            # 检查 num_heads 是否合理
            if num_heads > model_dim:
                # num_heads 不应大于 model_dim，否则会导致模型无法正确运行
                raise TrialPruned()

            # 其余超参数
            num_layers = trial.suggest_categorical('num_layers', [2, 4, 8, 16])
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
            patience = trial.suggest_int('early_stopping_patience', 5, 10)

            # SVR 超参数
            svr_kernel = trial.suggest_categorical('svr_kernel', ['poly', 'rbf', 'sigmoid'])
            svr_C = trial.suggest_float('svr_C', 1e-1, 1e2, log=True)
            svr_epsilon = trial.suggest_float('svr_epsilon', 0.0, 1.0)
            svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])

            # 如果 kernel 是 'poly'，则调优 degree 和 coef0
            if svr_kernel == 'poly':
                svr_degree = trial.suggest_int('svr_degree', 2, 5)
                svr_coef0 = trial.suggest_float('svr_coef0', 0.0, 1.0)
            else:
                svr_degree = 3  # 默认值
                svr_coef0 = 0.0  # 默认值

            # 收集当前超参数组合
            current_params = {
                'model_dim': model_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'early_stopping_patience': patience,
                'svr_kernel': svr_kernel,
                'svr_C': svr_C,
                'svr_epsilon': svr_epsilon,
                'svr_gamma': svr_gamma,
                'svr_degree': svr_degree,
                'svr_coef0': svr_coef0,
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

            # 检查提取的特征中是否存在 NaN
            if np.isnan(X_train_features).any() or np.isnan(X_test_features).any():
                print("提取的特征中包含 NaN，试验将被剪枝。")
                raise TrialPruned()

            # 逆标准化坐标目标变量
            y_train_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_train_coords[:, 0].reshape(-1, 1))
            y_train_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_train_coords[:, 1].reshape(-1, 1))
            y_train_coords_original = np.hstack((y_train_longitude_original, y_train_latitude_original))

            y_test_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_test_coords[:, 0].reshape(-1, 1))
            y_test_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_test_coords[:, 1].reshape(-1, 1))
            y_test_coords_original = np.hstack((y_test_longitude_original, y_test_latitude_original))

            # 定义 SVR 参数
            svr_params_dict = {
                'kernel': svr_kernel,
                'C': svr_C,
                'epsilon': svr_epsilon,
                'gamma': svr_gamma,
                'degree': svr_degree,
                'coef0': svr_coef0,
            }

            # 将 SVR 参数添加到 training_params 中
            training_params = current_params.copy()  # 使用当前参数作为训练参数

            # 训练模型
            regression_model, classification_model = train_and_evaluate_models(
                X_train_features, y_train_coords_original, y_train_floor,
                X_test_features, y_test_coords_original, y_test_floor,
                svr_params=svr_params_dict,
                training_params=training_params  # 传递训练参数
            )

            # 预测并评估
            y_pred_coords = regression_model.predict(X_test_features)
            error_distances = compute_error_distances(y_test_coords_original, y_pred_coords)
            mean_error_distance = np.mean(error_distances)

            # 报告中间结果并检查是否应该剪枝
            trial.report(mean_error_distance, step=0)

            if trial.should_prune():
                print("试验被剪枝。")
                raise TrialPruned()

            # 返回要最小化的目标值
            return mean_error_distance

        except NaNLossError:
            print("试验因 NaN 损失而被剪枝。")
            raise TrialPruned()
        except ValueError as e:
            if 'NaN' in str(e):
                print("试验因数据中存在 NaN 而被剪枝。")
                raise TrialPruned()
            else:
                raise e

    # === 创建和运行优化研究 ===
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

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
    svr_kernel = best_params['svr_kernel']
    svr_C = best_params['svr_C']
    svr_epsilon = best_params['svr_epsilon']
    svr_gamma = best_params['svr_gamma']
    svr_degree = best_params.get('svr_degree', 3)  # 如果不是poly核，默认degree为3
    svr_coef0 = best_params.get('svr_coef0', 0.0)  # 如果不是poly核，默认coef0为0.0

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

    # 检查提取的特征中是否存在 NaN
    if np.isnan(X_train_features).any() or np.isnan(X_test_features).any():
        print("提取的特征中包含 NaN，无法训练最佳模型。")
        exit(1)

    # 逆标准化坐标目标变量
    y_train_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_train_coords[:, 0].reshape(-1, 1))
    y_train_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_train_coords[:, 1].reshape(-1, 1))
    y_train_coords_original = np.hstack((y_train_longitude_original, y_train_latitude_original))

    y_test_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_test_coords[:, 0].reshape(-1, 1))
    y_test_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_test_coords[:, 1].reshape(-1, 1))
    y_test_coords_original = np.hstack((y_test_longitude_original, y_test_latitude_original))

    # 定义最佳 SVR 参数
    svr_params_best = {
        'kernel': svr_kernel,
        'C': svr_C,
        'epsilon': svr_epsilon,
        'gamma': svr_gamma,
        'degree': svr_degree,
        'coef0': svr_coef0,
    }

    # 将最佳参数组合成字典
    training_params = best_params.copy()  # 使用最佳参数作为训练参数

    # 训练最佳模型
    best_regression_model, best_classification_model = train_and_evaluate_models(
        X_train_features, y_train_coords_original, y_train_floor,
        X_test_features, y_test_coords_original, y_test_floor,
        svr_params=svr_params_best,
        training_params=training_params  # 传递训练参数
    )

    # === 保存模型和参数 ===
    # 保存最佳 Transformer 模型
    torch.save(best_model.state_dict(), 'best_transformer_autoencoder_optuna.pth')
    print("Transformer 自编码器模型已保存到 best_transformer_autoencoder_optuna.pth")

    # 保存最佳回归模型
    joblib.dump(best_regression_model, 'best_coordinate_regression_model_optuna.pkl')
    print("坐标回归模型已保存到 best_coordinate_regression_model_optuna.pkl")

    # 保存最佳分类模型
    joblib.dump(best_classification_model, 'best_floor_classification_model_optuna.pkl')
    print("楼层分类模型已保存到 best_floor_classification_model_optuna.pkl")

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
