# hyperparameter_tuning_classification.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
    train_autoencoder,
    extract_features,
    NaNLossError
)
import optuna
from optuna.exceptions import TrialPruned
import joblib
import json
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 设置随机种子以确保可重复性
    set_seed()

    # === 参数设置 ===
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    train_path = 'UJIndoorLoc/trainingData_building0.csv'
    test_path = 'UJIndoorLoc/validationData_building0.csv'

    # 固定训练参数
    epochs = 150  # 训练轮数
    n_trials = 100  # Optuna 试验次数，根据计算资源调整

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, _, y_train_floor, X_val, _, y_val_floor, X_test, _, y_test_floor, scaler_X, _, label_encoder = load_and_preprocess_data(train_path, test_path)

    # === 定义优化目标函数 ===
    def objective(trial):
        try:
            # Transformer 自编码器超参数
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64, 128])
            num_heads_options = [h for h in [2, 4, 8, 16] if model_dim % h == 0]
            if not num_heads_options:
                raise TrialPruned("model_dim 不可被任何 num_heads 整除。")
            num_heads = trial.suggest_categorical('num_heads', num_heads_options)
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
            patience = trial.suggest_int('early_stopping_patience', 5, 10)

            # SVC 超参数
            svc_C = trial.suggest_float('svc_C', 1e-1, 1e2, log=True)
            svc_kernel = trial.suggest_categorical('svc_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            svc_gamma = trial.suggest_categorical('svc_gamma', ['scale', 'auto'])
            if svc_kernel == 'poly':
                svc_degree = trial.suggest_int('svc_degree', 2, 5)
                svc_coef0 = trial.suggest_float('svc_coef0', 0.0, 1.0)
            else:
                svc_degree = 3  # 默认值
                svc_coef0 = 0.0  # 默认值

            # 收集当前超参数组合
            current_params = {
                'model_dim': model_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'early_stopping_patience': patience,
                'svc_C': svc_C,
                'svc_kernel': svc_kernel,
                'svc_gamma': svc_gamma,
                'svc_degree': svc_degree,
                'svc_coef0': svc_coef0,
            }

            # 打印当前超参数组合
            print(f"\n当前超参数组合:\n{json.dumps(current_params, indent=4, ensure_ascii=False)}")

            # 初始化 Transformer 自编码器模型
            model = WiFiTransformerAutoencoder(
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)

            # 训练自编码器
            model, _, _ = train_autoencoder(
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

            # 定义 SVC 参数
            from sklearn.svm import SVC

            svc_params = {
                'C': svc_C,
                'kernel': svc_kernel,
                'gamma': svc_gamma,
                'degree': svc_degree,
                'coef0': svc_coef0,
            }

            # 训练 SVC 模型
            classification_model = SVC(**svc_params)
            classification_model.fit(X_train_features, y_train_floor)

            # 预测并评估
            y_pred_floor = classification_model.predict(X_test_features)
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test_floor, y_pred_floor)

            # 返回准确率作为优化目标
            return accuracy

        except NaNLossError:
            raise TrialPruned()
        except ValueError as e:
            if 'NaN' in str(e):
                raise TrialPruned()
            else:
                raise e

    # === 创建并运行优化研究 ===
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # === 打印并保存最佳结果 ===
    print("\n=== 最佳参数 ===")
    best_trial = study.best_trial
    print(f"分类准确率: {best_trial.value:.4f}")
    print("最佳超参数:")
    print(json.dumps(best_trial.params, indent=4, ensure_ascii=False))

    # 保存最佳超参数
    with open('best_hyperparameters_classification.json', 'w', encoding='utf-8') as f:
        json.dump(best_trial.params, f, indent=4, ensure_ascii=False)
    print("最佳超参数已保存到 best_hyperparameters_classification.json。")

if __name__ == '__main__':
    main()
