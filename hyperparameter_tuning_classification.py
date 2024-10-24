# hyperparameter_tuning_classification.py

import os
import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
    train_autoencoder,
    extract_features,
    train_and_evaluate_classification_model,
    NaNLossError
)
import optuna
from optuna.exceptions import TrialPruned
import joblib
import json
import random
from datetime import datetime
import shutil

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():

    # 固定训练参数
    epochs = 5  # 训练轮数
    n_trials = 3  # Optuna 试验次数，根据计算资源调整

    # 设置随机种子以确保可重复性
    set_seed()

    # === 参数设置 ===
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    train_path = 'UJIndoorLoc/trainingData_building0.csv'
    test_path = 'UJIndoorLoc/validationData_building0.csv'

    # === 创建结果保存目录 ===
    results_dir = 'results'
    classification_results_dir = os.path.join(results_dir, 'classification')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = os.path.join(classification_results_dir, timestamp)
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"结果将保存到: {current_run_dir}")

    # === 定义模型保存目录 ===
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)  # 创建目录，如果已存在则不操作

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, _, y_train_floor, X_val, _, y_val_floor, X_test, _, y_test_floor, scaler_X, _, label_encoder = load_and_preprocess_data(train_path, test_path)

    # 初始化图片编号
    image_index = 1

    # === 定义优化目标函数 ===
    def objective(trial):
        nonlocal image_index  # 引入外部变量

        try:
            # Transformer 自编码器超参数
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64, 128])
            num_heads_options = [h for h in [2, 4, 8, 16] if model_dim % h == 0]
            if not num_heads_options:
                raise TrialPruned("model_dim 不可被任何 num_heads 整除。")
            num_heads = trial.suggest_categorical('num_heads', num_heads_options)
            num_layers = trial.suggest_int('num_layers', low=4, high=64, log=True)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
            patience = trial.suggest_int('early_stopping_patience', 3, 15)

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
            model, train_loss_list, val_loss_list = train_autoencoder(
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
            svc_params = {
                'C': svc_C,
                'kernel': svc_kernel,
                'gamma': svc_gamma,
                'degree': svc_degree,
                'coef0': svc_coef0,
            }

            # 训练并评估分类模型，包含可视化和打印输出
            classification_model, accuracy = train_and_evaluate_classification_model(
                X_train_features, y_train_floor,
                X_test_features, y_test_floor,
                svc_params=svc_params,
                training_params=current_params,
                train_loss_list=train_loss_list,
                val_loss_list=val_loss_list,
                label_encoder=label_encoder,
                output_dir=current_run_dir,
                image_index=trial.number + 1  # 使用 trial.number + 1 作为图片编号
            )

            # 保存模型到指定目录
            model_path = os.path.join(model_dir, f'classification_model_trial_{trial.number}.pkl')
            joblib.dump(classification_model, model_path)
            print(f"分类模型已保存到 {model_path}。")

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
    best_params_path = os.path.join(current_run_dir, 'best_hyperparameters_classification.json')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        json.dump(best_trial.params, f, indent=4, ensure_ascii=False)
    print(f"最佳超参数已保存到 {best_params_path}。")

    # === 保存最佳试验的结果图片为“0000.png” ===
    try:
        best_trial_number = best_trial.number
        best_image_index = best_trial_number + 1  # 与保存图片时的编号对应
        best_image_name = f"{best_image_index:04d}.png"
        best_image_path = os.path.join(current_run_dir, best_image_name)
        destination_image_path = os.path.join(current_run_dir, "0000.png")
        # 复制最佳试验的图片并重命名为“0000.png”
        shutil.copyfile(best_image_path, destination_image_path)
        print(f"最佳试验的结果图片已保存为 {destination_image_path}")
    except Exception as e:
        print(f"无法保存最佳试验的图片：{e}")

    # === 将最佳模型另存为指定文件名 ===
    best_model_path = os.path.join(model_dir, 'best_classification_model.pkl')
    trial_model_path = os.path.join(model_dir, f'classification_model_trial_{best_trial.number}.pkl')
    if os.path.exists(trial_model_path):
        shutil.copyfile(trial_model_path, best_model_path)
        print(f"最佳分类模型已保存为 {best_model_path}")
    else:
        print("未找到最佳分类模型，无法复制。")

if __name__ == '__main__':
    main()
