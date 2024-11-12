# hyperparameter_tuning_regression.py

import os
import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation_regression import (
    train_autoencoder,
    extract_features,
    train_and_evaluate_regression_model,
    NaNLossError
)
import optuna
from optuna.exceptions import TrialPruned
import joblib
import json
import random
from datetime import datetime
import shutil
import pandas as pd  # 确保导入 pandas 库

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():

    # 固定训练参数
    epochs = 30  # 训练轮数
    n_trials = 600  # Optuna 试验次数，根据计算资源调整

    # 设置随机种子以确保可重复性
    set_seed()

    # === 参数设置 ===
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    train_path = 'UJIndoorLoc/trainingData.csv'
    test_path = 'UJIndoorLoc/validationData.csv'

    # === 创建结果保存目录 ===
    results_dir = 'results'
    regression_results_dir = os.path.join(results_dir, 'regression')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = os.path.join(regression_results_dir, timestamp)
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"结果将保存到: {current_run_dir}")

    # === 定义模型保存目录 ===
    model_dir = 'best_models_pkl'
    os.makedirs(model_dir, exist_ok=True)  # 创建目录，如果已存在则不操作

    # 定义 CSV 文件路径，将其保存在 current_run_dir
    csv_file_path_regression = os.path.join(current_run_dir, 'regression_results.csv')

    # 定义 error.csv 文件路径
    error_csv_path = os.path.join(current_run_dir, 'error.csv')

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, y_train_coords, _, X_val, y_val_coords, _, X_test, y_test_coords, _, scaler_X, scaler_y, _ = load_and_preprocess_data(train_path, test_path)

    # 保存原始的测试集数据，用于记录到 error.csv
    test_data = pd.read_csv(test_path)

    # 用于保存最佳模型
    best_mean_error_distance = float('inf')
    best_regression_model = None

    # 定义最佳超参数文件路径
    best_params_path = os.path.join(current_run_dir, 'best_hyperparameters_regression.json')

    # === 定义优化目标函数 ===
    def objective(trial):
        nonlocal best_mean_error_distance
        nonlocal best_regression_model

        try:
            # Transformer 自编码器超参数

            # model_dim: Transformer模型的维度。这个参数决定了模型的大小和复杂度。
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64, 128])

            # num_heads_options: 根据model_dim的可整除性选择的头数选项，保证model_dim可以被num_heads整除。
            num_heads_options = [h for h in [2, 4, 8, 16] if model_dim % h == 0]

            # 如果没有有效的头数选项，则终止当前试验。
            if not num_heads_options:
                raise TrialPruned("model_dim 不可被任何 num_heads 整除。")

            # num_heads: 选择一个有效的头数，这影响到模型的并行处理能力。
            num_heads = trial.suggest_categorical('num_heads', num_heads_options)

            # num_layers: Transformer模型的层数。层数越多，模型通常越能捕获复杂的特征，但计算成本也越高。
            num_layers = trial.suggest_int('num_layers', low=4, high=64)

            # dropout: 在模型训练时随机丢弃节点的比例，用以防止过拟合。
            dropout = trial.suggest_float('dropout', 0.0, 0.5)

            # learning_rate: 学习率，使用对数标度选择，对模型训练速度和效果有重要影响。
            learning_rate = trial.suggest_float('learning_rate', 0.0005, 0.01, log=True)

            # batch_size: 批大小，可选值为[16, 32, 48, 64, 128, 256]，影响模型的内存需求和训练速度。
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64, 128, 256])

            # patience: 早停机制的耐心值，当验证损失在连续多个epoch内未改善时停止训练，范围从3到7。
            patience = trial.suggest_int('early_stopping_patience', 3, 7)

            # SVR 超参数

            # svr_kernel: SVR模型的核函数类型，可选['linear', 'poly', 'rbf', 'sigmoid']，影响模型处理数据的方式。
            svr_kernel = trial.suggest_categorical('svr_kernel', ['poly', 'rbf'])

            # svr_C: 正则化参数C，使用对数标度从1到200选择，C值越大，模型越复杂，容错率越低。
            svr_C = trial.suggest_float('svr_C', 100, 500, log=True)

            # svr_epsilon: SVR模型的epsilon，定义了不惩罚预测误差在此值内的观测，范围从0.01到3.0。
            svr_epsilon = trial.suggest_float('svr_epsilon', 0.01, 3.0)

            # svr_gamma: SVR核函数的gamma参数，可选['scale', 'auto']，影响核函数的形状和数据的映射。
            svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])

            # 如果核函数是多项式（poly），则需要选择多项式的度数和coef0参数。
            if svr_kernel == 'poly':
                svr_degree = trial.suggest_int('svr_degree', 2, 5)  # 多项式的度数
                svr_coef0 = trial.suggest_float('svr_coef0', 0.0, 1.0)  # 多项式核函数中的独立项系数
            else:
                svr_degree = 3  # 默认值为3
                svr_coef0 = 0.0  # 默认coef0为0.0

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

            # 逆标准化坐标目标变量
            y_train_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_train_coords[:, 0].reshape(-1, 1))
            y_train_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_train_coords[:, 1].reshape(-1, 1))
            y_train_coords_original = np.hstack((y_train_longitude_original, y_train_latitude_original))

            y_test_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_test_coords[:, 0].reshape(-1, 1))
            y_test_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_test_coords[:, 1].reshape(-1, 1))
            y_test_coords_original = np.hstack((y_test_longitude_original, y_test_latitude_original))

            # 定义 SVR 参数
            svr_params = {
                'kernel': svr_kernel,
                'C': svr_C,
                'epsilon': svr_epsilon,
                'gamma': svr_gamma,
                'degree': svr_degree,
                'coef0': svr_coef0,
            }

            # 训练并评估回归模型，包含可视化和打印输出
            regression_model, mean_error_distance, error_distances, y_pred_coords = train_and_evaluate_regression_model(
                X_train_features, y_train_coords_original,
                X_test_features, y_test_coords_original,
                svr_params=svr_params,
                training_params=current_params,
                train_loss_list=train_loss_list,
                val_loss_list=val_loss_list,
                output_dir=current_run_dir,
                image_index=trial.number + 1,  # 使用 trial.number + 1 作为图片编号
                csv_file_path=csv_file_path_regression  # 添加此行，指定CSV文件路径
            )

            # 如果当前模型表现更好，保存模型和超参数，并更新最佳图片
            if mean_error_distance < best_mean_error_distance:
                best_mean_error_distance = mean_error_distance
                best_regression_model = regression_model
                # 保存最佳模型
                best_model_path = os.path.join(model_dir, 'best_regression_model.pkl')
                joblib.dump(best_regression_model, best_model_path)
                print(f"最佳回归模型已保存到 {best_model_path}。")
                # 保存最佳超参数
                with open(best_params_path, 'w', encoding='utf-8') as f:
                    json.dump(current_params, f, indent=4, ensure_ascii=False)
                print(f"最佳超参数已保存到 {best_params_path}。")
                # 更新最佳试验的结果图片为“0000.png”
                try:
                    best_image_index = trial.number + 1  # 与保存图片时的编号对应
                    best_image_name = f"{best_image_index:04d}_regression.png"
                    best_image_path = os.path.join(current_run_dir, best_image_name)
                    destination_image_path = os.path.join(current_run_dir, "0000.png")
                    # 复制最佳试验的图片并重命名为“0000.png”
                    shutil.copyfile(best_image_path, destination_image_path)
                    print(f"最佳试验的结果图片已更新为 {destination_image_path}")
                except Exception as e:
                    print(f"无法更新最佳试验的图片：{e}")

                # 保存误差超过20米的样本到 error.csv
                error_threshold = 25.0  # 误差阈值（米）
                indices_high_error = np.where(error_distances > error_threshold)[0]
                if len(indices_high_error) > 0:
                    print(f"发现 {len(indices_high_error)} 个误差超过 {error_threshold} 米的样本。")
                    high_error_data = test_data.iloc[indices_high_error].copy()
                    # 添加预测的坐标和误差距离
                    high_error_data['Predicted_LONGITUDE'] = y_pred_coords[indices_high_error, 0]
                    high_error_data['Predicted_LATITUDE'] = y_pred_coords[indices_high_error, 1]
                    high_error_data['Error_Distance'] = error_distances[indices_high_error]
                    high_error_data.to_csv(error_csv_path, index=False)
                    print(f"高误差样本已保存到 {error_csv_path}")
                else:
                    print(f"没有误差超过 {error_threshold} 米的样本。")
            else:
                print("当前模型未超过最佳模型，不更新 error.csv。")

            # 返回平均误差距离作为优化目标
            return mean_error_distance

        except NaNLossError:
            raise TrialPruned()
        except ValueError as e:
            if 'NaN' in str(e):
                raise TrialPruned()
            else:
                raise e

    # === 创建并运行优化研究 ===
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # === 打印并保存最佳结果 ===
    print("\n=== 最佳参数 ===")
    best_trial = study.best_trial
    print(f"平均误差距离: {best_trial.value:.2f} 米")
    print("最佳超参数:")
    print(json.dumps(best_trial.params, indent=4, ensure_ascii=False))

    # 已在每次试验中实时保存最佳模型、超参数和图片，因此这里无需再次保存

if __name__ == '__main__':
    main()
