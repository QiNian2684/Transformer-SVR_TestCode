# combined_hyperparameter_tuning.py

import os
import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from TE_regression import (
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
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # 通用数据路径与设置
    train_path = 'UJIndoorLoc/trainingData.csv'
    test_path = 'UJIndoorLoc/validationData.csv'
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ========= 阶段1：Transformer调优 =========
    epochs = 75
    n_trials_transformer = 100  # 可根据需求调整
    transformer_results_dir = 'transformer_tuning_results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    transformer_run_dir = os.path.join(transformer_results_dir, timestamp)
    os.makedirs(transformer_run_dir, exist_ok=True)
    print(f"Transformer调优结果将保存到: {transformer_run_dir}")

    print("加载并预处理数据...")
    X_train, y_train_coords, _, X_val, y_val_coords, _, X_test, y_test_coords, _, scaler_X, scaler_y, _, filtered_test_indices = load_and_preprocess_data(train_path, test_path)

    best_val_loss = float('inf')
    best_transformer_params_path = os.path.join(transformer_run_dir, 'best_hyperparameters_transformer.json')
    best_transformer_model_path = os.path.join(transformer_run_dir, 'best_transformer_model.pt')

    def transformer_objective(trial):
        nonlocal best_val_loss

        try:
            # Transformer 参数调优
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64])
            num_heads_options = [h for h in [2, 4, 8, 16] if model_dim % h == 0]
            if not num_heads_options:
                raise TrialPruned("model_dim 不可被任何 num_heads 整除。")
            num_heads = trial.suggest_categorical('num_heads', num_heads_options)
            num_layers = trial.suggest_int('num_layers', low=4, high=64)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.03, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64, 128])
            patience = trial.suggest_int('early_stopping_patience', 5, 5)
            min_delta_ratio = trial.suggest_float('min_delta_ratio', 0.003, 0.003)

            current_params = {
                'model_dim': model_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'early_stopping_patience': patience,
                'min_delta_ratio': min_delta_ratio
            }

            print(f"\n当前Transformer超参数组合:\n{json.dumps(current_params, indent=4, ensure_ascii=False)}")

            model = WiFiTransformerAutoencoder(
                input_dim=X_train.shape[1],
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)

            model, train_loss_list, val_loss_list = train_autoencoder(
                model, X_train, X_val,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stopping_patience=patience,
                min_delta_ratio=min_delta_ratio
            )

            final_val_loss = val_loss_list[-1]

            if np.isnan(final_val_loss):
                raise TrialPruned("验证集Loss为NaN，剪枝试验。")

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                # 保存最佳Transformer模型及参数
                torch.save(model.state_dict(), best_transformer_model_path)
                with open(best_transformer_params_path, 'w', encoding='utf-8') as f:
                    json.dump(current_params, f, indent=4, ensure_ascii=False)
                print(f"最佳Transformer已保存到 {best_transformer_model_path}")
                print(f"最佳Transformer超参数已保存到 {best_transformer_params_path}")

            return final_val_loss

        except NaNLossError:
            raise TrialPruned()
        except ValueError as e:
            if 'NaN' in str(e):
                raise TrialPruned()
            else:
                raise e

    # 开始Transformer调优
    transformer_study = optuna.create_study(direction='minimize')
    transformer_study.optimize(transformer_objective, n_trials=n_trials_transformer, n_jobs=1)

    print("\n=== 最佳Transformer参数 ===")
    best_transformer_trial = transformer_study.best_trial
    print(f"最小验证Loss: {best_transformer_trial.value}")
    print("最佳Transformer超参数:")
    print(json.dumps(best_transformer_trial.params, indent=4, ensure_ascii=False))

    # ========= 阶段2：SVR调优 =========
    # 使用已确定的最佳Transformer模型进行特征提取，然后对SVR进行调优
    with open(best_transformer_params_path, 'r', encoding='utf-8') as f:
        transformer_params = json.load(f)

    # 加载最佳Transformer模型
    model = WiFiTransformerAutoencoder(
        input_dim=X_train.shape[1],
        model_dim=transformer_params['model_dim'],
        num_heads=transformer_params['num_heads'],
        num_layers=transformer_params['num_layers'],
        dropout=transformer_params['dropout']
    ).to(device)

    model.load_state_dict(torch.load(best_transformer_model_path, map_location=device))
    model.eval()
    print("已加载最佳Transformer模型用于特征提取。")

    batch_size = transformer_params['batch_size']
    X_train_features = extract_features(model, X_train, device=device, batch_size=batch_size)
    X_test_features = extract_features(model, X_test, device=device, batch_size=batch_size)

    y_train_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_train_coords[:, 0].reshape(-1, 1))
    y_train_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_train_coords[:, 1].reshape(-1, 1))
    y_train_coords_original = np.hstack((y_train_longitude_original, y_train_latitude_original))

    y_test_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_test_coords[:, 0].reshape(-1, 1))
    y_test_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_test_coords[:, 1].reshape(-1, 1))
    y_test_coords_original = np.hstack((y_test_longitude_original, y_test_latitude_original))

    # 创建SVR调优结果目录
    svr_results_dir = 'svr_tuning_results'
    svr_run_dir = os.path.join(svr_results_dir, timestamp)  # 与transformer统一时间戳
    os.makedirs(svr_run_dir, exist_ok=True)
    print(f"SVR调参结果将保存到: {svr_run_dir}")

    model_dir = os.path.join(svr_run_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    csv_file_path_regression = os.path.join(svr_run_dir, 'regression_results.csv')
    error_csv_path = os.path.join(svr_run_dir, 'error.csv')

    best_mean_error_distance = float('inf')
    best_regression_model = None
    best_svr_params_path = os.path.join(svr_run_dir, 'best_hyperparameters_svr.json')
    n_trials_svr = 100  # 根据需要调整SVR的试验次数

    def svr_objective(trial):
        nonlocal best_mean_error_distance
        nonlocal best_regression_model

        try:
            # SVR 参数调优
            svr_kernel = trial.suggest_categorical('svr_kernel', ['rbf'])
            svr_C = trial.suggest_float('svr_C', 100, 500, log=True)
            svr_epsilon = trial.suggest_float('svr_epsilon', 0.01, 3.0)
            svr_gamma = trial.suggest_categorical('svr_gamma', ['scale'])

            if svr_kernel == 'poly':
                svr_degree = trial.suggest_int('svr_degree', 2, 5)
                svr_coef0 = trial.suggest_float('svr_coef0', 0.0, 1.0)
            else:
                svr_degree = 3
                svr_coef0 = 0.0

            current_params = {
                'svr_kernel': svr_kernel,
                'svr_C': svr_C,
                'svr_epsilon': svr_epsilon,
                'svr_gamma': svr_gamma,
                'svr_degree': svr_degree,
                'svr_coef0': svr_coef0
            }

            print(f"\n当前SVR超参数组合:\n{json.dumps(current_params, indent=4, ensure_ascii=False)}")

            test_data = pd.read_csv(test_path)
            test_data = test_data.iloc[filtered_test_indices].reset_index(drop=True)

            # 不再训练Transformer，因此传空列表给train_loss_list和val_loss_list
            train_loss_list, val_loss_list = [], []

            regression_model, mean_error_distance, error_distances, y_pred_coords = train_and_evaluate_regression_model(
                X_train_features, y_train_coords_original,
                X_test_features, y_test_coords_original,
                svr_params=current_params,
                training_params=current_params,
                train_loss_list=train_loss_list,
                val_loss_list=val_loss_list,
                output_dir=svr_run_dir,
                image_index=trial.number + 1,
                csv_file_path=csv_file_path_regression
            )

            if mean_error_distance < best_mean_error_distance:
                best_mean_error_distance = mean_error_distance
                best_regression_model = regression_model
                best_model_path = os.path.join(model_dir, 'best_svr_model.pkl')
                joblib.dump(best_regression_model, best_model_path)
                print(f"最佳SVR模型已保存到 {best_model_path}。")

                with open(best_svr_params_path, 'w', encoding='utf-8') as f:
                    json.dump(current_params, f, indent=4, ensure_ascii=False)
                print(f"最佳SVR超参数已保存到 {best_svr_params_path}。")

                try:
                    best_image_index = trial.number + 1
                    best_image_name = f"{best_image_index:04d}_regression.png"
                    best_image_path = os.path.join(svr_run_dir, best_image_name)
                    destination_image_path = os.path.join(svr_run_dir, "0000.png")
                    shutil.copyfile(best_image_path, destination_image_path)
                    print(f"最佳试验的结果图片已更新为 {destination_image_path}")
                except Exception as e:
                    print(f"无法更新最佳试验的图片：{e}")

                error_threshold = 15.0
                indices_high_error = np.where(error_distances > error_threshold)[0]
                if len(indices_high_error) > 0:
                    print(f"发现 {len(indices_high_error)} 个误差超过 {error_threshold} 米的样本。")
                    high_error_data = test_data.iloc[indices_high_error].copy()
                    high_error_data['Predicted_LONGITUDE'] = y_pred_coords[indices_high_error, 0]
                    high_error_data['Predicted_LATITUDE'] = y_pred_coords[indices_high_error, 1]
                    high_error_data['Error_Distance'] = error_distances[indices_high_error]
                    high_error_data.to_csv(error_csv_path, index=False)
                    print(f"高误差样本已保存到 {error_csv_path}")
                else:
                    print(f"没有误差超过 {error_threshold} 米的样本。")

            return mean_error_distance

        except NaNLossError:
            raise TrialPruned()
        except ValueError as e:
            if 'NaN' in str(e):
                raise TrialPruned()
            else:
                raise e

    # 开始SVR调优
    svr_study = optuna.create_study(direction='minimize')
    svr_study.optimize(svr_objective, n_trials=n_trials_svr, n_jobs=1)

    print("\n=== 最佳SVR参数 ===")
    best_svr_trial = svr_study.best_trial
    print(f"平均误差距离: {best_svr_trial.value:.2f} 米")
    print("最佳SVR超参数:")
    print(json.dumps(best_svr_trial.params, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()
