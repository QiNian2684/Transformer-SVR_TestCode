# hyperparameter_tuning_regression.py

import os
import torch
import numpy as np
from data_preprocessing_no_kpca import load_and_preprocess_data
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
    # 固定训练参数
    epochs = 75
    n_trials = 600

    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_path = 'UJIndoorLoc/trainingData.csv'
    test_path = 'UJIndoorLoc/validationData.csv'

    results_dir = 'results'
    regression_results_dir = os.path.join(results_dir, 'regression')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = os.path.join(regression_results_dir, timestamp)
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"结果将保存到: {current_run_dir}")

    model_dir = 'best_models_pkl'
    os.makedirs(model_dir, exist_ok=True)

    csv_file_path_regression = os.path.join(current_run_dir, 'regression_results.csv')
    error_csv_path = os.path.join(current_run_dir, 'error.csv')

    print("加载并预处理数据...")
    X_train, y_train_coords, _, X_val, y_val_coords, _, X_test, y_test_coords, _, scaler_X, scaler_y, _, filtered_test_indices = load_and_preprocess_data(train_path, test_path)

    test_data = pd.read_csv(test_path)
    test_data = test_data.iloc[filtered_test_indices].reset_index(drop=True)

    best_mean_error_distance = float('inf')
    best_regression_model = None
    best_params_path = os.path.join(current_run_dir, 'best_hyperparameters_regression.json')

    def objective(trial):
        nonlocal best_mean_error_distance
        nonlocal best_regression_model

        try:
            # Transformer 参数
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64, 128])
            num_heads_options = [h for h in [2, 4, 8, 16] if model_dim % h == 0]
            if not num_heads_options:
                raise TrialPruned("model_dim 不可被任何 num_heads 整除。")
            num_heads = trial.suggest_categorical('num_heads', num_heads_options)
            num_layers = trial.suggest_int('num_layers', low=4, high=64)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.0005, 0.1, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64, 128])
            patience = trial.suggest_int('early_stopping_patience', 3, 7)
            min_delta_ratio = trial.suggest_float('min_delta_ratio', 0.001, 0.003)

            # SVR 参数
            svr_kernel = trial.suggest_categorical('svr_kernel', ['rbf', 'poly'])
            svr_C = trial.suggest_float('svr_C', 200, 500, log=True)
            svr_epsilon = trial.suggest_float('svr_epsilon', 0.01, 3.0)
            svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])
            if svr_kernel == 'poly':
                svr_degree = trial.suggest_int('svr_degree', 2, 5)
                svr_coef0 = trial.suggest_float('svr_coef0', 0.0, 1.0)
            else:
                svr_degree = 3
                svr_coef0 = 0.0

            current_params = {
                'model_dim': model_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'early_stopping_patience': patience,
                'min_delta_ratio': min_delta_ratio,  # 将min_delta_ratio加入参数中
                'svr_kernel': svr_kernel,
                'svr_C': svr_C,
                'svr_epsilon': svr_epsilon,
                'svr_gamma': svr_gamma,
                'svr_degree': svr_degree,
                'svr_coef0': svr_coef0,
            }

            print(f"\n当前超参数组合:\n{json.dumps(current_params, indent=4, ensure_ascii=False)}")

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
                min_delta_ratio=min_delta_ratio  # 在训练中使用该参数
            )

            X_train_features = extract_features(model, X_train, device=device, batch_size=batch_size)
            X_test_features = extract_features(model, X_test, device=device, batch_size=batch_size)

            if np.isnan(X_train_features).any() or np.isnan(X_test_features).any():
                print("提取的特征中包含 NaN，试验将被剪枝。")
                raise TrialPruned()

            y_train_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_train_coords[:, 0].reshape(-1, 1))
            y_train_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_train_coords[:, 1].reshape(-1, 1))
            y_train_coords_original = np.hstack((y_train_longitude_original, y_train_latitude_original))

            y_test_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_test_coords[:, 0].reshape(-1, 1))
            y_test_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_test_coords[:, 1].reshape(-1, 1))
            y_test_coords_original = np.hstack((y_test_longitude_original, y_test_latitude_original))

            svr_params = {
                'kernel': svr_kernel,
                'C': svr_C,
                'epsilon': svr_epsilon,
                'gamma': svr_gamma,
                'degree': svr_degree,
                'coef0': svr_coef0,
            }

            regression_model, mean_error_distance, error_distances, y_pred_coords = train_and_evaluate_regression_model(
                X_train_features, y_train_coords_original,
                X_test_features, y_test_coords_original,
                svr_params=svr_params,
                training_params=current_params,  # 将包含min_delta_ratio的params传给该函数
                train_loss_list=train_loss_list,
                val_loss_list=val_loss_list,
                output_dir=current_run_dir,
                image_index=trial.number + 1,
                csv_file_path=csv_file_path_regression
            )

            if mean_error_distance < best_mean_error_distance:
                best_mean_error_distance = mean_error_distance
                best_regression_model = regression_model
                best_model_path = os.path.join(model_dir, 'best_regression_model.pkl')
                joblib.dump(best_regression_model, best_model_path)
                print(f"最佳回归模型已保存到 {best_model_path}。")

                with open(best_params_path, 'w', encoding='utf-8') as f:
                    json.dump(current_params, f, indent=4, ensure_ascii=False)
                print(f"最佳超参数已保存到 {best_params_path}。")

                try:
                    best_image_index = trial.number + 1
                    best_image_name = f"{best_image_index:04d}_regression.png"
                    best_image_path = os.path.join(current_run_dir, best_image_name)
                    destination_image_path = os.path.join(current_run_dir, "0000.png")
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
            else:
                print("当前模型未超过最佳模型，不更新 error.csv。")

            return mean_error_distance

        except NaNLossError:
            raise TrialPruned()
        except ValueError as e:
            if 'NaN' in str(e):
                raise TrialPruned()
            else:
                raise e

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("\n=== 最佳参数 ===")
    best_trial = study.best_trial
    print(f"平均误差距离: {best_trial.value:.2f} 米")
    print("最佳超参数:")
    print(json.dumps(best_trial.params, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()
