# hyperparameter_tuning_classification.py

import os
import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation_classification import (
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
    n_trials = 500  # Optuna 试验次数，根据计算资源调整

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
    classification_results_dir = os.path.join(results_dir, 'classification')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = os.path.join(classification_results_dir, timestamp)
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"结果将保存到: {current_run_dir}")

    # === 定义模型保存目录 ===
    model_dir = 'best_models_pkl'
    os.makedirs(model_dir, exist_ok=True)  # 创建目录，如果已存在则不操作

    # 定义 CSV 文件路径，将其保存在 current_run_dir
    csv_file_path_classification = os.path.join(current_run_dir, 'classification_results.csv')

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, _, y_train_floor, X_val, _, y_val_floor, X_test, _, y_test_floor, scaler_X, _, label_encoder = load_and_preprocess_data(train_path, test_path)

    # 用于保存最佳模型
    best_accuracy = 0
    best_classification_model = None

    # 定义最佳超参数文件路径
    best_params_path = os.path.join(current_run_dir, 'best_hyperparameters_classification.json')

    # === 定义优化目标函数 ===
    def objective(trial):
        nonlocal best_accuracy
        nonlocal best_classification_model

        try:
            # Transformer 自编码器超参数

            # model_dim: 模型维度，决定了嵌入空间的大小，以及随后各层的大小。
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64, 128, 256, 512])

            # num_heads_options: 生成一个列表，包含可以整除model_dim的头数选项。
            num_heads_options = [h for h in [2, 4, 8, 16] if model_dim % h == 0]

            # 如果没有有效的头数选项，中止这次试验。
            if not num_heads_options:
                raise TrialPruned("model_dim 不可被任何 num_heads 整除。")

            # num_heads: 注意力机制中的头数。
            num_heads = trial.suggest_categorical('num_heads', num_heads_options)

            # num_layers: Transformer模型中的层数。
            num_layers = trial.suggest_int('num_layers', low=4, high=64)

            # dropout: 在模型训练时每个元素被随机丢弃的概率。
            dropout = trial.suggest_float('dropout', 0.0, 0.6)

            # learning_rate: 学习率。
            learning_rate = trial.suggest_float('learning_rate', 0.00003, 0.001, log=True)

            # batch_size: 批量大小。
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

            # patience: 早停策略中的耐心值。
            patience = trial.suggest_int('early_stopping_patience', 3, 5)

            # SVC 超参数

            # svc_C: 支持向量机的正则化参数C。
            svc_C = trial.suggest_float('svc_C', 1, 500, log=True)


            # svc_kernel: 支持向量机使用的核函数类型。
            svc_kernel = trial.suggest_categorical('svc_kernel', ['poly', 'rbf', 'sigmoid'])

            # svc_gamma: 核函数的系数。
            svc_gamma = trial.suggest_categorical('svc_gamma', ['scale', 'auto'])

            # 如果选用的是多项式核函数，还需要设置多项式的度数和系数。
            if svc_kernel == 'poly':
                # svc_degree: 多项式核函数的度数。
                svc_degree = trial.suggest_int('svc_degree', 2, 5)

                # svc_coef0: 多项式核函数的自由项系数。
                svc_coef0 = trial.suggest_float('svc_coef0', 0.0, 1.0)
            else:
                # 对于非多项式核，使用默认值。
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
            classification_model, accuracy, y_pred_floor = train_and_evaluate_classification_model(
                X_train_features, y_train_floor,
                X_test_features, y_test_floor,
                svc_params=svc_params,
                training_params=current_params,
                train_loss_list=train_loss_list,
                val_loss_list=val_loss_list,
                label_encoder=label_encoder,
                output_dir=current_run_dir,
                image_index=trial.number + 1,  # 使用 trial.number + 1 作为图片编号
                csv_file_path=csv_file_path_classification  # 添加此行，指定CSV文件路径
            )

            # 如果当前模型表现更好，保存模型和超参数，并更新最佳图片
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_classification_model = classification_model
                # 保存最佳模型
                best_model_path = os.path.join(model_dir, 'best_classification_model.pkl')
                joblib.dump(best_classification_model, best_model_path)
                print(f"最佳分类模型已保存到 {best_model_path}。")
                # 保存最佳超参数
                with open(best_params_path, 'w', encoding='utf-8') as f:
                    json.dump(current_params, f, indent=4, ensure_ascii=False)
                print(f"最佳超参数已保存到 {best_params_path}。")
                # 更新最佳试验的结果图片为“0000.png”
                try:
                    best_image_index = trial.number + 1  # 与保存图片时的编号对应
                    best_image_name = f"{best_image_index:04d}_classification.png"
                    best_image_path = os.path.join(current_run_dir, best_image_name)
                    destination_image_path = os.path.join(current_run_dir, "0000.png")
                    # 复制最佳试验的图片并重命名为“0000.png”
                    shutil.copyfile(best_image_path, destination_image_path)
                    print(f"最佳试验的结果图片已更新为 {destination_image_path}")
                except Exception as e:
                    print(f"无法更新最佳试验的图片：{e}")

                # 保存预测的楼层到 CSV 文件
                predicted_floor_csv_path = os.path.join(current_run_dir, 'predicted_floors.csv')
                test_data_with_predictions = pd.DataFrame({
                    'Actual_Floor': y_test_floor,
                    'Predicted_Floor': y_pred_floor
                })
                # 解码楼层标签
                test_data_with_predictions['Actual_Floor'] = label_encoder.inverse_transform(test_data_with_predictions['Actual_Floor'])
                test_data_with_predictions['Predicted_Floor'] = label_encoder.inverse_transform(test_data_with_predictions['Predicted_Floor'])
                test_data_with_predictions.to_csv(predicted_floor_csv_path, index=False)
                print(f"预测的楼层已保存到 {predicted_floor_csv_path}。")

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

    # 已在每次试验中实时保存最佳模型、超参数和图片，因此这里无需再次保存

if __name__ == '__main__':
    main()
