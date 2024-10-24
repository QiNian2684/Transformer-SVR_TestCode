# hyperparameter_tuning_regression.py

import os
import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
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
import shutil  # 新增，用于复制文件

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
    train_path = 'UJIndoorLoc/trainingData.csv'
    test_path = 'UJIndoorLoc/validationData.csv'

    # 固定训练参数
    epochs = 200  # 训练轮数
    n_trials = 500  # Optuna 试验次数，根据计算资源调整

    # === 创建结果保存目录 ===
    results_dir = 'results'
    regression_results_dir = os.path.join(results_dir, 'regression')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = os.path.join(regression_results_dir, timestamp)
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"结果将保存到: {current_run_dir}")

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, y_train_coords, _, X_val, y_val_coords, _, X_test, y_test_coords, _, scaler_X, scaler_y, _ = load_and_preprocess_data(train_path, test_path)

    # 初始化图片编号
    image_index = 1

    # === 定义优化目标函数 ===
    def objective(trial):
        nonlocal image_index  # 引入外部变量

        try:
            # Transformer 自编码器超参数

            # model_dim: 模型的维度，影响模型的大小和复杂性，可选值包括16, 32, 64, 128。
            # 选用较大的维度通常能提高模型的学习能力，但也会增加计算负担和过拟合的风险。
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64, 128])

            # num_heads_options: 根据 model_dim 的可整除性确定的注意力头数选项。
            # 注意力机制的头数影响模型对不同信息的并行处理能力，但必须能被 model_dim 整除。
            num_heads_options = [h for h in [2, 4, 8, 16] if model_dim % h == 0]
            if not num_heads_options:
                raise TrialPruned("model_dim 不可被任何 num_heads 整除。")

            # num_heads: 选择的注意力头数，从 num_heads_options 中选取，与模型并行处理信息的能力直接相关。
            num_heads = trial.suggest_categorical('num_heads', num_heads_options)

            # num_layers: 模型的层数，从1到32层，步长为8。
            # 层数增加可以增强模型的表达能力，但也会增加模型的复杂度和训练难度。
            num_layers = trial.suggest_int('num_layers', low=4, high=64, log=True)

            # dropout: 在模型训练时随机丢弃节点的比例，范围从0.1到0.5。
            # Dropout 可以防止模型过拟合，较高的值意味着更强的正则化效果。
            dropout = trial.suggest_float('dropout', 0.1, 0.5)

            # learning_rate: 模型学习率，采用对数刻度从1e-6到1e-2选择。
            # 学习率决定了模型参数更新的速度，较小的学习率可能导致学习缓慢，而较大的学习率可能导致训练不稳定。
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)

            # batch_size: 训练批次的大小，可选值为64, 128, 256。
            # 较大的批次可以提高内存利用率和训练速度，但可能影响模型最终的泛化能力。
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

            # patience: 早停机制的耐心值，即在一定数量的训练轮次内如果模型性能没有提升则停止训练。
            # 范围从5到10，这个参数用来防止过拟合并缩短无效训练时间。
            patience = trial.suggest_int('early_stopping_patience', 5, 10)

            # SVR 超参数

            # svr_kernel: 支持向量机的核函数类型，选择 'linear', 'poly', 'rbf', 'sigmoid' 中的一个。
            # 核函数类型决定了数据在更高维空间中的映射方式，影响模型的复杂度和训练效果。
            svr_kernel = trial.suggest_categorical('svr_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

            # svr_C: 正则化参数C，控制误差项和决策面边界的权衡，从0.1到100的对数刻度选择。
            # 较大的C值意味着赋予误差项更小的权重，通常会得到更紧的决策边界，但可能导致过拟合。
            svr_C = trial.suggest_float('svr_C', 1e-1, 1e2, log=True)

            # svr_epsilon: Epsilon in the SVR model, controls the width of the tube within which no penalty is associated in the training loss.
            # 值的范围从0.01到1.0，控制SVR模型中无惩罚区域的宽度，影响模型的灵敏度和泛化能力。
            svr_epsilon = trial.suggest_float('svr_epsilon', 0.01, 1.0)

            # svr_gamma: 核函数的系数，'scale' 和 'auto' 两种选择。
            # 这个参数决定了数据特征在高维空间的分布，与核函数的性能和模型的复杂度密切相关。
            svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])

            # 如果使用多项式核函数，则还需要设置以下参数：
            if svr_kernel == 'poly':
                # svr_degree: 多项式核函数的度数，从2到5选择。
                # 度数越高，模型能表达的复杂度越高，但计算成本也越大。
                svr_degree = trial.suggest_int('svr_degree', 2, 5)

                # svr_coef0: 多项式核函数的自由系数，从0.0到1.0选择。
                # 这个参数影响核函数的形状，进而影响模型的决策边界。
                svr_coef0 = trial.suggest_float('svr_coef0', 0.0, 1.0)
            else:
                # 默认值
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
            regression_model, mean_error_distance = train_and_evaluate_regression_model(
                X_train_features, y_train_coords_original,
                X_test_features, y_test_coords_original,
                svr_params=svr_params,
                training_params=current_params,
                train_loss_list=train_loss_list,
                val_loss_list=val_loss_list,
                output_dir=current_run_dir,
                image_index=trial.number + 1  # 使用 trial.number + 1 作为图片编号
            )

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

    # 保存最佳超参数
    best_params_path = os.path.join(current_run_dir, 'best_hyperparameters_regression.json')
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

if __name__ == '__main__':
    main()
