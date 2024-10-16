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
    epochs = 125  # 训练轮数
    n_trials = 500  # Optuna 试验次数，根据计算资源调整

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, y_train_coords, y_train_floor, X_val, y_val_coords, y_val_floor, X_test, y_test_coords, y_test_floor, scaler_X, scaler_y, label_encoder = load_and_preprocess_data(train_path, test_path)

    # === 定义优化目标函数 ===
    def objective(trial):
        try:
            # Transformer 自编码器超参数

            # 1. 选择 model_dim
            # 'model_dim' 是模型维度，它定义了嵌入向量的大小以及Transformer中所有线性层的输出尺寸。
            # 较大的 'model_dim' 可以增加模型的容量，允许模型捕捉更复杂的特征，但会增加计算量和内存消耗，可能导致训练更慢。
            # 较小的 'model_dim' 可以减少计算量和内存消耗，但可能限制模型学习复杂特征的能力。
            model_dim = trial.suggest_categorical('model_dim', [16, 32, 64])

            # 2. 选择 num_heads
            # 'num_heads' 指的是多头注意力机制中头的数量。每个头会独立学习输入数据的不同方面的信息。
            # 增加 'num_heads' 可以提高模型的表示能力，允许模型在不同的表示子空间中捕获信息，但同时会显著增加计算负担。
            # 减少 'num_heads' 可以减轻计算负担，但可能使得模型捕捉特征的能力下降。
            # 注意：'num_heads' 必须能整除 'model_dim'。
            num_heads_options = [h for h in [2, 4, 8, 16] if model_dim % h == 0]
            if not num_heads_options:
                raise TrialPruned("No valid num_heads for the selected model_dim.")
            num_heads = trial.suggest_categorical('num_heads', num_heads_options)

            # 3. 选择 num_layers
            # 'num_layers' 表示Transformer网络的层数。每一层都包括一个多头注意力和一个前向传播网络。
            # 增加 'num_layers' 可以提高模型的深度，理论上可以让模型学习更复杂的特征和更深层次的抽象，但也增加了过拟合的风险和训练难度。
            # 减少 'num_layers' 可以使模型更快地训练，但可能限制模型处理复杂数据的能力。
            num_layers = trial.suggest_categorical('num_layers', [2, 4, 8, 16])

            # 4. 设置 dropout
            # 'dropout' 是一种正则化技术，用于防止神经网络过拟合。它随机地关闭一部分神经元，使得网络不过分依赖于任何一条路径，增加泛化能力。
            # 较高的 'dropout' 值增加了正则化强度，有助于减少过拟合，但过高可能导致欠拟合。
            # 较低的 'dropout' 值减少了正则化强度，可以使网络在训练集上表现更好，但可能增加过拟合风险。
            dropout = trial.suggest_float('dropout', 0.1, 0.5)

            # 5. 设定学习率
            # 'learning_rate' 控制模型在每次迭代中更新的步长。合适的学习率可以使模型快速收敛，而不合适的学习率可能导致模型训练不稳定。
            # 较高的学习率可以加快训练进程，但过高可能导致训练过程中出现震荡或不收敛。
            # 较低的学习率确保了训练稳定性，但可能使训练过程变得非常缓慢，特别是在接近最优解时。
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)

            # 6. 选择批量大小
            # 'batch_size' 影响模型的训练速度和内存使用量。大批量可以提高内存利用率和训练速度，但可能影响模型最终的泛化能力。
            # 较大的 'batch_size' 可以实现更稳定的梯度下降，但需要更多的内存，并可能导致模型泛化性能下降。
            # 较小的 'batch_size' 可以提高模型的泛化能力，但可能使训练更加不稳定和耗时。
            batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

            # 7. 设置早停耐心值
            # 'patience' 是早停策略中的一个参数，如果在设定的连续几个训练周期内，验证集上的性能没有改善，则提前终止训练。
            # 较大的 'patience' 值允许模型在停止前有更多的时间来找到更好的解，但可能导致训练时间过长。
            # 较小的 'patience' 值可以减少训练时间，但可能导致模型未能充分训练。
            patience = trial.suggest_int('early_stopping_patience', 5, 10)

            # 8. SVR 超参数
            # 'svr_kernel' 表示SVR（支持向量回归）的核函数类型，不同的核函数可以适应不同的数据分布。
            # 'svr_C' 是正则化参数，控制错误项的惩罚力度。较大的C值可以减少训练误差，但可能导致过拟合。
            # 'svr_epsilon' 定义了不惩罚预测误差在该值范围内的模型，较大的epsilon可以减少模型的敏感度，提高模型的鲁棒性。
            # 'svr_gamma' 决定了核函数的形状，影响模型的复杂度和训练效果。
            svr_kernel = trial.suggest_categorical('svr_kernel', ['poly', 'rbf', 'sigmoid'])
            svr_C = trial.suggest_float('svr_C', 1e-1, 1e2, log=True)
            svr_epsilon = trial.suggest_float('svr_epsilon', 0.0, 1.0)
            svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])

            # 如果 kernel 是 'poly'，则调优 degree 和 coef0
            # 'svr_degree' 和 'svr_coef0' 分别是多项式核的度数和系数，影响核函数的形状和模型的非线性。
            if svr_kernel == 'poly':
                svr_degree = trial.suggest_int('svr_degree', 2, 5)
                svr_coef0 = trial.suggest_float('svr_coef0', 0.0, 1.0)
            else:
                svr_degree = 3  # 默认值
                svr_coef0 = 0.0  # 默认值

            # 9. SVC 超参数
            # 'svc_C' 是SVC（支持向量分类）的正则化参数。较大的C值可以优化分类准确率，但可能引起过拟合。
            # 'svc_kernel' 和 'svc_gamma' 的解释同SVR。
            # 'svc_degree' 和 'svc_coef0' 仅在选择多项式核时有意义，用于定义核函数的非线性程度和形状。
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
                'svr_kernel': svr_kernel,
                'svr_C': svr_C,
                'svr_epsilon': svr_epsilon,
                'svr_gamma': svr_gamma,
                'svr_degree': svr_degree,
                'svr_coef0': svr_coef0,
                'svc_C': svc_C,
                'svc_kernel': svc_kernel,
                'svc_gamma': svc_gamma,
                'svc_degree': svc_degree,
                'svc_coef0': svc_coef0,
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

            # 定义 SVC 参数
            svc_params_dict = {
                'C': svc_C,
                'kernel': svc_kernel,
                'gamma': svc_gamma,
                'degree': svc_degree,
                'coef0': svc_coef0,
            }

            # 将 SVR 和 SVC 参数添加到 training_params 中
            training_params = current_params.copy()  # 使用当前参数作为训练参数

            # 训练模型
            regression_model, classification_model, accuracy = train_and_evaluate_models(
                X_train_features, y_train_coords_original, y_train_floor,
                X_test_features, y_test_coords_original, y_test_floor,
                svr_params=svr_params_dict,
                svc_params=svc_params_dict,
                training_params=training_params  # 传递训练参数
            )

            # 预测并评估
            y_pred_coords = regression_model.predict(X_test_features)
            error_distances = compute_error_distances(y_test_coords_original, y_pred_coords)
            mean_error_distance = np.mean(error_distances)

            # 移除 trial.report() 和 trial.should_prune()，直接返回目标值

            # 返回要最小化和最大化的目标值
            return mean_error_distance, accuracy

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
    study = optuna.create_study(
        directions=['minimize', 'maximize'],
        pruner=optuna.pruners.NopPruner()  # 在多目标优化中，不使用剪枝
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # === 打印和保存最佳结果 ===
    print("\n=== 最佳参数 ===")

    # 获取所有试验
    all_trials = study.trials

    # 过滤出已完成的试验
    completed_trials = [trial for trial in all_trials if
                        trial.state == optuna.trial.TrialState.COMPLETE and trial.values is not None]

    if not completed_trials:
        print("没有完成的试验可供选择。")
        exit(1)

    # 获取 mean_error_distance 最小的试验
    best_error_trial = min(completed_trials, key=lambda t: t.values[0])

    # 获取 accuracy 最大的试验
    best_accuracy_trial = max(completed_trials, key=lambda t: t.values[1])

    # 基于加权综合指标选择最佳试验
    weight_error_distance = 0.5
    weight_accuracy = 0.5
    combined_score = lambda t: weight_error_distance * (-t.values[0]) + weight_accuracy * t.values[1]
    best_combined_trial = max(completed_trials, key=combined_score)

    # 打印最佳平均误差距离试验
    print("\n--- 最佳平均误差距离试验 ---")
    print("参数：")
    print(json.dumps(best_error_trial.params, indent=4, ensure_ascii=False))
    print(f"平均误差距离（米）：{best_error_trial.values[0]:.2f}")
    print(f"分类准确率：{best_error_trial.values[1]:.4f}")

    # 打印最佳分类准确率试验
    print("\n--- 最佳分类准确率试验 ---")
    print("参数：")
    print(json.dumps(best_accuracy_trial.params, indent=4, ensure_ascii=False))
    print(f"分类准确率：{best_accuracy_trial.values[1]:.4f}")
    print(f"平均误差距离（米）：{best_accuracy_trial.values[0]:.2f}")

    # 打印最佳综合得分试验
    best_combined_score = combined_score(best_combined_trial)
    print("\n--- 最佳综合得分（0.5 权重）试验 ---")
    print("参数：")
    print(json.dumps(best_combined_trial.params, indent=4, ensure_ascii=False))
    print(f"平均误差距离（米）：{best_combined_trial.values[0]:.2f}")
    print(f"分类准确率：{best_combined_trial.values[1]:.4f}")
    print(f"综合得分：{best_combined_score:.4f}")

    # === 使用最佳综合得分试验的超参数重新训练模型 ===
    print("\n=== 使用最佳综合得分（0.5 权重）试验的超参数重新训练模型 ===")
    best_params = best_combined_trial.params

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

    svc_C = best_params['svc_C']
    svc_kernel = best_params['svc_kernel']
    svc_gamma = best_params['svc_gamma']
    svc_degree = best_params.get('svc_degree', 3)  # 如果不是poly核，默认degree为3
    svc_coef0 = best_params.get('svc_coef0', 0.0)  # 如果不是poly核，默认coef0为0.0

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

    # 定义最佳 SVC 参数
    svc_params_best = {
        'C': svc_C,
        'kernel': svc_kernel,
        'gamma': svc_gamma,
        'degree': svc_degree,
        'coef0': svc_coef0,
    }

    # 将最佳参数组合成字典
    training_params = best_params.copy()  # 使用最佳参数作为训练参数

    # 训练最佳模型
    best_regression_model, best_classification_model, accuracy = train_and_evaluate_models(
        X_train_features, y_train_coords_original, y_train_floor,
        X_test_features, y_test_coords_original, y_test_floor,
        svr_params=svr_params_best,
        svc_params=svc_params_best,
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
    all_trials_info = []
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'params': trial.params,
            'values': trial.values,
            'state': trial.state.name,
            'duration': trial.duration.total_seconds() if trial.duration else None
        }
        all_trials_info.append(trial_info)

    with open('all_trials_optuna.json', 'w', encoding='utf-8') as f:
        json.dump(all_trials_info, f, indent=4, ensure_ascii=False)
    print("所有试验的详细信息已保存到 all_trials_optuna.json。")


if __name__ == '__main__':
    main()