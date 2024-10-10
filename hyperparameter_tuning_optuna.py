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
    train_path = 'UJIndoorLoc/trainingData_building2.csv'
    test_path = 'UJIndoorLoc/validationData_building2.csv'

    # 固定训练参数
    epochs = 200  # 训练轮数
    n_trials = 300  # Optuna 试验次数，根据计算资源调整

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_and_preprocess_data(train_path, test_path)

    # === 定义优化目标函数 ===
    def objective(trial):
        # Transformer 自编码器超参数
        model_dim = trial.suggest_categorical('model_dim', [16, 32, 64, 128])
        # model_dim: 模型维度，决定了模型的容量，更大的值意味着更强的学习能力，但也可能导致过拟合。

        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8, 16])
        # num_heads: 注意力机制中的头数，更多头数可以捕捉更丰富的信息，但计算量也更大。

        num_layers = trial.suggest_categorical('num_layers', [4, 8, 16, 32])
        # num_layers: 编码器和解码器中层的数量，层数越多，模型能表达的复杂度越高，但也容易过拟合。

        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        # dropout: 防止过拟合的技术，随机丢弃部分神经网络单元。增加dropout比率可以增强模型的泛化能力，但过高可能导致欠拟合。

        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.005, log=True)
        # learning_rate: 学习率决定了参数更新的步长，过高可能导致训练不稳定，过低可能导致训练速度过慢。

        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        # batch_size: 每次训练的样本数量，较大的batch size通常可以提高训练稳定性和效率，但也可能影响模型的泛化能力。

        patience = trial.suggest_int('early_stopping_patience', 5, 15)
        # patience: 早停的容忍度，若训练在指定的轮数内未见改善，则停止训练，有助于防止过拟合。

        # SVR 超参数
        svr_kernel = trial.suggest_categorical('svr_kernel', ['poly', 'rbf', 'sigmoid'])
        # svr_kernel: SVR中使用的核函数类型，不同的核函数适用于不同的数据分布。

        svr_C = trial.suggest_float('svr_C', 1e-1, 1e2, log=True)
        # svr_C: 错误项的惩罚系数。较大的C值可以减少训练误差，但可能增加泛化误差，反之则可能导致训练误差增大。

        svr_epsilon = trial.suggest_float('svr_epsilon', 0.0, 1.0)
        # svr_epsilon: 容忍误差，设置目标函数预测的自由区间，epsilon越大，模型越不敏感。

        svr_gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])
        # svr_gamma: 核函数的系数，仅对'rbf', 'poly'和'sigmoid'核有效。'scale'会自动调整，而'auto'使用特征数量的倒数。

        # 如果 kernel 是 'poly'，则调优 degree 和 coef0
        if svr_kernel == 'poly':
            svr_degree = trial.suggest_int('svr_degree', 2, 5)
            # svr_degree: 'poly'核函数的度数，度数越高，函数能拟合更复杂的曲线，但计算量也更大。

            svr_coef0 = trial.suggest_float('svr_coef0', 0.0, 1.0)
            # svr_coef0: 'poly'和'sigmoid'核的独立项系数，可以调整决策函数的形状。
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

        # 逆标准化目标变量
        y_train_original = scaler_y.inverse_transform(y_train)
        y_test_original = scaler_y.inverse_transform(y_test)

        # 定义 SVR 参数
        svr_params = {
            'kernel': svr_kernel,
            'C': svr_C,
            'epsilon': svr_epsilon,
            'gamma': svr_gamma,
            'degree': svr_degree,
            'coef0': svr_coef0,
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

    # 逆标准化目标变量
    y_train_original = scaler_y.inverse_transform(y_train)
    y_test_original = scaler_y.inverse_transform(y_test)

    # 定义最佳 SVR 参数
    svr_params = {
        'kernel': svr_kernel,
        'C': svr_C,
        'epsilon': svr_epsilon,
        'gamma': svr_gamma,
        'degree': svr_degree,
        'coef0': svr_coef0,
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
