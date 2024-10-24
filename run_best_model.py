# run_best_model.py

import os
import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
    train_autoencoder,
    extract_features,
    train_and_evaluate_classification_model,
    train_and_evaluate_regression_model,
    NaNLossError
)
import json
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """
    主程序，读取最佳超参数配置，运行分类和回归模型，并保存结果图片。
    """
    #设置epoch
    classification_epoch = 75
    regression_epoch = 75

    # 设置随机种子
    set_seed()

    # === 参数设置 ===
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据路径
    train_path = 'UJIndoorLoc/trainingData.csv'
    test_path = 'UJIndoorLoc/validationData.csv'

    # === 数据加载与预处理 ===
    print("加载并预处理数据...")
    X_train, y_train_coords, y_train_floor, X_val, y_val_coords, y_val_floor, X_test, y_test_coords, y_test_floor, scaler_X, scaler_y, label_encoder = load_and_preprocess_data(train_path, test_path)

    # === 读取最佳超参数 ===
    # 设置存放最佳模型参数的文件夹
    best_model_json_dir = 'best_model_json'

    # 获取分类模型的超参数文件路径
    classification_params_path = os.path.join(best_model_json_dir, 'best_hyperparameters_classification.json')
    if not os.path.exists(classification_params_path):
        print("没有找到分类的最佳超参数文件。")
        return

    # 获取回归模型的超参数文件路径
    regression_params_path = os.path.join(best_model_json_dir, 'best_hyperparameters_regression.json')
    if not os.path.exists(regression_params_path):
        print("没有找到回归的最佳超参数文件。")
        return

    # 加载最佳超参数
    with open(classification_params_path, 'r', encoding='utf-8') as f:
        classification_params = json.load(f)
    with open(regression_params_path, 'r', encoding='utf-8') as f:
        regression_params = json.load(f)

    # === 创建保存结果的文件夹 ===
    best_result_dir = 'best_result'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = os.path.join(best_result_dir, timestamp)
    os.makedirs(current_run_dir, exist_ok=True)
    print(f"结果将保存到: {current_run_dir}")

    # === 定义模型保存目录 ===
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)  # 创建目录，如果已存在则不操作

    # === 运行分类模型 ===
    print("运行最佳分类模型...")
    try:
        # 提取分类模型的参数
        model_dim = classification_params['model_dim']
        num_heads = classification_params['num_heads']
        num_layers = classification_params['num_layers']
        dropout = classification_params['dropout']
        learning_rate = classification_params['learning_rate']
        batch_size = classification_params['batch_size']
        early_stopping_patience = classification_params['early_stopping_patience']
        svc_C = classification_params['svc_C']
        svc_kernel = classification_params['svc_kernel']
        svc_gamma = classification_params['svc_gamma']
        svc_degree = classification_params.get('svc_degree', 3)
        svc_coef0 = classification_params.get('svc_coef0', 0.0)

        # 收集当前超参数组合
        current_params = classification_params

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
            epochs=classification_epoch,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience
        )

        # 提取特征
        X_train_features = extract_features(model, X_train, device=device, batch_size=batch_size)
        X_test_features = extract_features(model, X_test, device=device, batch_size=batch_size)

        # 定义 SVC 参数
        svc_params = {
            'C': svc_C,
            'kernel': svc_kernel,
            'gamma': svc_gamma,
            'degree': svc_degree,
            'coef0': svc_coef0,
        }

        # 训练并评估分类模型
        classification_model, accuracy = train_and_evaluate_classification_model(
            X_train_features, y_train_floor,
            X_test_features, y_test_floor,
            svc_params=svc_params,
            training_params=current_params,
            train_loss_list=train_loss_list,
            val_loss_list=val_loss_list,
            label_encoder=label_encoder,
            output_dir=current_run_dir,
            image_index=1  # 图片编号为1
        )

        # === 保存分类结果图片 ===
        classification_image_path = os.path.join(current_run_dir, 'classification_result.png')
        source_image_path = os.path.join(current_run_dir, f"{1:04d}.png")
        shutil.copyfile(source_image_path, classification_image_path)
        print(f"分类结果图片已保存为 {classification_image_path}")

        # 显示图片
        img = plt.imread(classification_image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"运行分类模型时发生错误：{e}")

    # === 运行回归模型 ===
    print("运行最佳回归模型...")
    try:
        # 提取回归模型的参数
        model_dim = regression_params['model_dim']
        num_heads = regression_params['num_heads']
        num_layers = regression_params['num_layers']
        dropout = regression_params['dropout']
        learning_rate = regression_params['learning_rate']
        batch_size = regression_params['batch_size']
        early_stopping_patience = regression_params['early_stopping_patience']
        svr_kernel = regression_params['svr_kernel']
        svr_C = regression_params['svr_C']
        svr_epsilon = regression_params['svr_epsilon']
        svr_gamma = regression_params['svr_gamma']
        svr_degree = regression_params.get('svr_degree', 3)
        svr_coef0 = regression_params.get('svr_coef0', 0.0)

        # 收集当前超参数组合
        current_params = regression_params

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
            epochs=regression_epoch,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience
        )

        # 提取特征
        X_train_features = extract_features(model, X_train, device=device, batch_size=batch_size)
        X_test_features = extract_features(model, X_test, device=device, batch_size=batch_size)

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

        # 训练并评估回归模型
        regression_model, mean_error_distance = train_and_evaluate_regression_model(
            X_train_features, y_train_coords_original,
            X_test_features, y_test_coords_original,
            svr_params=svr_params,
            training_params=current_params,
            train_loss_list=train_loss_list,
            val_loss_list=val_loss_list,
            output_dir=current_run_dir,
            image_index=2  # 图片编号为2
        )

        # === 保存回归结果图片 ===
        regression_image_path = os.path.join(current_run_dir, 'regression_result.png')
        source_image_path = os.path.join(current_run_dir, f"{2:04d}.png")
        shutil.copyfile(source_image_path, regression_image_path)
        print(f"回归结果图片已保存为 {regression_image_path}")

        # 显示图片
        img = plt.imread(regression_image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"运行回归模型时发生错误：{e}")

    print("最佳模型已运行完毕，结果已保存。")

if __name__ == '__main__':
    main()
