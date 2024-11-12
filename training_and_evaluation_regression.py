# training_and_evaluation_regression.py

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import pandas as pd

# 设置可视化字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


class NaNLossError(Exception):
    """自定义异常，当训练过程中出现 NaN 损失时抛出。"""
    pass


def compute_error_distances(y_true, y_pred):
    """
    计算真实位置和预测位置之间的欧氏距离（以米为单位），不包括楼层差异。

    参数：
    - y_true: 真实位置的数组，形状为 [n_samples, 2]，列分别为 [X_coordinate, Y_coordinate]
    - y_pred: 预测位置的数组，形状为 [n_samples, 2]

    返回：
    - distances: 距离数组，形状为 [n_samples,]，单位为米
    """
    # 计算水平距离（X 和 Y 坐标）
    distances = np.linalg.norm(y_true - y_pred, axis=1)
    return distances


def write_csv_row(csv_file_path, fieldnames, row_data):
    """
    写入一行数据到 CSV 文件，如果文件不存在则创建并写入表头。

    参数：
    - csv_file_path: CSV 文件路径
    - fieldnames: 表头列表
    - row_data: 字典形式的一行数据
    """
    # 检查 CSV 文件是否存在，如果不存在则创建并写入表头
    file_exists = os.path.exists(csv_file_path)
    if not file_exists:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # 写入一行结果
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row_data)


def train_autoencoder(model, X_train, X_val, device, epochs=50, batch_size=256, learning_rate=2e-4,
                      early_stopping_patience=5):
    """
    训练 Transformer 自编码器模型。

    参数：
    - model: Transformer 自编码器模型实例
    - X_train: 训练集特征
    - X_val: 验证集特征
    - device: 设备类型
    - epochs: 训练轮数
    - batch_size: 批大小
    - learning_rate: 学习率
    - early_stopping_patience: 早停的轮数

    返回：
    - model: 训练后的模型
    - train_loss_list: 每个 epoch 的训练损失列表
    - val_loss_list: 每个 epoch 的验证损失列表
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_train, dtype=torch.float32)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state_dict = None  # 用于保存最佳模型参数

    # 初始化列表以记录每个 epoch 的损失
    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            batch_size_current = X_batch.size(0)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, X_batch)
            loss.backward()

            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item() * batch_size_current
            total_train_samples += batch_size_current

        avg_train_loss = total_train_loss / total_train_samples

        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                batch_size_current = X_batch.size(0)
                outputs = model(X_batch)
                loss = criterion(outputs, X_batch)
                total_val_loss += loss.item() * batch_size_current
                total_val_samples += batch_size_current

        avg_val_loss = total_val_loss / total_val_samples

        print(f"Epoch [{epoch + 1}/{epochs}], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")

        # 将损失添加到列表中
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

        # 检查数据中是否存在 NaN
        if np.isnan(avg_train_loss) or np.isnan(avg_val_loss):
            print("发现 NaN 损失，停止训练。")
            raise NaNLossError("训练过程中出现 NaN 损失。")

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型参数到内存
            best_model_state_dict = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("早停触发，停止训练。")
                break

    # 加载最佳模型参数
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    else:
        print("未找到最佳模型参数，使用最终的模型参数。")

    return model, train_loss_list, val_loss_list


def extract_features(model, X_data, device, batch_size=256):
    """
    使用训练好的自编码器提取特征。

    参数：
    - model: 训练好的自编码器模型
    - X_data: 输入特征数据
    - device: 设备类型
    - batch_size: 批次大小

    返回：
    - features: 提取的特征数组
    """
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        torch.tensor(X_data, dtype=torch.float32), batch_size=batch_size
    )
    features = []
    with torch.no_grad():
        for X_batch in data_loader:
            X_batch = X_batch.to(device)
            encoded = model.encode(X_batch)
            features.append(encoded.cpu().numpy())
    return np.vstack(features)


def train_and_evaluate_regression_model(X_train_features, y_train_coords, X_test_features, y_test_coords,
                                        svr_params=None, training_params=None,
                                        train_loss_list=None, val_loss_list=None,
                                        output_dir=None, image_index=1, csv_file_path=None):
    """
    训练并评估回归模型，包括可视化和打印输出。

    参数：
    - X_train_features: 训练集特征
    - y_train_coords: 训练集坐标（经度、纬度）
    - X_test_features: 测试集特征
    - y_test_coords: 测试集坐标（经度、纬度）
    - svr_params: SVR模型的参数字典（可选）
    - training_params: 训练参数的字典，用于显示在图表中
    - train_loss_list: 每个 epoch 的训练损失列表（可选，自编码器的损失）
    - val_loss_list: 每个 epoch 的验证损失列表（可选，自编码器的损失）
    - output_dir: 保存结果图片的目录（可选）
    - image_index: 图片编号，默认从1开始
    - csv_file_path: 记录结果的 CSV 文件路径（可选）

    返回：
    - regression_model: 训练好的回归模型
    - mean_error_distance: 平均误差距离（用于超参数优化）
    - error_distances: 误差距离数组
    - y_pred_coords: 预测的坐标数组
    """
    try:
        # 坐标回归模型
        if svr_params is None:
            print("使用默认的 SVR 参数进行回归...")
            svr_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1,
                'gamma': 'scale'
            }

        svr = SVR(**svr_params)
        regression_model = MultiOutputRegressor(svr)
        regression_model.fit(X_train_features, y_train_coords)
        print("坐标回归模型训练完成。")

        # 预测测试数据
        print("开始预测测试数据...")
        y_pred_coords = regression_model.predict(X_test_features)
        print("预测完成。")

        # 评估回归模型
        print("评估回归模型...")
        mse = mean_squared_error(y_test_coords, y_pred_coords)
        mae = mean_absolute_error(y_test_coords, y_pred_coords)
        r2 = r2_score(y_test_coords, y_pred_coords)
        print("回归模型评估完成。")

        # 计算误差距离
        error_distances = compute_error_distances(y_test_coords, y_pred_coords)
        mean_error_distance = np.mean(error_distances)
        median_error_distance = np.median(error_distances)

        print(f"回归模型评估结果：")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R^2 Score: {r2:.6f}")
        print(f"平均误差距离（米）: {mean_error_distance:.2f}")
        print(f"中位数误差距离（米）: {median_error_distance:.2f}")

        # 生成可视化图表
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        # 第一行第一列：2D预测误差散点图
        ax1 = fig.add_subplot(gs[0, 0])
        error_x = y_pred_coords[:, 0] - y_test_coords[:, 0]
        error_y = y_pred_coords[:, 1] - y_test_coords[:, 1]
        error_distance = np.sqrt(error_x ** 2 + error_y ** 2)
        scatter = ax1.scatter(error_x, error_y, c=error_distance, cmap='viridis', alpha=0.6)
        cbar = fig.colorbar(scatter, ax=ax1)
        cbar.set_label('Error Distance (meters)')

        ax1.set_title('2D Prediction Errors')
        ax1.set_xlabel('Longitude Error (meters)')
        ax1.set_ylabel('Latitude Error (meters)')

        # 第一行第二列：训练参数和评估指标
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')  # 隐藏坐标轴

        # 格式化训练参数为多列文本
        params_text = "Training Parameters:\n"
        if training_params is not None:
            params_items = list(training_params.items())
            params_lines = []
            for i in range(0, len(params_items), 2):
                line = ''
                for j in range(2):
                    if i + j < len(params_items):
                        key, value = params_items[i + j]
                        line += f"{key}: {value}    "
                params_lines.append(line.strip())
            params_text += '\n'.join(params_lines)

        # 格式化评估指标为多列文本
        metrics_text = (
            f"Regression Model Evaluation Results:\n"
            f"MSE: {mse:.6f}    MAE: {mae:.6f}\n"
            f"R² Score: {r2:.6f}\n"
            f"Mean Error Dist (m): {mean_error_distance:.2f}    Median Error Dist (m): {median_error_distance:.2f}\n"
        )

        # 将训练参数和评估指标合并
        combined_text = params_text + "\n\n" + metrics_text

        # 设置字体大小并添加文本
        ax2.text(0.5, 0.5, combined_text, fontsize=12, ha='center', va='center', wrap=True)

        # 第二行：训练和验证损失曲线（来自自编码器训练）
        ax3 = fig.add_subplot(gs[1, :])
        if train_loss_list is not None and val_loss_list is not None:
            epochs_range = range(1, len(train_loss_list) + 1)
            ax3.plot(epochs_range, train_loss_list, 'r-', label='Autoencoder Training Loss')
            ax3.plot(epochs_range, val_loss_list, 'b-', label='Autoencoder Validation Loss')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.set_title('Autoencoder Training and Validation Loss')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No loss data available', fontsize=16, ha='center', va='center')
            ax3.axis('off')

        # 将模型参数和结果写入 CSV 文件
        if csv_file_path is not None:
            # 展开参数字典为单独的列
            row_data = {
                'Image_Index': f"{image_index:04d}",
                'MSE': mse,
                'MAE': mae,
                'R2_Score': r2,
                'Mean_Error_Distance': mean_error_distance,
                'Median_Error_Distance': median_error_distance
            }

            # 添加 svr_params
            if svr_params is not None:
                for key, value in svr_params.items():
                    row_data[f'svr_{key}'] = value

            # 添加 training_params
            if training_params is not None:
                for key, value in training_params.items():
                    row_data[key] = value

            # 构建字段名列表
            fieldnames = list(row_data.keys())

            write_csv_row(csv_file_path, fieldnames, row_data)
            print(f"结果已记录到 CSV 文件: {csv_file_path}")

        # 保存图片
        if output_dir is not None:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            # 修改这里，确保图片编号为四位数
            image_path = os.path.join(output_dir, f"{image_index:04d}_regression.png")
            plt.savefig(image_path)
            plt.close(fig)
            print(f"结果图片已保存到 {image_path}")
        else:
            plt.show()

        return regression_model, mean_error_distance, error_distances, y_pred_coords  # 返回 y_pred_coords

    except ValueError as e:
        if 'NaN' in str(e):
            print("训练过程中遇到 NaN，试验将被剪枝。")
            raise ValueError("训练失败，输入数据中包含 NaN。")
        else:
            raise e
