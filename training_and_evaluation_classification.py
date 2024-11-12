# training_and_evaluation_classification.py

import os
import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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


def train_and_evaluate_classification_model(X_train_features, y_train_floor, X_test_features, y_test_floor,
                                            svc_params=None, training_params=None,
                                            train_loss_list=None, val_loss_list=None, label_encoder=None,
                                            output_dir=None, image_index=1, csv_file_path=None):
    """
    训练并评估分类模型，包括可视化和打印输出。

    参数：
    - X_train_features: 训练集特征
    - y_train_floor: 训练集楼层标签
    - X_test_features: 测试集特征
    - y_test_floor: 测试集楼层标签
    - svc_params: SVC模型的参数字典（可选）
    - training_params: 训练参数的字典，用于显示在图表中
    - train_loss_list: 每个 epoch 的训练损失列表（可选，自编码器的损失）
    - val_loss_list: 每个 epoch 的验证损失列表（可选，自编码器的损失）
    - label_encoder: 标签编码器，用于解码楼层标签
    - output_dir: 保存结果图片的目录（可选）
    - image_index: 图片编号，默认从1开始
    - csv_file_path: 记录结果的 CSV 文件路径（可选）

    返回：
    - classification_model: 训练好的分类模型
    - accuracy: 分类准确率（用于超参数优化）
    - y_pred_floor: 预测的楼层标签数组
    """
    try:
        # 楼层分类模型
        print("训练楼层分类模型...")
        if svc_params is None:
            classification_model = SVC()
        else:
            classification_model = SVC(**svc_params)
        classification_model.fit(X_train_features, y_train_floor)
        print("楼层分类模型训练完成。")

        # 预测测试数据
        print("开始预测测试数据...")
        y_pred_floor = classification_model.predict(X_test_features)
        print("预测完成。")

        # 评估分类模型
        print("评估分类模型...")
        accuracy = accuracy_score(y_test_floor, y_pred_floor)
        class_report = classification_report(y_test_floor, y_pred_floor, zero_division=0)
        print("分类模型评估完成。")

        # 检查未被预测到的类别
        actual_classes = set(y_test_floor)
        predicted_classes = set(y_pred_floor)
        missing_classes = actual_classes - predicted_classes
        if missing_classes:
            print(f"警告：以下类别未被预测到：{missing_classes}")

        print(f"分类模型评估结果：")
        print(f"准确率: {accuracy:.4f}")
        print("分类报告：")
        print(class_report)

        # 定义所有可能的楼层标签（根据您的数据集调整）
        all_possible_floors = np.unique(np.concatenate((y_train_floor, y_test_floor)))

        # 计算每层楼的分类准确率
        per_floor_accuracies = {}
        for floor in all_possible_floors:
            indices = (y_test_floor == floor)
            if np.sum(indices) > 0:
                # 测试集中有该楼层的样本
                floor_accuracy = accuracy_score(y_test_floor[indices], y_pred_floor[indices])
                per_floor_accuracies[floor] = floor_accuracy
            else:
                # 测试集中没有该楼层的样本，标记为 -1
                per_floor_accuracies[floor] = -1

        print("每层楼的分类准确率：")
        for floor in sorted(per_floor_accuracies.keys()):
            acc = per_floor_accuracies[floor]
            if acc != -1:
                print(f"Floor {floor}: Accuracy {acc:.4f}")
            else:
                print(f"Floor {floor}: -1")

        # 生成可视化图表
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

        # 第一行第一列：分类报告
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')  # 隐藏坐标轴

        # 将分类报告格式化为多列文本
        class_report_lines = class_report.strip().split('\n')
        class_report_text = '\n'.join(class_report_lines)
        ax1.text(0.5, 0.5, f"Classification Report:\n{class_report_text}", fontsize=12, ha='center', va='center',
                 wrap=True)

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

        # 格式化评估指标为文本
        metrics_text = (
            f"Classification Model Evaluation Results:\n"
            f"Accuracy: {accuracy:.4f}\n"
        )

        if missing_classes:
            metrics_text += f"Missing Classes: {missing_classes}\n"

        # 添加每层楼的分类准确率
        metrics_text += "\nPer-floor Accuracies:\n"
        for floor in sorted(per_floor_accuracies.keys()):
            acc = per_floor_accuracies[floor]
            if acc != -1:
                metrics_text += f"Floor {floor}: {acc:.4f}\n"
            else:
                metrics_text += f"Floor {floor}: -1\n"

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
                'Accuracy': accuracy,
                'Missing_Classes': str(missing_classes) if missing_classes else 'None'
            }

            # 添加每层楼的分类准确率到 row_data
            for floor, acc in per_floor_accuracies.items():
                row_data[f'Accuracy_Floor_{floor}'] = acc

            # 添加 svc_params
            if svc_params is not None:
                for key, value in svc_params.items():
                    row_data[f'svc_{key}'] = value

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
            image_path = os.path.join(output_dir, f"{image_index:04d}_classification.png")
            plt.savefig(image_path)
            plt.close(fig)
            print(f"结果图片已保存到 {image_path}")
        else:
            plt.show()

        return classification_model, accuracy, y_pred_floor  # 返回 y_pred_floor

    except ValueError as e:
        if 'NaN' in str(e):
            print("训练过程中遇到 NaN，试验将被剪枝。")
            raise ValueError("训练失败，输入数据中包含 NaN。")
        else:
            raise e
