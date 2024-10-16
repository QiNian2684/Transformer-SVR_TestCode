# training_and_evaluation.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputRegressor
import joblib
import matplotlib.pyplot as plt

# 定义每层的高度（单位：米）
FLOOR_HEIGHT = 3  # 您可以根据需要调整此值

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

def train_autoencoder(model, X_train, X_val, device, epochs=50, batch_size=256, learning_rate=2e-4, early_stopping_patience=5):
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

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, X_batch)
            loss.backward()

            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, X_batch)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # 检查是否存在 NaN
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

    return model

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

def train_and_evaluate_models(X_train_features, y_train_coords, y_train_floor, X_test_features, y_test_coords, y_test_floor,
                              svr_params=None, FLOOR_HEIGHT=3.0, training_params=None):
    """
    训练并评估回归和分类模型。

    参数：
    - X_train_features: 训练集特征
    - y_train_coords: 训练集坐标（经度、纬度）
    - y_train_floor: 训练集楼层标签
    - X_test_features: 测试集特征
    - y_test_coords: 测试集坐标（经度、纬度）
    - y_test_floor: 测试集楼层标签
    - svr_params: SVR模型的参数字典（可选）
    - FLOOR_HEIGHT: 楼层高度，用于误差计算（默认为3.0米）
    - training_params: 训练参数的字典，用于显示在图表中

    返回：
    - regression_model: 训练好的坐标回归模型
    - classification_model: 训练好的楼层分类模型
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

        # 楼层分类模型
        print("训练楼层分类模型...")
        classification_model = SVC()
        classification_model.fit(X_train_features, y_train_floor)
        print("楼层分类模型训练完成。")

        # 预测测试数据
        print("开始预测测试数据...")
        y_pred_coords = regression_model.predict(X_test_features)
        y_pred_floor = classification_model.predict(X_test_features)
        print("预测完成。")

        # 评估回归模型
        print("评估回归模型...")
        mse = mean_squared_error(y_test_coords, y_pred_coords)
        mae = mean_absolute_error(y_test_coords, y_pred_coords)
        r2 = r2_score(y_test_coords, y_pred_coords)
        print("回归模型评估完成。")

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

        print(f"分类模型评估结果：")
        print(f"准确率: {accuracy:.4f}")
        print("分类报告：")
        print(class_report)

        # 保存模型
        print("保存模型...")
        joblib.dump(regression_model, 'coordinate_regression_model.pkl')
        joblib.dump(classification_model, 'floor_classification_model.pkl')
        print("模型已保存。")

        # 生成2D预测误差散点图并添加评价指标和训练参数
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

        # 第一个子图：2D散点图
        ax1 = fig.add_subplot(gs[:, 0])
        error_x = y_pred_coords[:, 0] - y_test_coords[:, 0]
        error_y = y_pred_coords[:, 1] - y_test_coords[:, 1]
        error_distance = np.sqrt(error_x ** 2 + error_y ** 2)
        scatter = ax1.scatter(error_x, error_y, c=error_distance, cmap='viridis', alpha=0.6)
        cbar = fig.colorbar(scatter, ax=ax1)
        cbar.set_label('Error distance (meters)')

        ax1.set_title('2D Prediction Errors')
        ax1.set_xlabel('Error in Longitude (meters)')
        ax1.set_ylabel('Error in Latitude (meters)')

        # 第二个子图：训练参数和评估指标
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
            f"Avg Error Dist (m): {mean_error_distance:.2f}    Median Error Dist (m): {median_error_distance:.2f}\n\n"
            f"Classification Model Evaluation Results:\n"
            f"Accuracy: {accuracy:.4f}\n"
        )

        if missing_classes:
            metrics_text += f"Missing Classes: {missing_classes}\n"

        # 将训练参数和评估指标合并
        combined_text = params_text + "\n\n" + metrics_text

        # 设置字体大小并添加文本
        ax2.text(0.5, 0.5, combined_text, fontsize=10, ha='center', va='center', wrap=True)

        # 第三个子图：分类报告
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')  # 隐藏坐标轴

        # 将分类报告格式化为多列文本
        class_report_lines = class_report.strip().split('\n')
        class_report_text = '\n'.join(class_report_lines)
        ax3.text(0.5, 0.5, f"Classification Report:\n{class_report_text}", fontsize=10, ha='center', va='center', wrap=True)

        plt.show()

        return regression_model, classification_model

    except ValueError as e:
        if 'NaN' in str(e):
            print("训练过程中遇到 NaN，试验将被剪枝。")
            raise ValueError("训练失败，输入数据中包含 NaN。")
        else:
            raise e

# 您需要在这里添加模型的定义、数据加载和预处理的代码，以及调用上述函数来完成整个训练和评估流程。
