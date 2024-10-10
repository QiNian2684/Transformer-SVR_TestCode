# training_and_evaluation.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import joblib
import matplotlib.pyplot as plt

def compute_error_distances(y_true, y_pred):
    """
    计算真实位置和预测位置之间的欧氏距离（以米为单位），包括楼层差异。

    参数：
    - y_true: 真实位置的数组，形状为 [n_samples, 3]，列分别为 [X_coordinate, Y_coordinate, FLOOR]
    - y_pred: 预测位置的数组，形状为 [n_samples, 3]

    返回：
    - distances: 距离数组，形状为 [n_samples,]，单位为米
    """
    # 计算水平距离（经度和纬度）
    horizontal_distances = np.linalg.norm(y_true[:, :2] - y_pred[:, :2], axis=1)
    # 计算垂直距离（楼层差异 * 4米）
    vertical_distances = np.abs(y_true[:, 2] - y_pred[:, 2]) * 4
    # 计算总欧氏距离
    distances = np.sqrt(horizontal_distances**2 + vertical_distances**2)
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

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, X_batch)
            loss.backward()
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

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型参数
            torch.save(model.state_dict(), 'best_transformer_autoencoder.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("早停触发，停止训练。")
                break

    # 加载最佳模型参数
    model.load_state_dict(torch.load('best_transformer_autoencoder.pth'))
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


def train_and_evaluate_svr(X_train_features, y_train, X_test_features, y_test, svr_params=None):
    """
    训练并评估 SVR 模型。

    参数：
    - X_train_features: 训练集特征
    - y_train: 训练集目标变量
    - X_test_features: 测试集特征
    - y_test: 测试集目标变量
    - svr_params: SVR模型的参数字典（可选）

    返回：
    - best_svr: 训练好的 SVR 模型
    """
    if svr_params is None:
        print("使用 RandomizedSearchCV 调优 SVR 模型参数...")
        param_dist = {
            'estimator__kernel': ['rbf', 'linear'],
            'estimator__C': [0.1, 1, 10, 100, 1000],
            'estimator__epsilon': [0.05, 0.1, 0.2, 0.5, 1.0]
        }
        svr = SVR()
        multi_svr = MultiOutputRegressor(svr)
        random_search = RandomizedSearchCV(
            multi_svr, param_distributions=param_dist, n_iter=20, cv=3,
            verbose=2, random_state=42, n_jobs=-1
        )
        random_search.fit(X_train_features, y_train)

        print(f"最佳参数: {random_search.best_params_}")
        best_svr = random_search.best_estimator_
    else:
        print("使用提供的 SVR 参数训练模型...")
        svr = SVR(**svr_params)
        best_svr = MultiOutputRegressor(svr)
        best_svr.fit(X_train_features, y_train)

    y_pred = best_svr.predict(X_test_features)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 计算误差距离
    error_distances = compute_error_distances(y_test, y_pred)
    mean_error_distance = np.mean(error_distances)
    median_error_distance = np.median(error_distances)

    print(f"SVR 回归模型评估结果：")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R^2 Score: {r2:.6f}")
    print(f"平均误差距离（米）: {mean_error_distance:.2f}")
    print(f"中位数误差距离（米）: {median_error_distance:.2f}")

    # 保存 SVR 模型
    joblib.dump(best_svr, 'best_svr_model.pkl')

    # 生成3D预测误差散点图
    error_x = y_pred[:, 0] - y_test[:, 0]
    error_y = y_pred[:, 1] - y_test[:, 1]
    error_floor = y_pred[:, 2] - y_test[:, 2]

    # 将楼层误差转换为米
    error_z = error_floor * 6

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 计算每个点的误差距离
    error_distance = np.sqrt(error_x ** 2 + error_y ** 2 + error_z ** 2)

    scatter = ax.scatter(error_x, error_y, error_z, c=error_distance, cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('误差距离 (米)')

    ax.set_title('3D Prediction Errors')
    ax.set_xlabel('Error in X coordinate (meters)')
    ax.set_ylabel('Error in Y coordinate (meters)')
    ax.set_zlabel('Error in Z coordinate (meters)')  # Z represents vertical error

    plt.show()

    return best_svr