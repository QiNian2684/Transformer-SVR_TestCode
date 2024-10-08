import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def train_transformer(model, X_train, y_train, X_val, y_val, device, epochs=10, batch_size=256, learning_rate=1e-3):
    """
    训练 Transformer 模型。

    参数：
    - model: Transformer 模型实例
    - X_train: 训练集特征
    - y_train: 训练集目标变量
    - X_val: 验证集特征
    - y_val: 验证集目标变量
    - device: 设备类型
    - epochs: 训练轮数
    - batch_size: 批次大小
    - learning_rate: 学习率

    返回：
    - model: 训练后的模型
    """
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")

    return model

def evaluate_model(model, X_test, y_test, device):
    """
    在测试集上评估模型性能。

    参数：
    - model: 训练好的模型
    - X_test: 测试集特征
    - y_test: 测试集目标变量
    - device: 设备类型

    返回：
    - 无
    """
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        y_pred = model(X_test).cpu().numpy()
        y_true = y_test.cpu().numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Transformer 模型评估结果：")
    print(f"均方误差（MSE）：{mse}")
    print(f"平均绝对误差（MAE）：{mae}")
    print(f"R² 分数：{r2}")

def train_other_regressors(X_train, y_train, X_test, y_test):
    """
    训练其他回归模型并评估性能，包括随机森林、XGBoost 和深度神经网络。

    参数：
    - X_train: 训练集特征
    - y_train: 训练集目标变量
    - X_test: 测试集特征
    - y_test: 测试集目标变量

    返回：
    - 无
    """
    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 导入 tqdm 库用于显示进度条
    from tqdm import tqdm

    # 1. 随机森林回归
    print("训练随机森林回归模型...")
    from sklearn.ensemble import RandomForestRegressor
    # 设置 warm_start=True，允许增量训练
    rf = RandomForestRegressor(warm_start=True, random_state=42)
    rf.n_estimators = 0  # 初始化 n_estimators

    # 使用 tqdm 显示训练进度
    for i in tqdm(range(100), desc="Random Forest Training"):
        rf.n_estimators += 1  # 每次增加一个决策树
        rf.fit(X_train_scaled, y_train)

    # 预测并评估
    y_pred_rf = rf.predict(X_test_scaled)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"随机森林回归模型评估结果：")
    print(f"MSE: {mse_rf}, MAE: {mae_rf}, R2: {r2_rf}")

    # 2. XGBoost 回归
    print("训练 XGBoost 回归模型...")
    import xgboost as xgb
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    # 设置 verbose=1 显示训练进度
    xgbr.fit(X_train_scaled, y_train, verbose=1)
    y_pred_xgb = xgbr.predict(X_test_scaled)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    print(f"XGBoost 回归模型评估结果：")
    print(f"MSE: {mse_xgb}, MAE: {mae_xgb}, R2: {r2_xgb}")

    # 3. 支持向量回归（SVR）模型调优
    print("使用 GridSearchCV 调优 SVR 模型参数...")
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR

    svr = SVR()
    parameters = {
        'kernel': ['rbf', 'linear'],
        'C': [1, 10],
        'epsilon': [0.1, 0.2]
    }
    # 设置 verbose=2 显示调优进度
    svr_grid = GridSearchCV(svr, parameters, cv=3, n_jobs=-1, verbose=2)
    svr_grid.fit(X_train_scaled, y_train[:, 0])  # 这里只调优经度的模型，纬度类似
    print(f"最佳参数: {svr_grid.best_params_}")
    y_pred_svr = svr_grid.predict(X_test_scaled)
    mse_svr = mean_squared_error(y_test[:, 0], y_pred_svr)
    mae_svr = mean_absolute_error(y_test[:, 0], y_pred_svr)
    r2_svr = r2_score(y_test[:, 0], y_pred_svr)
    print(f"SVR 模型评估结果（经度）：")
    print(f"MSE: {mse_svr}, MAE: {mae_svr}, R2: {r2_svr}")

    # 4. 简单的深度神经网络
    print("训练简单的深度神经网络...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2))
    model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3))

    # verbose=1 已经显示训练进度，不需要额外处理
    model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=10, batch_size=256, verbose=1)
    y_pred_nn = model.predict(X_test_scaled)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    print(f"深度神经网络模型评估结果：")
    print(f"MSE: {mse_nn}, MAE: {mae_nn}, R2: {r2_nn}")
