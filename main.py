# main.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import train_autoencoder, extract_features, train_and_evaluate_svr, compute_error_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def main():
    """
    主程序，执行数据加载、模型训练和评估的全过程。
    """
    # 检查是否有可用的 GPU，如果有则使用 GPU 训练模型，否则使用 CPU，输出详细的用于计算的设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 数据加载与预处理
    train_path = 'UJIndoorLoc/trainingData1.csv'
    test_path = 'UJIndoorLoc/validationData1.csv'
    print("加载并预处理数据...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y = load_and_preprocess_data(train_path, test_path)

    # 2. 初始化 Transformer 自编码器模型
    print("初始化 Transformer 自编码器模型...")
    model = WiFiTransformerAutoencoder().to(device)

    # 3. 训练 Transformer 自编码器模型
    print("训练 Transformer 自编码器模型...")
    model = train_autoencoder(
        model, X_train, X_val,
        device=device,
        epochs=50,
        batch_size=256,
        learning_rate=2e-4,
        early_stopping_patience=5
    )

    # 4. 提取特征
    print("提取训练集和测试集特征...")
    X_train_features = extract_features(model, X_train, device=device, batch_size=256)
    X_test_features = extract_features(model, X_test, device=device, batch_size=256)

    # 5. 训练和评估 SVR 模型
    print("训练和评估 SVR 回归模型...")
    # 逆标准化目标变量进行训练和评估
    y_train_original = scaler_y.inverse_transform(y_train)
    y_test_original = scaler_y.inverse_transform(y_test)

    # 定义 SVR 参数
    svr_params = {
        'kernel': 'rbf',
        'C': 1,
        'epsilon': 0.1
    }

    # 训练 MultiOutputRegressor SVR
    best_svr_model = train_and_evaluate_svr(X_train_features, y_train_original, X_test_features, y_test_original, svr_params=svr_params)

    # 预测并评估
    y_pred = best_svr_model.predict(X_test_features)

    mse = mean_squared_error(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)

    # 计算误差距离
    error_distances = compute_error_distances(y_test_original, y_pred)
    mean_error_distance = np.mean(error_distances)
    median_error_distance = np.median(error_distances)

    print(f"SVR 回归模型评估结果：")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R^2 Score: {r2:.6f}")
    print(f"平均误差距离（米）: {mean_error_distance:.2f}")
    print(f"中位数误差距离（米）: {median_error_distance:.2f}")

    # 保存模型和其他结果
    torch.save(model.state_dict(), 'best_transformer_autoencoder.pth')
    joblib.dump(best_svr_model, 'best_svr_model.pkl')

if __name__ == '__main__':
    main()
