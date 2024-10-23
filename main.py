# main.py

import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformerAutoencoder
from training_and_evaluation import (
    train_autoencoder,
    extract_features,
    train_and_evaluate_regression_model,
    train_and_evaluate_classification_model,
    NaNLossError
)
import joblib
import json

def main():
    """
    主程序，执行数据加载、模型训练和评估的全过程。
    """
    try:
        # 检查是否有可用的 GPU，如果有则使用 GPU 训练模型，否则使用 CPU，输出详细的用于计算的设备信息
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # 1. 数据加载与预处理
        train_path = 'UJIndoorLoc/trainingData.csv'  # 请确保文件路径正确
        test_path = 'UJIndoorLoc/validationData.csv'
        print("Loading and preprocessing data...")
        X_train, y_train_coords, y_train_floor, X_val, y_val_coords, y_val_floor, X_test, y_test_coords, y_test_floor, scaler_X, scaler_y, label_encoder = load_and_preprocess_data(train_path, test_path)

        # 2. 初始化 Transformer 自编码器模型
        print("Initializing Transformer Autoencoder model...")
        model = WiFiTransformerAutoencoder(
            input_dim=520,
            model_dim=128,
            num_heads=8,
            num_layers=2,
            dropout=0.1
        ).to(device)

        # 3. 训练 Transformer 自编码器模型
        print("Training Transformer Autoencoder model...")
        model, train_loss_list, val_loss_list = train_autoencoder(
            model, X_train, X_val,
            device=device,
            epochs=50,
            batch_size=256,
            learning_rate=2e-4,
            early_stopping_patience=5
        )

        # 4. 提取特征
        print("Extracting features from training and testing sets...")
        X_train_features = extract_features(model, X_train, device=device, batch_size=256)
        X_val_features = extract_features(model, X_val, device=device, batch_size=256)
        X_test_features = extract_features(model, X_test, device=device, batch_size=256)

        # 5. 逆标准化坐标目标变量
        y_train_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_train_coords[:, 0].reshape(-1, 1))
        y_train_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_train_coords[:, 1].reshape(-1, 1))
        y_train_coords_original = np.hstack((y_train_longitude_original, y_train_latitude_original))

        y_val_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_val_coords[:, 0].reshape(-1, 1))
        y_val_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_val_coords[:, 1].reshape(-1, 1))
        y_val_coords_original = np.hstack((y_val_longitude_original, y_val_latitude_original))

        y_test_longitude_original = scaler_y['scaler_y_longitude'].inverse_transform(y_test_coords[:, 0].reshape(-1, 1))
        y_test_latitude_original = scaler_y['scaler_y_latitude'].inverse_transform(y_test_coords[:, 1].reshape(-1, 1))
        y_test_coords_original = np.hstack((y_test_longitude_original, y_test_latitude_original))

        # 6. 定义 SVR 参数（可以根据需要调整）
        svr_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }

        # 7. 训练和评估回归模型
        print("Training and evaluating regression model...")
        regression_model, mean_error_distance = train_and_evaluate_regression_model(
            X_train_features, y_train_coords_original,
            X_test_features, y_test_coords_original,
            svr_params=svr_params,
            training_params={
                'model_dim': model.encoder_embedding.out_features,
                'num_heads': model.transformer_encoder.layers[0].self_attn.num_heads,
                'num_layers': len(model.transformer_encoder.layers),
                'dropout': model.transformer_encoder.layers[0].dropout.p,
                'learning_rate': 2e-4,
                'batch_size': 256,
                'early_stopping_patience': 5,
                'svr_kernel': svr_params['kernel'],
                'svr_C': svr_params['C'],
                'svr_epsilon': svr_params['epsilon'],
                'svr_gamma': svr_params['gamma']
            },
            train_loss_list=train_loss_list,
            val_loss_list=val_loss_list
        )

        # 8. 定义 SVC 参数（可以根据需要调整）
        svc_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale'
        }

        # 9. 训练和评估分类模型
        print("Training and evaluating classification model...")
        classification_model, accuracy = train_and_evaluate_classification_model(
            X_train_features, y_train_floor,
            X_test_features, y_test_floor,
            svc_params=svc_params,
            training_params={
                'model_dim': model.encoder_embedding.out_features,
                'num_heads': model.transformer_encoder.layers[0].self_attn.num_heads,
                'num_layers': len(model.transformer_encoder.layers),
                'dropout': model.transformer_encoder.layers[0].dropout.p,
                'learning_rate': 2e-4,
                'batch_size': 256,
                'early_stopping_patience': 5,
                'svc_C': svc_params['C'],
                'svc_kernel': svc_params['kernel'],
                'svc_gamma': svc_params['gamma']
            },
            train_loss_list=train_loss_list,
            val_loss_list=val_loss_list,
            label_encoder=label_encoder
        )

        # 10. 保存模型和其他结果
        torch.save(model.state_dict(), 'best_transformer_autoencoder.pth')
        joblib.dump(regression_model, 'coordinate_regression_model.pkl')
        joblib.dump(classification_model, 'floor_classification_model.pkl')

        # 另存缩放器和标签编码器
        joblib.dump(scaler_X, 'scaler_X.pkl')
        joblib.dump(scaler_y, 'scaler_y.pkl')
        joblib.dump(label_encoder, 'floor_label_encoder.pkl')

        # 保存预测结果（可选）
        y_pred_coords = regression_model.predict(X_test_features)
        y_pred_floor = classification_model.predict(X_test_features)
        y_pred_floor_labels = label_encoder.inverse_transform(y_pred_floor)

        # 合并预测结果
        y_pred_combined = np.hstack((y_pred_coords, y_pred_floor_labels.reshape(-1, 1)))
        np.savetxt('y_pred_final.csv', y_pred_combined, delimiter=',', header='LONGITUDE,LATITUDE,FLOOR', comments='')

        print("All models have been trained and results are saved.")

    except Exception as e:
        print(f"An unhandled exception occurred: {e}")

if __name__ == '__main__':
    main()
