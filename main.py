import torch
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_definition import WiFiTransformer
from training_and_evaluation import train_transformer, evaluate_model, train_other_regressors


def main():
    """
    主程序，执行数据加载、模型训练和评估的全过程。
    """
    # 设置设备类型，若有 GPU 可用则使用 GPU，否则使用 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 数据加载与预处理
    train_path = 'UJIndoorLoc/trainingData.csv'  # 训练数据路径
    test_path = 'UJIndoorLoc/validationData.csv'  # 测试数据路径
    print("加载并预处理数据...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(train_path, test_path)

    # 2. 初始化 Transformer 模型
    print("初始化 Transformer 模型...")
    model = WiFiTransformer().to(device)

    # 3. 训练 Transformer 模型
    print("训练 Transformer 模型...")
    model = train_transformer(model, X_train, y_train, X_val, y_val, device, epochs=10, batch_size=256,
                              learning_rate=1e-3)

    # 4. 在测试集上评估 Transformer 模型
    print("在测试集上评估 Transformer 模型...")
    evaluate_model(model, X_test, y_test, device)

    # 5. 训练和评估其他回归模型
    print("训练和评估其他回归模型...")
    # 首先，从训练好的 Transformer 模型中提取特征
    print("从 Transformer 模型中提取特征...")
    model.eval()
    with torch.no_grad():
        X_train_features = model.embedding(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().numpy()
        X_test_features = model.embedding(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    # 训练其他回归模型
    train_other_regressors(X_train_features, y_train, X_test_features, y_test)

    # 如果需要保存模型和其他结果，可以在此处添加代码
    # 例如，保存模型参数：
    # torch.save(model.state_dict(), 'wifi_transformer.pth')


if __name__ == '__main__':
    main()
