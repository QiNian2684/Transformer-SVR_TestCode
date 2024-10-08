import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def extract_features(model, X_data, batch_size=256, device='cpu'):
    """
    使用 Transformer 模型提取特征。

    参数：
    - model: 已定义的 Transformer 模型
    - X_data: 输入特征数据，形状为 [样本数, 特征维度]
    - batch_size: 批次大小，默认为 256
    - device: 设备类型，'cpu' 或 'cuda'

    返回：
    - features: 提取的特征数组，形状为 [样本数, 模型特征维度]
    """
    # 设置模型为评估模式
    model.eval()
    # 将输入数据转换为 Tensor，并创建数据加载器
    data_loader = DataLoader(TensorDataset(torch.tensor(X_data, dtype=torch.float32)), batch_size=batch_size)
    features = []
    # 禁用梯度计算，加速推理
    with torch.no_grad():
        for data in data_loader:
            # 获取输入，并将其移动到指定设备
            inputs = data[0].to(device)
            # 前向传播，提取特征
            outputs = model(inputs)
            # 将特征移动到 CPU 并转换为 NumPy 数组
            features.append(outputs.cpu().numpy())
    # 将所有批次的特征垂直堆叠，形成完整的特征矩阵
    return np.vstack(features)
