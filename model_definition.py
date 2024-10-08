import torch
import torch.nn as nn


class WiFiTransformer(nn.Module):
    """
    定义用于特征提取和回归的 Transformer 模型。

    参数：
    - input_dim: 输入特征维度，默认为 520
    - model_dim: 模型内部特征维度，默认为 128
    - num_heads: 多头注意力头数，默认为 8
    - num_layers: Transformer 编码器层数，默认为 2
    """

    def __init__(self, input_dim=520, model_dim=128, num_heads=8, num_layers=2):
        super(WiFiTransformer, self).__init__()
        # 线性层，将输入特征映射到模型维度
        self.embedding = nn.Linear(input_dim, model_dim)
        # 定义 Transformer 编码器层，设置 batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 自适应平均池化层，将序列长度维度缩减为 1
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 输出层，回归预测经度和纬度
        self.fc_out = nn.Linear(model_dim, 2)

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入张量，形状为 [batch_size, input_dim]

        返回：
        - x: 输出预测值，形状为 [batch_size, 2]
        """
        # 将输入映射到模型维度
        x = self.embedding(x)  # [batch_size, model_dim]
        # 添加序列长度维度，seq_len=1
        x = x.unsqueeze(1)  # [batch_size, seq_len=1, model_dim]
        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)  # [batch_size, seq_len=1, model_dim]
        # 调整维度以便池化
        x = x.permute(0, 2, 1)  # [batch_size, model_dim, seq_len=1]
        # 自适应平均池化，去除序列长度维度
        x = self.pool(x).squeeze(-1)  # [batch_size, model_dim]
        # 通过全连接层，输出经度和纬度的预测值
        x = self.fc_out(x)  # [batch_size, 2]
        return x
