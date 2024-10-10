# model_definition.py

import torch
import torch.nn as nn

class WiFiTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=520, model_dim=128, num_heads=8, num_layers=2, dropout=0.1):
        """
        定义用于特征提取的 Transformer 自编码器模型。

        参数：
        - input_dim: 输入特征维度，默认为 520
        - model_dim: 模型内部特征维度，默认为 128
        - num_heads: 多头注意力头数，默认为 8
        - num_layers: Transformer 编码器层数，默认为 2
        - dropout: Dropout 概率，默认为 0.1
        """
        super(WiFiTransformerAutoencoder, self).__init__()
        self.encoder_embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)
        self.decoder_fc = nn.Linear(model_dim, input_dim)
        self.activation = nn.ReLU()  # 保持 ReLU 激活函数

        # 移除自定义初始化，使用 PyTorch 默认初始化

    def encode(self, x):
        """
        编码器部分，提取特征。

        参数：
        - x: 输入张量，形状为 [batch_size, input_dim]

        返回：
        - x: 编码后的特征，形状为 [batch_size, model_dim]
        """
        x = self.encoder_embedding(x)
        x = self.activation(x)
        x = x.unsqueeze(1)  # [batch_size, 1, model_dim]
        x = self.transformer_encoder(x)  # [batch_size, 1, model_dim]
        x = x.permute(0, 2, 1)  # [batch_size, model_dim, 1]
        x = self.encoder_pool(x).squeeze(-1)  # [batch_size, model_dim]
        return x

    def decode(self, x):
        """
        解码器部分，重构输入。

        参数：
        - x: 编码后的特征，形状为 [batch_size, model_dim]

        返回：
        - x: 重构后的输入，形状为 [batch_size, input_dim]
        """
        x = self.decoder_fc(x)
        return x

    def forward(self, x):
        """
        前向传播函数，完成自编码器的编码和解码。

        参数：
        - x: 输入张量，形状为 [batch_size, input_dim]

        返回：
        - x: 重构后的输入，形状为 [batch_size, input_dim]
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded