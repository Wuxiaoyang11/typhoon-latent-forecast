import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """
    一个用于时间序列预测的LSTM（长短期记忆）模型。
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        初始化LSTM模型。

        参数:
            input_size (int): 输入序列的特征维度。
            hidden_size (int): LSTM隐藏层的维度。
            num_layers (int): LSTM的层数。
            output_size (int): 最终输出的维度。
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM层
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True, # 输入和输出张量的格式为 (批大小, 序列长度, 特征维度)
                            dropout=0.1) # 在除最后一层外的每层LSTM输出上应用dropout
        # 全连接层，用于将LSTM的输出映射到最终的预测维度
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        """
        模型的前向传播。

        参数:
            x (torch.Tensor): 输入的时间序列数据。

        返回:
            torch.Tensor: 模型的预测输出。
        """
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        # 我们只取序列最后一个时间步的输出，并通过全连接层进行预测
        out = self.fc(out[:, -1, :]).squeeze()

        return out
