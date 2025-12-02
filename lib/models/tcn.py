import torch.nn as nn
from pytorch_tcn import TCN


class TCNForecasting(nn.Module):
    """
    使用时间卷积网络（TCN）进行时间序列预测的模型。
    """
    def __init__(self,
                 input_size,
                 output_size) -> None:
        """
        初始化TCN预测模型。

        参数:
            input_size (int): 输入序列的特征维度。
            output_size (int): 输出预测的维度。
        """
        super().__init__()
        # 时间卷积网络（TCN）
        self.tcn = TCN(num_inputs=input_size,
                       num_channels=[1024, 1024, 1024, 1024, 1024], # 每个TCN层的通道数
                       #output_projection=output_size, # 输出投影层
                       causal=True, # 是否使用因果卷积
                       use_skip_connections=True, # 是否使用跳跃连接
                       input_shape="NLC") # 输入形状格式 (批大小, 序列长度, 通道数)

        # 自适应平均池化层，将序列输出转换为固定大小
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 多层感知机（MLP），用于最终的预测输出
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.Sigmoid(), # 使用Sigmoid激活函数将输出缩放到0-1之间
        )

    def forward(self, seq):
        """
        模型的前向传播。

        参数:
            seq (torch.Tensor): 输入的时间序列数据。

        返回:
            torch.Tensor: 模型的预测输出。
        """
        # 通过TCN网络
        y = self.tcn(seq)
        # (批大小, 序列长度, 通道数) -> (批大小, 通道数, 序列长度)
        # 通过自适应平均池化层
        y = self.avg_pool(y.transpose(1,2)).squeeze()
        # 通过MLP进行最终预测
        y = self.mlp(y)
        # 移除多余的维度并返回
        return y.squeeze()
