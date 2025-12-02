import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEMA(nn.Module):
    """
    一个使用指数移动平均（EMA）的孪生网络模型，常用于自监督学习（如MoCo）。
    """
    def __init__(self, base_encoder, dim=1024, out_dim=512, momentum=0.999) -> None:
        """
        初始化SiameseEMA模型。

        参数:
            base_encoder (nn.Module): 基础编码器（例如，ResNet）。
            dim (int): 投影头中间层的维度。
            out_dim (int): 最终输出的特征维度。
            momentum (float): EMA更新的动量。
        """
        super(SiameseEMA, self).__init__()
        self.momentum = momentum

        # 查询编码器 (encoder_q)，包含基础编码器和一个MLP投影头
        self.encoder_q = nn.Sequential(
            base_encoder,
            nn.Linear(out_dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),   # 隐藏层
            nn.Linear(dim, out_dim), # 输出层
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),   # 隐藏层
            nn.Linear(out_dim, out_dim) # 输出层
            )

        # 键编码器 (encoder_k)，结构与查询编码器相同，但参数通过动量更新
        self.encoder_k = copy.deepcopy(self.encoder_q)


    def forward(self, x1, x2):
        """
        模型的前向传播。

        参数:
            x1 (torch.Tensor): 查询图像。
            x2 (torch.Tensor): 键图像。

        返回:
            torch.Tensor, torch.Tensor: 查询向量 (q) 和键向量 (k)。
        """
        # 计算查询向量
        q = self.encoder_q(x1)
        q = F.normalize(q, dim=1) # L2归一化

        # 计算键向量（不计算梯度）
        with torch.no_grad():
            self._momentum_update() # 更新键编码器
            k = self.encoder_k(x2)
            k = F.normalize(k, dim=1) # L2归一化

        return q, k


    def _momentum_update(self):
        """
        使用动量更新键编码器的参数。
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


def infoNCELoss(q, k, queue, T):
    """
    计算InfoNCE损失函数。

    参数:
        q (torch.Tensor): 查询向量。
        k (torch.Tensor): 对应的正样本键向量。
        queue (torch.Tensor): 负样本队列。
        T (float): 温度超参数。

    返回:
        torch.Tensor: 计算出的InfoNCE损失。
    """
    # 计算正样本的点积 (logits)
    # l_pos: Nx1
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    # 计算负样本的点积 (logits)
    # l_neg: NxK
    l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])
    # 拼接正负样本的logits
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # 应用温度超参数
    logits /= T

    # 创建标签，正样本的索引为0
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

    # 计算交叉熵损失
    return F.cross_entropy(logits, labels)
