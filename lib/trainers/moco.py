from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from lib.models.feature_extractors import get_feature_extractor
from lib.models.siamese_ema import SiameseEMA, infoNCELoss
from lib.trainers.base import BaseTrainer


class MocoTrainer(BaseTrainer):
    """
    用于MoCo（Momentum Contrast）自监督学习的训练器。
    """
    def __init__(self, train_loader, val_loader, time_window_scheduler, args) -> None:
        """
        初始化MocoTrainer。

        参数:
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            time_window_scheduler: 用于动态调整时间窗口的调度器。
            args (Namespace): 包含所有配置参数的对象。
        """
        # 初始化SiameseEMA模型
        self.model = SiameseEMA(
            base_encoder=get_feature_extractor(args.backbone),
            out_dim=args.out_dim,
        )
        self.model = self.model.to(args.device)

        # 调用父类（BaseTrainer）的构造函数
        super().__init__(train_loader, val_loader, args)

        # 如果使用wandb，则初始化wandb
        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="moco",
                name=args.run_name,
                config=args.__dict__,
            )

        # 初始化用于存储负样本的队列
        self.queue_size = args.queue_size
        self.queue = torch.randn(args.out_dim, self.queue_size, device=args.device)
        self.queue = nn.functional.normalize(self.queue, dim=0) # 归一化队列
        self.queue_ptr = torch.zeros(1, dtype=torch.long) # 队列指针

        # 为验证集也创建一个队列
        self.queue_val = torch.randn(args.out_dim, 128, device=args.device)
        self.queue_val = nn.functional.normalize(self.queue_val, dim=0)
        self.queue_ptr_val = torch.zeros(1, dtype=torch.long)

        # InfoNCE损失的温度超参数
        self.T = args.temperature
        # 时间窗口调度器
        self.time_window_scheduler = time_window_scheduler


    def _train_batch_loss(self, batch, train=True):
        """
        计算单个批次的损失。
        """
        # 从批次中获取正样本对 (x1, x2)
        x1, x2 = batch
        # 通过模型前向传播得到查询向量q和键向量k
        q, k = self.model(x1.to(self.device), x2.to(self.device))

        # 根据是训练还是验证，选择对应的队列和指针
        queue = self.queue if train else self.queue_val
        queue_ptr = self.queue_ptr if train else self.queue_ptr_val

        # 计算InfoNCE损失
        loss = infoNCELoss(q, k, queue, self.T)

        # 将当前的键向量k入队
        self._dequeue_and_enqueue(k, queue, queue_ptr)

        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        """
        执行出队和入队操作，更新负样本队列。
        """
        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        queue_size = queue.shape[1]
        
        # 为简单起见，假设队列大小是批大小的整数倍
        assert queue_size % batch_size == 0

        # 在指针位置替换旧的键（出队），并存入新的键（入队）
        queue[:, ptr : ptr + batch_size] = keys.T
        # 移动指针
        ptr = (ptr + batch_size) % queue_size
        queue_ptr[0] = ptr


    def _run_train_epoch(self):
        """
        运行一个完整的训练轮次。
        """
        self.model.train() # 设置模型为训练模式
        # 使用tqdm创建进度条
        pbar = tqdm(self.train_loader, desc=f"训练 {self.epoch+1}/{self.num_epochs}")
        # 使用deque记录最近的损失值，用于计算滑动平均
        losses = dict(loss=deque(maxlen=self.log_interval))

        for i, batch in enumerate(pbar):
            self.opt.zero_grad() # 清空梯度
            loss = self._train_batch_loss(batch) # 计算损失
            loss.backward() # 反向传播
            self.opt.step() # 更新参数

            self.step += 1
            losses["loss"].append(loss.item())
            running_avg = np.mean(losses["loss"])
            # 更新进度条的后缀信息
            pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

            # 如果使用wandb，则记录训练损失
            if self.use_wandb and self.step % self.log_interval == 0:
                wandb.log(data={"train_loss": running_avg}, step=self.step)

        # 更新学习率和时间窗口
        self.lr_scheduler.step()
        self.time_window_scheduler.step()

    def _run_val_epoch(self):
        """
        运行一个完整的验证轮次。
        """
        self.model.eval() # 设置模型为评估模式
        pbar = tqdm(self.val_loader, desc=f"评估 {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list())

        with torch.no_grad(): # 在评估期间不计算梯度
            for i, batch in enumerate(pbar):
                loss = self._train_batch_loss(batch, train=False) # 计算验证损失

                losses["loss"].append(loss.item())
                running_avg = np.mean(losses["loss"])
                pbar.set_postfix(dict(loss=loss.item(), avg=running_avg))

        # 如果使用wandb，则记录验证损失
        if self.use_wandb:
            wandb.log(data={"val_loss": np.mean(losses["loss"])}, step=self.step)
