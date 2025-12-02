from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import sigmoid
from tqdm import tqdm

import wandb
from lib.models.lstm_predictor import LSTM
from lib.models.tcn import TCNForecasting
from lib.trainers.base import BaseTrainer


class TimeSeriesTrainer(BaseTrainer):
    """
    用于时间序列预测任务的训练器。
    """
    def __init__(self, train_loader, val_loader, args) -> None:
        """
        初始化TimeSeriesTrainer。

        参数:
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            args (Namespace): 包含所有配置参数的对象。
        """
        # 根据配置参数选择并初始化时间序列模型（LSTM或TCN）
        if args.ts_model == "lstm":
            self.model = LSTM(
                # 从数据集中获取输入特征的维度
                input_size=train_loader.dataset.dataset.get_input_size(),
                hidden_size=args.hidden_dim,
                num_layers=args.num_layers,
                # 从数据集中获取预测目标的维度
                output_size=train_loader.dataset.dataset.num_preds
            ).to(args.device)
        elif args.ts_model == "tcn":
            self.model = TCNForecasting(
                input_size = train_loader.dataset.dataset.get_input_size(),
                output_size = train_loader.dataset.dataset.get_output_size(),
            ).to(args.device)
        else:
            raise NotImplementedError(f"不支持的时间序列模型: {args.ts_model}")

        # 调用父类（BaseTrainer）的构造函数
        super().__init__(train_loader, val_loader, args)

        # 如果使用wandb，则初始化wandb
        if self.use_wandb:
            wandb.init(
                project="typhoon",
                group="time-series",
                name=args.run_name,
                config=args.__dict__,
            )

        # 定义回归任务的损失函数（均方误差）
        self.reg_criterion = nn.MSELoss()
        # 获取标签名称
        self.labels = args.labels


    def _run_train_epoch(self):
        """
        运行一个完整的训练轮次。
        """
        self.model.train() # 设置模型为训练模式
        pbar = tqdm(self.train_loader, desc=f"训练 {self.epoch+1}/{self.num_epochs}")
        # 使用deque记录损失值
        losses = dict(loss=deque(maxlen=self.log_interval))

        for batch in pbar:
            self.opt.zero_grad() # 清空梯度
            # 将输入和输出数据移动到指定设备
            inp, outs = batch[0].to(self.device), batch[1].to(self.device)
            # 模型前向传播得到预测值
            preds = self.model(inp)
            
            # 特殊处理：如果标签是'grade'，则将其视为一个二分类问题（是否达到等级6）
            if "grade" in self.labels:
                outs = outs == 6 # 目标标签变为布尔值
                preds = sigmoid(preds) # 预测值通过sigmoid激活
            
            # 计算损失
            loss = self.reg_criterion(preds, (outs).float().squeeze())
            losses["loss"].append(loss.item())

            # 反向传播和参数更新
            loss.backward()
            self.opt.step()

            self.step += 1

            # 计算并显示平均损失
            avg = {f"tr_{key}": np.mean(val) for key, val in losses.items()}
            pbar.set_postfix(dict(loss=loss.item(), **avg))

            # 如果使用wandb，则记录训练损失
            if self.use_wandb and self.step % self.log_interval == 0:
                wandb.log(data=avg, step=self.step)

        # 更新学习率
        self.lr_scheduler.step()

    def _run_val_epoch(self):
        """
        运行一个完整的验证轮次。
        """
        self.model.eval() # 设置模型为评估模式
        pbar = tqdm(self.val_loader, desc=f"评估 {self.epoch+1}/{self.num_epochs}")
        losses = dict(loss=list())

        with torch.no_grad(): # 在评估期间不计算梯度
            for batch in pbar:
                inp, outs = batch[0].to(self.device), batch[1].to(self.device)

                preds = self.model(inp)

                # 与训练时相同的特殊处理
                if "grade" in self.labels:
                    outs = outs == 6
                    preds = sigmoid(preds)
                
                # 计算损失
                loss = self.reg_criterion(preds, (outs).float().squeeze())
                losses["loss"].append(loss.item())

                # 计算并显示平均损失
                avg = {f"ev_{key}": np.mean(val) for key, val in losses.items()}
                pbar.set_postfix(dict(loss=loss.item(), **avg))

        # 如果使用wandb，则记录验证损失
        if self.use_wandb:
            wandb.log(data=avg, step=self.step)

        # 返回平均验证损失，用于早停判断
        return np.mean(losses["loss"])
