from abc import abstractmethod
from os import makedirs

import torch


class BaseTrainer:
    """
    一个抽象的训练器基类，定义了所有训练器通用的结构和方法。
    """
    def __init__(self,
                 train_loader,
                 val_loader,
                 args) -> None:
        """
        初始化训练器。

        参数:
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            args (Namespace): 包含所有配置参数的对象。
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 从参数中获取设备、学习率、批量大小等基本配置
        self.device = args.device
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.num_workers = args.num_workers

        # Weights & Biases (wandb) 相关配置
        self.use_wandb = args.use_wandb
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval

        # 设置并创建模型保存目录
        self.save_dir = f"{args.save_dir}/{args.experiment}/{args.run_name}"
        makedirs(self.save_dir, exist_ok=True)

        # 初始化优化器（AdamW）和学习率调度器（余弦退火）
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.num_epochs)

        # 如果指定了检查点，则加载；否则，从头开始训练
        if args.checkpoint is not None:
            self._load_checkpoint(args.checkpoint)
        else:
            self.model_params = list(self.model.parameters())
            self.name = args.run_name
            self.step = 0
            self.epoch = 0

        # 打印模型参数数量和检查点保存信息
        print(f"训练器准备就绪，模型参数量: {sum(p.numel() for p in self.model_params):,}")
        print(f"每 {self.save_interval} 轮保存一次检查点")

        # 早停（Early Stopping）相关配置
        self.early_stopping = args.es_patience > 0
        if self.early_stopping:
            self.best_val_loss = float("inf")
            self.patience = args.es_patience
            self.exasperation = 0 # “耐心耗尽”计数器
            print(f"本次运行将在验证集上连续 {args.es_patience} 轮无改善后停止")

    def train(self):
        """
        主训练循环。
        """
        train_epochs = range(self.epoch, self.num_epochs)
        for _ in train_epochs:
            # 运行一个训练轮次
            self._run_train_epoch()

            # 根据保存间隔保存检查点
            if self.epoch % self.save_interval == 0:
                self._save_checkpoint()

            # 运行一个验证轮次
            val_loss = self._run_val_epoch()
            self.epoch += 1

            # 检查是否需要早停
            if self.early_stopping:
                if val_loss < self.best_val_loss:
                    self.exasperation = 0
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best") # 保存最佳模型
                else:
                    self.exasperation += 1

                if self.exasperation == self.patience:
                    print("早停：已达到最大耐心值")
                    print(f"最佳验证轮次: {self.epoch - self.patience}")
                    break

            print(f"学习率: {self.lr_scheduler.get_last_lr()[0]:.5f}")

        # 训练结束后再保存一次最终模型
        self._save_checkpoint()

    @abstractmethod
    def _run_train_epoch(self):
        """
        （抽象方法）运行一个完整的训练轮次。子类必须实现此方法。
        """
        ...


    @abstractmethod
    def _run_val_epoch(self):
        """
        （抽象方法）运行一个完整的验证轮次。子类必须实现此方法。
        """
        ...


    def _save_checkpoint(self, name=None):
        """
        保存模型检查点。
        """
        # 对于MoCo实验，只保存主干编码器
        if "moco" in self.name:
            model_dict = self.model.encoder_q[0].state_dict()
        else:
            model_dict = self.model.state_dict()

        # 组织需要保存的数据
        data = dict(
            model_dict=model_dict,
            opt_dict=self.opt.state_dict(),
            epoch=self.epoch,
            step=self.step,
            name=self.name,
        )

        # 定义保存路径
        path = f"{self.save_dir}/checkpoint_{self.step if name is None else name}.pth"

        # 保存数据
        torch.save(data, path)
        print(f"检查点已保存至 {path}")


    def _load_checkpoint(self, path):
        """
        从文件加载模型检查点。
        """
        # 加载数据
        data = torch.load(path)
        # 恢复模型和优化器的状态
        self.model.load_state_dict(data["model_dict"])
        self.model_params = list(self.model.parameters())
        self.opt.load_state_dict(data["opt_dict"])
        # 恢复训练的轮次和步数
        self.epoch = data["epoch"]
        self.step = data["step"]
        # 更新运行名称以标识是恢复的训练
        self.name = f"{data['name']}_resumed"

        print("="*100)
        print(f"从检查点恢复训练: {path}")
        print("="*100)
