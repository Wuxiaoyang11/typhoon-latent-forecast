from os import makedirs

import numpy as np
import torch
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SimpleSeqDataset(Dataset):
    # ... 初始化 DigitalTyphoonDataset ...
    def __init__(self, prefix) -> None:
        # prefix is the path to the Digital Typhoon Dataset

        super().__init__()
        self.dataset = DigitalTyphoonDataset(
                get_images_by_sequence=True,        # <--- 关键点：按序列加载，而不是按单张图
                labels=[],
                filter_func= None,
                ignore_list=[],
                transform=None,
                verbose=False,
                image_dir=f"{prefix}/image/",
                metadata_dir=f"{prefix}/metadata/",
                metadata_json=f"{prefix}/metadata.json",
            )

    def __getitem__(self, index):
        return self.dataset.get_ith_sequence(index)     # 返回第 i 个完整的台风序列对象

    def __len__(self):
        return len(self.dataset)

def preprocess_images_sequences(model, out_dir, transform, device, dataset_path):
    """
    Projects all sequences of the Digital Typhoon Dataset

    Args:
        model (nn.Module): Image encoder
        out_dir (str): Directory to write all feature files
        transform (function): transform function to be applied to sequences
        device (str): device to use to for processung
    """
    # ... 创建输出目录 ...
    makedirs(out_dir, exist_ok=True)
    print(f"Writing feature files to {out_dir}")
    dataset = SimpleSeqDataset(dataset_path)
    # DataLoader 设置
    loader = DataLoader(dataset,
                        batch_size=1,   # <--- 必须是 1！因为每个台风的持续时长（序列长度）不一样，没法 batch 在一起
                        num_workers=16,
                        shuffle=False,
                        collate_fn=lambda x: x) # 不让 PyTorch 自动拼接，保持原样

    model.eval()        # 冻结模型（不训练）
    model = model.to(device)    # 放到 GPU 上
    with torch.no_grad():       # 不计算梯度（省显存，因为只是提取特征）
        for seq in tqdm(loader):
            seq = seq[0]        # 取出当前这个台风序列对象
            # 1. 加载该台风的所有原始图片
            images = seq.get_all_images_in_sequence()

            names = np.array([str(image.image_filepath).split("/")[-1].split(".")[0] for image in images])
            # 2. 预处理 (Resize, Normalize, 转 Tensor)
            # transform 函数是在外部定义的，通常包含单通道转三通道、归一化等
            images = torch.stack([transform(image.image()) for image in images])

            images = images.to(device)
            # 3.通过 CNN 模型提取特征
            # 输入是一堆图片，输出是一堆向量
            features = model(images).cpu().numpy()
            # 4. 存盘 (.npz 文件)
            # 文件名是台风的 ID (如 198001)
            np.savez(f"{out_dir}/{seq.sequence_str}", features, names)

