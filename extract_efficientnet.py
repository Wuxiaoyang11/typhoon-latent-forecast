import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# 导入您提供的工具函数
from lib.utils.preprocessing import preprocess_images_sequences


# === 自定义变换类：单通道转三通道 ===
# 作用：将灰度图复制3份，变成伪彩色(RGB)，以适配 ImageNet 预训练模型
class To3Channels(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        # img shape: [1, H, W] -> [3, H, W]
        return img.repeat(3, 1, 1)


def main():
    # === 1. 定义图像变换流程 (Transforms) ===
    # 这里的顺序非常重要
    transforms = T.Compose([
        T.ToTensor(),  # 转为 Tensor (C, H, W)
        T.CenterCrop(224),  # 核心步骤：裁剪中心 224x224 (依据 V1 论文结论)
        To3Channels(),  # 关键修改：1通道 -> 3通道 (适配 EfficientNet)

        # 新增：ImageNet 标准归一化
        # 因为我们用的是 ImageNet 权重，输入分布必须和 ImageNet 一致
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # === 2. 定义具体的物理预处理函数 ===
    def transform_func(img):
        # 输入 img 是 numpy array, 原始数值为开尔文温度(Kelvin)

        # 物理归一化：将 150K - 350K 映射到 0 - 1
        # 这是基于气象学常识：云顶温度通常在此范围内
        img_range = [150, 350]
        img = (img - img_range[0]) / (img_range[1] - img_range[0])

        # 转换为 float32 防止报错
        img = img.astype(np.float32)

        # 应用上面的 transforms (ToTensor -> Crop -> 3Channels -> Normalize)
        img = transforms(img)
        return img

    # === 3. 准备模型 (EfficientNet-B0) ===
    print("Loading EfficientNet-B0 with ImageNet weights...")

    # 直接加载官方预训练权重
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # 关键修改：去掉分类头 (Classifier Head)
    # 我们不需要它做 1000 类分类，只需要提取特征
    # 将分类器替换为 Identity (恒等映射)，输出层即为池化后的 1280 维向量
    model.classifier = nn.Identity()

    # 注意：这里我们不需要 feature_extractors.py 里的 _wrap_model_1to3channels
    # 因为我们在 transform 里已经手动转成 3 通道了，直接喂给原版模型即可

    print(f"Model ready! Output dimension should be 1280.")

    # === 4. 配置路径并执行 ===
    # 请修改为您实际的数据集路径
    dataset_path = "/home/wxy/dataset/WP"

    # 输出目录名 (建议重命名以区分原版 ResNet)
    output_dir = "efficientnet_b0_imagenet_features"

    # 开始提取
    # device="cuda:0" 请根据您的显卡编号修改
    preprocess_images_sequences(model,
                                output_dir,
                                transform_func,
                                device="cuda:0",
                                dataset_path=dataset_path)


if __name__ == "__main__":
    main()