import ssl

import torch
import torch.nn as nn
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)
from torchvision.models.vgg import VGG11_BN_Weights, vgg11_bn

from lib.models.siamese_ema import SiameseEMA

# 解决在某些环境下下载预训练模型时可能出现的SSL证书验证问题
ssl._create_default_https_context = ssl._create_unverified_context


def _load_checkpoint(model, path):
    """
    从指定路径加载模型检查点。

    Args:
        model (nn.Module): 需要加载权重的目标模型。
        path (str): 检查点文件的路径。

    Returns:
        nn.Module: 加载了权重的模型。
    """
    # 从CPU加载数据，以避免因设备不匹配（如在无GPU环境加载GPU模型）导致的问题
    data = torch.load(path, map_location="cpu")
    # 加载模型的状态字典
    model.load_state_dict(data["model_dict"])

    print("="*100)
    print(f"从检查点加载模型: {path}")
    print("="*100)

    return model


def _wrap_model_1to3channels(model):
    """
    将一个为单通道输入设计的模型包装起来，使其能够接受三通道的输入。
    这对于将在单通道图像（如卫星云图）上使用为三通道（如ImageNet）设计的预训练模型非常有用。

    Args:
        model (nn.Module): 原始的单通道模型。

    Returns:
        nn.Module: 包装后的、可以接受三通道输入的模型。
    """
    return nn.Sequential(
        # 添加一个1x1卷积层，它会学习如何将1个输入通道映射到3个输出通道
        nn.Conv2d(1, 3, kernel_size=1, bias=False),
        model
    )


def get_resnet18():
    """
    获取一个在ImageNet上预训练的ResNet-18模型，并移除其最后的分类层。

    Returns:
        nn.Module: 作为特征提取器的ResNet-18模型。
    """
    # 加载使用ImageNet V1权重预训练的ResNet-18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # 将全连接层（fc）替换为一个恒等映射层，这样模型的输出就是卷积部分的特征图
    model.fc = nn.Identity()
    return _wrap_model_1to3channels(model)


def get_resnet18_3channels():
    """
    获取一个为三通道输入设计的、在ImageNet上预训练的ResNet-18模型。

    Returns:
        nn.Module: 作为特征提取器的ResNet-18模型。
    """
    # 加载使用ImageNet V1权重预训练的ResNet-18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # 移除全连接层
    model.fc = nn.Identity()
    return model


def get_moco_encoder(backbone: str, weights: str, dataparallel=False):
    """
    加载一个通过MoCo自监督学习预训练的编码器。

    Args:
        backbone (str): 主干网络的名称 (例如, 'resnet34')。
        weights (str): 预训练权重的路径。
        dataparallel (bool): 模型是否使用了nn.DataParallel进行包装。

    Returns:
        nn.Module: 训练好的编码器（特征提取器）。
    """
    # 创建一个SiameseEMA模型实例，这是MoCo训练时使用的结构
    model = SiameseEMA(
            base_encoder=get_feature_extractor(backbone),
            out_dim=512,
    )
    # 如果训练时使用了DataParallel，则在加载权重前也需要包装
    if dataparallel:
        model = nn.DataParallel(model)
    # 加载预训练的权重
    model = _load_checkpoint(model, weights)
    # 从SiameseEMA结构中提取出真正的编码器部分 (encoder_q的主干网络)
    if dataparallel:
        model = model.module.encoder_q[0]
    else:
        model = model.encoder_q[0]
    # 将模型设置为评估模式，这会禁用dropout和batch normalization的更新
    model.eval()

    return model


def get_encoder(backbone: str, weights: str):
    """
    加载一个通用的、预训练好的编码器模型。

    Args:
        backbone (str): 主干网络的名称。
        weights (str): 预训练权重的路径。

    Returns:
        nn.Module: 训练好的编码器。
    """
    # 根据名称获取基础的特征提取器模型
    model = get_feature_extractor(backbone)
    # 加载指定的权重
    model = _load_checkpoint(model, weights)
    # 设置为评估模式
    model.eval()

    return model


def get_resnet34():
    """
    获取一个ResNet-34模型，并将其适配为单通道输入。

    Returns:
        nn.Module: 修改后的ResNet-34模型。
    """
    # 加载一个没有预训练权重的ResNet-34模型
    model = resnet34(weights=None)
    # 移除全连接层
    model.fc = nn.Identity()

    # 包装模型以接受单通道输入
    return _wrap_model_1to3channels(model)

def get_resnet50():
    """
    获取一个在ImageNet上预训练的ResNet-50模型，并将其适配为单通道输入。

    Returns:
        nn.Module: 修改后的ResNet-50模型。
    """
    # 加载使用默认ImageNet权重预训练的ResNet-50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # 移除全连接层
    model.fc = nn.Identity()

    # 包装模型以接受单通道输入
    return _wrap_model_1to3channels(model)

def get_vgg11():
    """
    获取一个在ImageNet上预训练的VGG-11模型，并将其适配为单通道输入。

    Returns:
        nn.Module: 修改后的VGG-11模型。
    """
    # 加载带批量归一化（Batch Normalization）的预训练VGG-11
    model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
    # 移除分类器部分的最后一层（通常是输出层）
    del model.classifier[-1]
    # 包装模型以接受单通道输入
    return _wrap_model_1to3channels(model)


# 一个字典，用于将字符串名称映射到对应的特征提取器获取函数
_feature_extractors = dict(
    resnet18=get_resnet18,
    resnet18_3c=get_resnet18_3channels,
    resnet34=get_resnet34,
    resnet50=get_resnet50,
    vgg11=get_vgg11,
)


def get_feature_extractor(name):
    """
    根据给定的名称，从字典中查找并返回一个特征提取器模型实例。

    Args:
        name (str): 特征提取器的名称。

    Returns:
        nn.Module: 对应的特征提取器模型。
    """
    # 从字典中获取函数并调用，返回模型实例
    return _feature_extractors[name]()
