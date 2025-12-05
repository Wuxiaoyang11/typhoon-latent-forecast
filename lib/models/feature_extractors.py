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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from lib.models.siamese_ema import SiameseEMA


ssl._create_default_https_context = ssl._create_unverified_context


def _load_checkpoint(model, path):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_dict"])          # 提取 "model_dict" 键值下的权重

    print("="*100)
    print(f"Loading model from checkpoint {path}")
    print("="*100)

    return model


def _wrap_model_1to3channels(model):
    return nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=1, bias=False),
        model
    )


def get_resnet18():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)         # 加载预训练权重
    model.fc = nn.Identity()                                         # 去掉全连接层
    return model


def get_resnet18_3channels():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model

def get_resnet34():
    model = resnet34(weights=None)
    model.fc = nn.Identity()

    return _wrap_model_1to3channels(model)

def get_resnet50():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()

    return _wrap_model_1to3channels(model)

def get_vgg11():
    model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT)
    #print(model.classifier[-1])
    del model.classifier[-1]
    return _wrap_model_1to3channels(model)


# 定义获取函数 (仿照 get_resnet/get_vgg 写法)
def get_efficientnet_b0():
    # 加载带有 ImageNet 预训练权重的模型
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # --- 关键修改：去掉分类头 ---
    # EfficientNet 的分类头叫 'classifier'，是一个 Sequential 结构
    # 我们把它替换为 Identity，这样输出的就是池化后的特征向量
    model.classifier = nn.Identity()

    # --- 关键修改：加上单通道适配器 ---
    # 因为 EfficientNet 第一层也需要 3 通道输入
    return _wrap_model_1to3channels(model)

#使用MoCo
# def get_moco_encoder(backbone: str, weights: str, dataparallel=False):
#     model = SiameseEMA(
#             base_encoder=get_feature_extractor(backbone),
#             out_dim=384 if backbone == "vit_small" else 512,
#     )
#     if dataparallel:
#         model = nn.DataParallel(model)
#     model = _load_checkpoint(model, weights)    # 加载预训练权重
#     if dataparallel:
#         model = model.module.encoder_q[0]
#     else:
#         model = model.encoder_q[0]
#     model.eval()
#
#     return model                                 # 返回用于提取特征的编码器

#不使用MoCo
def get_encoder(backbone: str, weights: str):
    model = get_feature_extractor(backbone)             # 1. 获取模型骨架
    model = _load_checkpoint(model, weights)            # 2. 加载指定路径的权重
    model.eval()                                        # 3. 设为评估模式 (冻结 BatchNorm 等)

    return model

_feature_extractors = dict(
    resnet18=get_resnet18,
    resnet18_3c=get_resnet18_3channels,
    resnet34=get_resnet34,
    resnet50=get_resnet50,
    vgg11=get_vgg11,
    efficientnet_b0=get_efficientnet_b0,
)


def get_feature_extractor(name):
    return _feature_extractors[name]()
