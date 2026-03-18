"""
Vision Mamba (Vim) - 基于 Mamba SSM 的视觉特征提取器
参考：https://github.com/hustvl/Vim
"""
import math
from functools import partial

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Please install mamba-ssm: pip install mamba-ssm")


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b]", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PatchEmbed(nn.Module):
    """图像转 Patch 嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MambaBlock(nn.Module):
    """Mamba Block"""
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path=0., 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mamba(self.norm(x)))
        return x


class DropPath(nn.Module):
    """Stochastic Depth"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class VisionMamba(nn.Module):
    """
    Vision Mamba 特征提取器
    
    Args:
        img_size: 输入图像尺寸 (默认 224x224)
        patch_size: Patch 大小 (默认 16)
        in_chans: 输入通道数 (默认 1，适用于红外图像)
        embed_dim: 嵌入维度
        depth: Mamba 层数
        d_state: Mamba 状态维度
        d_conv: Mamba 卷积核大小
        expand: Mamba 扩展因子
        drop_path_rate: Stochastic depth 比率
        norm_layer: 归一化层
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=1, 
                 embed_dim=768, 
                 depth=12,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        # Patch 嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Mamba 层
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, C, H, W), C=1 为单通道红外图像
        Returns:
            特征向量 (B, embed_dim)
        """
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # 返回 CLS token 作为全局特征
        return x[:, 0]


def vim_tiny(patch_size=16, **kwargs):
    """Vision Mamba Tiny"""
    model = VisionMamba(
        patch_size=patch_size, 
        embed_dim=192, 
        depth=12,
        d_state=16,
        d_conv=4,
        expand=2,
        in_chans=1,  # 单通道红外图像
        **kwargs
    )
    return model


def vim_small(patch_size=16, **kwargs):
    """Vision Mamba Small"""
    model = VisionMamba(
        patch_size=patch_size, 
        embed_dim=384, 
        depth=12,
        d_state=16,
        d_conv=4,
        expand=2,
        in_chans=1,  # 单通道红外图像
        **kwargs
    )
    return model


def vim_base(patch_size=16, **kwargs):
    """Vision Mamba Base"""
    model = VisionMamba(
        patch_size=patch_size, 
        embed_dim=768, 
        depth=12,
        d_state=16,
        d_conv=4,
        expand=2,
        in_chans=1,  # 单通道红外图像
        **kwargs
    )
    return model
