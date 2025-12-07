import torch
import torch.nn.functional as F

# 生成一个用于鱼眼变换的网格
def fisheye_grid(width, height, alpha):
    """
    生成一个应用了鱼眼效果的采样网格。

    参数:
        width (int): 网格的宽度。
        height (int): 网格的高度。
        alpha (float): 鱼眼效果的强度。值越大，畸变越强。

    返回:
        torch.Tensor: 形状为 (height, width, 2) 的鱼眼变换网格。
    """
    # 创建一个标准化的坐标网格，范围从-1到1
    x, y = torch.meshgrid(torch.linspace(-1, 1, width), torch.linspace(-1, 1, height), indexing="ij")
    coords = torch.stack((y, x), dim=-1)

    # 对坐标应用鱼眼变换
    # 计算每个点到中心的距离
    r = torch.sqrt(coords[:, :, 0]**2 + coords[:, :, 1]**2)
    # 计算径向缩放因子
    radial_scale = torch.pow(r, alpha)
    # 处理中心点r=0的情况，避免除以零
    radial_scale[r == 0] = 1.0
    # 将缩放因子应用到坐标上
    fisheye_coords = coords * torch.unsqueeze(radial_scale, -1)

    # 将变换后的坐标限制在[-1, 1]的范围内
    fisheye_coords = torch.clamp(fisheye_coords, min=-1, max=1)

    return fisheye_coords

# 一个PyTorch模块，用于将鱼眼变换应用于图像
class FishEye(torch.nn.Module):
    def __init__(self, size, alpha):
        """
        初始化鱼眼变换模块。

        参数:
            size (int): 输入图像的尺寸（假设为正方形）。
            alpha (float): 鱼眼效果的强度。
        """
        super().__init__()
        # 预先计算鱼眼网格
        self.grid = fisheye_grid(size, size, alpha)

    def forward(self, img):
        """
        对输入图像应用鱼眼变换。

        参数:
            img (torch.Tensor): 输入图像张量，形状可以是 (C, H, W) 或 (B, C, H, W)。

        返回:
            torch.Tensor: 经过鱼眼变换后的图像张量。
        """
        # 如果输入图像没有batch维度，则添加一个
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        B, _, _, _ = img.shape
        # 将预计算的网格扩展到与输入batch相同的大小
        # 使用 F.grid_sample 进行重采样，实现鱼眼效果
        fish = F.grid_sample(img, self.grid.unsqueeze(0).repeat(B, 1, 1, 1), align_corners=True).squeeze(0)
        return fish
