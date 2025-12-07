"""
本脚本用于在测试集上评估已训练好的 LSTM 模型，并生成预测结果报告和可视化图表。
包含自动反归一化 (Denormalization) 功能，将结果还原为 hPa 单位。
"""
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# 项目内部引用
from parse import parse_args
from lib.utils.dataloaders import get_dataloaders
from lib.models.lstm_predictor import LSTM

# =========================================================
# 1. 强力补丁：强制指定配置文件 (解决 KeyError: None)
# =========================================================
# 检查是否传入了 config_file，如果没有，强行加上
has_config = False
for arg in sys.argv:
    if "--config_file" in arg:
        has_config = True
        break

if not has_config:
    print("⚠️ 未检测到配置文件参数，正在强制加载默认配置...")
    sys.argv.append("--config_file")
    # 确保路径正确指向你的配置文件
    sys.argv.append("configs/train_lstm.conf")

# =========================================================
# 2. 配置区域
# =========================================================
# 使用相对路径，指向你仅存的 checkpoint_1000.pth
CHECKPOINT_PATH = "models/ts/lstm_efficientnet/checkpoint_best.pth"
PLOT_SAVE_PATH = "result/prediction_result.png"

# === 归一化参数 (反归一化用) ===
# 你的预测值在 -3 到 1 之间，说明是 Z-Score 标准化。
# 这里使用西太平洋台风气压的经验统计值。
# 如果你的 dataset.py 里有准确的 mean/std，请在这里修改。
NORM_MEAN = 965.0  # 气压均值 (hPa)
NORM_STD = 20.0    # 气压标准差 (hPa)


def denormalize(data):
    """
    将模型输出的归一化数值还原为真实的 hPa 气压值。
    公式: Real = Norm * Std + Mean
    """
    return data * NORM_STD + NORM_MEAN


def setup_environment():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载配置: {args.config_file}")
    print(f"使用设备: {args.device}")
    return args


def load_test_data(args):
    print("正在加载测试集数据 (这可能需要几秒钟)...")
    _, _, test_loader = get_dataloaders(args)
    print(f"测试集包含 {len(test_loader.dataset)} 个序列样本。")
    return test_loader


def build_and_load_model(args, test_loader):
    print("正在初始化 LSTM 模型...")
    # 自动获取输入输出维度
    input_size = test_loader.dataset.dataset.get_input_size()
    num_preds = test_loader.dataset.dataset.num_preds

    model = LSTM(
        input_size=input_size,
        hidden_size=args.hidden_dim, # 1024
        num_layers=args.num_layers,  # 3
        output_size=num_preds
    ).to(args.device)

    print(f"正在加载权重文件: {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=args.device)
        # base.py 保存时把参数包在了 'model_dict' 键里
        model.load_state_dict(checkpoint['model_dict'])
        print("✅ 权重加载成功！")
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {CHECKPOINT_PATH}")
        print("请确认 lib/models/ts/lstm_efficientnet/ 下是否存在 checkpoint_1000.pth")
        return None

    model.eval()
    return model


def run_inference(model, dataloader, device):
    all_preds = []
    all_targets = []
    print("🚀 开始在测试集上进行推理...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # 拼接所有 Batch
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return all_targets, all_preds


def report_metrics(targets, preds):
    """
    计算并打印评估指标 (反归一化后的真实 hPa)。
    """
    # 1. 形状处理 (N, 4, 1) -> (N, 4)
    if targets.ndim == 3:
        targets = targets.squeeze(-1)
    if preds.ndim == 3:
        preds = preds.squeeze(-1)

    # 2. 反归一化 (还原为真实气压)
    targets_hpa = denormalize(targets)
    preds_hpa = denormalize(preds)

    # 3. 计算总体指标
    mse = mean_squared_error(targets_hpa, preds_hpa)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_hpa, preds_hpa)

    print("\n" + "=" * 40)
    print("       🎉 最终测试报告 (Test Set) 🎉")
    print("=" * 40)
    print(f"RMSE (均方根误差): {rmse:.4f} hPa  <-- (越小越好)")
    print(f"MAE  (平均绝对误差): {mae:.4f} hPa")
    print("-" * 40)

    # 4. 计算分步指标 (Step-wise)
    num_steps = preds.shape[1]
    for i in range(num_steps):
        step_mse = mean_squared_error(targets_hpa[:, i], preds_hpa[:, i])
        step_rmse = np.sqrt(step_mse)
        print(f"Step {i+1} (未来 {(i+1)*3}h): RMSE = {step_rmse:.4f} hPa")
    print("=" * 40)

    return targets_hpa, preds_hpa


def plot_results(targets, preds, save_path):
    print("正在绘制预测对比图...")
    # 只取前 100 个样本画图
    subset_size = len(targets)

    # 如果是多步，只画第1步 (未来3小时)
    if preds.ndim > 1:
        targets_plot = targets[:, 0]
        preds_plot = preds[:, 0]
    else:
        targets_plot = targets
        preds_plot = preds

    t = np.arange(subset_size)

    plt.figure(figsize=(12, 6), dpi=100)
    # 真实值 (红实线)
    plt.plot(t, targets_plot[:subset_size], color='red', linestyle='-', linewidth=2, label='Ground Truth (Real)')
    # 预测值 (蓝虚线)
    plt.plot(t, preds_plot[:subset_size], color='blue', linestyle='--', linewidth=2, label='Prediction (LSTM)')

    plt.title('Typhoon Intensity Prediction (Central Pressure)', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Pressure (hPa)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 可视化图片已保存至: {save_path}")
    print("请在左侧文件列表双击打开该图片查看。")


def main():
    args = setup_environment()
    test_loader = load_test_data(args)
    model = build_and_load_model(args, test_loader)

    if model is None:
        return

    # 推理 -> 拿到归一化的数据
    raw_targets, raw_preds = run_inference(model, test_loader, args.device)

    # 报告 -> 内部会进行反归一化并打印真实 hPa 误差
    real_targets, real_preds = report_metrics(raw_targets, raw_preds)

    # 画图 -> 使用反归一化后的数据画图
    plot_results(real_targets, real_preds, PLOT_SAVE_PATH)


if __name__ == "__main__":
    main()