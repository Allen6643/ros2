#!/usr/bin/env python3
"""
离线推理测试脚本（验证 model.pth 是否可用）
==========================================
**不启动 ROS、不启动 MuJoCo**，直接加载 `train.py` 训练出的 `model.pth`，
对数据集中的图像做前向推理，打印预测的 [前进推力, 转向] 并与标签对比。

这是作业的硬性要求之一：证明权重文件真实可用、可复现。

用法
----
    # 随机抽取 10 张数据集图像做推理并统计误差
    python3 src/water/rov_e2e/rov_e2e/test_infer.py

    # 指定权重、样本数与数据目录
    python3 src/water/rov_e2e/rov_e2e/test_infer.py \
        --model model.pth --samples 20 --data dataset

    # 只测单张任意图片（此时无标签，只打印预测值）
    python3 src/water/rov_e2e/rov_e2e/test_infer.py --image some_frame.png
"""

import argparse
import os
import sys

import numpy as np

try:
    import torch
except ImportError:
    sys.stderr.write('\n【环境依赖缺失】未检测到 PyTorch。请先 pip3 install torch torchvision\n\n')
    raise

# 允许直接以脚本方式运行（无需 colcon 安装）：把外层包目录加入 sys.path
try:
    from rov_e2e.train import (ROVSteeringNet, ROVDataset, preprocess_image,
                               DEFAULT_WIDTH, DEFAULT_HEIGHT)
except ImportError:
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from rov_e2e.train import (ROVSteeringNet, ROVDataset, preprocess_image,
                               DEFAULT_WIDTH, DEFAULT_HEIGHT)

from PIL import Image


def load_model(model_path, device):
    """加载 model.pth，返回 (model, width, height)。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'找不到权重文件 {model_path}\n'
            f'请先运行 train.py 训练：python3 src/water/rov_e2e/rov_e2e/train.py')

    ckpt = torch.load(model_path, map_location=device)
    if not isinstance(ckpt, dict) or 'state_dict' not in ckpt:
        raise RuntimeError(
            f'{model_path} 不是本脚本保存的权重格式（缺少 state_dict）。')

    width = int(ckpt.get('width', DEFAULT_WIDTH))
    height = int(ckpt.get('height', DEFAULT_HEIGHT))

    model = ROVSteeringNet().to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, width, height, ckpt


def infer_array(model, arr, device):
    """对已预处理好的 HWC float32 数组做推理，返回 (forward, turn)。"""
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)[0].cpu().numpy()
    return float(out[0]), float(out[1])


def main(argv=None):
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(pkg_dir, '..'))

    parser = argparse.ArgumentParser(description='离线测试 ROV 端到端控制网络')
    parser.add_argument('--model', default=os.path.join(default_root, 'model.pth'),
                        help='权重文件路径')
    parser.add_argument('--data', default=os.path.join(default_root, 'dataset'),
                        help='数据集目录（用于取样本与标签）')
    parser.add_argument('--image', default=None,
                        help='只测试单张图片（无标签时仅打印预测值）')
    parser.add_argument('--samples', type=int, default=10,
                        help='随机抽取的样本数（默认 10）')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', action='store_true',
                        help='保存预测值 vs 真值散点图 test_infer_result.png')
    args = parser.parse_args(argv)

    device = torch.device('cpu')
    model, width, height, ckpt = load_model(os.path.abspath(args.model), device)

    print('=' * 66)
    print('  ROV 端到端控制网络 — 离线推理测试')
    print('=' * 66)
    print(f'  权重文件    : {os.path.abspath(args.model)}')
    print(f'  输入分辨率  : {width}x{height}')
    print(f'  训练轮数    : {ckpt.get("epochs", "未知")}')
    best = ckpt.get('best_val_mse')
    print(f'  最佳验证MSE : {"%.6f" % best if isinstance(best, (int, float)) else "未记录"}')
    print('-' * 66)

    # ---------------- 单张图片模式 ----------------
    if args.image:
        img_path = os.path.abspath(args.image)
        if not os.path.exists(img_path):
            print(f'❌ 图片不存在: {img_path}')
            return 1
        arr = preprocess_image(Image.open(img_path), width, height)
        forward, turn = infer_array(model, arr, device)
        print(f'  图片: {img_path}')
        print(f'  ┌─────────────────────────────────────────┐')
        print(f'  │  预测 前进推力 forward = {forward:+.4f}      │')
        print(f'  │  预测 转向     turn    = {turn:+.4f}      │')
        print(f'  └─────────────────────────────────────────┘')
        print('\n  说明：输出经 tanh 限幅在 [-1,1]，可直接作为 /cmd_vel 的'
              'linear.x 与 angular.z。')
        return 0

    # ---------------- 数据集批量模式 ----------------
    ds = ROVDataset(os.path.abspath(args.data), width=width, height=height)
    n = min(args.samples, len(ds))
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(ds), size=n, replace=False)

    print(f'  数据集样本: {len(ds)}，本次抽取 {n} 张')
    print('-' * 66)
    print(f'  {"文件":<16}{"预测(前进,转向)":<24}{"真值(前进,转向)":<24}{"误差":<10}')
    print('-' * 66)

    preds, gts = [], []
    for i in idxs:
        path, gt_f, gt_t = ds.samples[i]
        arr = preprocess_image(Image.open(path), width, height)
        p_f, p_t = infer_array(model, arr, device)
        err = float(np.hypot(p_f - gt_f, p_t - gt_t))
        preds.append((p_f, p_t))
        gts.append((gt_f, gt_t))
        print(f'  {os.path.basename(path):<16}'
              f'({p_f:+.3f}, {p_t:+.3f})'.ljust(24) +
              f'({gt_f:+.3f}, {gt_t:+.3f})'.ljust(24) +
              f'{err:.4f}')

    preds = np.asarray(preds)
    gts = np.asarray(gts)
    mae_f = float(np.mean(np.abs(preds[:, 0] - gts[:, 0])))
    mae_t = float(np.mean(np.abs(preds[:, 1] - gts[:, 1])))
    mse = float(np.mean((preds - gts) ** 2))

    print('-' * 66)
    print(f'  平均绝对误差  forward MAE = {mae_f:.4f}')
    print(f'  平均绝对误差  turn    MAE = {mae_t:.4f}')
    print(f'  整体均方误差  MSE         = {mse:.6f}')
    print('-' * 66)
    if mae_f < 0.25 and mae_t < 0.25:
        print('  ✅ 网络输出与人工标签一致度良好，权重可正常用于在线推理。')
    else:
        print('  ⚠ 误差偏大：建议增加采集样本量、确认采集时是否长时间停留在'
              '单一动作，或增加训练轮数后重新测试。')

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, k, name in zip(axes, (0, 1), ('forward', 'turn')):
                ax.scatter(gts[:, k], preds[:, k], alpha=0.7, edgecolors='k')
                lo, hi = -1.05, 1.05
                ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
                ax.set_xlabel('Ground Truth')
                ax.set_ylabel('Prediction')
                ax.set_title(name)
                ax.grid(alpha=0.3)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
            fig.suptitle('Offline Inference: Prediction vs Ground Truth')
            fig.tight_layout()
            out = os.path.join(os.path.dirname(os.path.abspath(args.model)),
                               'test_infer_result.png')
            fig.savefig(out, dpi=140)
            plt.close(fig)
            print(f'  ✅ 散点图已保存: {out}')
        except ImportError:
            print('  （未安装 matplotlib，跳过绘图）')

    print('=' * 66)
    return 0


if __name__ == '__main__':
    sys.exit(main())
