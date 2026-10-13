#!/usr/bin/env python3
"""
端到端控制网络训练脚本
======================
读取 `collect_data.py` 采集的数据集（图像 + [前进推力, 转向] 标签），
训练一个小型 CNN 回归网络，输出 `model.pth`。

网络：输入 RGB 图像 → 3 层卷积 → 全局平均池化 → 2 层全连接 → tanh → 2 个连续值
损失：MSE（均方误差）
输出：权重文件 `model.pth`（**由本脚本在你自己的机器上训练产生**）

用法
----
    # 默认读取 <包目录>/dataset，权重写到 <包目录>/model.pth
    python3 src/water/rov_e2e/rov_e2e/train.py

    # 指定轮数与数据目录
    python3 src/water/rov_e2e/rov_e2e/train.py --epochs 40 --data ~/my_dataset

说明
----
本文件同时定义网络结构 `ROVSteeringNet`，供 `test_infer.py` 与
`infer_node.py` 通过 `from rov_e2e.train import ROVSteeringNet` 复用，
从而保证"训练用的网络"与"推理用的网络"严格是同一个定义。
matplotlib 只在绘图函数内部导入，避免推理节点被迫依赖它。
"""

import argparse
import csv
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    sys.stderr.write(
        '\n【环境依赖缺失】未检测到 PyTorch。请先安装（Ubuntu 20.04 + Python 3.8 需固定版本）：\n'
        '    pip3 install "torch<=2.4.1" torchvision pillow numpy\n'
        '    # 国内加速：-i https://pypi.tuna.tsinghua.edu.cn/simple\n\n'
    )
    raise

from PIL import Image

#: 默认输入分辨率（与 collect_data.py 的 --width/--height 保持一致）
DEFAULT_WIDTH = 160
DEFAULT_HEIGHT = 120

#: 训练/验证时的图像归一化参数（ImageNet 统计量，通用且稳定）
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


class ROVSteeringNet(nn.Module):
    """小型卷积回归网络：单目 RGB 图像 → [前进推力, 转向]。

    结构（参数量约 2.2 万，CPU 上可快速训练）：

        Conv(3→16, 3x3, stride2) + ReLU
        Conv(16→32, 3x3, stride2) + ReLU
        Conv(32→64, 3x3, stride2) + ReLU
        AdaptiveAvgPool2d(1)              # 对输入分辨率不敏感
        Flatten → Linear(64→32) + ReLU → Linear(32→2) → Tanh

    末层用 ``tanh`` 把输出限制在 ``[-1, 1]``，与标签量纲一致
    （标签即归一化的推力/转向指令），也便于在推理节点直接做限幅。
    """

    def __init__(self, in_channels=3, hidden=64):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_channels, 16),
            block(16, 32),
            block(32, hidden),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.head(self.features(x))


def preprocess_image(img, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """把 ``PIL.Image`` 转成网络输入的 ``(H, W, 3)`` float32 数组。

    训练（:class:`ROVDataset`）与推理（``test_infer.py`` / ``infer_node.py``）
    共用本函数，保证 **训练与部署的预处理完全一致**——这是端到端策略
    最容易出错的地方（缩放方式、通道顺序、归一化统计量不一致）。
    """
    img = img.convert('RGB').resize((width, height))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(NORM_MEAN, dtype=np.float32)) / \
          np.asarray(NORM_STD, dtype=np.float32)
    return arr


class ROVDataset(Dataset):
    """读取 ``dataset/labels.csv`` 与 ``dataset/images/*.png``。"""

    def __init__(self, root, indices=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.width = width
        self.height = height

        labels_path = os.path.join(root, 'labels.csv')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f'找不到标签文件 {labels_path}，请先运行 collect_data.py 采集数据集')

        rows = []
        with open(labels_path, 'r', newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                path = os.path.join(self.img_dir, row['filename'])
                if os.path.exists(path):
                    rows.append((path, float(row['forward']), float(row['turn'])))
        if not rows:
            raise RuntimeError(f'{labels_path} 中没有可用样本')

        self.samples = [rows[i] for i in indices] if indices is not None else rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, forward, turn = self.samples[idx]
        arr = preprocess_image(Image.open(path), self.width, self.height)
        x = torch.from_numpy(arr.transpose(2, 0, 1))          # HWC -> CHW
        y = torch.tensor([forward, turn], dtype=torch.float32)
        return x, y


def save_checkpoint(path, model, width, height, extra=None):
    """保存权重与推理所需的元信息，供 test_infer.py / infer_node.py 直接加载。"""
    payload = {
        'state_dict': model.state_dict(),
        'width': width,
        'height': height,
        'norm_mean': list(NORM_MEAN),
        'norm_std': list(NORM_STD),
        'arch': 'ROVSteeringNet',
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def plot_curve(history, out_path):
    """绘制训练/验证损失曲线（matplotlib 延迟导入）。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as err:      # 未安装、或版本与 numpy 不兼容，都不应影响训练产物
        print(f'  （matplotlib 不可用，跳过损失曲线绘制: {err}）')
        return False

    epochs = range(1, len(history['train']) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history['train'], marker='o', label='Train MSE')
    if history['val']:
        plt.plot(epochs[:len(history['val'])], history['val'],
                 marker='s', label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('ROV End-to-End Training Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return True


def evaluate(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += criterion(model(x), y).item() * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def main(argv=None):
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.abspath(os.path.join(pkg_dir, '..', 'dataset'))
    default_out = os.path.abspath(os.path.join(pkg_dir, '..', 'model.pth'))

    parser = argparse.ArgumentParser(description='训练 ROV 端到端控制网络')
    parser.add_argument('--data', default=default_data, help='数据集目录')
    parser.add_argument('--out', default=default_out, help='输出权重路径 model.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH)
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 数据集划分 ----
    full = ROVDataset(args.data, width=args.width, height=args.height)
    n_total = len(full)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    if n_train < 1:
        print(f'样本数过少（{n_total}），请先采集更多数据。')
        return 1

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = ROVDataset(args.data, train_idx, args.width, args.height)
    val_ds = ROVDataset(args.data, val_idx, args.width, args.height) if n_val else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

    device = torch.device('cpu')          # 教学虚拟机无独立显卡，固定 CPU
    model = ROVSteeringNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print('=' * 62)
    print('  ROV 端到端控制网络训练')
    print('=' * 62)
    print(f'  数据集      : {args.data}')
    print(f'  样本总数    : {n_total}  (训练 {n_train} / 验证 {n_val})')
    print(f'  输入分辨率  : {args.width}x{args.height}')
    print(f'  网络参数量  : {n_params:,}')
    print(f'  设备        : {device}  轮数: {args.epochs}  批大小: {args.batch_size}')
    print('-' * 62)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {'train': [], 'val': []}
    best_val = float('inf')
    best_state = None
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / max(seen, 1)
        history['train'].append(train_loss)

        msg = f'  Epoch {epoch:3d}/{args.epochs}  train_MSE={train_loss:.6f}'
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            history['val'].append(val_loss)
            msg += f'  val_MSE={val_loss:.6f}'
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        scheduler.step()
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    save_checkpoint(out_path, model, args.width, args.height,
                    extra={'epochs': args.epochs,
                           'best_val_mse': None if best_val == float('inf') else best_val,
                           'n_samples': n_total})
    print('-' * 62)
    print(f'  ✅ 权重已保存: {out_path}')
    if best_val != float('inf'):
        print(f'     最佳验证 MSE: {best_val:.6f}')

    curve_path = os.path.join(os.path.dirname(out_path), 'training_curve.png')
    if plot_curve(history, curve_path):
        print(f'  ✅ 损失曲线: {curve_path}')
    print('=' * 62)
    return 0


if __name__ == '__main__':
    sys.exit(main())
