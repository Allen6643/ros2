"""
训练脚本
========
读取 `collect_data.py` 生成的两个数据集，分别训练

  1. `SensorPerceptionNet` —— 障碍物方位三分类（交叉熵损失）
  2. `TrajectoryControlNet` —— 控制量回归（均方误差损失）

训练结束后输出 `model_sensor.pth`、`model_traj.pth` 以及两条损失曲线。
权重文件由本脚本在**你自己的机器上训练产生**，不入版本库。

用法
----
    python3 train.py                       # 使用默认 dataset/ 与轮数
    python3 train.py --epochs 60 --batch-size 128
"""

import argparse
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    sys.stderr.write(
        '\n【依赖缺失】需要 PyTorch（Python 3.8 请装 CPU 版）：\n'
        '    pip3 install torch==2.4.1+cpu torchvision==0.19.1+cpu \\\n'
        '        -i https://mirrors.aliyun.com/pypi/simple/ \\\n'
        '        --extra-index-url https://download.pytorch.org/whl/cpu\n\n')
    raise

try:
    from rov_mujoco_sim_teleop.networks import (
        build_sensor_net, build_traj_net,
        normalize_sensor_features, normalize_traj_features)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rov_mujoco_sim_teleop.networks import (
        build_sensor_net, build_traj_net,
        normalize_sensor_features, normalize_traj_features)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_npz(path):
    """读取 npz 数据集，缺少时给出可操作的报错。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'找不到数据集 {path}\n'
            f'请先运行采集脚本：\n'
            f'    python3 src/water/rov_mujoco_sim_teleop/rov_mujoco_sim_teleop/collect_data.py --samples 4000')
    d = np.load(path)
    return d['X'], d['y']


def split(n, val_ratio, seed):
    """按比例划分训练/验证索引（固定种子保证可复现）。"""
    g = np.random.RandomState(seed)
    idx = g.permutation(n)
    n_val = int(n * val_ratio)
    return idx[n_val:], idx[:n_val]


def normalize_batch(X, fn):
    """对整批特征做标准化（调用与推理端同一个函数）。"""
    return np.stack([fn(x) for x in X]).astype(np.float32)


def train_one(name, X, y, net, criterion, epochs, batch_size, lr, val_ratio,
              seed, out_path, is_classification):
    """训练单个网络并保存权重，返回训练历史。"""
    device = torch.device('cpu')
    n = len(X)
    tr_idx, va_idx = split(n, val_ratio, seed)

    Xtr = torch.from_numpy(X[tr_idx])
    ytr = torch.from_numpy(y[tr_idx])
    Xva = torch.from_numpy(X[va_idx])
    yva = torch.from_numpy(y[va_idx])

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {'train': [], 'val': []}
    best_val, best_state = float('inf'), None

    print(f'\n--- 训练 {name} ---')
    print(f'  样本 {n}（训练 {len(tr_idx)} / 验证 {len(va_idx)}）| 参数量 '
          f'{sum(p.numel() for p in net.parameters()):,}')

    for epoch in range(1, epochs + 1):
        net.train()
        run, seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            run += float(loss.item()) * xb.size(0)
            seen += xb.size(0)
        tr_loss = run / max(seen, 1)

        net.eval()
        with torch.no_grad():
            va_out = net(Xva)
            va_loss = float(criterion(va_out, yva).item())

        history['train'].append(tr_loss)
        history['val'].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            extra = ''
            if is_classification:
                acc = float((va_out.argmax(dim=1) == yva).float().mean().item())
                extra = f'  val_acc={acc * 100:.1f}%'
            print(f'  Epoch {epoch:3d}/{epochs}  train={tr_loss:.6f}  val={va_loss:.6f}{extra}')

    if best_state is not None:
        net.load_state_dict(best_state)

    torch.save({
        'state_dict': net.state_dict(),
        'arch': name,
        'val_loss': best_val,
        'epochs': epochs,
        'is_classification': is_classification,
    }, out_path)
    print(f'  ✅ 权重已保存: {out_path}  (最佳验证损失 {best_val:.6f})')
    return history


def plot_curves(histories, out_path):
    """绘制训练/验证损失曲线。

    中文字体不可用时会自动改用英文标签，避免图上出现方框。
    matplotlib 不可用时只跳过绘图，不影响训练产物。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as err:  # noqa: BLE001
        print(f'  （matplotlib 不可用，跳过损失曲线: {err}）')
        return False

    from rov_mujoco_sim_teleop.networks import setup_chinese_font
    zh_font = setup_chinese_font()
    zh = zh_font is not None
    if not zh:
        print('  （未找到中文字体，损失曲线改用英文标签）')

    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4.2))
    if len(histories) == 1:
        axes = [axes]
    for ax, (name, h) in zip(axes, histories.items()):
        ep = range(1, len(h['train']) + 1)
        ax.plot(ep, h['train'], marker='o', ms=3, label='训练集' if zh else 'Train')
        ax.plot(ep, h['val'], marker='s', ms=3, label='验证集' if zh else 'Val')
        ax.set_xlabel('训练轮次 (Epoch)' if zh else 'Epoch',
                      fontproperties=zh_font)
        ax.set_ylabel('损失 (Loss)' if zh else 'Loss', fontproperties=zh_font)
        # 图标题里的中文在无字体时会变成方框，因此无字体时用英文
        if not zh:
            name = {'传感器感知网络 (3类)': 'Sensor Perception Net (3-class)',
                    '轨迹跟踪网络 (回归)': 'Trajectory Control Net (regression)'}.get(name, name)
        ax.set_title(name, fontproperties=zh_font)
        ax.grid(alpha=0.3)
        ax.legend(prop=zh_font)
    fig.suptitle('ROV MuJoCo 神经网络训练损失曲线' if zh
                 else 'ROV MuJoCo Neural Network Training Loss',
                 fontproperties=zh_font)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description='训练传感器感知网络与轨迹跟踪网络')
    parser.add_argument('--data', default=os.path.abspath(os.path.join(HERE, '..', 'dataset')))
    parser.add_argument('--out', default=os.path.abspath(os.path.join(HERE, '..')))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 66)
    print('  ROV MuJoCo 神经网络训练')
    print('=' * 66)
    print(f'  数据集目录: {args.data}')
    print(f'  输出目录  : {out_dir}')
    print(f'  轮数 {args.epochs} | 批大小 {args.batch_size} | 学习率 {args.lr}')
    print('=' * 66)

    histories = {}

    # ---------- 1) 传感器感知网络（三分类）----------
    Xs, ys = load_npz(os.path.join(args.data, 'sensor_dataset.npz'))
    Xs = normalize_batch(Xs, normalize_sensor_features)
    net_s = build_sensor_net()
    criterion = nn.CrossEntropyLoss()
    histories['传感器感知网络 (3类)'] = train_one(
        'SensorPerceptionNet', Xs, ys.astype(np.int64), net_s, criterion,
        args.epochs, args.batch_size, args.lr, args.val_ratio, args.seed,
        os.path.join(out_dir, 'model_sensor.pth'), is_classification=True)

    # ---------- 2) 轨迹跟踪网络（回归）----------
    Xt, yt = load_npz(os.path.join(args.data, 'traj_dataset.npz'))
    Xt = normalize_batch(Xt, normalize_traj_features)
    net_t = build_traj_net()
    criterion_t = nn.MSELoss()
    histories['轨迹跟踪网络 (回归)'] = train_one(
        'TrajectoryControlNet', Xt, yt.astype(np.float32), net_t, criterion_t,
        args.epochs, args.batch_size, args.lr, args.val_ratio, args.seed,
        os.path.join(out_dir, 'model_traj.pth'), is_classification=False)

    curve = os.path.join(out_dir, 'training_curve.png')
    if plot_curves(histories, curve):
        print(f'\n  ✅ 损失曲线: {curve}')

    print('=' * 66)
    print('  训练完成。请接着运行离线测试验证权重：')
    print('    python3 test_infer.py')
    print('=' * 66)
    return 0


if __name__ == '__main__':
    sys.exit(main())
