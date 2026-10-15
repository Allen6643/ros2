"""
训练脚本
========
读取 `collect_data.py` 生成的两个数据集，分别训练

  1. `VisualE2ENet`       —— 单目图像到 [前进推力, 转向] 的端到端回归
  2. `ObstacleMapNet`     —— 声呐测距扇到开阔方向的三分类

训练结束后输出 `model_e2e.pth`、`model_seg.pth` 与两条损失曲线。
权重由本脚本在**你自己的机器上训练产生**，不入版本库。

用法
----
    python3 train.py --epochs 40
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
    from rov_vision_e2e_nav.networks import (
        build_visual_e2e_net, build_obstacle_map_net,
        preprocess_image, normalize_sonar_features, setup_chinese_font)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rov_vision_e2e_nav.networks import (
        build_visual_e2e_net, build_obstacle_map_net,
        preprocess_image, normalize_sonar_features, setup_chinese_font)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'找不到数据集 {path}\n'
            f'请先采集：python3 src/water/rov_vision_e2e_nav/rov_vision_e2e_nav/collect_data.py --samples 3000')
    return np.load(path)


def split(n, val_ratio, seed):
    g = np.random.RandomState(seed)
    idx = g.permutation(n)
    n_val = int(n * val_ratio)
    return idx[n_val:], idx[:n_val]


def plot_curves(histories, out_path):
    """绘制损失曲线；无中文字体时自动改用英文标签。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as err:  # noqa: BLE001
        print(f'  （matplotlib 不可用，跳过绘图: {err}）')
        return False

    zh_font = setup_chinese_font()
    zh = zh_font is not None
    if not zh:
        print('  （未找到中文字体，改用英文标签）')

    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4.2))
    if len(histories) == 1:
        axes = [axes]
    for ax, (name, h) in zip(axes, histories.items()):
        ep = range(1, len(h['train']) + 1)
        ax.plot(ep, h['train'], marker='o', ms=3, label='训练集' if zh else 'Train')
        ax.plot(ep, h['val'], marker='s', ms=3, label='验证集' if zh else 'Val')
        ax.set_xlabel('训练轮次 (Epoch)' if zh else 'Epoch', fontproperties=zh_font)
        ax.set_ylabel('损失 (Loss)' if zh else 'Loss', fontproperties=zh_font)
        ax.set_title(name, fontproperties=zh_font)
        ax.grid(alpha=0.3)
        ax.legend(prop=zh_font)
    fig.suptitle('ROV 视觉端到端与声呐感知网络训练损失曲线' if zh
                 else 'ROV Visual E2E & Sonar Network Training Loss',
                 fontproperties=zh_font)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description='训练视觉端到端网络与声呐感知网络')
    parser.add_argument('--data', default=os.path.abspath(os.path.join(HERE, '..', 'dataset')))
    parser.add_argument('--out', default=os.path.abspath(os.path.join(HERE, '..')))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 66)
    print('  ROV 视觉端到端 + 声呐感知 神经网络训练')
    print('=' * 66)
    print(f'  数据集: {args.data}')
    print(f'  轮数 {args.epochs} | 批大小 {args.batch_size} | 学习率 {args.lr}')
    print('=' * 66)

    histories = {}

    # ---------------- 1) 视觉端到端网络 ----------------
    d = load_npz(os.path.join(args.data, 'e2e_dataset.npz'))
    images, e2e_y = d['images'], d['y']
    print(f'\n--- 训练 视觉端到端网络 ---')
    print(f'  样本 {len(images)} | 图像 {images.shape[1:]}')

    # 逐张预处理（复用小批次处理，避免一次性占用过多内存）
    X_img = np.stack([preprocess_image(im) for im in images]).astype(np.float32)

    tr, va = split(len(X_img), args.val_ratio, args.seed)
    net_e = build_visual_e2e_net()
    print(f'  参数量 {sum(p.numel() for p in net_e.parameters()):,}')

    Xtr = torch.from_numpy(X_img[tr]); ytr = torch.from_numpy(e2e_y[tr])
    Xva = torch.from_numpy(X_img[va]); yva = torch.from_numpy(e2e_y[va])
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(net_e.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=12, gamma=0.5)

    h = {'train': [], 'val': []}
    best, best_state = float('inf'), None
    for ep in range(1, args.epochs + 1):
        net_e.train()
        run, seen = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(net_e(xb), yb)
            loss.backward()
            opt.step()
            run += float(loss.item()) * xb.size(0)
            seen += xb.size(0)
        tr_loss = run / max(seen, 1)

        net_e.eval()
        with torch.no_grad():
            va_loss = float(criterion(net_e(Xva), yva).item())
        h['train'].append(tr_loss); h['val'].append(va_loss)
        if va_loss < best:
            best = va_loss
            best_state = {k: v.clone() for k, v in net_e.state_dict().items()}
        sched.step()
        if ep % 10 == 0 or ep == 1:
            print(f'  Epoch {ep:3d}/{args.epochs}  train={tr_loss:.6f}  val={va_loss:.6f}')

    if best_state:
        net_e.load_state_dict(best_state)
    torch.save({'state_dict': net_e.state_dict(), 'arch': 'VisualE2ENet',
                'val_loss': best, 'epochs': args.epochs,
                'width': 160, 'height': 120},
               os.path.join(out_dir, 'model_e2e.pth'))
    histories['视觉端到端网络 (回归)'] = h
    print(f'  ✅ 权重已保存: model_e2e.pth (最佳验证损失 {best:.6f})')

    # ---------------- 2) 声呐感知网络 ----------------
    ds = load_npz(os.path.join(args.data, 'sonar_dataset.npz'))
    sonar_X, sonar_y = ds['X'], ds['y']
    Xs = np.stack([normalize_sonar_features(x) for x in sonar_X]).astype(np.float32)
    tr2, va2 = split(len(Xs), args.val_ratio, args.seed)

    net_s = build_obstacle_map_net()
    print(f'\n--- 训练 声呐感知网络 ---')
    print(f'  样本 {len(Xs)} | 参数量 {sum(p.numel() for p in net_s.parameters()):,}')

    Xtr2 = torch.from_numpy(Xs[tr2]); ytr2 = torch.from_numpy(sonar_y[tr2].astype(np.int64))
    Xva2 = torch.from_numpy(Xs[va2]); yva2 = torch.from_numpy(sonar_y[va2].astype(np.int64))
    loader2 = DataLoader(TensorDataset(Xtr2, ytr2), batch_size=args.batch_size, shuffle=True)

    criterion2 = nn.CrossEntropyLoss()
    opt2 = torch.optim.Adam(net_s.parameters(), lr=args.lr)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=12, gamma=0.5)

    h2 = {'train': [], 'val': []}
    best2, best_state2 = float('inf'), None
    for ep in range(1, args.epochs + 1):
        net_s.train()
        run, seen = 0.0, 0
        for xb, yb in loader2:
            opt2.zero_grad()
            loss = criterion2(net_s(xb), yb)
            loss.backward()
            opt2.step()
            run += float(loss.item()) * xb.size(0)
            seen += xb.size(0)
        tr_loss = run / max(seen, 1)
        net_s.eval()
        with torch.no_grad():
            out_va = net_s(Xva2)
            va_loss = float(criterion2(out_va, yva2).item())
            acc = float((out_va.argmax(dim=1) == yva2).float().mean().item())
        h2['train'].append(tr_loss); h2['val'].append(va_loss)
        if va_loss < best2:
            best2 = va_loss
            best_state2 = {k: v.clone() for k, v in net_s.state_dict().items()}
        sched2.step()
        if ep % 10 == 0 or ep == 1:
            print(f'  Epoch {ep:3d}/{args.epochs}  train={tr_loss:.6f}  val={va_loss:.6f}  val_acc={acc*100:.1f}%')

    if best_state2:
        net_s.load_state_dict(best_state2)
    torch.save({'state_dict': net_s.state_dict(), 'arch': 'ObstacleMapNet',
                'val_loss': best2, 'epochs': args.epochs},
               os.path.join(out_dir, 'model_seg.pth'))
    histories['声呐感知网络 (3类)'] = h2
    print(f'  ✅ 权重已保存: model_seg.pth (最佳验证损失 {best2:.6f})')

    curve = os.path.join(out_dir, 'training_curve.png')
    if plot_curves(histories, curve):
        print(f'\n  ✅ 损失曲线: {curve}')

    print('=' * 66)
    print('  训练完成。接着运行离线测试：python3 test_infer.py')
    print('=' * 66)
    return 0


if __name__ == '__main__':
    sys.exit(main())
