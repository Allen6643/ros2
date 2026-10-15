"""
离线测试脚本
============
**不启动 ROS、不启动 MuJoCo**，直接加载训练好的权重对数据集做推理，
量化两个网络的精度，证明权重真实可用、可复现。

输出：
  * 视觉端到端网络：两个控制量的 MAE、整体 MSE、逐维散点图
  * 声呐感知网络  ：分类准确率与混淆矩阵
  * `test_infer_result.png`

用法
----
    python3 test_infer.py
"""

import argparse
import os
import sys

import numpy as np

try:
    import torch
except ImportError:
    sys.stderr.write('\n【依赖缺失】需要 PyTorch：pip3 install "torch==2.4.1+cpu"\n\n')
    raise

try:
    from rov_vision_e2e_nav.networks import (
        build_visual_e2e_net, build_obstacle_map_net, preprocess_image,
        normalize_sonar_features, setup_chinese_font,
        E2E_OUTPUT_DIMS, OPEN_DIRECTION_CLASSES)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rov_vision_e2e_nav.networks import (
        build_visual_e2e_net, build_obstacle_map_net, preprocess_image,
        normalize_sonar_features, setup_chinese_font,
        E2E_OUTPUT_DIMS, OPEN_DIRECTION_CLASSES)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_weights(path, builder, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'找不到权重 {path}\n'
            f'请先训练：python3 src/water/rov_vision_e2e_nav/rov_vision_e2e_nav/train.py')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    net = builder()
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    print(f'  已加载 {name}: {os.path.basename(path)}'
          f'（训练 {ckpt.get("epochs", "?")} 轮，验证损失 {ckpt.get("val_loss", float("nan")):.6f}）')
    return net


def main(argv=None):
    parser = argparse.ArgumentParser(description='离线测试视觉端到端网络与声呐感知网络')
    parser.add_argument('--data', default=os.path.abspath(os.path.join(HERE, '..', 'dataset')))
    parser.add_argument('--out', default=os.path.abspath(os.path.join(HERE, '..')))
    parser.add_argument('--samples', type=int, default=0, help='0 = 全部样本')
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.out)

    print('=' * 70)
    print('  ROV 视觉端到端 + 声呐感知 神经网络 —— 离线推理测试')
    print('=' * 70)

    # ---------------- 1) 视觉端到端网络 ----------------
    net_e = load_weights(os.path.join(out_dir, 'model_e2e.pth'),
                         build_visual_e2e_net, '视觉端到端网络')
    d = np.load(os.path.join(data_dir, 'e2e_dataset.npz'))
    images, y = d['images'], d['y']
    n = len(images) if args.samples <= 0 else min(args.samples, len(images))
    idx = np.arange(n)

    X = np.stack([preprocess_image(images[i]) for i in idx]).astype(np.float32)
    with torch.no_grad():
        pred = net_e(torch.from_numpy(X)).numpy()
    gt = y[idx]

    mae = np.abs(pred - gt).mean(axis=0)
    mse = float(((pred - gt) ** 2).mean())

    print(f'\n  样本数: {n}')
    print(f'  {"输出维度":<10}{"MAE":>12}{"真值范围":>22}')
    for i, name in enumerate(E2E_OUTPUT_DIMS):
        print(f'  {name:<10}{mae[i]:>12.4f}'
              f'{f"[{gt[:, i].min():+.3f}, {gt[:, i].max():+.3f}]":>22}')
    print(f'  整体 MSE: {mse:.6f}')
    print(f'  输出维度平均 MAE: {mae.mean():.4f}')

    # ---------------- 2) 声呐感知网络 ----------------
    net_s = load_weights(os.path.join(out_dir, 'model_seg.pth'),
                         build_obstacle_map_net, '声呐感知网络')
    ds = np.load(os.path.join(data_dir, 'sonar_dataset.npz'))
    Xs, ys = ds['X'], ds['y']
    Xs_n = np.stack([normalize_sonar_features(x) for x in Xs]).astype(np.float32)
    with torch.no_grad():
        pred_s = net_s(torch.from_numpy(Xs_n)).argmax(dim=1).numpy()

    acc = float((pred_s == ys).mean())
    nc = len(OPEN_DIRECTION_CLASSES)
    conf = np.zeros((nc, nc), dtype=int)
    for t, p in zip(ys, pred_s):
        conf[t, p] += 1

    print(f'\n  样本数: {len(ys)}')
    print(f'  分类准确率: {acc * 100:.2f}%')
    print('\n  混淆矩阵（行=真值，列=预测）:')
    print('        ' + ''.join(f'{c:>12}' for c in OPEN_DIRECTION_CLASSES))
    for i, c in enumerate(OPEN_DIRECTION_CLASSES):
        print(f'  {c:<6}' + ''.join(f'{conf[i, j]:>12d}' for j in range(nc)))

    print('-' * 70)
    ok = acc > 0.85 and mae.mean() < 0.20
    print('  ✅ 两个网络精度均达标，权重可正常用于在线推理。' if ok
          else '  ⚠ 精度偏低：建议增加采样条数或训练轮数。')
    print('=' * 70)

    # ---------------- 3) 散点图 ----------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        zh_font = setup_chinese_font()
        zh = zh_font is not None

        fig, axes = plt.subplots(1, len(E2E_OUTPUT_DIMS), figsize=(5 * len(E2E_OUTPUT_DIMS), 4.2))
        for ax, i in zip(axes, range(len(E2E_OUTPUT_DIMS))):
            name = E2E_OUTPUT_DIMS[i] if zh else ('forward', 'turn')[i]
            ax.scatter(gt[:, i], pred[:, i], s=5, alpha=0.35)
            ax.plot([-1.05, 1.05], [-1.05, 1.05], 'r--', lw=1)
            ax.set_xlabel('真值' if zh else 'True', fontproperties=zh_font)
            ax.set_ylabel('预测' if zh else 'Pred', fontproperties=zh_font)
            ax.set_title(name, fontsize=11, fontproperties=zh_font)
            ax.grid(alpha=0.25)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
        fig.suptitle('视觉端到端网络：预测值 vs 真值' if zh
                     else 'Visual E2E Net: Prediction vs Ground Truth',
                     fontproperties=zh_font)
        fig.tight_layout()
        png = os.path.join(out_dir, 'test_infer_result.png')
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f'  ✅ 散点图已保存: {png}')
    except Exception as err:  # noqa: BLE001
        print(f'  （绘图跳过: {err}）')

    return 0


if __name__ == '__main__':
    sys.exit(main())
