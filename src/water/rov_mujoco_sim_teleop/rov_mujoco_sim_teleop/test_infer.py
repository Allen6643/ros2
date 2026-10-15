"""
离线测试脚本
============
**不启动 ROS、不启动 MuJoCo**，直接加载训练好的权重对数据集做推理，
量化网络精度。用于证明权重文件真实可用、结果可复现。

输出：
  * 感知网络：分类准确率、混淆矩阵
  * 轨迹网络：各控制维度的 MAE、整体 MSE
  * `test_infer_result.png`：轨迹网络预测值 vs 真值散点图

用法
----
    python3 test_infer.py
    python3 test_infer.py --data dataset --out . --plot
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
    from rov_mujoco_sim_teleop.networks import (
        build_sensor_net, build_traj_net, OBSTACLE_CLASSES, CONTROL_DIMS,
        normalize_sensor_features, normalize_traj_features)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rov_mujoco_sim_teleop.networks import (
        build_sensor_net, build_traj_net, OBSTACLE_CLASSES, CONTROL_DIMS,
        normalize_sensor_features, normalize_traj_features)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_weights(path, builder, name):
    """加载权重并构造对应网络。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'找不到权重文件 {path}\n'
            f'请先训练：python3 src/water/rov_mujoco_sim_teleop/rov_mujoco_sim_teleop/train.py')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    net = builder()
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    print(f'  已加载 {name}: {os.path.basename(path)}（训练 {ckpt.get("epochs", "?")} 轮，'
          f'验证损失 {ckpt.get("val_loss", float("nan")):.6f}）')
    return net


def main(argv=None):
    parser = argparse.ArgumentParser(description='离线测试两个神经网络的权重')
    parser.add_argument('--data', default=os.path.abspath(os.path.join(HERE, '..', 'dataset')))
    parser.add_argument('--out', default=os.path.abspath(os.path.join(HERE, '..')))
    parser.add_argument('--samples', type=int, default=0, help='0 = 用全部样本')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plot', action='store_true', default=True)
    args = parser.parse_args(argv)

    data_dir = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.out)

    print('=' * 70)
    print('  ROV MuJoCo 神经网络 —— 离线推理测试')
    print('=' * 70)

    # ---------------- 1) 传感器感知网络 ----------------
    net_s = load_weights(os.path.join(out_dir, 'model_sensor.pth'),
                         build_sensor_net, '传感器感知网络')
    ds = np.load(os.path.join(data_dir, 'sensor_dataset.npz'))
    Xs, ys = ds['X'], ds['y']
    Xs_n = np.stack([normalize_sensor_features(x) for x in Xs]).astype(np.float32)

    with torch.no_grad():
        logits = net_s(torch.from_numpy(Xs_n))
        pred = logits.argmax(dim=1).numpy()

    acc = float((pred == ys).mean())
    n_cls = len(OBSTACLE_CLASSES)
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(ys, pred):
        conf[t, p] += 1

    print(f'\n  样本数: {len(ys)}')
    print(f'  分类准确率: {acc * 100:.2f}%')
    print('\n  混淆矩阵（行=真值，列=预测）:')
    header = '        ' + ''.join(f'{c:>10}' for c in OBSTACLE_CLASSES)
    print(header)
    for i, c in enumerate(OBSTACLE_CLASSES):
        row = ''.join(f'{conf[i, j]:>10d}' for j in range(n_cls))
        print(f'  {c:<6}{row}')

    # ---------------- 2) 轨迹跟踪网络 ----------------
    net_t = load_weights(os.path.join(out_dir, 'model_traj.pth'),
                         build_traj_net, '轨迹跟踪网络')
    dt = np.load(os.path.join(data_dir, 'traj_dataset.npz'))
    Xt, yt = dt['X'], dt['y']
    Xt_n = np.stack([normalize_traj_features(x) for x in Xt]).astype(np.float32)

    with torch.no_grad():
        yp = net_t(torch.from_numpy(Xt_n)).numpy()

    mae = np.abs(yp - yt).mean(axis=0)
    mse = float(((yp - yt) ** 2).mean())

    print(f'\n  样本数: {len(yt)}')
    print(f'  {"控制维度":<10}{"MAE":>12}{"真值范围":>20}')
    for i, name in enumerate(CONTROL_DIMS):
        print(f'  {name:<10}{mae[i]:>12.4f}'
              f'{f"[{yt[:, i].min():+.3f}, {yt[:, i].max():+.3f}]":>20}')
    print(f'  整体 MSE: {mse:.6f}')
    print(f'  控制维度平均 MAE: {mae.mean():.4f}')

    # ---------------- 3) 判定 ----------------
    print('-' * 70)
    ok = acc > 0.85 and mae.mean() < 0.15
    if ok:
        print('  ✅ 两个网络精度均达标，权重可正常用于在线推理。')
    else:
        print('  ⚠ 精度偏低：建议增加采样条数或训练轮数后重新测试。')
    print('=' * 70)

    # ---------------- 4) 散点图 ----------------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from rov_mujoco_sim_teleop.networks import setup_chinese_font
            zh_font = setup_chinese_font()
            zh = zh_font is not None

            fig, axes = plt.subplots(1, len(CONTROL_DIMS), figsize=(4 * len(CONTROL_DIMS), 3.8))
            for ax, i in zip(axes, range(len(CONTROL_DIMS))):
                name = CONTROL_DIMS[i] if zh else \
                    ('forward', 'lateral', 'vertical', 'yaw')[i]
                ax.scatter(yt[:, i], yp[:, i], s=4, alpha=0.35)
                ax.plot([-1.05, 1.05], [-1.05, 1.05], 'r--', lw=1)
                ax.set_xlabel('真值' if zh else 'True', fontproperties=zh_font)
                ax.set_ylabel('预测' if zh else 'Pred', fontproperties=zh_font)
                ax.set_title(name, fontsize=10, fontproperties=zh_font)
                ax.grid(alpha=0.25)
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
            fig.suptitle('轨迹跟踪网络：预测值 vs 真值' if zh
                         else 'Trajectory Control Net: Prediction vs Ground Truth',
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
