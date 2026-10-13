#!/usr/bin/env python3
"""
端到端控制性能评价脚本
======================
对应作业要求「推荐使用多个场景进行测试，且包含性能评价」。

提供四个模式：

    # 1) 开环评价：模型精度与推理耗时（不需要 ROS / MuJoCo）
    python3 evaluate.py offline --model model.pth --data dataset

    # 2) 记录一次闭环航行的里程计（需要仿真在跑）
    python3 evaluate.py record --out runs/S1_odom.csv --duration 30

    # 3) 分析某次闭环航行，给出闭环性能指标
    python3 evaluate.py report --log runs/S1_odom.csv \
        --start 0 0 -1 --target-distance 5.0 --lateral-tol 0.5

    # 4) 汇总多场景结果，生成对比表（写入 Markdown）
    python3 evaluate.py compare --reports runs/*_report.json --out runs/compare.md

设计说明
--------
- `offline` 与 `report`/`compare` **不需要 ROS 与 MuJoCo**，只要 numpy（`offline` 另需 torch），
  便于在没有仿真的机器上做回归测试；
- `record` 才需要 rclpy，因此 `rclpy` 在函数内部延迟导入，
  避免因为缺少 ROS 环境导致 `--help` 都用不了；
- 所有指标输出为 JSON（供程序读取）+ Markdown 表格（供直接粘进文档）。
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def quat_to_yaw(qw, qx, qy, qz):
    """四元数 → 偏航角（绕 Z 轴，弧度）。"""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def load_odom_csv(path):
    """读取 record 模式写出的里程计 CSV，返回结构化数组。"""
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if not rows:
        raise RuntimeError(f'{path} 中没有数据')
    out = {}
    for k in rows[0].keys():
        out[k] = np.array([float(r[k]) for r in rows], dtype=np.float64)
    return out


def markdown_table(headers, rows):
    """把二维数据渲染成 Markdown 表格文本。"""
    lines = ['| ' + ' | '.join(str(h) for h in headers) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# 1) 开环评价
# --------------------------------------------------------------------------- #
def cmd_offline(args):
    """在测试集上评价 model.pth 的精度与推理耗时。"""
    try:
        import torch
    except ImportError:
        sys.stderr.write('\n【依赖缺失】开环评价需要 PyTorch：pip3 install "torch<=2.4.1"\n\n')
        return 1

    try:
        from rov_e2e.train import ROVSteeringNet, ROVDataset, preprocess_image
    except ImportError:
        sys.path.insert(0, HERE)
        from rov_e2e.train import ROVSteeringNet, ROVDataset, preprocess_image

    from PIL import Image

    device = torch.device('cpu')
    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        sys.stderr.write(f'找不到权重 {model_path}，请先运行 train.py\n')
        return 1

    ckpt = torch.load(model_path, map_location=device)
    width = int(ckpt.get('width', 160))
    height = int(ckpt.get('height', 120))

    model = ROVSteeringNet().to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    ds = ROVDataset(os.path.abspath(args.data), width=width, height=height)
    n = min(args.samples, len(ds)) if args.samples > 0 else len(ds)
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(ds), size=n, replace=False) if n < len(ds) else np.arange(len(ds))

    preds, gts, latencies = [], [], []
    for i in idxs:
        path, gf, gt = ds.samples[i]
        arr = preprocess_image(Image.open(path), width, height)
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)[0].cpu().numpy()
        latencies.append((time.perf_counter() - t0) * 1000.0)   # ms
        preds.append(out)
        gts.append((gf, gt))

    preds = np.asarray(preds)
    gts = np.asarray(gts)
    lat = np.asarray(latencies)

    metrics = {
        'model_path': model_path,
        'n_samples': int(len(idxs)),
        'input_size': f'{width}x{height}',
        'n_parameters': int(sum(p.numel() for p in model.parameters())),
        'weight_size_kb': round(os.path.getsize(model_path) / 1024.0, 1),
        'mae_forward': float(np.mean(np.abs(preds[:, 0] - gts[:, 0]))),
        'mae_turn': float(np.mean(np.abs(preds[:, 1] - gts[:, 1]))),
        'mse_overall': float(np.mean((preds - gts) ** 2)),
        'latency_mean_ms': float(lat.mean()),
        'latency_p50_ms': float(np.percentile(lat, 50)),
        'latency_p95_ms': float(np.percentile(lat, 95)),
        'latency_max_ms': float(lat.max()),
        'max_latency_hz': float(1000.0 / max(lat.mean(), 1e-9)),
    }

    print('=' * 68)
    print('  开环性能评价（精度 + 推理耗时，CPU）')
    print('=' * 68)
    print(markdown_table(
        ['指标', '数值'],
        [['样本数', metrics['n_samples']],
         ['输入分辨率', metrics['input_size']],
         ['网络参数量', f"{metrics['n_parameters']:,}"],
         ['权重体积 (KB)', metrics['weight_size_kb']],
         ['前进 MAE', f"{metrics['mae_forward']:.4f}"],
         ['转向 MAE', f"{metrics['mae_turn']:.4f}"],
         ['整体 MSE', f"{metrics['mse_overall']:.6f}"],
         ['单帧推理均值 (ms)', f"{metrics['latency_mean_ms']:.2f}"],
         ['单帧推理 P95 (ms)', f"{metrics['latency_p95_ms']:.2f}"],
         ['理论最高推理频率 (Hz)', f"{metrics['max_latency_hz']:.1f}"]]))
    print('=' * 68)

    out = os.path.abspath(args.out or
                          os.path.join(os.path.dirname(model_path),
                                       'evaluation_report.json'))
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f'  ✅ 指标已写入: {out}')
    return 0


# --------------------------------------------------------------------------- #
# 2) 记录闭环里程计
# --------------------------------------------------------------------------- #
def cmd_record(args):
    """订阅 /rov/odom，把一次闭环航行的位姿写成 CSV。"""
    try:
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import Odometry
    except ImportError:
        sys.stderr.write('\n【依赖缺失】record 模式需要 ROS 2：source /opt/ros/humble/setup.bash\n\n')
        return 1

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    class OdomRecorder(Node):
        def __init__(self):
            super().__init__('rov_e2e_odom_recorder')
            self.sub = self.create_subscription(Odometry, args.topic, self._cb, 10)
            self.t0 = None

        def _cb(self, msg: Odometry):
            now = time.time()
            if self.t0 is None:
                self.t0 = now
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            self.writer.writerow([
                f'{now - self.t0:.4f}',
                f'{p.x:.6f}', f'{p.y:.6f}', f'{p.z:.6f}',
                f'{o.w:.6f}', f'{o.x:.6f}', f'{o.y:.6f}', f'{o.z:.6f}',
                f'{v.x:.6f}', f'{v.y:.6f}', f'{v.z:.6f}'])

    rclpy.init()
    node = OdomRecorder()
    fh = open(out, 'w', newline='', encoding='utf-8')
    node.writer = csv.writer(fh)
    node.writer.writerow(['t', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz'])

    print(f'开始记录 {args.topic} → {out}（{args.duration:.0f} 秒，Ctrl+C 可提前结束）')
    try:
        deadline = time.time() + args.duration
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        fh.close()
        node.destroy_node()
        rclpy.shutdown()

    print(f'  ✅ 已记录到 {out}')
    return 0


# --------------------------------------------------------------------------- #
# 3) 闭环性能指标
# --------------------------------------------------------------------------- #
def cmd_report(args):
    """从里程计 CSV 计算闭环性能指标。"""
    d = load_odom_csv(os.path.abspath(args.log))
    p = np.stack([d['x'], d['y'], d['z']], axis=1)
    t = d['t']

    # 初始艏向作为「前进方向」，与其垂直的方向作为「横向」
    yaw0 = quat_to_yaw(d['qw'][0], d['qx'][0], d['qy'][0], d['qz'][0])
    u = np.array([math.cos(yaw0), math.sin(yaw0)])      # 前进方向
    v = np.array([-math.sin(yaw0), math.cos(yaw0)])     # 横向

    rel = p[:, :2] - p[0, :2]
    along = rel @ u                     # 沿前进方向的位移
    lateral = rel @ v                   # 横向偏差

    forward_distance = float(along[-1])
    duration = float(t[-1] - t[0])
    lateral_abs = np.abs(lateral)
    depth = p[:, 2]

    reached = forward_distance >= args.target_distance
    kept_line = float(lateral_abs.max()) <= args.lateral_tol
    success = bool(reached and kept_line)

    metrics = {
        'log': os.path.abspath(args.log),
        'duration_s': round(duration, 2),
        'forward_distance_m': round(forward_distance, 3),
        'mean_forward_speed_ms': round(forward_distance / duration, 4) if duration > 0 else 0.0,
        'lateral_mae_m': round(float(lateral_abs.mean()), 4),
        'lateral_max_m': round(float(lateral_abs.max()), 4),
        'lateral_std_m': round(float(lateral.std()), 4),
        'depth_mean_m': round(float(depth.mean()), 3),
        'depth_std_m': round(float(depth.std()), 4),
        'target_distance_m': args.target_distance,
        'lateral_tol_m': args.lateral_tol,
        'reached_target': reached,
        'kept_lateral_tol': kept_line,
        'success': success,
    }

    print('=' * 68)
    print(f'  闭环性能评价 — {os.path.basename(args.log)}')
    print('=' * 68)
    print(markdown_table(
        ['指标', '数值'],
        [['航行时长 (s)', metrics['duration_s']],
         ['前进距离 (m)', metrics['forward_distance_m']],
         ['平均前进速度 (m/s)', metrics['mean_forward_speed_ms']],
         ['横向偏差 平均 (m)', metrics['lateral_mae_m']],
         ['横向偏差 最大 (m)', metrics['lateral_max_m']],
         ['横向偏差 标准差 (m)', metrics['lateral_std_m']],
         ['深度 平均 (m)', metrics['depth_mean_m']],
         ['深度 标准差 (m)', metrics['depth_std_m']],
         [f"到达目标 {args.target_distance} m", '是' if reached else '否'],
         [f"横向偏差 ≤ {args.lateral_tol} m", '是' if kept_line else '否'],
         ['任务成功', '✅ 是' if success else '❌ 否']]))
    print('=' * 68)

    out = os.path.abspath(args.out or
                          os.path.splitext(os.path.abspath(args.log))[0] + '_report.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f'  ✅ 指标已写入: {out}')

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(along, lateral)
            ax[0].axhline(args.lateral_tol, color='r', ls='--', lw=1)
            ax[0].axhline(-args.lateral_tol, color='r', ls='--', lw=1)
            ax[0].set_xlabel('Forward distance (m)')
            ax[0].set_ylabel('Lateral deviation (m)')
            ax[0].set_title('Lateral tracking')
            ax[0].grid(alpha=0.3)
            ax[1].plot(t, depth, label='depth')
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Depth (m)')
            ax[1].set_title('Depth')
            ax[1].grid(alpha=0.3)
            fig.tight_layout()
            png = os.path.splitext(out)[0] + '.png'
            fig.savefig(png, dpi=140)
            plt.close(fig)
            print(f'  ✅ 曲线图: {png}')
        except ImportError:
            print('  （未安装 matplotlib，跳过绘图）')
    return 0


# --------------------------------------------------------------------------- #
# 4) 多场景汇总对比
# --------------------------------------------------------------------------- #
def cmd_compare(args):
    """汇总多个场景的 report.json，生成横向对比表。"""
    rows = []
    for path in args.reports:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            print(f'  跳过（不存在）: {path}')
            continue
        with open(path, 'r', encoding='utf-8') as fh:
            m = json.load(fh)
        name = os.path.basename(path).replace('_report.json', '')
        rows.append([
            name,
            m.get('forward_distance_m', '-'),
            m.get('mean_forward_speed_ms', '-'),
            m.get('lateral_mae_m', '-'),
            m.get('lateral_max_m', '-'),
            m.get('depth_std_m', '-'),
            '✅' if m.get('success') else '❌',
        ])

    if not rows:
        print('没有可汇总的结果。')
        return 1

    table = markdown_table(
        ['场景', '前进距离(m)', '平均速度(m/s)', '横向MAE(m)',
         '横向最大(m)', '深度σ(m)', '成功'], rows)

    print('=' * 78)
    print('  多场景性能对比')
    print('=' * 78)
    print(table)
    print('=' * 78)

    n_success = sum(1 for r in rows if r[-1] == '✅')
    print(f'  成功率: {n_success}/{len(rows)} = {n_success / len(rows) * 100:.1f}%')

    if args.out:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write('## 多场景性能评价\n\n')
            fh.write(table + '\n\n')
            fh.write(f'成功率：**{n_success}/{len(rows)} '
                     f'= {n_success / len(rows) * 100:.1f}%**\n')
        print(f'  ✅ 对比表已写入: {out}')
    return 0


# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        prog='evaluate.py',
        description='rov_e2e 端到端控制性能评价',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    sub = parser.add_subparsers(dest='mode')

    p = sub.add_parser('offline', help='开环评价：精度与推理耗时')
    p.add_argument('--model', default=os.path.join(HERE, '..', 'model.pth'))
    p.add_argument('--data', default=os.path.join(HERE, '..', 'dataset'))
    p.add_argument('--samples', type=int, default=0, help='0 = 使用全部样本')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_offline)

    p = sub.add_parser('record', help='记录一次闭环航行的 /rov/odom 到 CSV')
    p.add_argument('--topic', default='/rov/odom')
    p.add_argument('--out', required=True)
    p.add_argument('--duration', type=float, default=30.0)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser('report', help='从里程计 CSV 计算闭环性能指标')
    p.add_argument('--log', required=True)
    p.add_argument('--target-distance', type=float, default=5.0,
                   help='期望前进距离 (m)')
    p.add_argument('--lateral-tol', type=float, default=0.5,
                   help='允许的最大横向偏差 (m)')
    p.add_argument('--out', default=None)
    p.add_argument('--plot', action='store_true')
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('compare', help='汇总多场景结果生成对比表')
    p.add_argument('--reports', nargs='+', required=True)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'mode', None):
        parser.print_help()
        return 1
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
