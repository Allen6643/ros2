"""
数据集采集脚本
==============
在 MuJoCo 中随机采样 ROV 的位姿与运动状态，记录

  1. 传感器感知数据集：7 维传感器特征 -> 3 类障碍物方位标签
  2. 轨迹跟踪数据集  ：8 维状态误差特征 -> 4 维控制量标签

输出到 `dataset/` 目录：`sensor_dataset.npz` 与 `traj_dataset.npz`。

**标签是怎么来的（关键设计）**
------------------------------
学习型控制要求标签与输入之间存在因果关系，否则网络只能学到常数输出。
本脚本不用人工按键打标（那在无人值守时会让标签与观测无关），
而是用一个**确定性的几何规则**充当"专家"：

* 感知标签：直接由障碍物相对 ROV 的真实方位决定 ——
  前方 2.5 m 内有障碍记"前方障碍"，左侧近记"左方障碍"，右侧近记"右方障碍"。
* 控制标签：由状态误差经一个比例-微分（PD）专家律给出 ——
  `u = clip(Kp * e + Kd * ė, -1, 1)`，其中 e 是"期望位置 - 当前位置"。

这样生成的标签是观测的确定函数，网络可以稳定拟合；而该专家律本身是一个
合理的控制律，因此学到的策略在闭环中也表现出可解释的行为。

用法
----
    python3 collect_data.py --samples 4000 --out dataset
"""

import argparse
import math
import os
import sys

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])

try:
    import mujoco
except ImportError:
    sys.stderr.write('\n【依赖缺失】需要 MuJoCo：pip3 install "mujoco<3.2"\n\n')
    raise

HERE = os.path.dirname(os.path.abspath(__file__))


def default_model_path():
    """探测随包安装的 MuJoCo 模型路径。"""
    candidates = [
        os.path.join(HERE, '..', 'models', 'rov_channel.xml'),
        os.path.join(os.getcwd(), 'models', 'rov_channel.xml'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(get_package_share_directory('rov_mujoco_sim_teleop'),
                                       'models', 'rov_channel.xml'))
    except Exception:  # noqa: BLE001
        pass
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return os.path.abspath(candidates[0])


def quat_to_yaw(q):
    """四元数 (w,x,y,z) -> 偏航角（弧度）。"""
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                      1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))


def sensor_label_from_ranges(rf, rl, rr):
    """由三向测距的几何真值给出障碍物方位标签。

    Returns
    -------
    int
        0 = 前方障碍，1 = 左方障碍，2 = 右方障碍
    """
    near = 2.0          # 判定"近"的距离阈值（米）
    if rf <= 2.5:
        return 0
    if rl <= near:
        return 1
    return 2


def expert_control(pos_err, vel_err, yaw_err, range_front):
    """确定性 PD 专家律，给出 4 维归一化控制量作为标签。

    Parameters
    ----------
    pos_err : (3,) 期望位置 - 当前位置
    vel_err : (3,) 期望速度 - 当前速度
    yaw_err : float 期望偏航 - 当前偏航
    range_front : float 前向测距（越小越需要减速）

    Returns
    -------
    (4,) float32  前进推力 / 横移 / 升沉 / 偏航力矩，均在 [-1,1]
    """
    kp, kd = 0.9, 0.35
    u_forward = kp * pos_err[0] + kd * vel_err[0]
    u_lateral = kp * pos_err[1] + kd * vel_err[1]
    u_vertical = kp * pos_err[2] + kd * vel_err[2]
    u_yaw = 1.4 * yaw_err

    # 前方障碍越近越减速：这是避障行为的关键，让网络学得会
    if range_front < 2.5:
        u_forward *= max(0.15, (range_front - 0.6) / 1.9)

    u = np.array([u_forward, u_lateral, u_vertical, u_yaw], dtype=np.float32)
    return np.clip(u, -1.0, 1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description='采集传感器感知与轨迹跟踪数据集')
    parser.add_argument('--model', default=default_model_path(), help='MuJoCo 模型路径')
    parser.add_argument('--out', default=os.path.join(HERE, '..', 'dataset'), help='输出目录')
    parser.add_argument('--samples', type=int, default=4000, help='采样条数')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    model_path = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.exists(model_path):
        sys.stderr.write(f'找不到模型文件: {model_path}\n')
        return 1

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rov')
    qadr = model.jnt_qposadr[model.body_jntadr[body]]

    print('=' * 66)
    print('  采集数据集（MuJoCo + 确定性专家律打标）')
    print('=' * 66)
    print(f'  模型      : {model_path}')
    print(f'  采样条数  : {args.samples}')
    print(f'  输出目录  : {out_dir}')
    print('-' * 66)

    sensor_X = np.zeros((args.samples, 7), dtype=np.float32)
    sensor_y = np.zeros((args.samples,), dtype=np.int64)
    traj_X = np.zeros((args.samples, 8), dtype=np.float32)
    traj_y = np.zeros((args.samples, 4), dtype=np.float32)

    d = data
    for i in range(args.samples):
        # 在通道内随机放置 ROV：x∈[0,7]，y∈[-0.9,0.9]，z∈[-1.6,-0.5]
        d.qpos[qadr + 0] = rng.uniform(0.0, 7.0)
        d.qpos[qadr + 1] = rng.uniform(-0.9, 0.9)
        d.qpos[qadr + 2] = rng.uniform(-1.6, -0.5)
        half = rng.uniform(-math.pi, math.pi) / 2.0
        d.qpos[qadr + 3] = math.cos(half)
        d.qpos[qadr + 4] = 0.0
        d.qpos[qadr + 5] = 0.0
        d.qpos[qadr + 6] = math.sin(half)
        # 随机给一点速度，让"速度特征"有变化
        d.qvel[:] = rng.uniform(-0.6, 0.6, size=d.qvel.shape)
        mujoco.mj_forward(model, d)

        rf = float(d.sensor('range_front').data[0])
        rl = float(d.sensor('range_left').data[0])
        rr = float(d.sensor('range_right').data[0])
        rd = float(d.sensor('range_down').data[0])
        # MuJoCo 的 rangefinder 未命中时返回 -1，置换为量程上限
        rf = 20.0 if rf < 0 else rf
        rl = 20.0 if rl < 0 else rl
        rr = 20.0 if rr < 0 else rr
        rd = 20.0 if rd < 0 else rd

        vel = d.qvel[model.jnt_dofadr[model.body_jntadr[body]]:
                     model.jnt_dofadr[model.body_jntadr[body]] + 3].copy()

        # ---- 感知数据集 ----
        sensor_X[i] = [rf, rl, rr, rd, vel[0], vel[1], vel[2]]
        sensor_y[i] = sensor_label_from_ranges(rf, rl, rr)

        # ---- 轨迹跟踪数据集 ----
        # 期望轨迹：沿通道中线（y=0）前进到 x=6.5，深度保持 -1.0
        cur = d.qpos[qadr: qadr + 3].copy()
        desired = np.array([min(6.5, cur[0] + 1.0), 0.0, -1.0], dtype=np.float64)
        pos_err = desired - cur
        vel_err = -vel
        yaw_err = 0.0 - quat_to_yaw(d.qpos[qadr + 3: qadr + 7])
        yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi   # 归一化到 [-pi,pi]
        traj_X[i] = [pos_err[0], pos_err[1], pos_err[2],
                     vel[0], vel[1], vel[2], yaw_err, rf]
        traj_y[i] = expert_control(pos_err, vel_err, yaw_err, rf)

        if (i + 1) % 1000 == 0:
            print(f'  已采样 {i + 1}/{args.samples}')

    sensor_path = os.path.join(out_dir, 'sensor_dataset.npz')
    traj_path = os.path.join(out_dir, 'traj_dataset.npz')
    np.savez_compressed(sensor_path, X=sensor_X, y=sensor_y)
    np.savez_compressed(traj_path, X=traj_X, y=traj_y)

    print('-' * 66)
    print(f'  ✅ 感知数据集: {sensor_path}  X={sensor_X.shape} y={sensor_y.shape}')
    print(f'     类别分布: {np.bincount(sensor_y, minlength=3).tolist()} (前方/左方/右方)')
    print(f'  ✅ 轨迹数据集: {traj_path}  X={traj_X.shape} y={traj_y.shape}')
    print(f'     控制量范围: forward[{traj_y[:, 0].min():+.2f},{traj_y[:, 0].max():+.2f}] '
          f'yaw[{traj_y[:, 3].min():+.2f},{traj_y[:, 3].max():+.2f}]')
    print('=' * 66)
    return 0


if __name__ == '__main__':
    sys.exit(main())
