"""
数据集采集脚本
==============
在 MuJoCo 中随机采样 ROV 位姿，渲染前视相机图像并采集声呐测距，
同时用确定性规则生成标签，产出两个数据集：

  1. 视觉端到端数据集：图像 -> [前进推力, 转向]
  2. 声呐建图数据集  ：8 维测距 -> 开阔方向三分类

**标签怎么来（关键设计）**
--------------------------
学习型控制要求标签与输入存在因果关系，否则网络只能学到常数输出。

* **视觉端到端标签**：用一个"视觉伺服"规则充当专家 —— 以图像底部区域的
  水平亮度重心作为航向误差代理量，令 ROV 转向使该重心回到画面中央。
  这个标签**确实是图像的函数**，因此 CNN 可以稳定拟合。
* **声呐标签**：由三个方向的测距几何关系直接判定哪一侧更开阔：
  前向测距最大记"正前开阔"，左侧平均测距最大记"左侧开阔"，否则"右侧开阔"。

用法
----
    python3 collect_data.py --samples 3000 --out dataset
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
    candidates = [
        os.path.join(HERE, '..', 'models', 'rov_vision_channel.xml'),
        os.path.join(os.getcwd(), 'models', 'rov_vision_channel.xml'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.join(get_package_share_directory('rov_vision_e2e_nav'),
                                       'models', 'rov_vision_channel.xml'))
    except Exception:  # noqa: BLE001
        pass
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return os.path.abspath(candidates[0])


def visual_servo_label(frame, gain=3.0):
    """视觉伺服专家：由**画面本身**给出 [前进推力, 转向] 标签。

    伺服律：以图像灰度水平亮度重心作为航向误差代理量，令其趋近画面中心。

        e = (Σ_x x·I(x) / Σ_x I(x)) / W − 0.5
        turn    = clip(−k · e, −1, 1)
        forward = clip(1 − 0.5·|turn|, 0, 1)      # 转弯减速

    标签是图像的可微函数，因此 CNN 能稳定拟合；
    该伺服律本身是合理的控制律，因此学到的策略在闭环中行为可解释。
    """
    gray = frame.astype(np.float32).mean(axis=2)
    col = gray.mean(axis=0)
    w = col.shape[0]
    centroid = float((col * np.arange(w)).sum() / (col.sum() + 1e-6) / w)
    err = centroid - 0.5
    turn = float(np.clip(-gain * err, -1.0, 1.0))
    forward = float(np.clip(1.0 - 0.5 * abs(turn), 0.0, 1.0))
    return forward, turn


def sonar_label(ranges):
    """由声呐测距扇判定哪一侧更开阔。

    Parameters
    ----------
    ranges : dict
        键为 'f','fl','l','bl','fr','r','br','down'，值为距离（米）

    Returns
    -------
    int
        0 = 左侧开阔，1 = 正前开阔，2 = 右侧开阔
    """
    left = np.mean([ranges['fl'], ranges['l'], ranges['bl']])
    right = np.mean([ranges['fr'], ranges['r'], ranges['br']])
    front = ranges['f']
    if front >= max(left, right):
        return 1
    if left >= right:
        return 0
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description='采集视觉端到端与声呐建图数据集')
    parser.add_argument('--model', default=default_model_path())
    parser.add_argument('--out', default=os.path.join(HERE, '..', 'dataset'))
    parser.add_argument('--samples', type=int, default=3000)
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    model_path = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.exists(model_path):
        sys.stderr.write(f'找不到模型: {model_path}\n')
        return 1
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rov')
    qadr = model.jnt_qposadr[model.body_jntadr[body]]

    print('=' * 66)
    print('  采集数据集（视觉端到端 + 声呐建图）')
    print('=' * 66)
    print(f'  模型: {model_path}')
    print(f'  采样: {args.samples} 条 | 图像 {args.width}x{args.height}')
    print('-' * 66)

    try:
        renderer = mujoco.Renderer(model, args.height, args.width)
    except Exception as err:  # noqa: BLE001
        sys.stderr.write(
            f'!! 相机渲染器创建失败: {err}\n'
            f'   无显卡环境请设置 MUJOCO_GL=osmesa 并安装 libosmesa6\n')
        return 1

    rng = np.random.RandomState(args.seed)
    images = np.zeros((args.samples, args.height, args.width, 3), dtype=np.uint8)
    e2e_y = np.zeros((args.samples, 2), dtype=np.float32)
    sonar_X = np.zeros((args.samples, 8), dtype=np.float32)
    sonar_y = np.zeros((args.samples,), dtype=np.int64)
    dirs = ('f', 'fl', 'l', 'bl', 'fr', 'r', 'br', 'down')

    for i in range(args.samples):
        # 随机位姿（在通道直行段内）
        data.qpos[qadr + 0] = rng.uniform(0.0, 6.0)
        data.qpos[qadr + 1] = rng.uniform(-1.0, 1.0)
        data.qpos[qadr + 2] = rng.uniform(-1.5, -0.6)
        half = rng.uniform(-math.pi, math.pi) / 2.0
        data.qpos[qadr + 3] = math.cos(half)
        data.qpos[qadr + 4] = 0.0
        data.qpos[qadr + 5] = 0.0
        data.qpos[qadr + 6] = math.sin(half)
        data.qvel[:] = rng.uniform(-0.4, 0.4, size=data.qvel.shape)
        mujoco.mj_forward(model, data)

        # 渲染前视相机
        renderer.update_scene(data, camera='forward_cam')
        frame = renderer.render()
        images[i] = frame
        e2e_y[i] = visual_servo_label(frame)

        # 声呐测距
        rr = {}
        for d in dirs:
            v = float(data.sensor(f'sonar_range_{d}').data[0])
            rr[d] = 20.0 if v < 0 else v
        sonar_X[i] = [rr[d] for d in dirs]
        sonar_y[i] = sonar_label(rr)

        if (i + 1) % 500 == 0:
            print(f'  已采样 {i + 1}/{args.samples}')

    e2e_path = os.path.join(out_dir, 'e2e_dataset.npz')
    sonar_path = os.path.join(out_dir, 'sonar_dataset.npz')
    np.savez_compressed(e2e_path, images=images, y=e2e_y)
    np.savez_compressed(sonar_path, X=sonar_X, y=sonar_y)

    print('-' * 66)
    print(f'  ✅ 视觉端到端数据集: {e2e_path}')
    print(f'     图像 {images.shape} | 标签 {e2e_y.shape}')
    print(f'     前进推力范围 [{e2e_y[:, 0].min():+.3f}, {e2e_y[:, 0].max():+.3f}]')
    print(f'     转向范围     [{e2e_y[:, 1].min():+.3f}, {e2e_y[:, 1].max():+.3f}]')
    print(f'  ✅ 声呐建图数据集: {sonar_path}')
    print(f'     特征 {sonar_X.shape} | 标签分布 {np.bincount(sonar_y, minlength=3).tolist()}')
    print('=' * 66)
    return 0


if __name__ == '__main__':
    sys.exit(main())
