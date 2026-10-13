#!/usr/bin/env python3
"""
数据集采集脚本（单目视觉 → 前进推力 / 转向 的行为克隆）
=========================================================
通过键盘手动操纵 MuJoCo 中的水下 ROV，同步保存 **前视相机画面** 与
**当时的控制标签 [前进推力, 转向]**，构成端到端模仿学习数据集。

核心设计：直接复用真实仿真节点
------------------------------
本脚本 **不重新实现一份物理**，而是直接实例化 `rov_mujoco.mujoco_sim_node.MujocoSimNode`，
并采用 **单线程 `spin_once` 循环** 交替推进物理与采样图像：

    while running:
        rclpy.spin_once(node, timeout_sec=0.0)   # 触发 500Hz 物理步进定时器
        key = read_key()                         # 读键盘
        node.cmd_vel = ...                       # 注入控制指令
        if 到采样周期:
            frame = node.camera.render()         # 与仿真完全同源的相机
            保存 (frame, label)

这样做的好处：
1. **动力学零差异**：训练数据与 `ros2 launch` 部署时的物理完全一致，
   不存在"采集用一套动力学、部署用另一套"的隐性偏差；
2. **单线程安全**：MuJoCo 不是线程安全的，用 `spin_once` 交替而非
   多线程并发，避免渲染与步进竞争；
3. **无需 rclpy 多进程编排**，一条命令即可采集。

按键（**锁存式**：按一次持续生效，直到按其他方向键或空格）
--------------------------------------------------------
    W / ↑    前进推力 +1
    S / ↓    后退推力 -1
    A / ←    左转 +1
    D / →    右转 -1
    空格     全部归零（悬停）
    R        重置 ROV 到初始位姿
    Q        保存并退出

注意：本脚本把 A/D 用作 **转向**（与 `keyboard_teleop_node` 中 A/D 为横向平移不同），
因为端到端任务只需要"前进 + 转向"两个自由度，这样单手即可完成录制。

用法
----
    source /opt/ros/humble/setup.bash
    source install/setup.bash

    # 静水环境采集 600 帧到默认 dataset/ 目录
    python3 src/water/rov_e2e/rov_e2e/collect_data.py --frames 600

    # 无显卡虚拟机：必须让 MuJoCo 走软件渲染
    export MUJOCO_GL=osmesa
"""

import argparse
import csv
import os
import select
import sys
import termios
import time
import tty

# ---------------------------------------------------------------------------
# 关键：MUJOCO_GL 必须在 `import mujoco` 之前生效。
# 无 GPU 的虚拟机若不设置，mujoco.Renderer 会因拿不到 GL 上下文而失败。
# ---------------------------------------------------------------------------
os.environ.setdefault('MUJOCO_GL', 'osmesa')

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

try:
    import rclpy
except ImportError:
    sys.stderr.write(
        '\n【环境依赖缺失】未检测到 rclpy。请先加载 ROS 2 环境：\n'
        '    source /opt/ros/humble/setup.bash\n'
        '    source install/setup.bash\n\n'
    )
    raise


#: 锁存键位表：按键 → (前进推力, 转向)
KEY_TO_ACTION = {
    'w': (1.0, 0.0), 'KEY_UP': (1.0, 0.0),
    's': (-1.0, 0.0), 'KEY_DOWN': (-1.0, 0.0),
    'a': (0.0, 1.0), 'KEY_LEFT': (0.0, 1.0),
    'd': (0.0, -1.0), 'KEY_RIGHT': (0.0, -1.0),
}

#: 同时按前进+转向的复合键位（斜向行驶，让数据覆盖转弯中的前进）
KEY_TO_ACTION.update({
    'q': (1.0, 1.0),
    'e': (1.0, -1.0),
})


def visual_servo_action(frame, gain=20.0):
    """视觉伺服教师：由**画面本身**给出 ``[前进推力, 转向]`` 标签。

    为什么要这个模式
    ----------------
    行为克隆要求标签与观测之间存在因果关系。若在无人值守时"盲打"按键，
    标签与画面无关，网络只能学到常数输出。本函数用一个确定性的视觉伺服律
    充当"专家"，其输出是图像的函数，因此 CNN 可以稳定拟合它。

    伺服律：以灰度图的**水平亮度重心**作为艏向误差代理量，令其趋近画面中心。

        e = (Σ_x x·I(x) / Σ_x I(x)) / W − 0.5
        turn    = clip(−k · e, −1, 1)
        forward = clip(1 − 0.5·|turn|, 0, 1)      # 转弯减速

    即"画面重心偏哪边就往反方向转"，并使 ROV 保持视野平衡地前进。

    Parameters
    ----------
    frame : np.ndarray
        ``(H, W, 3)`` 的 uint8 RGB 图像。
    gain : float
        转向增益 k（重心偏移通常在 ±0.05 量级，故 k 取 20 左右）。

    Returns
    -------
    (forward, turn) : tuple[float, float]
        归一化到 [-1, 1] 的控制标签。
    """
    gray = frame.astype(np.float32).mean(axis=2)
    col = gray.mean(axis=0)
    w = col.shape[0]
    centroid = float((col * np.arange(w)).sum() / (col.sum() + 1e-6) / w)
    err = centroid - 0.5
    turn = float(np.clip(-gain * err, -1.0, 1.0))
    # 转向越急越减速：既符合"转弯收油门"的驾驶直觉，
    # 也让前进通道成为图像的非平凡函数（否则 forward 恒为 1，网络只学到常数）
    forward = float(np.clip(1.0 - 0.5 * abs(turn), 0.0, 1.0))
    return forward, turn


class TerminalKeyboard:
    """终端原始模式非阻塞按键读取（支持 ANSI 方向键转义序列）。

    与 `rov_mujoco.keyboard_teleop_node` 中的实现保持一致，
    退出时确保恢复终端属性，避免把用户终端留在 raw 模式。
    """

    def __init__(self):
        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except Exception:
            self.settings = None

    def read_key(self, timeout=0.0):
        if self.settings is None:
            return ''
        try:
            tty.setraw(sys.stdin.fileno())
            key = ''
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                c = sys.stdin.read(1)
                if c == '\x1b':
                    r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r2 and sys.stdin.read(1) == '[':
                        r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r3:
                            c3 = sys.stdin.read(1)
                            key = {'A': 'KEY_UP', 'B': 'KEY_DOWN',
                                   'C': 'KEY_RIGHT', 'D': 'KEY_LEFT'}.get(c3, '')
                else:
                    key = c
            return key
        except Exception:
            return ''
        finally:
            if self.settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
                except Exception:
                    pass

    def restore(self):
        if self.settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            except Exception:
                pass


def banner(paths):
    print("""
╔══════════════════════════════════════════════════════════════╗
║   水下 ROV 端到端数据集采集（前视相机 → 前进/转向）           ║
╠══════════════════════════════════════════════════════════════╣
║  W / ↑  前进      S / ↓  后退      A / ←  左转   D / →  右转 ║
║  Q / E  前进+左转 / 前进+右转                                ║
║  空格   归零悬停            R  重置位姿            X  退出    ║
╠══════════════════════════════════════════════════════════════╣
║  按键为【锁存式】：按一次持续生效，直到按其他方向键或空格。    ║
╚══════════════════════════════════════════════════════════════╝""")
    print(f'  图像目录: {paths["images"]}')
    print(f'  标签文件: {paths["labels"]}\n')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='采集水下 ROV 单目视觉端到端控制数据集')
    parser.add_argument('--output', default=None,
                        help='数据集根目录（默认 <包目录>/dataset）')
    parser.add_argument('--frames', type=int, default=600,
                        help='目标采集帧数（默认 600，课程作业 300~600 足够）')
    parser.add_argument('--sample-rate', type=float, default=10.0,
                        help='图像采样频率 Hz（默认 10；CPU 软件渲染下不要调太高）')
    parser.add_argument('--width', type=int, default=160, help='图像宽度')
    parser.add_argument('--height', type=int, default=120, help='图像高度')
    parser.add_argument('--ocean-current', action='store_true',
                        help='启用洋流扰动（默认关闭 = 静水环境，便于学习）')
    parser.add_argument('--reset-every', type=int, default=0,
                        help='每采集 N 帧自动重置位姿（0 = 不自动重置）')
    parser.add_argument('--auto', action='store_true',
                        help='自动标注模式：用视觉伺服教师（见 visual_servo_action）'
                             '由画面生成标签，适合无人值守批量采数据')
    parser.add_argument('--auto-gain', type=float, default=20.0,
                        help='自动标注模式的转向增益 k（默认 20）')
    args = parser.parse_args(argv)

    # ---- 数据集路径 ----
    if args.output:
        root = os.path.abspath(args.output)
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'dataset')
        root = os.path.abspath(root)
    img_dir = os.path.join(root, 'images')
    os.makedirs(img_dir, exist_ok=True)
    label_path = os.path.join(root, 'labels.csv')
    paths = {'images': img_dir, 'labels': label_path}
    banner(paths)

    # ---- 导入并实例化真实仿真节点 ----
    from rov_mujoco.mujoco_sim_node import MujocoSimNode

    rclpy.init()
    node = MujocoSimNode()

    # 静水环境：关闭洋流（配置里默认开启，这里显式覆盖）
    if not args.ocean_current:
        node.ocean_current = None
        node.get_logger().info('静水环境：洋流模型已关闭')

    # 相机：必须显式开启真实渲染器。rov_mujoco.camera_sensor.UnderwaterCamera 的
    # enable_renderer 默认为 False，此时 render() 会返回**固定的合成底色图像**
    # （只有蓝绿渐变，不含任何场景信息），用它训练出来的网络学不到任何视觉线索。
    from rov_mujoco.camera_sensor import UnderwaterCamera
    node.camera = UnderwaterCamera(node.model, node.data,
                                   camera_name='forward_cam',
                                   width=args.width, height=args.height,
                                   enable_renderer=True)
    if node.camera.renderer is None:
        node.get_logger().error('相机渲染器初始化失败，无法采集真实场景图像。')
        node.get_logger().error(
            '无显卡虚拟机请执行: export MUJOCO_GL=osmesa，'
            '并安装 sudo apt install libosmesa6 与 pip3 install PyOpenGL')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    kb = TerminalKeyboard()
    forward, turn = 0.0, 0.0
    saved = 0
    sample_period = 1.0 / max(args.sample_rate, 0.1)
    next_sample = time.time()
    label_fh = open(label_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(label_fh)
    writer.writerow(['filename', 'forward', 'turn'])

    node.get_logger().info(
        f'开始采集：目标 {args.frames} 帧 @ {args.sample_rate} Hz，'
        f'分辨率 {args.width}x{args.height}')

    try:
        while saved < args.frames and rclpy.ok():
            # 1) 推进物理：定时器在 spin_once 中触发，单线程安全
            rclpy.spin_once(node, timeout_sec=0.0)

            # 2) 读键盘并更新锁存指令
            key = kb.read_key(timeout=0.0)
            if key in KEY_TO_ACTION:
                forward, turn = KEY_TO_ACTION[key]
            elif key == ' ':
                forward, turn = 0.0, 0.0
            elif key == 'r':
                node._init_robot_pose()
                node.target_depth = -1.0
                forward, turn = 0.0, 0.0
                node.get_logger().info('已重置 ROV 位姿')
            elif key == 'x':
                node.get_logger().info('收到退出指令，提前结束采集')
                break

            # 3) 把锁存指令写入仿真节点（linear.x=推力, angular.z=转向）
            node.cmd_vel.linear.x = forward
            node.cmd_vel.linear.y = 0.0
            node.cmd_vel.linear.z = 0.0
            node.cmd_vel.angular.z = turn

            # 4) 到采样周期就存一帧
            now = time.time()
            if now >= next_sample:
                next_sample = now + sample_period
                frame = node.camera.render()
                if frame is None:
                    continue
                # 自动标注模式：标签由画面本身经视觉伺服给出，保证标签与观测
                # 存在因果关系（否则网络只能学到常数输出）
                if args.auto:
                    forward, turn = visual_servo_action(frame, args.auto_gain)
                name = f'{saved:06d}.png'
                Image.fromarray(frame).save(os.path.join(img_dir, name))
                writer.writerow([name, f'{forward:.3f}', f'{turn:.3f}'])
                saved += 1

                if saved % 20 == 0:
                    pos = node.data.sensor('pos').data[:3]
                    node.get_logger().info(
                        f'已采集 {saved}/{args.frames} 帧 | '
                        f'位置=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}) | '
                        f'指令=({forward:+.1f}, {turn:+.1f})')

                if args.reset_every and saved % args.reset_every == 0:
                    node._init_robot_pose()
                    node.target_depth = -1.0

    except KeyboardInterrupt:
        pass
    finally:
        label_fh.close()
        kb.restore()
        try:
            node.camera.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

    print(f'\n采集结束：共保存 {saved} 帧')
    print(f'  图像: {img_dir}')
    print(f'  标签: {label_path}')
    if saved < 30:
        print('  提示：样本过少，建议至少采集 300 帧再训练。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
