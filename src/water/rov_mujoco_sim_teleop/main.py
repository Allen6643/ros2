#!/usr/bin/env python3
"""
rov_mujoco_sim_teleop 统一入口（main 脚本）
===========================================
作业要求模块入口以 ``main.`` 开头，且可通过 main 脚本直接跑通整个模块。

    python3 src/water/rov_mujoco_sim_teleop/main.py            # = run，一键启动仿真
    python3 src/water/rov_mujoco_sim_teleop/main.py collect    # 采集数据集
    python3 src/water/rov_mujoco_sim_teleop/main.py train      # 训练神经网络
    python3 src/water/rov_mujoco_sim_teleop/main.py test       # 离线测试权重
    python3 src/water/rov_mujoco_sim_teleop/main.py run        # 启动仿真+遥控
    python3 src/water/rov_mujoco_sim_teleop/main.py auto       # 启动仿真+神经网络闭环

也可以使用 shell 包装（会先 source ROS 环境并设置 MUJOCO_GL=osmesa）：

    bash src/water/rov_mujoco_sim_teleop/main.sh run

设计说明
--------
``collect`` / ``train`` / ``test`` 三个子命令**在本进程内**直接调用对应模块的
``main()``，保持异常栈与日志完整；``run`` / ``auto`` 需要 ROS 2 launch 的进程
编排能力，因此转交 ``ros2 launch``。
"""

import argparse
import os
import subprocess
import sys

# 无显卡虚拟机必须使用 OSMesa 软件渲染；必须在 import mujoco 之前生效
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_NAME = 'rov_mujoco_sim_teleop'


def _run_module_main(module_name):
    """在本进程内调用 rov_mujoco_sim_teleop.<module>:main()。"""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    mod = __import__(f'{PKG_NAME}.{module_name}', fromlist=['main'])
    return mod.main()


def cmd_collect(args):
    sys.argv = ['collect_data'] + args.rest
    return _run_module_main('collect_data')


def cmd_train(args):
    sys.argv = ['train'] + args.rest
    return _run_module_main('train')


def cmd_test(args):
    sys.argv = ['test_infer'] + args.rest
    return _run_module_main('test_infer')


def _launch(extra, rest):
    cmd = ['ros2', 'launch', PKG_NAME, 'main.launch.py'] + extra + rest
    print('[rov_mujoco_sim_teleop] $ ' + ' '.join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        sys.stderr.write(
            '\n找不到 ros2 命令。请先加载 ROS 2 环境：\n'
            '    source /opt/ros/humble/setup.bash\n'
            '    source install/setup.bash\n'
            '或使用 shell 包装：bash src/water/rov_mujoco_sim_teleop/main.sh run\n\n')
        return 127


def cmd_run(args):
    """启动仿真 + 键盘遥控（人工操控）。"""
    return _launch(['mode:=manual'], args.rest)


def cmd_auto(args):
    """启动仿真 + 神经网络感知 + 神经网络轨迹跟踪（自动闭环）。"""
    return _launch(['mode:=auto'], args.rest)


def build_parser():
    p = argparse.ArgumentParser(
        prog='main.py',
        description='水下 ROV MuJoCo 仿真、键盘遥控与神经网络感知控制 —— 模块统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python3 main.py collect --samples 4000   采集数据集\n'
            '  python3 main.py train --epochs 50        训练两个神经网络\n'
            '  python3 main.py test                     离线测试权重\n'
            '  python3 main.py run                      启动仿真+键盘遥控\n'
            '  python3 main.py auto                     启动仿真+神经网络闭环\n'))
    sub = p.add_subparsers(dest='command')

    for name, fn, help_text, tail in (
            ('collect', cmd_collect, '采集数据集（传感器感知 + 轨迹控制）', '采集参数'),
            ('train', cmd_train, '训练传感器感知网络与轨迹跟踪网络', '训练参数'),
            ('test', cmd_test, '离线测试两个网络的权重', '测试参数'),
            ('run', cmd_run, '一键启动 MuJoCo 仿真 + 键盘遥控', 'launch 参数'),
            ('auto', cmd_auto, '一键启动 MuJoCo 仿真 + 神经网络闭环', 'launch 参数')):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument('rest', nargs=argparse.REMAINDER, help=f'透传给{tail}')
        sp.set_defaults(func=fn)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        # 不带子命令时默认执行 run，方便「直接跑整个模块」
        argv = list(argv) if argv is not None else sys.argv[1:]
        args = parser.parse_args(['run'] + argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
