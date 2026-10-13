#!/usr/bin/env python3
"""
rov_e2e 统一入口（main 脚本）
=============================
按作业要求，本模块的入口以 ``main.`` 开头，可通过本脚本直接运行整个模块：

    python3 src/water/rov_e2e/main.py            # 默认：一键启动完整闭环
    python3 src/water/rov_e2e/main.py collect    # 采集数据集
    python3 src/water/rov_e2e/main.py train      # 训练网络 → model.pth
    python3 src/water/rov_e2e/main.py test       # 离线测试 model.pth
    python3 src/water/rov_e2e/main.py run        # 启动 MuJoCo 仿真 + 推理节点

也可以使用 shell 包装（会先 source ROS 环境并设置 MUJOCO_GL）：

    bash src/water/rov_e2e/main.sh run

设计说明
--------
``collect`` / ``train`` / ``test`` 三个子命令**直接在本进程内调用**对应模块的 ``main()``，
而不是起子进程，这样异常栈与日志保持完整、便于排查；
``run`` 子命令需要 ROS 2 launch 的进程编排能力，因此走 ``ros2 launch``。
"""

import argparse
import os
import subprocess
import sys

# 无显卡虚拟机需要 OSMesa 软件渲染；必须在 import mujoco 之前生效
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(HERE, 'rov_e2e')          # 内层同名包目录（放 Python 源码）


def _run_module_main(module_name):
    """在本进程内调用 rov_e2e.<module>:main()。"""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    mod = __import__(f'rov_e2e.{module_name}', fromlist=['main'])
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


def cmd_run(args):
    """通过 ros2 launch 启动 MuJoCo 仿真 + 端到端推理节点（完整闭环）。"""
    cmd = ['ros2', 'launch', 'rov_e2e', 'main.launch.py'] + args.rest
    print('[rov_e2e] $ ' + ' '.join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        sys.stderr.write(
            '\n找不到 ros2 命令。请先加载 ROS 2 环境：\n'
            '    source /opt/ros/humble/setup.bash\n'
            '    source install/setup.bash\n'
            '或直接使用 shell 包装：bash src/water/rov_e2e/main.sh run\n\n')
        return 127


def cmd_evaluate(args):
    """性能评价：开环精度/推理耗时、闭环指标、多场景汇总对比。"""
    sys.argv = ['evaluate'] + args.rest
    return _run_module_main('evaluate')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='水下 ROV 单目视觉端到端控制 — 模块统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python3 main.py collect --frames 600   采集 600 帧数据集\n'
            '  python3 main.py train --epochs 30      训练并生成 model.pth\n'
            '  python3 main.py test --samples 10      离线验证 model.pth\n'
            '  python3 main.py evaluate offline        性能评价（精度/推理耗时）\n'
            '  python3 main.py run                    一键启动仿真 + 推理闭环\n'))
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('collect', help='采集数据集（键盘操纵 ROV）')
    p.add_argument('rest', nargs=argparse.REMAINDER,
                   help='透传给 collect_data.py 的参数，如 --frames 600')
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser('train', help='训练端到端网络，产出 model.pth')
    p.add_argument('rest', nargs=argparse.REMAINDER,
                   help='透传给 train.py 的参数，如 --epochs 30')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('test', help='离线测试 model.pth（不启动 ROS）')
    p.add_argument('rest', nargs=argparse.REMAINDER,
                   help='透传给 test_infer.py 的参数，如 --samples 10 --plot')
    p.set_defaults(func=cmd_test)

    p = sub.add_parser('run', help='一键启动 MuJoCo 仿真 + 推理节点（默认）')
    p.add_argument('rest', nargs=argparse.REMAINDER,
                   help='透传给 launch 的参数，如 model_path:=/abs/model.pth')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('evaluate', help='性能评价：开环精度/耗时、闭环指标、多场景对比')
    p.add_argument('rest', nargs=argparse.REMAINDER,
                   help='透传给 evaluate.py 的参数，如 offline / report --log runs/a.csv')
    p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        # 不带子命令时默认为 run，方便「直接跑整个模块」
        argv = list(argv) if argv is not None else sys.argv[1:]
        args = parser.parse_args(['run'] + argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
