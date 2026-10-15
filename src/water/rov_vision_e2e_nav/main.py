#!/usr/bin/env python3
"""
rov_vision_e2e_nav 统一入口（main 脚本）
========================================
作业要求模块入口以 ``main.`` 开头，且可通过 main 脚本直接跑通整个模块。

    python3 src/water/rov_vision_e2e_nav/main.py            # = run，一键启动完整闭环
    python3 src/water/rov_vision_e2e_nav/main.py collect    # 采集数据集
    python3 src/water/rov_vision_e2e_nav/main.py train      # 训练两个网络
    python3 src/water/rov_vision_e2e_nav/main.py test       # 离线测试权重
    python3 src/water/rov_vision_e2e_nav/main.py run        # 仿真 + 端到端 + 建图导航
    python3 src/water/rov_vision_e2e_nav/main.py sim        # 只启动仿真

也可以使用 shell 包装：bash src/water/rov_vision_e2e_nav/main.sh run
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_NAME = 'rov_vision_e2e_nav'


def _run_module_main(module_name):
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
    print(f'[{PKG_NAME}] $ ' + ' '.join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        sys.stderr.write(
            '\n找不到 ros2 命令。请先加载 ROS 2 环境：\n'
            '    source /opt/ros/humble/setup.bash\n'
            '    source install/setup.bash\n'
            f'或使用：bash src/water/{PKG_NAME}/main.sh run\n\n')
        return 127


def cmd_run(args):
    """一键启动完整闭环：仿真 + 端到端视觉控制 + 声呐建图导航。"""
    return _launch(['mode:=all'], args.rest)


def cmd_sim(args):
    """只启动仿真（便于手动订阅话题检查）。"""
    return _launch(['mode:=sim'], args.rest)


def build_parser():
    p = argparse.ArgumentParser(
        prog='main.py',
        description='水下 ROV 单目视觉端到端控制与声呐建图导航 —— 模块统一入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python3 main.py collect --samples 3000   采集数据集\n'
            '  python3 main.py train --epochs 40        训练视觉端到端与声呐网络\n'
            '  python3 main.py test                     离线测试权重\n'
            '  python3 main.py run                      一键启动完整闭环\n'
            '  python3 main.py sim                      只启动仿真\n'))
    sub = p.add_subparsers(dest='command')
    for name, fn, text, tail in (
            ('collect', cmd_collect, '采集视觉端到端与声呐建图数据集', '采集参数'),
            ('train', cmd_train, '训练视觉端到端网络与声呐感知网络', '训练参数'),
            ('test', cmd_test, '离线测试两个网络的权重', '测试参数'),
            ('run', cmd_run, '一键启动仿真 + 端到端 + 建图导航', 'launch 参数'),
            ('sim', cmd_sim, '只启动 MuJoCo 仿真', 'launch 参数')):
        sp = sub.add_parser(name, help=text)
        sp.add_argument('rest', nargs=argparse.REMAINDER, help=f'透传给{tail}')
        sp.set_defaults(func=fn)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        argv = list(argv) if argv is not None else sys.argv[1:]
        args = parser.parse_args(['run'] + argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
