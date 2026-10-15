#!/usr/bin/env bash
# rov_mujoco_sim_teleop 模块 Shell 入口（main 脚本）
# ==================================================
# 加载 ROS 2 与工作空间环境、设置 MuJoCo 软件渲染后端，然后调用 main.py。
#
# 用法（在工作空间根目录，例如 ~/rov2_ws）：
#     bash src/water/rov_mujoco_sim_teleop/main.sh              # 一键启动仿真+遥控
#     bash src/water/rov_mujoco_sim_teleop/main.sh collect --samples 4000
#     bash src/water/rov_mujoco_sim_teleop/main.sh train --epochs 50
#     bash src/water/rov_mujoco_sim_teleop/main.sh test
#     bash src/water/rov_mujoco_sim_teleop/main.sh auto         # 神经网络闭环
#
# 无显卡虚拟机必须使用 OSMesa 软件渲染；若已开启 3D 加速，
# 可先 export MUJOCO_GL=glfw 覆盖。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) ROS 2 环境（未安装时给出提示但不直接失败，便于查看 --help）
if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    echo "[main.sh] 警告：未找到 /opt/ros/humble/setup.bash" >&2
fi

# 2) 当前工作空间的 colcon 构建产物
if [ -f "${PWD}/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "${PWD}/install/setup.bash"
fi

# 3) MuJoCo 渲染后端（CPU 软件渲染，无显卡环境必需）
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"

# 4) 转交统一 Python 入口
exec python3 "${SCRIPT_DIR}/main.py" "$@"
