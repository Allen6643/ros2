#!/usr/bin/env bash
# rov_vision_e2e_nav 模块 Shell 入口（main 脚本）
# ==============================================
# 加载 ROS 2 与工作空间环境、设置 MuJoCo 软件渲染后端，然后调用 main.py。
#
# 用法（在工作空间根目录，例如 ~/rov2_ws）：
#     bash src/water/rov_vision_e2e_nav/main.sh              # 一键启动完整闭环
#     bash src/water/rov_vision_e2e_nav/main.sh collect --samples 3000
#     bash src/water/rov_vision_e2e_nav/main.sh train --epochs 40
#     bash src/water/rov_vision_e2e_nav/main.sh test
#     bash src/water/rov_vision_e2e_nav/main.sh sim

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
else
    echo "[main.sh] 警告：未找到 /opt/ros/humble/setup.bash" >&2
fi

if [ -f "${PWD}/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "${PWD}/install/setup.bash"
fi

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"

exec python3 "${SCRIPT_DIR}/main.py" "$@"
