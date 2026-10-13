"""
一键启动入口：MuJoCo 水下 ROV 仿真 + 端到端视觉推理节点
=======================================================
老师验收作业时使用的唯一入口命令：

    ros2 launch rov_e2e main.launch.py

启动内容
--------
1. `mujoco_sim_node`  — 来自 `rov_mujoco` 包，加载 MuJoCo 水下 ROV 模型，
   500 Hz 物理步进，发布 `/rov/camera/image_raw` 等传感器话题，订阅 `/cmd_vel`；
2. `infer_node`       — 来自本包，订阅相机图像、加载 `model.pth` 推理，
   输出 [前进推力, 转向] 发布到 `/cmd_vel`，构成闭环自动航行。

为什么直接起 `mujoco_sim_node` 而不 include `rov_mujoco/main.launch.py`
---------------------------------------------------------------------
`rov_mujoco` 的 launch 会 **无条件** 一并拉起 `keyboard_teleop_node`，
它同样向 `/cmd_vel` 发指令，会与推理节点抢控制权，导致机器人动作抖动。
因此这里直接实例化仿真节点，只按需（`enable_teleop:=true`）追加遥控节点。

常用覆盖方式
------------
    # 指定自己训练出来的权重
    ros2 launch rov_e2e main.launch.py model_path:=$PWD/model.pth

    # 调试：只起仿真与相机，不起推理节点
    ros2 launch rov_e2e main.launch.py enable_infer:=false

    # 同时起键盘遥控节点（人工接管，会与推理节点抢 /cmd_vel）
    ros2 launch rov_e2e main.launch.py enable_teleop:=true

无显卡虚拟机注意
----------------
MuJoCo 离屏渲染需要 OpenGL 后端。本 launch 默认注入 `MUJOCO_GL=osmesa`
（纯 CPU 软件渲染），必须在 `import mujoco` 之前生效，所以用
`SetEnvironmentVariable` 在节点进程创建前设置。若虚拟机已开启 3D 加速：

    ros2 launch rov_e2e main.launch.py mujoco_gl:=glfw
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    e2e_share = get_package_share_directory('rov_e2e')
    mujoco_share = get_package_share_directory('rov_mujoco')

    default_model = ''   # 空 = 自动探测源码树中的 model.pth（见 infer_node.py）
    mujoco_xml = os.path.join(mujoco_share, 'models', 'underwater_rov_with_arm.xml')
    mujoco_cfg = os.path.join(mujoco_share, 'models', 'config.json')

    # ---------------- 启动参数 ----------------
    args = [
        DeclareLaunchArgument(
            'model_path', default_value=default_model,
            description='端到端控制网络权重 model.pth 的路径；'
                        '留空则自动在源码树中探测'),
        DeclareLaunchArgument(
            'mujoco_gl', default_value='osmesa',
            description='MuJoCo 渲染后端：无显卡虚拟机用 osmesa，有 3D 加速用 glfw/egl'),
        DeclareLaunchArgument(
            'mujoco_model', default_value=mujoco_xml,
            description='MuJoCo 模型 XML 路径'),
        DeclareLaunchArgument(
            'mujoco_config', default_value=mujoco_cfg,
            description='MuJoCo 仿真配置 config.json 路径'),
        DeclareLaunchArgument(
            'enable_infer', default_value='true',
            description='是否启动端到端推理节点'),
        DeclareLaunchArgument(
            'enable_teleop', default_value='false',
            description='是否同时启动键盘遥控节点（会与推理节点争用 /cmd_vel）'),
        DeclareLaunchArgument(
            'enable_camera_renderer', default_value='true',
            description='是否让仿真节点真实渲染相机画面。'
                        '必须为 true，否则 /rov/camera/image_raw 只发固定合成底色图，'
                        '端到端网络将学不到任何视觉线索'),
        DeclareLaunchArgument('forward_gain', default_value='1.0',
                              description='前进推力输出增益'),
        DeclareLaunchArgument('turn_gain', default_value='1.0',
                              description='转向输出增益'),
        DeclareLaunchArgument('sim_rate', default_value='500.0',
                              description='物理仿真频率 (Hz)'),
        DeclareLaunchArgument('publish_rate', default_value='50.0',
                              description='传感器话题发布频率 (Hz)'),
    ]

    # ---------------- 渲染后端（必须早于 import mujoco）----------------
    set_gl = SetEnvironmentVariable('MUJOCO_GL', LaunchConfiguration('mujoco_gl'))
    set_pygl = SetEnvironmentVariable('PYOPENGL_PLATFORM',
                                      LaunchConfiguration('mujoco_gl'))

    # ---------------- 1) MuJoCo 仿真节点 ----------------
    sim = Node(
        package='rov_mujoco',
        executable='mujoco_sim_node',
        name='mujoco_sim_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('mujoco_model'),
            'config_path': LaunchConfiguration('mujoco_config'),
            'sim_rate': LaunchConfiguration('sim_rate'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'use_ocean_current': False,     # 静水环境
            'auto_trajectory_mode': False,  # 由本包的推理节点闭环控制
            # 关键：不打开则相机只发固定合成底色图，视觉闭环不成立
            'enable_camera_renderer': LaunchConfiguration('enable_camera_renderer'),
        }],
    )

    # ---------------- 2) 端到端推理节点 ----------------
    infer = Node(
        package='rov_e2e',
        executable='infer_node',
        name='rov_e2e_infer_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_infer')),
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'forward_gain': LaunchConfiguration('forward_gain'),
            'turn_gain': LaunchConfiguration('turn_gain'),
        }],
    )

    # ---------------- 3) 可选：键盘遥控（人工接管 / 录屏对比）----------------
    teleop = Node(
        package='rov_mujoco',
        executable='keyboard_teleop_node',
        name='keyboard_teleop_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_teleop')),
        parameters=[{'linear_scale': 1.0, 'angular_scale': 1.0}],
    )

    return LaunchDescription(args + [set_gl, set_pygl, sim, infer, teleop])
