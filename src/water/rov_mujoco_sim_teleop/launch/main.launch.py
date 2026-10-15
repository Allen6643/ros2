"""
一键启动入口：MuJoCo 水下 ROV 仿真 + 键盘遥控 + 神经网络感知与控制
==================================================================
作业要求每个模块的入口以 ``main.`` 开头，并可通过 main 脚本跑通整个模块。

    ros2 launch rov_mujoco_sim_teleop main.launch.py

启动的节点
----------
1. ``mujoco_sim_node``      —— MuJoCo 物理仿真内核（发布传感器、接收 /cmd_vel）
2. ``keyboard_teleop_node`` —— 键盘遥控（任务①：手动运动控制）
3. ``sensor_perception_node`` —— 神经网络传感器感知（任务②：感知）
4. ``traj_control_node``    —— 神经网络轨迹跟踪控制（任务②：给定轨迹控制）

默认只启动 1 和 2（便于人工遥控）；用 ``enable_perception:=true`` /
``enable_traj_control:=true`` 打开神经网络节点。
也可以直接 ``ros2 launch rov_mujoco_sim_teleop main.launch.py mode:=auto``
一次拉起全部四个节点，观察"仿真 + 感知 + 控制"完整闭环。
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('rov_mujoco_sim_teleop')
    default_model = os.path.join(pkg_share, 'models', 'rov_channel.xml')

    args = [
        DeclareLaunchArgument('model_path', default_value=default_model,
                              description='MuJoCo 模型 XML 路径'),
        # 渲染后端：无显卡虚拟机必须用 osmesa（纯 CPU 软件渲染）
        DeclareLaunchArgument('mujoco_gl', default_value='osmesa',
                              description='MuJoCo 渲染后端：osmesa / egl / glfw'),
        # 运行模式：manual=只起仿真+键盘遥控；auto=再叠加神经网络感知与控制
        DeclareLaunchArgument('mode', default_value='manual',
                              description='manual（人工遥控）/ auto（神经网络闭环）'),
        DeclareLaunchArgument('enable_perception', default_value='false',
                              description='是否启动神经网络传感器感知节点'),
        DeclareLaunchArgument('enable_traj_control', default_value='false',
                              description='是否启动神经网络轨迹跟踪控制节点'),
        DeclareLaunchArgument('model_sensor_path', default_value='',
                              description='感知网络权重路径（留空则自动探测）'),
        DeclareLaunchArgument('model_traj_path', default_value='',
                              description='轨迹控制网络权重路径（留空则自动探测）'),
        DeclareLaunchArgument('sim_rate', default_value='500.0',
                              description='物理仿真频率 Hz'),
        DeclareLaunchArgument('publish_rate', default_value='50.0',
                              description='传感器发布频率 Hz'),
        DeclareLaunchArgument('target_depth', default_value='-1.0',
                              description='期望巡航深度（米）'),
        DeclareLaunchArgument('camera_width', default_value='160'),
        DeclareLaunchArgument('camera_height', default_value='120'),
    ]

    # MUJOCO_GL 必须在 import mujoco 之前生效，因此在节点进程创建前设置
    set_gl = SetEnvironmentVariable('MUJOCO_GL', LaunchConfiguration('mujoco_gl'))
    set_pygl = SetEnvironmentVariable('PYOPENGL_PLATFORM', LaunchConfiguration('mujoco_gl'))

    # ---------------- 1) 仿真内核 ----------------
    sim = Node(
        package='rov_mujoco_sim_teleop',
        executable='mujoco_sim_node',
        name='mujoco_sim_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'sim_rate': LaunchConfiguration('sim_rate'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'target_depth': LaunchConfiguration('target_depth'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'),
            'enable_camera': True,
        }],
    )

    # ---------------- 2) 键盘遥控 ----------------
    teleop = Node(
        package='rov_mujoco_sim_teleop',
        executable='keyboard_teleop_node',
        name='keyboard_teleop_node',
        output='screen',
        parameters=[{'linear_scale': 1.0, 'angular_scale': 1.0}],
    )

    # ---------------- 3) 神经网络传感器感知 ----------------
    perception = Node(
        package='rov_mujoco_sim_teleop',
        executable='sensor_perception_node',
        name='sensor_perception_node',
        output='screen',
        parameters=[{'model_path': LaunchConfiguration('model_sensor_path')}],
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration('enable_perception'), "' == 'true' or '",
             LaunchConfiguration('mode'), "' == 'auto'"])),
    )

    # ---------------- 4) 神经网络轨迹跟踪控制 ----------------
    traj = Node(
        package='rov_mujoco_sim_teleop',
        executable='traj_control_node',
        name='traj_control_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_traj_path'),
            'target_depth': LaunchConfiguration('target_depth'),
        }],
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration('enable_traj_control'), "' == 'true' or '",
             LaunchConfiguration('mode'), "' == 'auto'"])),
    )

    return LaunchDescription(args + [set_gl, set_pygl, sim, teleop, perception, traj])
