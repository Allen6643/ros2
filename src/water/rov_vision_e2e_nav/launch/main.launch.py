"""
一键启动入口：MuJoCo 视觉与声呐仿真 + 端到端视觉控制 + 声呐建图导航
==================================================================
作业要求模块入口以 ``main.`` 开头，并可通过 main 脚本跑通整个模块。

    ros2 launch rov_vision_e2e_nav main.launch.py

启动的节点
----------
1. ``vision_sim_node``   —— MuJoCo 仿真内核（相机 + 8 向声呐测距扇）
2. ``e2e_control_node``  —— 单目视觉端到端控制：图像 → /cmd_vel（任务④）
3. ``sonar_slam_node``   —— 声呐占据栅格建图 + 神经网络避障导航（任务③）

运行模式
--------
* ``mode:=e2e``  —— 只起仿真 + 端到端视觉控制（观察图像直接驱动 ROV）
* ``mode:=nav``  —— 只起仿真 + 声呐建图导航（观察建图与避障）
* ``mode:=all``  —— 三者全起（默认，完整闭环）
* ``mode:=sim``  —— 只起仿真（便于手动订阅话题检查）
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('rov_vision_e2e_nav')
    default_model = os.path.join(pkg_share, 'models', 'rov_vision_channel.xml')

    args = [
        DeclareLaunchArgument('model_path', default_value=default_model,
                              description='MuJoCo 模型 XML 路径'),
        DeclareLaunchArgument('mujoco_gl', default_value='osmesa',
                              description='MuJoCo 渲染后端：无显卡虚拟机用 osmesa'),
        DeclareLaunchArgument('mode', default_value='all',
                              description='sim / e2e / nav / all'),
        DeclareLaunchArgument('model_e2e_path', default_value='',
                              description='端到端网络权重路径（留空则自动探测）'),
        DeclareLaunchArgument('model_seg_path', default_value='',
                              description='声呐感知网络权重路径（留空则自动探测）'),
        DeclareLaunchArgument('sim_rate', default_value='500.0'),
        DeclareLaunchArgument('publish_rate', default_value='50.0'),
        DeclareLaunchArgument('target_depth', default_value='-1.0'),
        DeclareLaunchArgument('camera_width', default_value='160'),
        DeclareLaunchArgument('camera_height', default_value='120'),
        DeclareLaunchArgument('forward_gain', default_value='1.0',
                              description='端到端网络前进推力增益'),
        DeclareLaunchArgument('turn_gain', default_value='1.0',
                              description='端到端网络转向增益'),
    ]

    # MUJOCO_GL 必须在 import mujoco 之前生效
    set_gl = SetEnvironmentVariable('MUJOCO_GL', LaunchConfiguration('mujoco_gl'))
    set_pygl = SetEnvironmentVariable('PYOPENGL_PLATFORM', LaunchConfiguration('mujoco_gl'))

    # ---------------- 1) 仿真内核 ----------------
    sim = Node(
        package='rov_vision_e2e_nav',
        executable='vision_sim_node',
        name='vision_sim_node',
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

    # ---------------- 2) 端到端视觉控制（任务④）----------------
    e2e = Node(
        package='rov_vision_e2e_nav',
        executable='e2e_control_node',
        name='e2e_control_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_e2e_path'),
            'forward_gain': LaunchConfiguration('forward_gain'),
            'turn_gain': LaunchConfiguration('turn_gain'),
        }],
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration('mode'), "' == 'e2e' or '",
             LaunchConfiguration('mode'), "' == 'all'"])),
    )

    # ---------------- 3) 声呐建图导航（任务③）----------------
    slam = Node(
        package='rov_vision_e2e_nav',
        executable='sonar_slam_node',
        name='sonar_slam_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_seg_path'),
        }],
        condition=IfCondition(PythonExpression(
            ["'", LaunchConfiguration('mode'), "' == 'nav' or '",
             LaunchConfiguration('mode'), "' == 'all'"])),
    )

    return LaunchDescription(args + [set_gl, set_pygl, sim, e2e, slam])
