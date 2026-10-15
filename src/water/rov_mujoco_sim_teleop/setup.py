from setuptools import setup
import os
from glob import glob

package_name = 'rov_mujoco_sim_teleop'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    # launch 目录必须注册，否则 ros2 launch 找不到 main.launch.py
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    # 配置
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    # MuJoCo 模型（自包含，不含外部网格资产）
    (os.path.join('share', package_name, 'models'), glob('models/*.xml') + glob('models/*.json')),
]

# 训练产出的权重若已存在则一并安装；不存在时 glob 返回空列表，不影响 colcon build
for pat in ('model_sensor.pth', 'model_traj.pth', 'training_curve_sensor.png',
            'training_curve_traj.png'):
    if glob(pat):
        data_files.append((os.path.join('share', package_name), glob(pat)))

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Allen6643',
    maintainer_email='1769670648@qq.com',
    description='水下机器人（ROV）MuJoCo 物理仿真、键盘运动控制、多传感器感知与神经网络轨迹跟踪控制',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 仿真内核：加载 MuJoCo 模型、步进物理、发布传感器、接收 /cmd_vel
            'mujoco_sim_node = rov_mujoco_sim_teleop.mujoco_sim_node:main',
            # 键盘遥控
            'keyboard_teleop_node = rov_mujoco_sim_teleop.keyboard_teleop_node:main',
            # 传感器感知节点（神经网络）
            'sensor_perception_node = rov_mujoco_sim_teleop.sensor_perception_node:main',
            # 轨迹跟踪控制器（神经网络）
            'traj_control_node = rov_mujoco_sim_teleop.traj_control_node:main',
        ],
    },
)
