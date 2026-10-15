from setuptools import setup
import os
from glob import glob

package_name = 'rov_vision_e2e_nav'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    (os.path.join('share', package_name, 'models'), glob('models/*.xml')),
]

for pat in ('model_e2e.pth', 'model_seg.pth', 'training_curve_e2e.png',
            'training_curve_seg.png'):
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
    description='水下 ROV 单目视觉端到端神经网络控制与声呐占据栅格建图避障导航',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 仿真内核：MuJoCo 场景 + 相机 + 声呐测距扇
            'vision_sim_node = rov_vision_e2e_nav.vision_sim_node:main',
            # 端到端控制节点：图像 -> 控制指令
            'e2e_control_node = rov_vision_e2e_nav.e2e_control_node:main',
            # 声呐建图与导航节点
            'sonar_slam_node = rov_vision_e2e_nav.sonar_slam_node:main',
        ],
    },
)
