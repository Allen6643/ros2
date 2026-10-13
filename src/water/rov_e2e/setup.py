import os
from glob import glob

from setuptools import setup

package_name = 'rov_e2e'

# 训练产出的权重若已存在，一并安装到 share 目录；不存在时 glob 返回空列表，
# 不影响 colcon build（权重由 train.py 在本机训练生成，不入 git 仓库）。
model_files = glob('model.pth')
curve_files = glob('training_curve.png')

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    # launch 目录必须注册，否则 ros2 launch 找不到 main.launch.py
    (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    # 多场景测试配置
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
]

if model_files:
    data_files.append((os.path.join('share', package_name), model_files))
if curve_files:
    data_files.append((os.path.join('share', package_name), curve_files))

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Allen6643',
    maintainer_email='1769670648@qq.com',
    description='静水环境下单目视觉水下 ROV 前进-转向端到端神经网络控制',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 数据集采集：键盘操纵 ROV，保存图像 + [前进推力, 转向] 标签
            'collect_data = rov_e2e.collect_data:main',
            # 训练：生成 model.pth
            'train = rov_e2e.train:main',
            # 离线测试：不启动 ROS，验证 model.pth
            'test_infer = rov_e2e.test_infer:main',
            # 在线推理节点：/rov/camera/image_raw -> /cmd_vel
            'infer_node = rov_e2e.infer_node:main',
            # 性能评价：开环精度/推理耗时、闭环指标、多场景汇总对比
            'evaluate = rov_e2e.evaluate:main',
        ],
    },
)
