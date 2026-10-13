# rov_e2e — 静水环境下单目视觉水下 ROV 前进-转向端到端神经网络控制

基于 MuJoCo + ROS 2 Humble + PyTorch 的端到端（End-to-End）水下机器人控制作业。

**输入**：ROV 前视相机单目 RGB 图像
**输出**：两个连续控制量 `[前进推力, 转向]` → 发布为 `/cmd_vel`

## 功能包结构

```
rov_e2e/
├── package.xml
├── setup.py
├── requirements.txt
├── rov_e2e/                 # Python 源码
│   ├── collect_data.py      # ① 数据集采集（键盘操纵，存图像 + 标签）
│   ├── train.py             # ② 训练网络 → model.pth（含网络结构定义）
│   ├── test_infer.py        # ③ 离线测试 model.pth（不启动 ROS）
│   └── infer_node.py        # ④ ROS 2 在线推理节点
├── launch/
│   └── main.launch.py       # ⑤ 一键启动入口
└── dataset/                 # 采集到的图像与 labels.csv（不入版本库）
```

## 统一入口（main 脚本）

本模块入口以 `main.` 开头，可直接运行整个模块：

```bash
bash src/water/rov_e2e/main.sh              # 一键启动完整闭环（自动加载 ROS 环境）
bash src/water/rov_e2e/main.sh collect --frames 600
bash src/water/rov_e2e/main.sh train --epochs 30
bash src/water/rov_e2e/main.sh test --samples 10 --plot
```

也可直接用 ROS 2 标准入口：`ros2 launch rov_e2e main.launch.py`。

## 依赖

```bash
sudo apt install -y libosmesa6 libgl1 libglfw3    # 无显卡虚拟机的软件渲染后端
pip3 install -r src/water/rov_e2e/requirements.txt
```

## 快速开始

```bash
cd ~/ros2
source /opt/ros/humble/setup.bash
colcon build --packages-select rov_mujoco rov_e2e --symlink-install
source install/setup.bash

# 无显卡虚拟机：使用 CPU 软件渲染，否则相机拿不到图像
export MUJOCO_GL=osmesa

# ① 采集数据集（键盘 W/S 前进后退，A/D 转向，空格归零，X 退出）
python3 src/water/rov_e2e/rov_e2e/collect_data.py --frames 600

# ② 训练，产出 model.pth 与 training_curve.png
python3 src/water/rov_e2e/rov_e2e/train.py

# ③ 离线测试权重（不启动 ROS）
python3 src/water/rov_e2e/rov_e2e/test_infer.py --samples 10 --plot

# ④ 一键启动仿真 + 推理闭环
ros2 launch rov_e2e main.launch.py
```

## 多场景测试与性能评价

`config/scenarios.yaml` 定义了 6 个测试场景（静水/洋流、不同初始深度与艏向）。

`main.py evaluate` 提供四种模式：

| 模式 | 命令 | 说明 |
| :--- | :--- | :--- |
| offline | `main.py evaluate offline` | 开环精度与单帧推理耗时（不需 ROS） |
| record | `main.py evaluate record --out runs/S1.csv --duration 30` | 记录一次闭环航行的 `/rov/odom` |
| report | `main.py evaluate report --log runs/S1.csv --plot` | 闭环指标：前进距离、横向偏差、深度波动、成功率 |
| compare | `main.py evaluate compare --reports runs/*_report.json` | 生成多场景对比表 |

指标定义与完整流程见 `docs/water/rov_e2e.md` 第 3.5 与 6.8 节。

## 话题接口

| 话题 | 类型 | 方向 | 说明 |
| :--- | :--- | :--- | :--- |
| `/rov/camera/image_raw` | `sensor_msgs/Image` | 订阅 | MuJoCo `forward_cam` 前视相机 |
| `/cmd_vel` | `geometry_msgs/Twist` | 发布 | `linear.x` 前进推力，`angular.z` 转向 |

详细原理、实验步骤与结果见 `docs/water/rov_e2e.md`。
