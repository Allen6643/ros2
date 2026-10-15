# rov_mujoco_sim_teleop — 水下 ROV MuJoCo 仿真与神经网络感知控制

基于 **MuJoCo + ROS 2 Humble + PyTorch** 的水下机器人（ROV）仿真与控制功能包，
对应作业任务（1）**物理仿真与键盘运动控制** 与任务（2）**传感器感知与给定轨迹控制**。

> **本包完全自包含**：MuJoCo 模型只用基础几何体（box/cylinder/plane），
> 不引用任何外部网格与纹理文件；不依赖本仓库中任何其他功能包。
> 可以单独 `colcon build` + `ros2 launch` 运行。

## 一、运行环境

| 项目 | 版本 |
| :--- | :--- |
| 操作系统 | Ubuntu 20.04 LTS |
| ROS 2 | Humble（`/opt/ros/humble`） |
| Python | 3.8.10 |
| MuJoCo | 3.1.6（注意：`mujoco>=3.2` 要求 Python≥3.9） |
| PyTorch | 2.4.1+cpu / torchvision 0.19.1+cpu（CPU 训练，无需显卡） |
| 渲染后端 | `MUJOCO_GL=osmesa`（无显卡虚拟机必需） |

## 二、安装依赖

```bash
# 1) 系统库：MuJoCo 离屏渲染所需的 OSMesa 软件渲染后端
sudo apt install -y libosmesa6 libgl1 libglfw3

# 2) Python 依赖
pip3 install -i https://mirrors.aliyun.com/pypi/simple/ \
    "mujoco<3.2" PyOpenGL numpy pillow pyyaml matplotlib

# 3) PyTorch（CPU 版；Python 3.8 最后一个支持的版本是 2.4.1）
pip3 install torch==2.4.1+cpu torchvision==0.19.1+cpu \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

> `requirements.txt` 里列出了同样的依赖，也可以直接用
> `pip3 install -r requirements.txt`。

## 三、编译

```bash
cd ~/ros2_ws          # 你的 ROS 2 工作空间根目录
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rov_mujoco_sim_teleop
source install/setup.bash
```

## 四、运行步骤

### 步骤 1：一键启动仿真 + 键盘遥控（任务①）

```bash
export MUJOCO_GL=osmesa          # 无显卡环境必需
python3 src/water/rov_mujoco_sim_teleop/main.py run
# 或
bash src/water/rov_mujoco_sim_teleop/main.sh run
# 或
ros2 launch rov_mujoco_sim_teleop main.launch.py
```

**键盘操作**（锁存式，按一次持续生效）：

| 按键 | 功能 | 按键 | 功能 |
| :--- | :--- | :--- | :--- |
| `W` / `↑` | 前进 | `S` / `↓` | 后退 |
| `A` / `←` | 原地左转 | `D` / `→` | 原地右转 |
| `Q` | 上浮 | `E` | 下潜 |
| `Z` / `X` | 左横移 / 右横移 | `空格` | 全部归零（悬停） |
| `+` / `-` | 调节推力比例 | `Ctrl+C` | 退出 |

### 步骤 2：采集数据集并训练神经网络

```bash
# 采集 4000 条样本（生成 sensor_dataset.npz 与 traj_dataset.npz）
python3 src/water/rov_mujoco_sim_teleop/main.py collect --samples 4000

# 训练两个网络（产出 model_sensor.pth / model_traj.pth / training_curve.png）
python3 src/water/rov_mujoco_sim_teleop/main.py train --epochs 50

# 离线测试权重（不启动 ROS，输出准确率与 MAE）
python3 src/water/rov_mujoco_sim_teleop/main.py test
```

### 步骤 3：一键启动神经网络闭环（任务②）

```bash
python3 src/water/rov_mujoco_sim_teleop/main.py auto
# 或
ros2 launch rov_mujoco_sim_teleop main.launch.py mode:=auto
```

该模式下会同时启动 4 个节点：

| 节点 | 作用 |
| :--- | :--- |
| `mujoco_sim_node` | MuJoCo 物理仿真内核（发布传感器、接收 `/cmd_vel`） |
| `keyboard_teleop_node` | 键盘遥控（任务①） |
| `sensor_perception_node` | **神经网络**传感器感知：测距数据 → 障碍物方位（任务②） |
| `traj_control_node` | **神经网络**轨迹跟踪控制：状态误差 → 四维控制量（任务②） |

### 步骤 4：验证话题数据

```bash
ros2 topic list
ros2 topic echo /rov/odom --once
ros2 topic echo /rov/perception/obstacle_class
ros2 topic hz /rov/camera/image_raw
```

## 五、话题接口

| 话题 | 类型 | 方向 | 说明 |
| :--- | :--- | :--- | :--- |
| `/cmd_vel` | `geometry_msgs/Twist` | 订阅 | 归一化控制指令 |
| `/rov/odom` | `nav_msgs/Odometry` | 发布 | 位姿与速度 |
| `/rov/depth` | `std_msgs/Float64` | 发布 | 深度（米） |
| `/rov/imu` | `sensor_msgs/Imu` | 发布 | 姿态与角速度、加速度 |
| `/rov/camera/image_raw` | `sensor_msgs/Image` | 发布 | 前视相机 rgb8 图像 |
| `/rov/range/{front,left,right,down}` | `sensor_msgs/Range` | 发布 | 四向测距 |
| `/rov/perception/obstacle_class` | `std_msgs/Int32` | 发布 | 神经网络判定：0前方/1左方/2右方 |

## 六、实测结果

在 Ubuntu 20.04.6 + ROS 2 Humble + MuJoCo 3.1.6 + PyTorch 2.4.1+cpu 的
VMware 虚拟机（无独立显卡）中实测：

| 指标 | 数值 |
| :--- | :--- |
| colcon build | 通过，`1 package finished`，退出码 0 |
| 数据集 | 各 4000 条样本（感知 7 维特征 / 轨迹 8 维特征） |
| 感知网络分类准确率 | **97.67%** |
| 轨迹网络控制维度平均 MAE | **0.0253**（整体 MSE 0.002439） |
| 闭环自主航行 | 沿通道前进 **4.776 m**，横向偏移最大 0.693 m |
| 深度控制精度 | 目标 −1.000 m，实测波动范围仅 **0.025 m** |

详细原理、公式推导与源码解析见 `docs/water/rov_mujoco_sim_teleop.md`。
