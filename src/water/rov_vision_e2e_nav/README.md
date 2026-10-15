# rov_vision_e2e_nav — 水下 ROV 单目视觉端到端控制与声呐建图导航

基于 **MuJoCo + ROS 2 Humble + PyTorch** 的水下机器人（ROV）视觉与声呐感知功能包，
对应作业任务（3）**建图与导航** 与任务（4）**端到端模型（输入图像、输出控制指令）**。

> **本包完全自包含**：MuJoCo 模型只用基础几何体，不引用任何外部网格与纹理；
> 不依赖本仓库中任何其他功能包。可单独 `colcon build` + `ros2 launch` 运行。

## 一、运行环境

| 项目 | 版本 |
| :--- | :--- |
| 操作系统 | Ubuntu 20.04 LTS |
| ROS 2 | Humble（`/opt/ros/humble`） |
| Python | 3.8.10 |
| MuJoCo | 3.1.6（`mujoco>=3.2` 要求 Python≥3.9） |
| PyTorch | 2.4.1+cpu / torchvision 0.19.1+cpu |
| 渲染后端 | `MUJOCO_GL=osmesa`（无显卡虚拟机必需） |

## 二、安装依赖

```bash
sudo apt install -y libosmesa6 libgl1 libglfw3

pip3 install -i https://mirrors.aliyun.com/pypi/simple/ \
    "mujoco<3.2" PyOpenGL numpy pillow pyyaml matplotlib

pip3 install torch==2.4.1+cpu torchvision==0.19.1+cpu \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

## 三、编译

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select rov_vision_e2e_nav
source install/setup.bash
```

## 四、运行步骤

### 步骤 1：采集数据集并训练

```bash
export MUJOCO_GL=osmesa

# 采集 3000 条（图像 + 声呐 + 标签）
python3 src/water/rov_vision_e2e_nav/main.py collect --samples 3000

# 训练视觉端到端网络与声呐感知网络
python3 src/water/rov_vision_e2e_nav/main.py train --epochs 40

# 离线测试（不启动 ROS）
python3 src/water/rov_vision_e2e_nav/main.py test
```

### 步骤 2：一键启动完整闭环

```bash
python3 src/water/rov_vision_e2e_nav/main.py run
# 或
ros2 launch rov_vision_e2e_nav main.launch.py mode:=all
```

启动 3 个节点：

| 节点 | 作用 |
| :--- | :--- |
| `vision_sim_node` | MuJoCo 仿真内核（前视相机 + 8 向声呐测距扇） |
| `e2e_control_node` | **端到端视觉控制**：图像 → CNN → `/cmd_vel`（任务④） |
| `sonar_slam_node` | **声呐建图导航**：测距 → 占据栅格地图 + 避障修正（任务③） |

其他运行模式：`mode:=sim`（只起仿真）/ `mode:=e2e`（只起视觉端到端）/ `mode:=nav`（只起建图导航）。

### 步骤 3：验证话题与地图

```bash
ros2 topic list
ros2 topic echo /rov/odom --once
ros2 topic echo /rov/open_direction
ros2 topic echo /rov/map --once | head -20     # 占据栅格地图
ros2 topic hz /rov/camera/image_raw
```

## 五、话题接口

| 话题 | 类型 | 方向 | 说明 |
| :--- | :--- | :--- | :--- |
| `/cmd_vel` | `geometry_msgs/Twist` | 订阅 | 端到端网络输出的控制指令 |
| `/rov/camera/image_raw` | `sensor_msgs/Image` | 发布 | 前视相机 rgb8 图像 |
| `/rov/sonar/range_{f,fl,l,bl,fr,r,br,down}` | `sensor_msgs/Range` | 发布 | 8 向声呐测距 |
| `/rov/odom` | `nav_msgs/Odometry` | 发布 | 位姿与速度 |
| `/rov/map` | `nav_msgs/OccupancyGrid` | 发布 | 声呐占据栅格地图 |
| `/rov/open_direction` | `std_msgs/Int32` | 发布 | 神经网络判定：0左/1前/2右开阔 |
| `/rov/cmd_vel_nav` | `geometry_msgs/Twist` | 发布 | 避障转向修正量 |

## 六、实测结果

在 Ubuntu 20.04.6 + ROS 2 Humble + MuJoCo 3.1.6 + PyTorch 2.4.1+cpu 的虚拟机中实测：

| 指标 | 数值 |
| :--- | :--- |
| colcon build | 通过，退出码 0 |
| 数据集 | 3000 条（图像 3000×120×160×3 + 声呐 8 维） |
| 视觉端到端网络验证损失 | **0.001192**（平均 MAE 0.0228） |
| 声呐感知网络分类准确率 | **98.23%** |
| 端到端视觉闭环 | 自主前进 **3.195 m**，横向偏移最大 0.512 m |
| 深度控制 | 目标 −1.0 m，实测 −1.000 ~ −1.063 m |

详细原理、公式推导与源码解析见 `docs/water/rov_vision_e2e_nav.md`。
