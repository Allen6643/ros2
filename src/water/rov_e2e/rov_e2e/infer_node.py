#!/usr/bin/env python3
"""
ROS 2 端到端推理节点
====================
订阅 MuJoCo 仿真发布的前视相机图像，送入 `model.pth` 做单次前向推理，
把输出的 **[前进推力, 转向]** 换算为 `geometry_msgs/Twist` 发布到 `/cmd_vel`，
由 `mujoco_sim_node` 施加到 ROV 上，形成"感知—决策—控制"闭环。

话题接口（与 `rov_mujoco` 包的约定一致）
----------------------------------------
    订阅  /rov/camera/image_raw   sensor_msgs/Image
    发布  /cmd_vel                geometry_msgs/Twist
          linear.x  ← 前进推力 (tanh 输出，[-1,1])
          angular.z ← 转向     (tanh 输出，[-1,1])

注意：本节点面向仓库内已有的 `rov_mujoco` 仿真包，**不需要**
`mujoco_ros2_control` 等第三方 ros2_control 桥接。

用法
----
    ros2 run rov_e2e infer_node --ros-args -p model_path:=/abs/path/model.pth

    或直接由一键入口启动（推荐）：
    ros2 launch rov_e2e main.launch.py
"""

import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

try:
    import torch
except ImportError:
    sys.stderr.write('\n【环境依赖缺失】未检测到 PyTorch。请先 pip3 install torch torchvision\n\n')
    raise

from PIL import Image as PILImage

try:
    from rov_e2e.train import ROVSteeringNet, preprocess_image
except ImportError:
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    from rov_e2e.train import ROVSteeringNet, preprocess_image


class InferNode(Node):
    """单目视觉端到端控制推理节点。"""

    def __init__(self):
        super().__init__('rov_e2e_infer_node')

        # ---------------- 参数 ----------------
        self.declare_parameter('model_path', '')
        self.declare_parameter('image_topic', '/rov/camera/image_raw')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('forward_gain', 1.0)
        self.declare_parameter('turn_gain', 1.0)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('max_forward', 1.0)
        self.declare_parameter('max_turn', 1.0)
        self.declare_parameter('log_interval', 20)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.forward_gain = self.get_parameter('forward_gain').get_parameter_value().double_value
        self.turn_gain = self.get_parameter('turn_gain').get_parameter_value().double_value
        self.max_forward = self.get_parameter('max_forward').get_parameter_value().double_value
        self.max_turn = self.get_parameter('max_turn').get_parameter_value().double_value
        pub_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.log_interval = max(1, self.get_parameter('log_interval').get_parameter_value().integer_value)

        # ---------------- 加载权重 ----------------
        # `model_path` 留空时自动探测：优先源码树（--symlink-install 下
        # 安装目录即源码目录），其次当前工作目录，最后包内 share 目录。
        if not model_path:
            here = os.path.dirname(os.path.abspath(__file__))
            cwd = os.getcwd()
            candidates = [
                os.path.join(here, '..', 'model.pth'),
                os.path.join(here, 'model.pth'),
                os.path.join(cwd, 'model.pth'),
                # 源码树（最常见：在工作空间根目录下执行 launch）
                os.path.join(cwd, 'src', 'rov_e2e', 'model.pth'),
                os.path.join(cwd, 'src', 'water', 'rov_e2e', 'model.pth'),
                # colcon --symlink-install 的安装目录
                os.path.join(cwd, 'install', 'rov_e2e', 'share', 'rov_e2e', 'model.pth'),
            ]
            # --symlink-install 下 __file__ 可能落在 build/ 目录，再兜底查包 share 目录
            try:
                from ament_index_python.packages import get_package_share_directory
                candidates.append(
                    os.path.join(get_package_share_directory('rov_e2e'), 'model.pth'))
            except Exception:  # noqa: BLE001
                pass

            found = next((os.path.abspath(c) for c in candidates
                          if os.path.exists(c)), None)
            if found is None:
                self.get_logger().error(
                    '未找到 model.pth，已尝试以下位置：\n  ' +
                    '\n  '.join(os.path.abspath(c) for c in candidates))
                raise FileNotFoundError('model.pth')
            model_path = found
            self.get_logger().info(f'未指定 model_path，自动探测到: {model_path}')
        model_path = os.path.abspath(os.path.expanduser(model_path))

        if not os.path.exists(model_path):
            self.get_logger().error(
                f'找不到权重文件: {model_path}\n'
                f'请先采集数据并训练（训练完成后无需重新 colcon build）：\n'
                f'  python3 src/water/rov_e2e/rov_e2e/collect_data.py --frames 600\n'
                f'  python3 src/water/rov_e2e/rov_e2e/train.py\n'
                f'或用 ros2 launch rov_e2e main.launch.py model_path:=/绝对路径/model.pth')
            raise FileNotFoundError(model_path)

        self.device = torch.device('cpu')
        ckpt = torch.load(model_path, map_location=self.device)
        self.width = int(ckpt.get('width', 160))
        self.height = int(ckpt.get('height', 120))

        self.model = ROVSteeringNet().to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

        self.get_logger().info(
            f'权重已加载: {model_path} | 输入 {self.width}x{self.height} '
            f'| 训练轮数={ckpt.get("epochs", "未知")}')

        # ---------------- 状态 ----------------
        self.forward = 0.0
        self.turn = 0.0
        self.frame_count = 0
        self.infer_count = 0
        self.last_stamp = None

        # ---------------- 通信 ----------------
        # 相机图像用 BestEffort（与多数仿真相机发布者兼容），避免 QoS 不匹配
        img_qos = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)

        self.sub_image = self.create_subscription(
            Image, image_topic, self._image_callback, img_qos)
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)
        self.timer = self.create_timer(1.0 / max(pub_rate, 1.0), self._publish_callback)

        self.get_logger().info(
            f'端到端推理节点已启动 | 订阅 {image_topic} → 发布 {cmd_topic} '
            f'| 增益 forward={self.forward_gain} turn={self.turn_gain} '
            f'| 发布频率 {pub_rate} Hz')

    # ------------------------------------------------------------------ #
    def _image_callback(self, msg: Image):
        """收到相机图像 → 预处理 → 推理 → 缓存控制量。"""
        self.frame_count += 1
        try:
            frame = self._msg_to_array(msg)
        except Exception as err:  # noqa: BLE001
            self.get_logger().warn(f'图像解码失败（已跳过该帧）: {err}')
            return

        arr = preprocess_image(PILImage.fromarray(frame), self.width, self.height)
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)[0].cpu().numpy()

        forward = float(np.clip(out[0] * self.forward_gain,
                                -self.max_forward, self.max_forward))
        turn = float(np.clip(out[1] * self.turn_gain,
                             -self.max_turn, self.max_turn))
        self.forward, self.turn = forward, turn
        self.infer_count += 1
        self.last_stamp = msg.header.stamp

        if self.infer_count % self.log_interval == 0:
            self.get_logger().info(
                f'推理 #{self.infer_count} | 图像 {msg.width}x{msg.height} '
                f'| 前进推力={forward:+.4f} | 转向={turn:+.4f}')

    @staticmethod
    def _msg_to_array(msg: Image):
        """把 `sensor_msgs/Image` 转为 ``(H, W, 3)`` 的 uint8 RGB 数组。"""
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError(
                f'暂不支持的图像编码: {msg.encoding}（当前支持 rgb8 / bgr8）')

        channels = 3
        expected = msg.height * msg.step
        buf = np.frombuffer(msg.data, dtype=np.uint8)

        # 按 step 重新分行，兼容存在行填充（padding）的情况
        if msg.step == msg.width * channels:
            img = buf.reshape(msg.height, msg.width, channels)
        else:
            img = buf.reshape(msg.height, msg.step)[:, :msg.width * channels]
            img = img.reshape(msg.height, msg.width, channels)

        img = img.copy()
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        return img

    def _publish_callback(self):
        """按固定频率发布控制指令（新图像未到达时保持上一次输出）。"""
        cmd = Twist()
        cmd.linear.x = float(self.forward)
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = float(self.turn)
        self.pub_cmd.publish(cmd)

    def destroy_node(self):
        # 退出前发一次零指令，避免仿真里机器人带着上一帧推力继续冲
        try:
            self.pub_cmd.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = InferNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        # 权重缺失时已在 __init__ 中打印明确指引
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
