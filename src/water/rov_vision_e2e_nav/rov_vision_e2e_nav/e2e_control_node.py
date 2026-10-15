"""
端到端视觉控制节点
==================
订阅 MuJoCo 仿真发布的前视相机图像，送入训练好的 `VisualE2ENet`，
把输出的 [前进推力, 转向] 发布为 `/cmd_vel`，形成

    图像 → CNN → 控制指令 → 仿真

的单目视觉端到端闭环。这是作业任务(4)"每种机器人端到端模型
（输入图像、输出控制指令）"的实现。

话题接口
--------
    订阅  /rov/camera/image_raw   sensor_msgs/Image
    发布  /cmd_vel                geometry_msgs/Twist
          linear.x  ← 前进推力
          angular.z ← 转向
"""

import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

try:
    import torch
except ImportError:
    sys.stderr.write('\n【依赖缺失】需要 PyTorch：pip3 install "torch==2.4.1+cpu"\n\n')
    raise

from PIL import Image as PILImage

try:
    from rov_vision_e2e_nav.networks import (
        build_visual_e2e_net, preprocess_image, E2E_OUTPUT_DIMS)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rov_vision_e2e_nav.networks import (
        build_visual_e2e_net, preprocess_image, E2E_OUTPUT_DIMS)


class E2EControlNode(Node):
    """单目视觉端到端控制节点。"""

    def __init__(self):
        super().__init__('e2e_control_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('image_topic', '/rov/camera/image_raw')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('forward_gain', 1.0)
        self.declare_parameter('turn_gain', 1.0)
        self.declare_parameter('forward_offset', 0.0)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('log_interval', 20)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.forward_gain = self.get_parameter('forward_gain').get_parameter_value().double_value
        self.turn_gain = self.get_parameter('turn_gain').get_parameter_value().double_value
        self.forward_offset = self.get_parameter('forward_offset').get_parameter_value().double_value
        pub_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.log_interval = max(1, self.get_parameter('log_interval').get_parameter_value().integer_value)

        if not model_path:
            model_path = self._find_model()
        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path):
            self.get_logger().error(
                f'找不到权重文件: {model_path}\n'
                f'请先采集并训练：\n'
                f'  python3 src/water/rov_vision_e2e_nav/rov_vision_e2e_nav/collect_data.py --samples 3000\n'
                f'  python3 src/water/rov_vision_e2e_nav/rov_vision_e2e_nav/train.py\n'
                f'或用 -p model_path:=/绝对路径/model_e2e.pth 指定')
            raise FileNotFoundError(model_path)

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self.width = int(ckpt.get('width', 160))
        self.height = int(ckpt.get('height', 120))
        self.net = build_visual_e2e_net()
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.eval()
        self.get_logger().info(
            f'端到端网络已加载: {model_path} | 输入 {self.width}x{self.height} '
            f'| 训练 {ckpt.get("epochs", "?")} 轮 | 验证损失 {ckpt.get("val_loss", float("nan")):.6f}')

        self.forward = 0.0
        self.turn = 0.0
        self.frame_count = 0
        self.infer_count = 0

        # 相机图像用 BEST_EFFORT，避免与仿真发布者 QoS 不兼容收不到数据
        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, image_topic, self._on_image, img_qos)
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 10)
        self.create_timer(1.0 / max(pub_rate, 1.0), self._publish)

        self.get_logger().info(
            f'端到端视觉控制节点已启动 | 订阅 {image_topic} → 发布 {cmd_topic} '
            f'| 增益 forward={self.forward_gain} turn={self.turn_gain}')

    @staticmethod
    def _find_model():
        here = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        candidates = [
            os.path.join(here, '..', 'model_e2e.pth'),
            os.path.join(here, 'model_e2e.pth'),
            os.path.join(cwd, 'model_e2e.pth'),
            os.path.join(cwd, 'src', 'rov_vision_e2e_nav', 'model_e2e.pth'),
            os.path.join(cwd, 'src', 'water', 'rov_vision_e2e_nav', 'model_e2e.pth'),
            os.path.join(cwd, 'install', 'rov_vision_e2e_nav', 'share',
                         'rov_vision_e2e_nav', 'model_e2e.pth'),
        ]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('rov_vision_e2e_nav'), 'model_e2e.pth'))
        except Exception:  # noqa: BLE001
            pass
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c
        return candidates[0]

    def _on_image(self, msg: Image):
        """收到图像 → 预处理 → CNN 推理 → 缓存控制量。"""
        self.frame_count += 1
        try:
            frame = self._decode(msg)
        except Exception as err:  # noqa: BLE001
            self.get_logger().warn(f'图像解码失败（已跳过）: {err}')
            return

        x = torch.from_numpy(preprocess_image(PILImage.fromarray(frame),
                                              self.width, self.height)).unsqueeze(0)
        with torch.no_grad():
            out = self.net(x)[0].numpy()

        self.forward = float(np.clip(out[0] * self.forward_gain + self.forward_offset, -1.0, 1.0))
        self.turn = float(np.clip(out[1] * self.turn_gain, -1.0, 1.0))
        self.infer_count += 1

        if self.infer_count % self.log_interval == 0:
            self.get_logger().info(
                f'推理 #{self.infer_count} | 图像 {msg.width}x{msg.height} | '
                f'{E2E_OUTPUT_DIMS[0]}={self.forward:+.4f} {E2E_OUTPUT_DIMS[1]}={self.turn:+.4f}')

    @staticmethod
    def _decode(msg: Image):
        """把 sensor_msgs/Image 解码为 (H, W, 3) 的 uint8 RGB 数组。

        按 step 重新分行，兼容存在行填充（padding）的情况；
        同时支持 rgb8 与 bgr8 两种编码。
        """
        if msg.encoding not in ('rgb8', 'bgr8'):
            raise ValueError(f'暂不支持的图像编码: {msg.encoding}（支持 rgb8 / bgr8）')
        ch = 3
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.step == msg.width * ch:
            img = buf.reshape(msg.height, msg.width, ch)
        else:
            img = buf.reshape(msg.height, msg.step)[:, :msg.width * ch]
            img = img.reshape(msg.height, msg.width, ch)
        img = img.copy()
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        return img

    def _publish(self):
        """按固定频率发布控制指令。

        不在图像回调里直接发布：相机帧率抖动时保持上一次输出，
        避免推力频繁断续；节点退出时还会补发一次零指令。
        """
        cmd = Twist()
        cmd.linear.x = self.forward
        cmd.angular.z = self.turn
        self.pub_cmd.publish(cmd)

    def destroy_node(self):
        try:
            self.pub_cmd.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = E2EControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
