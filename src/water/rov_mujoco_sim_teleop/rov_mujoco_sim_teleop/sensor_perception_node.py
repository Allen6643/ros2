"""
传感器感知节点（神经网络）
==========================
订阅 MuJoCo 仿真发布的四向测距与里程计数据，用训练好的
`SensorPerceptionNet` 判断障碍物的方位，并把结果打印/发布出去。

这是作业要求的"**使用传感器获取的数据进行感知**"部分：
输入是测距传感器数据（雷达类传感器），输出是障碍物方位分类，
整个感知过程由一个神经网络完成。

话题接口
--------
    订阅  /rov/range/front, /rov/range/left, /rov/range/right, /rov/range/down
          /rov/odom
    发布  /rov/perception/obstacle_class   std_msgs/Int32（0=前方 1=左方 2=右方）
"""

import os
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Range
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32

try:
    import torch
except ImportError:
    sys.stderr.write('\n【依赖缺失】需要 PyTorch：pip3 install "torch==2.4.1+cpu"\n\n')
    raise

try:
    from rov_mujoco_sim_teleop.networks import (
        build_sensor_net, OBSTACLE_CLASSES, normalize_sensor_features)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rov_mujoco_sim_teleop.networks import (
        build_sensor_net, OBSTACLE_CLASSES, normalize_sensor_features)


class SensorPerceptionNode(Node):
    """用神经网络把测距数据解释为障碍物方位。"""

    def __init__(self):
        super().__init__('sensor_perception_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('log_interval', 20)
        self.declare_parameter('publish_rate', 10.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.log_interval = max(1, self.get_parameter('log_interval').get_parameter_value().integer_value)
        pub_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        if not model_path:
            model_path = self._find_model()
        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path):
            self.get_logger().error(
                f'找不到权重文件: {model_path}\n'
                f'请先训练：\n'
                f'  python3 src/water/rov_mujoco_sim_teleop/rov_mujoco_sim_teleop/collect_data.py --samples 4000\n'
                f'  python3 src/water/rov_mujoco_sim_teleop/rov_mujoco_sim_teleop/train.py\n'
                f'或用 ros2 run 时指定 -p model_path:=/绝对路径/model_sensor.pth')
            raise FileNotFoundError(model_path)

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self.net = build_sensor_net()
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.eval()
        self.get_logger().info(
            f'感知网络已加载: {model_path} | 训练 {ckpt.get("epochs", "?")} 轮 '
            f'| 验证损失 {ckpt.get("val_loss", float("nan")):.6f}')

        # ---------------- 状态 ----------------
        self.ranges = {'front': 20.0, 'left': 20.0, 'right': 20.0, 'down': 20.0}
        self.vel = np.zeros(3, dtype=np.float32)
        self.count = 0
        self.latest_class = -1

        # ---------------- 通信 ----------------
        rng_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Range, '/rov/range/front', self._mk_range_cb('front'), rng_qos)
        self.create_subscription(Range, '/rov/range/left', self._mk_range_cb('left'), rng_qos)
        self.create_subscription(Range, '/rov/range/right', self._mk_range_cb('right'), rng_qos)
        self.create_subscription(Range, '/rov/range/down', self._mk_range_cb('down'), rng_qos)
        self.create_subscription(Odometry, '/rov/odom', self._on_odom, 10)

        self.pub_class = self.create_publisher(Int32, '/rov/perception/obstacle_class', 10)
        self.create_timer(1.0 / max(pub_rate, 1.0), self._on_timer)

        self.get_logger().info('传感器感知节点已启动（订阅四向测距 → 输出障碍物方位）')

    @staticmethod
    def _find_model():
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, '..', 'model_sensor.pth'),
            os.path.join(os.getcwd(), 'model_sensor.pth'),
            os.path.join(os.getcwd(), 'src', 'rov_mujoco_sim_teleop', 'model_sensor.pth'),
        ]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('rov_mujoco_sim_teleop'), 'model_sensor.pth'))
        except Exception:  # noqa: BLE001
            pass
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c
        return candidates[0]

    def _mk_range_cb(self, key):
        """生成对应方向的测距回调。"""
        def cb(msg: Range):
            self.ranges[key] = float(msg.range)
        return cb

    def _on_odom(self, msg: Odometry):
        v = msg.twist.twist.linear
        self.vel = np.array([v.x, v.y, v.z], dtype=np.float32)

    def _on_timer(self):
        """组特征 → 推理 → 发布分类结果。"""
        feat = np.array([self.ranges['front'], self.ranges['left'], self.ranges['right'],
                         self.ranges['down'], self.vel[0], self.vel[1], self.vel[2]],
                        dtype=np.float32)
        feat_n = normalize_sensor_features(feat)

        with torch.no_grad():
            logits = self.net(torch.from_numpy(feat_n).unsqueeze(0))
            prob = torch.softmax(logits, dim=1)[0].numpy()
            cls = int(prob.argmax())

        self.latest_class = cls
        self.count += 1

        msg = Int32()
        msg.data = cls
        self.pub_class.publish(msg)

        if self.count % self.log_interval == 0:
            self.get_logger().info(
                f'感知 #{self.count} | 距离(前/左/右/下)='
                f'{self.ranges["front"]:.2f}/{self.ranges["left"]:.2f}/'
                f'{self.ranges["right"]:.2f}/{self.ranges["down"]:.2f} m '
                f'| 判定: {OBSTACLE_CLASSES[cls]} (置信度 {prob[cls] * 100:.1f}%)')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SensorPerceptionNode()
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
