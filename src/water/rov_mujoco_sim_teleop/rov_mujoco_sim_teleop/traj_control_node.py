"""
轨迹跟踪控制节点（神经网络）
============================
订阅 ROV 的里程计与测距数据，构造状态误差特征，送入
`TrajectoryControlNet` 得到 4 维控制量，再发布为 `/cmd_vel`
驱动 MuJoCo 仿真中的 ROV 沿给定轨迹前进。

这是作业要求的"**使用神经网络实现给定轨迹的控制**"部分：
与传统 PID 不同，这里的控制律完全由一个训练得到的 MLP 给出。

期望轨迹
--------
沿水下通道中线前进：目标点为 `(x_target, 0, -1.0)`，
其中 `x_target` 随进度推进（每步沿 +X 前进一点），
形成"给定轨迹"，控制器负责把 ROV 拉回中线并保持深度。

话题接口
--------
    订阅  /rov/odom, /rov/range/front
    发布  /cmd_vel  geometry_msgs/Twist
"""

import math
import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist

try:
    import torch
except ImportError:
    sys.stderr.write('\n【依赖缺失】需要 PyTorch：pip3 install "torch==2.4.1+cpu"\n\n')
    raise

try:
    from rov_mujoco_sim_teleop.networks import (
        build_traj_net, CONTROL_DIMS, normalize_traj_features)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rov_mujoco_sim_teleop.networks import (
        build_traj_net, CONTROL_DIMS, normalize_traj_features)


def yaw_from_quat(q):
    """四元数 (w,x,y,z) -> 偏航角（弧度）。"""
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                      1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))


def wrap_pi(a):
    """把角度归一化到 [-pi, pi]。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class TrajControlNode(Node):
    """用神经网络做轨迹跟踪的闭环控制器。"""

    def __init__(self):
        super().__init__('traj_control_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('forward_step', 0.8)     # 期望点每步前进距离
        self.declare_parameter('target_y', 0.0)         # 通道中线
        self.declare_parameter('target_depth', -1.0)    # 期望深度
        self.declare_parameter('x_max', 6.5)            # 终点
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('log_interval', 20)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.forward_step = self.get_parameter('forward_step').get_parameter_value().double_value
        self.target_y = self.get_parameter('target_y').get_parameter_value().double_value
        self.target_depth = self.get_parameter('target_depth').get_parameter_value().double_value
        self.x_max = self.get_parameter('x_max').get_parameter_value().double_value
        pub_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.log_interval = max(1, self.get_parameter('log_interval').get_parameter_value().integer_value)

        if not model_path:
            model_path = self._find_model()
        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path):
            self.get_logger().error(
                f'找不到权重文件: {model_path}\n'
                f'请先训练：python3 src/water/rov_mujoco_sim_teleop/rov_mujoco_sim_teleop/train.py')
            raise FileNotFoundError(model_path)

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self.net = build_traj_net()
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.eval()
        self.get_logger().info(
            f'轨迹控制网络已加载: {model_path} | 训练 {ckpt.get("epochs", "?")} 轮 '
            f'| 验证损失 {ckpt.get("val_loss", float("nan")):.6f}')

        # ---------------- 状态 ----------------
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw = 0.0
        self.range_front = 20.0
        self.x_target = 0.0
        self.count = 0
        self.got_odom = False

        # ---------------- 通信 ----------------
        self.create_subscription(Odometry, '/rov/odom', self._on_odom, 10)
        self.create_subscription(Range, '/rov/range/front', self._on_range, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / max(pub_rate, 1.0), self._on_timer)

        self.get_logger().info(
            f'神经网络轨迹控制节点已启动 | 目标: 沿 y={self.target_y} 通道中线前进至 x={self.x_max}，'
            f'深度保持 {self.target_depth} m')

    @staticmethod
    def _find_model():
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, '..', 'model_traj.pth'),
            os.path.join(os.getcwd(), 'model_traj.pth'),
            os.path.join(os.getcwd(), 'src', 'rov_mujoco_sim_teleop', 'model_traj.pth'),
        ]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('rov_mujoco_sim_teleop'), 'model_traj.pth'))
        except Exception:  # noqa: BLE001
            pass
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c
        return candidates[0]

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self.pos = np.array([p.x, p.y, p.z])
        self.vel = np.array([v.x, v.y, v.z])
        self.yaw = yaw_from_quat([o.w, o.x, o.y, o.z])
        self.got_odom = True

    def _on_range(self, msg: Range):
        self.range_front = float(msg.range)

    def _on_timer(self):
        """构造状态误差 → 神经网络推理 → 发布控制指令。"""
        if not self.got_odom:
            return

        # 期望点随进度推进，形成给定轨迹
        if self.pos[0] > self.x_target:
            self.x_target = min(self.x_max, self.pos[0] + self.forward_step)
        desired = np.array([self.x_target, self.target_y, self.target_depth])

        pos_err = desired - self.pos
        vel_err = -self.vel
        yaw_err = wrap_pi(0.0 - self.yaw)

        feat = np.array([pos_err[0], pos_err[1], pos_err[2],
                         vel_err[0], vel_err[1], vel_err[2],
                         yaw_err, self.range_front], dtype=np.float32)
        feat_n = normalize_traj_features(feat)

        with torch.no_grad():
            u = self.net(torch.from_numpy(feat_n).unsqueeze(0))[0].numpy()

        cmd = Twist()
        cmd.linear.x = float(u[0])          # 前进推力
        cmd.linear.y = float(u[1])          # 横移推力
        cmd.linear.z = float(u[2]) * 0.15   # 升沉（小幅，主要靠定深控制器）
        cmd.angular.z = float(u[3])         # 偏航力矩
        self.pub_cmd.publish(cmd)

        self.count += 1
        if self.count % self.log_interval == 0:
            self.get_logger().info(
                f'控制 #{self.count} | 位置=({self.pos[0]:+.2f},{self.pos[1]:+.2f},{self.pos[2]:+.2f}) '
                f'| 目标x={self.x_target:.2f} | 误差=({pos_err[0]:+.2f},{pos_err[1]:+.2f},{pos_err[2]:+.2f}) '
                f'| 输出: ' + ' '.join(f'{n}={cmd_val:+.2f}'
                                        for n, cmd_val in zip(CONTROL_DIMS, u)))

    def destroy_node(self):
        # 退出前发一次零指令，避免 ROV 带着上一帧推力继续冲撞
        try:
            self.pub_cmd.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TrajControlNode()
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
