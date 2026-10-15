"""
声呐建图与避障导航节点
======================
订阅声呐测距扇与里程计，完成两件事：

1. **占据栅格建图**：用 7 向水平声呐的读数 + ROV 当前位姿做射线投射，
   在二维栅格地图上把射线经过的格子标记为空闲、命中点标记为占据，
   并把栅格地图发布为 `nav_msgs/OccupancyGrid`（可在 RViz 中查看）。
2. **避障导航**：用 `ObstacleMapNet` 判断哪一侧更开阔，据此生成转向修正量，
   与端到端网络给出的前进推力融合，输出到 `/cmd_vel`。

这是作业任务(3)"在虚拟环境中实现对机器人的建图和导航"的实现。

话题接口
--------
    订阅  /rov/odom, /rov/sonar/range_{f,fl,l,bl,fr,r,br,down}
    发布  /rov/map            nav_msgs/OccupancyGrid
          /rov/cmd_vel_nav    geometry_msgs/Twist   （避障修正量）
          /rov/open_direction std_msgs/Int32        （开阔方向分类结果）
"""

import math
import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32

try:
    import torch
except ImportError:
    sys.stderr.write('\n【依赖缺失】需要 PyTorch：pip3 install "torch==2.4.1+cpu"\n\n')
    raise

try:
    from rov_vision_e2e_nav.networks import (
        build_obstacle_map_net, normalize_sonar_features, OPEN_DIRECTION_CLASSES)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rov_vision_e2e_nav.networks import (
        build_obstacle_map_net, normalize_sonar_features, OPEN_DIRECTION_CLASSES)

#: 声呐方向与其在机体系中的朝向（弧度，0 = +X 前方，逆时针为正）
#: 说明：'down' 是垂直向下的测距，只参与神经网络特征、不参与二维射线投射，
#: 因此这里给它一个占位朝向（0.0），并在 _update_map 中显式跳过。
SONAR_BEARINGS = {
    'f':  0.0,
    'fl': math.radians(30),
    'l':  math.radians(90),
    'bl': math.radians(150),
    'fr': math.radians(-30),
    'r':  math.radians(-90),
    'br': math.radians(-150),
    'down': 0.0,
}

#: 参与二维占据栅格射线投射的水平方向（不含垂直向下的 down）
HORIZONTAL_SONAR = ('f', 'fl', 'l', 'bl', 'fr', 'r', 'br')

#: 送入神经网络的 8 维声呐特征顺序（与 collect_data.py 保持一致）
SONAR_FEATURE_ORDER = ('f', 'fl', 'l', 'bl', 'fr', 'r', 'br', 'down')


def yaw_from_quat(q):
    """四元数 (w,x,y,z) -> 偏航角（弧度）。"""
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                      1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))


class SonarSlamNode(Node):
    """声呐占据栅格建图 + 神经网络避障导航。"""

    def __init__(self):
        super().__init__('sonar_slam_node')

        # ---------------- 参数 ----------------
        self.declare_parameter('model_path', '')
        self.declare_parameter('map_width', 200)      # 栅格数（X 方向）
        self.declare_parameter('map_height', 200)     # 栅格数（Y 方向）
        self.declare_parameter('map_resolution', 0.1)  # 每格 0.1 m
        self.declare_parameter('map_origin_x', -5.0)
        self.declare_parameter('map_origin_y', -10.0)
        self.declare_parameter('publish_rate', 2.0)   # 地图发布频率
        self.declare_parameter('enable_nav', True)    # 是否输出避障修正
        self.declare_parameter('nav_turn_gain', 0.8)
        self.declare_parameter('log_interval', 20)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.map_w = self.get_parameter('map_width').get_parameter_value().integer_value
        self.map_h = self.get_parameter('map_height').get_parameter_value().integer_value
        self.res = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.ox = self.get_parameter('map_origin_x').get_parameter_value().double_value
        self.oy = self.get_parameter('map_origin_y').get_parameter_value().double_value
        pub_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.enable_nav = self.get_parameter('enable_nav').get_parameter_value().bool_value
        self.nav_turn_gain = self.get_parameter('nav_turn_gain').get_parameter_value().double_value
        self.log_interval = max(1, self.get_parameter('log_interval').get_parameter_value().integer_value)

        # ---------------- 加载声呐感知网络 ----------------
        if not model_path:
            model_path = self._find_model()
        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path):
            self.get_logger().error(
                f'找不到权重文件: {model_path}\n'
                f'请先训练：python3 src/water/rov_vision_e2e_nav/rov_vision_e2e_nav/train.py')
            raise FileNotFoundError(model_path)

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self.net = build_obstacle_map_net()
        self.net.load_state_dict(ckpt['state_dict'])
        self.net.eval()
        self.get_logger().info(
            f'声呐感知网络已加载: {model_path} | 训练 {ckpt.get("epochs", "?")} 轮 '
            f'| 验证损失 {ckpt.get("val_loss", float("nan")):.6f}')

        # ---------------- 占据栅格地图 ----------------
        # 约定：-1 = 未知，0 = 空闲，100 = 占据
        self.grid = np.full((self.map_h, self.map_w), -1, dtype=np.int8)
        self.cells_touched = 0

        # ---------------- 状态 ----------------
        self.ranges = {d: 20.0 for d in SONAR_BEARINGS}
        self.pos = np.zeros(3)
        self.yaw = 0.0
        self.got_odom = False
        self.count = 0
        self.open_dir = 1

        # ---------------- 通信 ----------------
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        for d in SONAR_BEARINGS:
            self.create_subscription(Range, f'/rov/sonar/range_{d}',
                                     self._mk_cb(d), qos)
        self.create_subscription(Odometry, '/rov/odom', self._on_odom, 10)

        self.pub_map = self.create_publisher(OccupancyGrid, '/rov/map', 1)
        self.pub_nav = self.create_publisher(Twist, '/rov/cmd_vel_nav', 10)
        self.pub_dir = self.create_publisher(Int32, '/rov/open_direction', 10)
        self.create_timer(1.0 / max(pub_rate, 1.0), self._on_publish)

        self.get_logger().info(
            f'声呐建图与导航节点已启动 | 地图 {self.map_w}x{self.map_h} @ {self.res} m/格 '
            f'| 避障修正 {"开启" if self.enable_nav else "关闭"}')

    @staticmethod
    def _find_model():
        here = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        candidates = [
            os.path.join(here, '..', 'model_seg.pth'),
            os.path.join(here, 'model_seg.pth'),
            os.path.join(cwd, 'model_seg.pth'),
            os.path.join(cwd, 'src', 'rov_vision_e2e_nav', 'model_seg.pth'),
            os.path.join(cwd, 'src', 'water', 'rov_vision_e2e_nav', 'model_seg.pth'),
        ]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('rov_vision_e2e_nav'), 'model_seg.pth'))
        except Exception:  # noqa: BLE001
            pass
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c
        return candidates[0]

    def _mk_cb(self, d):
        def cb(msg: Range):
            self.ranges[d] = float(msg.range)
        return cb

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self.pos = np.array([p.x, p.y, p.z])
        self.yaw = yaw_from_quat([o.w, o.x, o.y, o.z])
        self.got_odom = True

    # ------------------------------------------------------------------ #
    def _world_to_grid(self, x, y):
        """世界坐标 -> 栅格索引，越界返回 None。"""
        gx = int((x - self.ox) / self.res)
        gy = int((y - self.oy) / self.res)
        if 0 <= gx < self.map_w and 0 <= gy < self.map_h:
            return gx, gy
        return None

    def _update_map(self):
        """用一帧声呐读数做射线投射，更新占据栅格。

        只使用水平方向（HORIZONTAL_SONAR）做二维投射；
        垂直向下的 down 不参与建图。
        """
        for d in HORIZONTAL_SONAR:
            r = self.ranges[d]
            if r >= 19.9:          # 未命中任何物体，跳过
                continue
            bearing = SONAR_BEARINGS[d]
            ang = self.yaw + bearing
            # 从机体当前位置沿该方向按栅格步长行走，沿途标空闲，终点标占据
            steps = int(r / self.res)
            for i in range(steps):
                dist = i * self.res
                wx = self.pos[0] + dist * math.cos(ang)
                wy = self.pos[1] + dist * math.sin(ang)
                cell = self._world_to_grid(wx, wy)
                if cell is None:
                    continue
                gx, gy = cell
                if self.grid[gy, gx] == -1:
                    self.grid[gy, gx] = 0
                    self.cells_touched += 1
            # 命中点标为占据
            wx = self.pos[0] + r * math.cos(ang)
            wy = self.pos[1] + r * math.sin(ang)
            cell = self._world_to_grid(wx, wy)
            if cell is not None:
                gx, gy = cell
                self.grid[gy, gx] = 100

    def _on_publish(self):
        if not self.got_odom:
            return

        self._update_map()

        # ---------------- 神经网络判定开阔方向 ----------------
        feat = np.array([self.ranges[d] for d in SONAR_FEATURE_ORDER],
                        dtype=np.float32)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(normalize_sonar_features(feat)).unsqueeze(0))
            prob = torch.softmax(logits, dim=1)[0].numpy()
            self.open_dir = int(prob.argmax())

        msg_dir = Int32()
        msg_dir.data = self.open_dir
        self.pub_dir.publish(msg_dir)

        # ---------------- 发布栅格地图 ----------------
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'world'
        grid_msg.info.resolution = float(self.res)
        grid_msg.info.width = self.map_w
        grid_msg.info.height = self.map_h
        grid_msg.info.origin.position.x = float(self.ox)
        grid_msg.info.origin.position.y = float(self.oy)
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = self.grid.flatten().tolist()
        self.pub_map.publish(grid_msg)

        # ---------------- 避障修正 ----------------
        if self.enable_nav:
            nav = Twist()
            # 前方受阻就减速；哪侧开阔就往哪侧转
            front = self.ranges['f']
            nav.linear.x = 0.0     # 前进推力由端到端节点负责，这里只给转向修正
            if front < 2.5:
                if self.open_dir == 0:        # 左侧开阔 → 左转
                    nav.angular.z = +self.nav_turn_gain
                elif self.open_dir == 2:      # 右侧开阔 → 右转
                    nav.angular.z = -self.nav_turn_gain
                else:
                    nav.angular.z = 0.0       # 正前开阔 → 不修正
            self.pub_nav.publish(nav)

        self.count += 1
        if self.count % self.log_interval == 0:
            known = int(np.count_nonzero(self.grid != -1))
            occupied = int(np.count_nonzero(self.grid == 100))
            self.get_logger().info(
                f'建图 #{self.count} | 位置=({self.pos[0]:+.2f},{self.pos[1]:+.2f}) '
                f'yaw={math.degrees(self.yaw):+.1f}° | 已探索 {known} 格（占据 {occupied}）'
                f' | 前方 {self.ranges["f"]:.2f} m | 判定: '
                f'{OPEN_DIRECTION_CLASSES[self.open_dir]} (置信度 {prob[self.open_dir] * 100:.1f}%)')

    def destroy_node(self):
        try:
            self.pub_nav.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SonarSlamNode()
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
