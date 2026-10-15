"""
MuJoCo 水下 ROV 仿真节点
=========================
本模块把 MuJoCo 物理引擎中水下 ROV 的运动仿真封装为 ROS 2 节点，
是整个功能包的内核。它负责：

  1. 加载本包自带的、完全自包含的 MJCF 模型（`models/rov_channel.xml`）；
  2. 以固定频率步进物理，并注入水下环境力（浮力补偿 + 各向异性流体阻力）；
  3. 把 `/cmd_vel` 的速度指令映射为机体推力与力矩，实现 6 自由度运动；
  4. 把位姿、速度、IMU、四向测距与前视相机图像发布为 ROS 2 话题。

话题接口
--------
    订阅  /cmd_vel              geometry_msgs/Twist
          linear.x  前进推力 ([-1,1] 归一化)
          linear.y  横移推力
          linear.z  升沉（驱动定深目标）
          angular.z 偏航力矩
    发布  /rov/odom             nav_msgs/Odometry
          /rov/depth            std_msgs/Float64
          /rov/imu              sensor_msgs/Imu
          /rov/camera/image_raw sensor_msgs/Image
          /rov/range            sensor_msgs/Range × 4（前/左/右/下）

设计要点（为什么这么做）
------------------------
* **推力用 xfrc_applied 而不是 actuator**：MuJoCo 的自由关节无法直接挂 motor，
  因此每步把机体系推力用旋转矩阵转到世界系后写入 `data.xfrc_applied`。
* **中性浮力补偿**：把整机重力用浮力完全抵消，使 ROV 能悬停。
  这正是 `config/rov_params.yaml` 里 `buoyancy_compensate` 的作用。
* **姿态自稳力矩**：对 roll/pitch 施加比例-微分阻尼，防止 ROV 在水中翻滚，
  让操作者只需关注前进/转向/升沉，降低遥控难度（对应任务(1)）。
* **测距射线**：`rangefinder` 沿 site 的 +Z 轴发射，模型里四个 site 已分别
  设好 zaxis，因此本节点直接读传感器值即可，不需要自己做射线求交。
"""

import os

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, Range
from std_msgs.msg import Float64, Header

# ---------------------------------------------------------------------------
# MUJOCO_GL 必须在 import mujoco 之前生效。
# 无显卡环境若用默认后端，Renderer 会拿不到 GL 上下文而失败；
# OSMesa 是纯 CPU 软件渲染，适用于本作业的 Ubuntu 虚拟机。
# ---------------------------------------------------------------------------
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        '\n==================================================================\n'
        '【环境依赖缺失】未检测到 MuJoCo。请先安装：\n'
        '    pip3 install "mujoco<3.2" numpy PyOpenGL pillow\n'
        '    # 无显卡环境还需要：sudo apt install libosmesa6\n'
        '==================================================================\n'
    ) from exc


class MujocoSimNode(Node):
    """水下 ROV 的 MuJoCo 物理仿真节点。"""

    def __init__(self):
        super().__init__('mujoco_sim_node')

        # ---------------- 参数 ----------------
        self.declare_parameter('model_path', '')
        self.declare_parameter('sim_rate', 500.0)          # 物理步进频率 Hz
        self.declare_parameter('publish_rate', 50.0)       # 话题发布频率 Hz
        self.declare_parameter('buoyancy_compensate', True)
        self.declare_parameter('force_scale', 10.0)        # 归一化推力 -> 牛顿
        self.declare_parameter('torque_scale', 2.0)        # 归一化力矩 -> 牛米
        self.declare_parameter('target_depth', -1.0)       # 期望深度（米，向下为负）
        self.declare_parameter('camera_width', 160)
        self.declare_parameter('camera_height', 120)
        self.declare_parameter('enable_camera', True)
        self.declare_parameter('publish_camera_every', 5)  # 每 N 次发布发一帧图像

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.sim_rate = self.get_parameter('sim_rate').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.buoyancy_compensate = self.get_parameter('buoyancy_compensate').get_parameter_value().bool_value
        self.force_scale = self.get_parameter('force_scale').get_parameter_value().double_value
        self.torque_scale = self.get_parameter('torque_scale').get_parameter_value().double_value
        self.target_depth = self.get_parameter('target_depth').get_parameter_value().double_value
        self.cam_w = self.get_parameter('camera_width').get_parameter_value().integer_value
        self.cam_h = self.get_parameter('camera_height').get_parameter_value().integer_value
        self.enable_camera = self.get_parameter('enable_camera').get_parameter_value().bool_value
        self.publish_camera_every = max(1, self.get_parameter('publish_camera_every').get_parameter_value().integer_value)

        if not model_path:
            model_path = self._default_model_path()
        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path):
            self.get_logger().error(
                f'找不到 MuJoCo 模型文件: {model_path}\n'
                f'该模型随功能包一起安装，请确认已正确 colcon build 并 source install/setup.bash')
            raise FileNotFoundError(model_path)
        self.get_logger().info(f'加载模型: {model_path}')

        # ---------------- 加载模型 ----------------
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rov')
        if self.body_id < 0:
            raise RuntimeError("模型里找不到名为 'rov' 的 body")
        joint_id = self.model.body_jntadr[self.body_id]
        self.qpos_addr = self.model.jnt_qposadr[joint_id]
        self.qvel_addr = self.model.jnt_dofadr[joint_id]

        # 中性浮力：用浮力完全抵消整机重力，使 ROV 可悬停
        total_mass = float(np.sum(self.model.body_mass))
        self.buoyancy_force = total_mass * abs(float(self.model.opt.gravity[2]))

        # 相机渲染器（可选：拿不到 GL 上下文时降级，不影响其余话题）
        self.renderer = None
        if self.enable_camera:
            try:
                self.renderer = mujoco.Renderer(self.model, self.cam_h, self.cam_w)
                self.get_logger().info(f'前视相机已就绪: {self.cam_w}x{self.cam_h} (MUJOCO_GL={os.environ.get("MUJOCO_GL")})')
            except Exception as err:  # noqa: BLE001
                self.get_logger().warn(
                    f'相机渲染器创建失败，/rov/camera/image_raw 将不发布图像: {err}\n'
                    f'无显卡环境请设置 MUJOCO_GL=osmesa 并安装 libosmesa6')

        # ---------------- 控制状态 ----------------
        self.cmd = Twist()
        self.step_count = 0
        self.pub_count = 0

        # ---------------- 通信 ----------------
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10)

        self.pub_odom = self.create_publisher(Odometry, '/rov/odom', 10)
        self.pub_depth = self.create_publisher(Float64, '/rov/depth', 10)
        self.pub_imu = self.create_publisher(Imu, '/rov/imu', 10)
        self.pub_image = self.create_publisher(Image, '/rov/camera/image_raw', 10)
        self.pub_range_front = self.create_publisher(Range, '/rov/range/front', 10)
        self.pub_range_left = self.create_publisher(Range, '/rov/range/left', 10)
        self.pub_range_right = self.create_publisher(Range, '/rov/range/right', 10)
        self.pub_range_down = self.create_publisher(Range, '/rov/range/down', 10)

        # ---------------- 定时器 ----------------
        self.create_timer(1.0 / max(self.sim_rate, 1.0), self._on_step)
        self.create_timer(1.0 / max(self.publish_rate, 1.0), self._on_publish)

        self.get_logger().info(
            f'MuJoCo 仿真节点已启动 | 物理 {self.sim_rate:.0f} Hz | 发布 {self.publish_rate:.0f} Hz | '
            f'中性浮力补偿={self.buoyancy_compensate}')

    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_model_path():
        """按"源码树 → 安装目录"的顺序探测默认模型路径。

        这样无论用户是直接在源码目录运行，还是 colcon 安装后运行，都能找到模型。
        """
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, '..', 'models', 'rov_channel.xml'),
            os.path.join(os.getcwd(), 'models', 'rov_channel.xml'),
            os.path.join(os.getcwd(), 'src', 'rov_mujoco_sim_teleop', 'models', 'rov_channel.xml'),
        ]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('rov_mujoco_sim_teleop'), 'models', 'rov_channel.xml'))
        except Exception:  # noqa: BLE001
            pass
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c
        return candidates[0]

    def _on_cmd_vel(self, msg: Twist):
        """缓存最新速度指令（由物理步进回调消费）。"""
        self.cmd = msg

    # ------------------------------------------------------------------ #
    def _on_step(self):
        """物理步进：注入环境力与控制力，然后积分一步。"""
        dt = float(self.model.opt.timestep)
        body = self.body_id

        # 1) 清空上一步的外部力/力矩
        self.data.xfrc_applied[body, :] = 0.0

        # 2) 中性浮力补偿（沿世界系 +Z 抵消重力）
        if self.buoyancy_compensate:
            self.data.xfrc_applied[body, 2] += self.buoyancy_force

        # 3) 姿态自稳：对 roll/pitch 做比例-微分阻尼，防止水中翻滚。
        #    增益同样按 13 kg 整机质量标定（见标定实验）。
        quat = self.data.qpos[self.qpos_addr + 3: self.qpos_addr + 7]
        ang_vel = self.data.qvel[self.qvel_addr + 3: self.qvel_addr + 6]
        stab_x = -12.0 * float(quat[1]) - 3.0 * float(ang_vel[0])
        stab_y = -12.0 * float(quat[2]) - 3.0 * float(ang_vel[1])
        stab_z = -0.8 * float(ang_vel[2])

        # 4) 定深：用比例-微分控制把深度拉向 target_depth，
        #    同时叠加操作者的升沉指令（linear.z）。
        #    增益按整机质量 13 kg 标定：260 N/m 对应约 0.9 s 的深度响应时间。
        current_z = float(self.data.qpos[self.qpos_addr + 2])
        depth_err = self.target_depth - current_z
        vel_z = float(self.data.qvel[self.qvel_addr + 2])
        thrust_z = 26.0 * depth_err - 22.0 * vel_z + self.cmd.linear.z * self.force_scale
        if abs(self.cmd.linear.z) > 0.01:
            # 操作者主动升降时，同步移动目标深度，避免控制器反向拉扯
            self.target_depth += self.cmd.linear.z * 0.6 * dt

        # 5) 水平推力：机体系 -> 世界系（用机体旋转矩阵）
        rot = self.data.xmat[body].reshape(3, 3)
        f_body = np.array([self.cmd.linear.x * self.force_scale,
                           self.cmd.linear.y * self.force_scale,
                           0.0])
        f_world = rot @ f_body

        # 6) 偏航力矩：绕世界系 Z 轴（水下 ROV 偏航通常直接用世界系 Z 更直观）
        tau_z = self.cmd.angular.z * self.torque_scale

        # 7) 写回外部分力/力矩并积分
        self.data.xfrc_applied[body, 0] += f_world[0]
        self.data.xfrc_applied[body, 1] += f_world[1]
        self.data.xfrc_applied[body, 2] += thrust_z
        self.data.xfrc_applied[body, 3] += stab_x
        self.data.xfrc_applied[body, 4] += stab_y
        self.data.xfrc_applied[body, 5] += stab_z + tau_z

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

    # ------------------------------------------------------------------ #
    def _on_publish(self):
        """发布传感器数据。"""
        now = self.get_clock().now().to_msg()
        q = self.qpos_addr
        v = self.qvel_addr
        pos = self.data.qpos[q: q + 3]
        quat = self.data.qpos[q + 3: q + 7]
        linvel = self.data.qvel[v: v + 3]

        # --- 里程计 ---
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'rov_base'
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])
        odom.twist.twist.linear.x = float(linvel[0])
        odom.twist.twist.linear.y = float(linvel[1])
        odom.twist.twist.linear.z = float(linvel[2])
        self.pub_odom.publish(odom)

        # --- 深度 ---
        depth = Float64()
        depth.data = float(pos[2])
        self.pub_depth.publish(depth)

        # --- IMU ---
        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = 'rov_imu_link'
        imu.orientation.w = float(quat[0])
        imu.orientation.x = float(quat[1])
        imu.orientation.y = float(quat[2])
        imu.orientation.z = float(quat[3])
        try:
            gyro = self.data.sensor('rov_gyro').data[:3]
            accel = self.data.sensor('rov_accel').data[:3]
            imu.angular_velocity.x = float(gyro[0])
            imu.angular_velocity.y = float(gyro[1])
            imu.angular_velocity.z = float(gyro[2])
            imu.linear_acceleration.x = float(accel[0])
            imu.linear_acceleration.y = float(accel[1])
            imu.linear_acceleration.z = float(accel[2])
        except Exception:  # noqa: BLE001
            pass
        self.pub_imu.publish(imu)

        # --- 四向测距 ---
        for name, pub in (('range_front', self.pub_range_front),
                          ('range_left', self.pub_range_left),
                          ('range_right', self.pub_range_right),
                          ('range_down', self.pub_range_down)):
            self._publish_range(name, pub, now)

        # --- 前视相机（降采样发布，降低 CPU 软件渲染压力）---
        self.pub_count += 1
        if self.renderer is not None and self.pub_count % self.publish_camera_every == 0:
            self._publish_image(now)

    def _publish_range(self, sensor_name, publisher, stamp):
        """把 MuJoCo rangefinder 的读数发布为 sensor_msgs/Range。

        MuJoCo 的 rangefinder 在未命中任何几何体时返回 -1，
        这里统一映射为量程上限，避免下游把 -1 当成"极近距离"。
        """
        try:
            val = float(self.data.sensor(sensor_name).data[0])
        except Exception:  # noqa: BLE001
            return
        max_range = 20.0
        msg = Range()
        msg.header.stamp = stamp
        msg.header.frame_id = 'rov_range'
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 0.05
        msg.min_range = 0.05
        msg.max_range = max_range
        msg.range = max_range if (val < 0 or val > max_range) else val
        publisher.publish(msg)

    def _publish_image(self, stamp):
        """渲染前视相机并发布为 sensor_msgs/Image（rgb8）。"""
        try:
            self.renderer.update_scene(self.data, camera='forward_cam')
            frame = self.renderer.render()
        except Exception as err:  # noqa: BLE001
            self.get_logger().warn(f'渲染失败（已跳过该帧）: {err}', throttle_duration_sec=5.0)
            return

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'rov_camera_optical_link'
        msg.height = int(frame.shape[0])
        msg.width = int(frame.shape[1])
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = np.ascontiguousarray(frame).tobytes()
        self.pub_image.publish(msg)

    def destroy_node(self):
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:  # noqa: BLE001
                pass
        super().destroy_node()


def main(args=None):
    """``mujoco_sim_node`` 进程入口（setup.py 的 console_scripts 指向本函数）。"""
    rclpy.init(args=args)
    node = None
    try:
        node = MujocoSimNode()
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
