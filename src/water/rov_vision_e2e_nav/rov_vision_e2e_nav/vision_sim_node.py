"""
MuJoCo 视觉与声呐仿真节点
==========================
把 MuJoCo 中水下 ROV 的**前视相机**与**声呐测距扇**封装为 ROS 2 节点，
为单目视觉端到端控制（任务④）与声呐建图导航（任务③）提供仿真数据源。

话题接口
--------
    订阅  /cmd_vel                  geometry_msgs/Twist
    发布  /rov/odom                 nav_msgs/Odometry
          /rov/depth                std_msgs/Float64
          /rov/imu                  sensor_msgs/Imu
          /rov/camera/image_raw     sensor_msgs/Image        （前视相机）
          /rov/sonar/range_f …      sensor_msgs/Range × 8    （声呐测距扇）

设计要点
--------
* **推力用 xfrc_applied**：MuJoCo 自由关节无法直接挂 actuator，
  每步把机体系推力经旋转矩阵变换到世界系后写入 `data.xfrc_applied`。
* **声呐测距扇**：模型内置 7 向水平扇（前/左前/左/左后/右前/右/右后）
  与 1 向向下，用于占据栅格建图；`rangefinder` 沿 site 的 +Z 轴发射。
* **推力已标定**：`force_scale` 默认 10 N（标定实验见 PR1 的文档）。
"""

import os

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, Range
from std_msgs.msg import Float64

# MUJOCO_GL 必须在 import mujoco 之前生效
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        '\n==================================================================\n'
        '【环境依赖缺失】未检测到 MuJoCo。请先安装：\n'
        '    pip3 install "mujoco<3.2" numpy PyOpenGL pillow\n'
        '    sudo apt install libosmesa6\n'
        '==================================================================\n'
    ) from exc

#: 声呐测距扇的方向键（与模型中的 site 名称一一对应）
SONAR_DIRECTIONS = ('f', 'fl', 'l', 'bl', 'fr', 'r', 'br', 'down')


class VisionSimNode(Node):
    """水下 ROV 的相机与声呐仿真节点。"""

    def __init__(self):
        super().__init__('vision_sim_node')

        # ---------------- 参数 ----------------
        self.declare_parameter('model_path', '')
        self.declare_parameter('sim_rate', 500.0)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('buoyancy_compensate', True)
        self.declare_parameter('force_scale', 10.0)      # 已标定
        self.declare_parameter('torque_scale', 2.0)
        self.declare_parameter('target_depth', -1.0)
        self.declare_parameter('camera_width', 160)
        self.declare_parameter('camera_height', 120)
        self.declare_parameter('camera_name', 'forward_cam')
        self.declare_parameter('enable_camera', True)
        self.declare_parameter('camera_every', 5)
        self.declare_parameter('max_range', 20.0)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.sim_rate = self.get_parameter('sim_rate').get_parameter_value().double_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.buoyancy_compensate = self.get_parameter('buoyancy_compensate').get_parameter_value().bool_value
        self.force_scale = self.get_parameter('force_scale').get_parameter_value().double_value
        self.torque_scale = self.get_parameter('torque_scale').get_parameter_value().double_value
        self.target_depth = self.get_parameter('target_depth').get_parameter_value().double_value
        self.cam_w = self.get_parameter('camera_width').get_parameter_value().integer_value
        self.cam_h = self.get_parameter('camera_height').get_parameter_value().integer_value
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        self.enable_camera = self.get_parameter('enable_camera').get_parameter_value().bool_value
        self.camera_every = max(1, self.get_parameter('camera_every').get_parameter_value().integer_value)
        self.max_range = self.get_parameter('max_range').get_parameter_value().double_value

        if not model_path:
            model_path = self._find_model()
        model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(model_path):
            self.get_logger().error(f'找不到 MuJoCo 模型: {model_path}')
            raise FileNotFoundError(model_path)
        self.get_logger().info(f'加载模型: {model_path}')

        # ---------------- 加载模型 ----------------
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'rov')
        if self.body_id < 0:
            raise RuntimeError("模型里找不到名为 'rov' 的 body")
        jnt = self.model.body_jntadr[self.body_id]
        self.qpos_addr = self.model.jnt_qposadr[jnt]
        self.qvel_addr = self.model.jnt_dofadr[jnt]

        total_mass = float(np.sum(self.model.body_mass))
        self.buoyancy_force = total_mass * abs(float(self.model.opt.gravity[2]))

        # 相机渲染器（可选，失败时降级不影响其余话题）
        self.renderer = None
        if self.enable_camera:
            try:
                self.renderer = mujoco.Renderer(self.model, self.cam_h, self.cam_w)
                self.get_logger().info(
                    f'前视相机已就绪: {self.cam_w}x{self.cam_h} @ {self.camera_name} '
                    f'(MUJOCO_GL={os.environ.get("MUJOCO_GL")})')
            except Exception as err:  # noqa: BLE001
                self.get_logger().warn(
                    f'相机渲染器创建失败，/rov/camera/image_raw 将不发布图像: {err}\n'
                    f'无显卡环境请设置 MUJOCO_GL=osmesa 并安装 libosmesa6')

        # ---------------- 控制与状态 ----------------
        self.cmd = Twist()
        self.step_count = 0
        self.pub_count = 0

        # ---------------- 通信 ----------------
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)

        self.pub_odom = self.create_publisher(Odometry, '/rov/odom', 10)
        self.pub_depth = self.create_publisher(Float64, '/rov/depth', 10)
        self.pub_imu = self.create_publisher(Imu, '/rov/imu', 10)
        self.pub_image = self.create_publisher(Image, '/rov/camera/image_raw', 10)
        self.pub_sonar = {
            d: self.create_publisher(Range, f'/rov/sonar/range_{d}', 10)
            for d in SONAR_DIRECTIONS
        }

        # ---------------- 定时器 ----------------
        self.create_timer(1.0 / max(self.sim_rate, 1.0), self._on_step)
        self.create_timer(1.0 / max(self.publish_rate, 1.0), self._on_publish)

        self.get_logger().info(
            f'视觉与声呐仿真节点已启动 | 物理 {self.sim_rate:.0f} Hz | 发布 {self.publish_rate:.0f} Hz '
            f'| 声呐 {len(SONAR_DIRECTIONS)} 向')

    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_model():
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, '..', 'models', 'rov_vision_channel.xml'),
            os.path.join(os.getcwd(), 'models', 'rov_vision_channel.xml'),
            os.path.join(os.getcwd(), 'src', 'rov_vision_e2e_nav', 'models', 'rov_vision_channel.xml'),
        ]
        try:
            from ament_index_python.packages import get_package_share_directory
            candidates.append(os.path.join(
                get_package_share_directory('rov_vision_e2e_nav'),
                'models', 'rov_vision_channel.xml'))
        except Exception:  # noqa: BLE001
            pass
        for c in candidates:
            c = os.path.abspath(c)
            if os.path.exists(c):
                return c
        return candidates[0]

    def _on_cmd(self, msg: Twist):
        self.cmd = msg

    # ------------------------------------------------------------------ #
    def _on_step(self):
        """物理步进：注入浮力、姿态自稳、定深与推进力。"""
        dt = float(self.model.opt.timestep)
        b = self.body_id

        self.data.xfrc_applied[b, :] = 0.0

        # 中性浮力补偿
        if self.buoyancy_compensate:
            self.data.xfrc_applied[b, 2] += self.buoyancy_force

        # 姿态自稳（增益按 13 kg 整机质量标定）
        quat = self.data.qpos[self.qpos_addr + 3: self.qpos_addr + 7]
        av = self.data.qvel[self.qvel_addr + 3: self.qvel_addr + 6]
        stab_x = -12.0 * float(quat[1]) - 3.0 * float(av[0])
        stab_y = -12.0 * float(quat[2]) - 3.0 * float(av[1])
        stab_z = -0.8 * float(av[2])

        # 定深 PD
        cur_z = float(self.data.qpos[self.qpos_addr + 2])
        vel_z = float(self.data.qvel[self.qvel_addr + 2])
        thrust_z = 26.0 * (self.target_depth - cur_z) - 22.0 * vel_z \
            + self.cmd.linear.z * self.force_scale
        if abs(self.cmd.linear.z) > 0.01:
            self.target_depth += self.cmd.linear.z * 0.6 * dt

        # 水平推力：机体系 -> 世界系
        rot = self.data.xmat[b].reshape(3, 3)
        f_body = np.array([self.cmd.linear.x * self.force_scale,
                           self.cmd.linear.y * self.force_scale, 0.0])
        f_world = rot @ f_body

        self.data.xfrc_applied[b, 0] += f_world[0]
        self.data.xfrc_applied[b, 1] += f_world[1]
        self.data.xfrc_applied[b, 2] += thrust_z
        self.data.xfrc_applied[b, 3] += stab_x
        self.data.xfrc_applied[b, 4] += stab_y
        self.data.xfrc_applied[b, 5] += stab_z + self.cmd.angular.z * self.torque_scale

        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

    # ------------------------------------------------------------------ #
    def _on_publish(self):
        now = self.get_clock().now().to_msg()
        q, v = self.qpos_addr, self.qvel_addr
        pos = self.data.qpos[q: q + 3]
        quat = self.data.qpos[q + 3: q + 7]
        linvel = self.data.qvel[v: v + 3]

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

        depth = Float64()
        depth.data = float(pos[2])
        self.pub_depth.publish(depth)

        imu = Imu()
        imu.header.stamp = now
        imu.header.frame_id = 'rov_imu_link'
        imu.orientation.w = float(quat[0])
        imu.orientation.x = float(quat[1])
        imu.orientation.y = float(quat[2])
        imu.orientation.z = float(quat[3])
        try:
            g = self.data.sensor('rov_gyro').data[:3]
            a = self.data.sensor('rov_accel').data[:3]
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = \
                float(g[0]), float(g[1]), float(g[2])
            imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = \
                float(a[0]), float(a[1]), float(a[2])
        except Exception:  # noqa: BLE001
            pass
        self.pub_imu.publish(imu)

        # 声呐测距扇
        for d in SONAR_DIRECTIONS:
            self._publish_sonar(d, now)

        # 前视相机（降采样发布）
        self.pub_count += 1
        if self.renderer is not None and self.pub_count % self.camera_every == 0:
            self._publish_image(now)

    def _publish_sonar(self, direction, stamp):
        """把 MuJoCo rangefinder 读数发布为 sensor_msgs/Range。

        MuJoCo 未命中时返回 -1，这里统一映射为量程上限，
        避免下游把 -1 误当作"极近距离"。
        """
        name = f'sonar_range_{direction}'
        try:
            val = float(self.data.sensor(name).data[0])
        except Exception:  # noqa: BLE001
            return
        msg = Range()
        msg.header.stamp = stamp
        msg.header.frame_id = 'rov_sonar_link'
        msg.radiation_type = Range.ULTRASOUND   # 声呐用超声类型
        msg.field_of_view = 0.6
        msg.min_range = 0.05
        msg.max_range = float(self.max_range)
        msg.range = self.max_range if (val < 0 or val > self.max_range) else val
        self.pub_sonar[direction].publish(msg)

    def _publish_image(self, stamp):
        """渲染前视相机并发布 rgb8 图像。"""
        try:
            self.renderer.update_scene(self.data, camera=self.camera_name)
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
    """``vision_sim_node`` 进程入口。"""
    rclpy.init(args=args)
    node = None
    try:
        node = VisionSimNode()
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
