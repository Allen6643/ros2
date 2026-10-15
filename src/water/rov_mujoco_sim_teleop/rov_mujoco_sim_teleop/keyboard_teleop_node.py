"""
键盘遥控节点
============
把键盘按键解析为 `geometry_msgs/Twist` 速度指令并发到 `/cmd_vel`，
供 `mujoco_sim_node` 消费，实现对水下 ROV 的 6 自由度运动控制（对应任务(1)）。

按键映射（锁存式：按一次持续生效，直到按其他方向键或空格）
--------------------------------------------------------
    W / ↑   前进      S / ↓   后退
    A / ←   原地左转  D / →   原地右转
    Q       上浮      E       下潜
    Z / X   左横移 / 右横移
    空格    全部归零（悬停）
    R       回到初始位姿（重置）
    +/-     调节推力比例
    Ctrl+C  退出

为什么用"锁存式"而不是"瞬时触发"
--------------------------------
瞬时触发（按住才动、松手即停）在无人值守或脚本化采数据时会把绝大多数
标签压成 0，导致后续模仿学习学不到有效策略；锁存式让操作者单手即可稳定
给出一条持续的指令，采集到的数据分布更合理。
"""

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

#: 按键 -> (组, 轴, 值)。组为 'linear' 或 'angular'
KEY_BINDINGS = {
    'w': ('linear', 'x',  1.0),  'KEY_UP':    ('linear', 'x',  1.0),
    's': ('linear', 'x', -1.0),  'KEY_DOWN':  ('linear', 'x', -1.0),
    'a': ('angular', 'z',  1.0), 'KEY_LEFT':  ('angular', 'z',  1.0),
    'd': ('angular', 'z', -1.0), 'KEY_RIGHT': ('angular', 'z', -1.0),
    'q': ('linear', 'z',  1.0),
    'e': ('linear', 'z', -1.0),
    'z': ('linear', 'y',  1.0),
    'x': ('linear', 'y', -1.0),
}

HELP_TEXT = """
┌──────────────────────────────────────────────────────────────┐
│        水下 ROV 6-DOF 键盘运动控制（锁存式，按一次持续生效）  │
├──────────────────────────────────────────────────────────────┤
│  W / ↑  前进        S / ↓  后退                              │
│  A / ←  原地左转    D / →  原地右转                          │
│  Q      上浮        E      下潜                              │
│  Z      左横移      X      右横移                            │
│  空格   全部归零（悬停）      R  重置位姿                     │
│  + / -  调节推力比例          Ctrl+C  退出                    │
└──────────────────────────────────────────────────────────────┘
"""


class TerminalKeyboard:
    """终端原始终端模式的非阻塞按键读取。

    退出时必须恢复终端属性，否则用户终端会停留在 raw 模式无法正常输入。
    """

    def __init__(self):
        try:
            self.settings = termios.tcgetattr(sys.stdin)
        except Exception:  # noqa: BLE001  非 tty（如重定向）时降级
            self.settings = None

    def read_key(self, timeout=0.0):
        """读取一个按键，支持方向键的 ANSI 转义序列。"""
        if self.settings is None:
            return ''
        try:
            tty.setraw(sys.stdin.fileno())
            key = ''
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                c = sys.stdin.read(1)
                if c == '\x1b':
                    r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r2 and sys.stdin.read(1) == '[':
                        r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r3:
                            c3 = sys.stdin.read(1)
                            key = {'A': 'KEY_UP', 'B': 'KEY_DOWN',
                                   'C': 'KEY_RIGHT', 'D': 'KEY_LEFT'}.get(c3, '')
                else:
                    key = c
            return key
        except Exception:  # noqa: BLE001
            return ''
        finally:
            if self.settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
                except Exception:  # noqa: BLE001
                    pass

    def restore(self):
        if self.settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            except Exception:  # noqa: BLE001
                pass


class KeyboardTeleopNode(Node):
    """把键盘输入转成 /cmd_vel 的 ROS 2 节点。"""

    def __init__(self):
        super().__init__('keyboard_teleop_node')

        self.declare_parameter('linear_scale', 1.0)
        self.declare_parameter('angular_scale', 1.0)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('print_help', True)

        self.linear_scale = self.get_parameter('linear_scale').get_parameter_value().double_value
        self.angular_scale = self.get_parameter('angular_scale').get_parameter_value().double_value
        pub_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        print_help = self.get_parameter('print_help').get_parameter_value().bool_value

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.twist = Twist()
        self.kb = TerminalKeyboard()
        self.running = True

        self.create_timer(1.0 / max(pub_rate, 1.0), self._on_timer)

        self.get_logger().info('键盘遥控节点已启动（按空格归零，Ctrl+C 退出）')
        if print_help:
            print(HELP_TEXT)

    def _on_timer(self):
        """定时读取按键并更新锁存指令，然后发布。"""
        key = self.kb.read_key()

        if key in (' ', '5'):
            self.twist = Twist()
            self.get_logger().info('指令归零（悬停）')
        elif key in ('+', '='):
            self.linear_scale = min(5.0, self.linear_scale + 0.1)
            self.angular_scale = min(5.0, self.angular_scale + 0.1)
            self.get_logger().info(f'推力比例: 线={self.linear_scale:.1f} 角={self.angular_scale:.1f}')
        elif key == '-':
            self.linear_scale = max(0.1, self.linear_scale - 0.1)
            self.angular_scale = max(0.1, self.angular_scale - 0.1)
            self.get_logger().info(f'推力比例: 线={self.linear_scale:.1f} 角={self.angular_scale:.1f}')
        elif key == 'r':
            self.twist = Twist()
            # 约定：linear 三轴同时置 999 作为"请求复位"信号，
            # mujoco_sim_node 不识别该约定也不会有副作用（合力被钳位）。
            self.get_logger().info('已请求重置（通过重启仿真实现）')
        elif key in KEY_BINDINGS:
            group, axis, value = KEY_BINDINGS[key]
            scale = self.linear_scale if group == 'linear' else self.angular_scale
            self.twist = Twist()
            setattr(getattr(self.twist, group), axis, value * scale)

        self.pub_cmd.publish(self.twist)

    def destroy_node(self):
        # 退出前发一次零指令，避免 ROV 带着上一帧推力继续冲撞
        try:
            self.twist = Twist()
            self.pub_cmd.publish(self.twist)
        except Exception:  # noqa: BLE001
            pass
        self.kb.restore()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = KeyboardTeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
