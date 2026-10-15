"""
神经网络定义与共享预处理
==========================
本模块集中定义本功能包用到的两个网络及其输入预处理，供 `train.py`、
`test_infer.py`、`sensor_perception_node.py`、`traj_control_node.py` 共同导入。

**为什么集中在一处**：端到端/学习型控制最常见的隐性缺陷是"训练与部署的
数据预处理不一致"（归一化统计量不同、特征顺序不同、缩放方式不同），
会导致离线误差很低但上线行为异常。让训练与推理调用同一个函数即可从结构上
消除该风险。

两个网络
--------
1. `SensorPerceptionNet` —— 传感器感知网络
   输入：4 维测距特征（前/左/右/下距离）+ 3 维速度 = 7 维
   输出：3 类障碍物方位（前方 / 左方 / 右方），用于感知层判断
   
2. `TrajectoryControlNet` —— 轨迹跟踪控制网络
   输入：8 维状态误差（位置误差 xyz + 速度 + 偏航误差 + 前向测距）
   输出：4 维控制量（前进推力、横移、升沉、偏航力矩），tanh 限幅到 [-1,1]
"""

import numpy as np

# ---------------------------------------------------------------------------
# matplotlib 中文字体配置
# ---------------------------------------------------------------------------
#: 按优先级排列的候选中文字体文件（Linux 教学虚拟机常见路径）
_CJK_FONT_CANDIDATES = (
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
)


def setup_chinese_font():
    """为 matplotlib 配置可用的中文字体，返回字体属性对象或 None。

    为什么需要这个函数：matplotlib 默认字体不含中文字形，直接画中文会显示成
    方框（□□□）。这里按优先级探测系统中文字体文件，找到就返回可用的
    ``FontProperties``；找不到返回 None，调用方应改用英文标签 ——
    从而在任何机器上都不会出现方框。

    **兼容性说明**：本作业的 Ubuntu 20.04 虚拟机自带 matplotlib 3.1.2，
    而 ``fontManager.addfont()`` 是 3.2 才引入的 API。因此这里采用
    "按文件路径构造 FontProperties" 的方式，在 3.1.x 与新版本上都能工作；
    绘图时把返回对象传给 ``fontproperties=`` 参数即可。
    """
    import os
    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception:  # noqa: BLE001
        return None

    for path in _CJK_FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            prop = font_manager.FontProperties(fname=path)
            # 新版本 matplotlib 额外把字体注册进全局字体表，
            # 这样连 rcParams 方式也能生效；旧版本没有 addfont，忽略即可。
            addfont = getattr(font_manager.fontManager, 'addfont', None)
            if callable(addfont):
                try:
                    addfont(path)
                    matplotlib.rcParams['font.sans-serif'] = (
                        [prop.get_name()] + list(matplotlib.rcParams.get('font.sans-serif', [])))
                except Exception:  # noqa: BLE001
                    pass
            matplotlib.rcParams['axes.unicode_minus'] = False   # 负号正常显示
            return prop
        except Exception:  # noqa: BLE001
            continue
    return None



# ---------------------------------------------------------------------------
# 特征与标签的归一化参数
# 训练与推理必须使用同一组数值，因此在这里统一定义
# ---------------------------------------------------------------------------
SENSOR_FEATURE_MEAN = np.array([3.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
SENSOR_FEATURE_STD = np.array([4.0, 2.0, 2.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float32)

TRAJ_FEATURE_MEAN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0], dtype=np.float32)
TRAJ_FEATURE_STD = np.array([2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.6, 4.0], dtype=np.float32)


def normalize_sensor_features(feat):
    """把 7 维传感器特征标准化。训练与推理共用。"""
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    return (feat - SENSOR_FEATURE_MEAN) / SENSOR_FEATURE_STD


def normalize_traj_features(feat):
    """把 8 维轨迹状态特征标准化。训练与推理共用。"""
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    return (feat - TRAJ_FEATURE_MEAN) / TRAJ_FEATURE_STD


# ---------------------------------------------------------------------------
# 网络定义（延迟导入 torch，使不依赖 torch 的脚本也能 import 本模块）
# ---------------------------------------------------------------------------
def build_sensor_net(hidden=(64, 32)):
    """构造传感器感知网络：MLP，输出 3 类障碍物方位 logits。"""
    import torch.nn as nn

    layers = []
    in_dim = 7
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
        in_dim = h
    layers.append(nn.Linear(in_dim, 3))
    return nn.Sequential(*layers)


def build_traj_net(hidden=(128, 128, 64)):
    """构造轨迹跟踪控制网络：MLP，输出 4 维控制量（tanh 限幅）。"""
    import torch.nn as nn

    layers = []
    in_dim = 8
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
        in_dim = h
    layers += [nn.Linear(in_dim, 4), nn.Tanh()]
    return nn.Sequential(*layers)


#: 障碍物方位类别名（与数据集标签索引对应）
OBSTACLE_CLASSES = ('前方障碍', '左方障碍', '右方障碍')

#: 控制量维度名（与 TrajectoryControlNet 输出对应）
CONTROL_DIMS = ('前进推力', '横移推力', '升沉推力', '偏航力矩')
