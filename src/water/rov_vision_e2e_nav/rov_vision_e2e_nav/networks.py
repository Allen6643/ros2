"""
神经网络定义与共享预处理
==========================
本模块集中定义本功能包用到的两个网络及其输入预处理，供 `train.py`、
`test_infer.py`、`e2e_control_node.py`、`sonar_slam_node.py` 共同导入。

**为什么集中在一处**：视觉端到端控制最常见的隐性缺陷是"训练与部署的
图像预处理不一致"（缩放插值方式、通道顺序、归一化统计量不同），
会导致离线指标很好但上线行为异常。让训练与推理调用同一个函数，
即可从结构上消除该风险。

两个网络
--------
1. `VisualE2ENet` —— 单目视觉端到端控制网络
   输入：3 通道 RGB 图像（120×160）
   输出：2 个连续控制量 [前进推力, 转向]，tanh 限幅到 [-1,1]
   这是作业任务(4)要求的"输入图像、输出控制指令"的端到端模型。

2. `ObstacleMapNet` —— 声呐测距扇到障碍物方位分布的映射网络
   输入：8 维声呐测距（7 向水平扇 + 1 向下）
   输出：3 类（左侧开阔 / 正前开阔 / 右侧开阔），供建图与避障决策使用
"""

import numpy as np

# ---------------------------------------------------------------------------
# matplotlib 中文字体配置
# ---------------------------------------------------------------------------
#: 按优先级排列的候选中文字体文件
_CJK_FONT_CANDIDATES = (
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
)


def setup_chinese_font():
    """为 matplotlib 配置可用的中文字体，返回字体属性对象或 None。

    虚拟机自带 matplotlib 3.1.2，而 ``fontManager.addfont()`` 是 3.2 才引入的，
    因此这里用"按文件路径构造 FontProperties"的方式，在新旧版本上都能工作。
    找不到中文字体时返回 None，调用方应改用英文标签，避免出现方框。
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
            addfont = getattr(font_manager.fontManager, 'addfont', None)
            if callable(addfont):
                try:
                    addfont(path)
                    matplotlib.rcParams['font.sans-serif'] = (
                        [prop.get_name()] + list(matplotlib.rcParams.get('font.sans-serif', [])))
                except Exception:  # noqa: BLE001
                    pass
            matplotlib.rcParams['axes.unicode_minus'] = False
            return prop
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# 图像与声呐特征的归一化参数（训练与推理必须一致）
# ---------------------------------------------------------------------------
#: 图像像素归一化统计量（通用 RGB 统计量，稳定且与 PyTorch 生态一致）
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: 声呐测距特征标准化参数（8 维：7 向水平 + 1 向下）
SONAR_MEAN = np.array([4.0, 3.5, 3.0, 3.5, 3.5, 3.0, 3.5, 1.0], dtype=np.float32)
SONAR_STD = np.array([5.0, 3.0, 2.0, 3.0, 3.0, 2.0, 3.0, 1.0], dtype=np.float32)

#: 输入图像尺寸（与仿真节点 camera_width/height 一致）
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120


def preprocess_image(img, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """把 PIL 图像转为网络输入张量所需的 (3, H, W) float32 数组。

    训练与推理共用本函数，保证缩放方式、通道顺序、归一化完全一致。
    """
    from PIL import Image as PILImage
    if not isinstance(img, PILImage.Image):
        img = PILImage.fromarray(np.asarray(img, dtype=np.uint8))
    img = img.convert('RGB').resize((width, height))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))   # HWC -> CHW


def normalize_sonar_features(feat):
    """把 8 维声呐特征标准化。训练与推理共用。"""
    feat = np.asarray(feat, dtype=np.float32).reshape(-1)
    return (feat - SONAR_MEAN) / SONAR_STD


# ---------------------------------------------------------------------------
# 网络定义（延迟导入 torch）
# ---------------------------------------------------------------------------
def build_visual_e2e_net(hidden=64):
    """构造单目视觉端到端网络：RGB 图像 -> [前进推力, 转向]。

    结构（参数量约 2.6 万，CPU 可快速训练）：

        Conv(3→16, 3×3, s2) + BN + ReLU
        Conv(16→32, 3×3, s2) + BN + ReLU
        Conv(32→64, 3×3, s2) + BN + ReLU
        AdaptiveAvgPool2d(1) → FC(64→32) + ReLU → FC(32→2) → Tanh

    末层 tanh 把输出限制在 [-1,1]，与归一化控制量量纲一致，
    也天然完成控制量限幅，避免推理时输出发散。
    """
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        block(3, 16),
        block(16, 32),
        block(32, hidden),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(hidden, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, 2),
        nn.Tanh(),
    )


def build_obstacle_map_net(hidden=(64, 32)):
    """构造声呐障碍物方位判别网络：8 维测距 -> 3 类开阔方向。"""
    import torch.nn as nn

    layers = []
    in_dim = 8
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
        in_dim = h
    layers.append(nn.Linear(in_dim, 3))
    return nn.Sequential(*layers)


#: 端到端控制输出维度名
E2E_OUTPUT_DIMS = ('前进推力', '转向')

#: 声呐开阔方向类别名
OPEN_DIRECTION_CLASSES = ('左侧开阔', '正前开阔', '右侧开阔')
