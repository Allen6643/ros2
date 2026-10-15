"""rov_vision_e2e_nav — 水下 ROV 单目视觉端到端控制与声呐建图导航。

模块划分：
    vision_sim_node      MuJoCo 仿真内核（前视相机 + 8 向声呐测距扇）
    networks             两个神经网络的定义与共享预处理
    collect_data         数据集采集（视觉伺服专家 + 声呐几何标签）
    train                训练视觉端到端网络与声呐感知网络
    test_infer           离线测试权重（不启动 ROS）
    e2e_control_node     单目视觉端到端控制节点（任务④）
    sonar_slam_node      声呐占据栅格建图 + 避障导航节点（任务③）
"""
