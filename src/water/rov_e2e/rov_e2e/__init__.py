"""rov_e2e — 静水环境下单目视觉水下 ROV 端到端神经网络控制。

模块划分：
    collect_data  数据集采集（键盘操纵 + 图像/标签同步保存）
    train         训练脚本，产出 model.pth（含网络结构 ROVSteeringNet）
    test_infer    离线推理测试，验证 model.pth 可用
    infer_node    ROS 2 在线推理节点（相机话题 → /cmd_vel）
"""
