"""rov_mujoco_sim_teleop — 水下 ROV MuJoCo 仿真、键盘遥控与神经网络感知控制。

模块划分：
    mujoco_sim_node         MuJoCo 物理仿真内核（发布传感器、接收 /cmd_vel）
    keyboard_teleop_node    键盘遥控（任务① 手动运动控制）
    networks                两个神经网络的定义与共享预处理
    collect_data            数据集采集（确定性专家律打标）
    train                   训练两个网络，产出 model_sensor.pth / model_traj.pth
    test_infer              离线测试权重（不启动 ROS）
    sensor_perception_node  神经网络传感器感知（任务② 感知）
    traj_control_node       神经网络轨迹跟踪控制（任务② 给定轨迹控制）
"""
