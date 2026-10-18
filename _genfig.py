#!/usr/bin/env python3
"""生成 PR1 演示图：相机渲染帧 + 轨迹图"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/ros2_ws/src/water/mjc_rov_current_teleop"))
os.environ["MUJOCO_GL"]="osmesa"; os.environ["PYOPENGL_PLATFORM"]="osmesa"
import numpy as np
import mujoco
from mjc_rov_current_teleop import common
from mjc_rov_current_teleop.cruise_net import _make_sim, _read_state, _expert_cmd, EXPERT_WAYPOINTS

params = dict(buoyancy_z=10.0, damping_lin=2.5, damping_ang=1.5,
              stab_roll=4.0, stab_pitch=4.0,
              current_amp_x=0.15, current_amp_y=0.10,
              current_w_x=0.30, current_w_y=0.23,
              max_force_x=5.0, max_force_y=5.0, max_torque_yaw=1.0)
model, data, dyn = _make_sim(params)
renderer = mujoco.Renderer(model, height=120, width=160)

# 跑一段轨迹记录位置
traj = []
wp_idx = 0
for i in range(6000):
    t = i*0.002
    state = _read_state(model, data)
    wp = EXPERT_WAYPOINTS[wp_idx]
    if np.hypot(wp[0]-state[0], wp[1]-state[1]) < 0.4:
        wp_idx = (wp_idx+1)%4
        wp = EXPERT_WAYPOINTS[wp_idx]
    cmd = _expert_cmd(state, wp)
    twist = np.array([cmd[0],cmd[1],0,0,0,0])
    dyn.apply(data, t, model.body('rov').id, twist)
    mujoco.mj_step(model, data)
    if i % 50 == 0:
        traj.append([state[0], state[1]])

# 渲染相机帧
renderer.update_scene(data, camera="forward_cam")
img = renderer.render()
import cv2
cv2.imwrite("/tmp/pr1_camera.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("camera saved")

# 画轨迹图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
traj = np.array(traj)
plt.figure(figsize=(6,4))
plt.plot(traj[:,0], traj[:,1], 'b-', label='ROV trajectory')
plt.scatter(EXPERT_WAYPOINTS[:,0], EXPERT_WAYPOINTS[:,1], c='r', s=100, marker='*', label='Waypoints')
plt.scatter([0],[0], c='g', s=100, marker='o', label='Start')
plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.title('ROV waypoint navigation under ocean current')
plt.legend(); plt.axis('equal'); plt.grid(True)
plt.savefig("/tmp/pr1_trajectory.png", dpi=100, bbox_inches='tight')
print("trajectory saved")
