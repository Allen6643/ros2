#!/bin/bash
{
set -e
source /opt/ros/humble/setup.bash
rm -rf ~/ros2_ws/src/water/mjc_rov_ppo_tracking
tar -xzf /tmp/_pkg3.tar.gz -C ~/ros2_ws/src/water
cd ~/ros2_ws
colcon build --symlink-install --packages-select mjc_rov_ppo_tracking
source install/setup.bash
echo "=== executables ==="
ros2 pkg executables mjc_rov_ppo_tracking
echo "=== offline: train+test ==="
cd ~/ros2_ws/src/water/mjc_rov_ppo_tracking
python3 main.py 2>&1 | tail -30
echo "=== ALL DONE ==="
} > /tmp/run_p3.log 2>&1
