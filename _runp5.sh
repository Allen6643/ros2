#!/bin/bash
{
set -e
source /opt/ros/humble/setup.bash
rm -rf ~/ros2_ws/src/water/mjc_rov_rl_compare
tar -xzf /tmp/_pkg5.tar.gz -C ~/ros2_ws/src/water
cd ~/ros2_ws
colcon build --symlink-install --packages-select mjc_rov_rl_compare
source install/setup.bash
echo "=== executables ==="
ros2 pkg executables mjc_rov_rl_compare
echo "=== offline: compare ==="
cd ~/ros2_ws/src/water/mjc_rov_rl_compare
python3 main.py 2>&1 | tail -10
echo "=== ALL DONE ==="
} > /tmp/run_p5.log 2>&1
