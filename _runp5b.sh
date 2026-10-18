#!/bin/bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select mjc_rov_rl_compare > /tmp/p5build.log 2>&1
echo "=== build result ==="
tail -10 /tmp/p5build.log
source install/setup.bash
echo "=== executables ==="
ros2 pkg executables mjc_rov_rl_compare
echo "=== compare ==="
cd src/water/mjc_rov_rl_compare
python3 main.py 2>&1 | tail -10
echo "=== DONE ==="
