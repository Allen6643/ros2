#!/bin/bash
{
set -e
source /opt/ros/humble/setup.bash
rm -rf ~/ros2_ws/src/water/mjc_rov_docking_nav
tar -xzf /tmp/_pkg4.tar.gz -C ~/ros2_ws/src/water
cd ~/ros2_ws
colcon build --symlink-install --packages-select mjc_rov_docking_nav
source install/setup.bash
echo "=== executables ==="
ros2 pkg executables mjc_rov_docking_nav
echo "=== offline: train+test ==="
cd ~/ros2_ws/src/water/mjc_rov_docking_nav
python3 main.py 2>&1 | tail -20
echo "=== ALL DONE ==="
} > /tmp/run_p4.log 2>&1
