#!/bin/bash
{
set -e
source /opt/ros/humble/setup.bash
rm -rf ~/ros2_ws/src/water/mjc_rov_sonar_cnn
tar -xzf /tmp/_pkg2.tar.gz -C ~/ros2_ws/src/water
cd ~/ros2_ws
colcon build --symlink-install --packages-select mjc_rov_sonar_cnn
source install/setup.bash
echo "=== executables ==="
ros2 pkg executables mjc_rov_sonar_cnn
echo "=== offline: train+test ==="
cd ~/ros2_ws/src/water/mjc_rov_sonar_cnn
python3 main.py 2>&1 | tail -20
echo "=== ALL DONE ==="
} > /tmp/run_p2.log 2>&1
