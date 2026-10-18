#!/bin/bash
{
source /opt/ros/humble/setup.bash
cd ~/ros2_ws/src/water/mjc_rov_current_teleop
python3 /tmp/_genfig.py 2>&1
echo "=== test output ==="
python3 main.py test 2>&1 | tail -20
} > /tmp/genfig.log 2>&1
