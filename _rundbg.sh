#!/bin/bash
{
source /opt/ros/humble/setup.bash
cd ~/ros2_ws/src/water/mjc_rov_current_teleop
python3 /tmp/_debug1.py 2>&1
} > /tmp/debug1.log 2>&1
