"""
MPPI 空中跟踪演示 launch 文件。
同时启动无人机跟踪节点和预配好的 rviz2。

使用方法（在仓库根目录）：
  ros2 launch src/mpc/aerial_pursuit.launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    here = os.path.dirname(os.path.abspath(__file__))
    demo_py = os.path.join(here, 'aerial_pursuit_demo.py')
    rviz_cfg = os.path.join(here, 'aerial_pursuit.rviz')

    demo = ExecuteProcess(cmd=['python3', demo_py], output='screen')
    rviz = ExecuteProcess(cmd=['rviz2', '-d', rviz_cfg], output='screen')

    return LaunchDescription([demo, rviz])
