"""闭环规划可视化 launch：plan_viz_node + rviz2。

用法：
  ros2 launch src/mpc/plan_viz.launch.py                       # 默认 outputs/runs/mppi/run01
  ros2 launch src/mpc/plan_viz.launch.py dump_dir:=/path/to/dir
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_launch_description():
    here = os.path.dirname(os.path.abspath(__file__))
    dump_dir = LaunchConfiguration('dump_dir')

    return LaunchDescription([
        DeclareLaunchArgument('dump_dir', default_value=str(PROJECT_ROOT / 'outputs/runs/mppi/run01')),
        ExecuteProcess(
            # 钉死系统 python：rclpy 的 C 扩展是 3.10 编的，conda 的 python 会 import 失败
            cmd=['/usr/bin/python3', os.path.join(here, 'plan_viz_node.py'),
                 '--dump-dir', dump_dir],
            output='screen'),
        ExecuteProcess(
            cmd=['rviz2', '-d', os.path.join(here, 'plan_viz.rviz')],
            output='screen'),
    ])
