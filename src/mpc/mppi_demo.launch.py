"""
MPPI 演示 launch 文件。
同时启动 MPPI 节点和预配好的 rviz2。

使用方法（在仓库根目录）：
  ros2 launch src/mpc/mppi_demo.launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    here = os.path.dirname(os.path.abspath(__file__))
    demo_py = os.path.join(here, 'mppi_demo.py')
    rviz_cfg = os.path.join(here, 'mppi_demo.rviz')

    # 用 ExecuteProcess 直接跑脚本，无需 colcon 构建成包
    mppi = ExecuteProcess(
        cmd=['python3', demo_py],
        output='screen',
    )
    rviz = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_cfg],
        output='screen',
    )

    return LaunchDescription([mppi, rviz])
