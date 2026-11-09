"""三机任务回放：三套 plan_viz_node + 一套静态体素地图 + RViz2。"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_launch_description():
    here = os.path.dirname(os.path.abspath(__file__))
    run_dir = LaunchConfiguration('run_dir')
    processes = []
    colors = ((0.0, 0.67, 1.0), (1.0, 0.72, 0.05), (1.0, 0.12, 0.72))
    for index, color in enumerate(colors, 1):
        name = f'Drone{index}'
        processes.append(ExecuteProcess(
            cmd=[
                '/usr/bin/python3', os.path.join(here, 'plan_viz_node.py'),
                '--dump-dir', PathJoinSubstitution([run_dir, name, 'viz_dump']),
                '--prefix-dump', PathJoinSubstitution([run_dir, name, 'coordination_dump']),
                '--node-name', f'plan_viz_drone{index}',
                '--topic-prefix', f'/drone{index}',
                '--drone-color', *(str(value) for value in color),
            ],
            output='screen'))

    return LaunchDescription([
        DeclareLaunchArgument('run_dir'),
        DeclareLaunchArgument('profile', default_value='airport'),
        *processes,
        ExecuteProcess(
            cmd=[
                '/usr/bin/python3', os.path.join(here, 'map_view.py'),
                '--profile', LaunchConfiguration('profile'),
            ],
            output='screen'),
        ExecuteProcess(
            cmd=['rviz2', '-d', str(PROJECT_ROOT / 'configs/rviz/multi.rviz')],
            output='screen'),
    ])
