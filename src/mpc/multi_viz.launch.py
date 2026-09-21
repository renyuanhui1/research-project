"""三机任务回放：三套 plan_viz_node + 一套静态体素地图 + RViz2。

用法：
  ros2 launch /home/hui/my/src/mpc/multi_viz.launch.py \
    run_dir:=/home/hui/my/airbase/outputs/runs/multi_servo/three_aircraft_MMDD_HHMMSS
"""

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
    map_file = LaunchConfiguration('map_file')
    default_map = PROJECT_ROOT / 'airbase/outputs/maps/voxel_probe/global/occupancy_packed.npz'
    detail_root = PROJECT_ROOT / 'airbase/outputs/maps/voxel_probe/fine_targets'
    detail_args = []
    for actor_name in ('actor_22', 'actor_18', 'actor_16', 'actor_20'):
        detail_args.extend([
            '--detail-map', str(detail_root / actor_name / 'occupancy_packed.npz')])

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
        DeclareLaunchArgument('map_file', default_value=str(default_map)),
        *processes,
        ExecuteProcess(
            cmd=[
                '/usr/bin/python3', os.path.join(here, 'voxel_map_viz_node.py'),
                '--map-file', map_file, '--objects-only', *detail_args,
            ],
            output='screen'),
        ExecuteProcess(
            cmd=['rviz2', '-d', os.path.join(here, 'multi_viz.rviz')],
            output='screen'),
    ])
