"""启动 ProjectAirSim 体素地图单次播放节点，可选同时启动 RViz2。

用法：
  rviz2 -d /home/hui/my/src/mpc/voxel_map_viz.rviz
  ros2 launch src/mpc/voxel_map_viz.launch.py
  ros2 launch src/mpc/voxel_map_viz.launch.py start_rviz:=true
  ros2 launch src/mpc/voxel_map_viz.launch.py preview:=true start_rviz:=true
  ros2 launch src/mpc/voxel_map_viz.launch.py preview:=true frustum_image:=true start_rviz:=true
"""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_launch_description():
    here = os.path.dirname(os.path.abspath(__file__))
    map_file = LaunchConfiguration('map_file')
    min_altitude = LaunchConfiguration('min_altitude')
    ground_max_altitude = LaunchConfiguration('ground_max_altitude')
    start_rviz = LaunchConfiguration('start_rviz')
    preview = LaunchConfiguration('preview')
    frustum_image = LaunchConfiguration('frustum_image')
    default_map = PROJECT_ROOT / 'airbase/outputs/maps/voxel_probe/global/occupancy_packed.npz'
    trajectory_dirs = (
        PROJECT_ROOT / 'airbase/outputs/runs/servo/机头_0805_103801',
        PROJECT_ROOT / 'airbase/outputs/runs/servo/红星_0722_194121',
    )
    detail_root = PROJECT_ROOT / 'airbase/outputs/maps/voxel_probe/fine_targets'
    detail_args = []
    for actor_name in ('actor_22', 'actor_18', 'actor_16', 'actor_20'):
        detail_args.extend([
            '--detail-map',
            str(detail_root / actor_name / 'occupancy_packed.npz'),
        ])

    return LaunchDescription([
        DeclareLaunchArgument('map_file', default_value=str(default_map)),
        DeclareLaunchArgument('min_altitude', default_value='0.0'),
        DeclareLaunchArgument('ground_max_altitude', default_value='0.5'),
        DeclareLaunchArgument('start_rviz', default_value='false'),
        DeclareLaunchArgument('preview', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('frustum_image', default_value='false', choices=['true', 'false']),
        ExecuteProcess(
            cmd=[
                '/usr/bin/python3', os.path.join(here, 'voxel_map_viz_node.py'),
                '--map-file', map_file,
                '--min-altitude', min_altitude,
                '--ground-max-altitude', ground_max_altitude,
                '--objects-only',
                '--exit-after-playback',
                '--preview', preview,
                '--frustum-image', frustum_image,
                *detail_args,
                *[
                    item
                    for trajectory_dir in trajectory_dirs
                    for item in ('--trajectory-dir', str(trajectory_dir))
                ],
            ],
            output='screen',
        ),
        ExecuteProcess(
            cmd=['rviz2', '-d', PythonExpression([
                repr(os.path.join(here, 'voxel_map_preview.rviz')),
                " if '", preview, "' == 'true' else ",
                repr(os.path.join(here, 'voxel_map_viz.rviz')),
            ])],
            output='screen',
            condition=IfCondition(start_rviz),
        ),
    ])
