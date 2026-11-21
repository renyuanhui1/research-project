"""Launch airport 3D mesh/occupancy map + 6-UAV 3D trajectory replay in RViz2."""

from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in HERE.parents if (p / "environment.yml").is_file())


def generate_launch_description() -> LaunchDescription:
    default_catalog = str(PROJECT_ROOT / "data/templates/airport_targets_20261116/catalog.json")
    default_rviz = str(PROJECT_ROOT / "configs/rviz/multi_uav_coordination.rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument("trajectories", description="Path to runs/coordination/<id>/trajectories.json"),
            DeclareLaunchArgument("speed", default_value="4.0"),
            DeclareLaunchArgument("static", default_value="false"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("target_catalog", default_value=default_catalog),
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3",
                    str(HERE / "map_view.py"),
                    "--profile",
                    "airport",
                    "--target-catalog",
                    LaunchConfiguration("target_catalog"),
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3",
                    str(HERE / "multi_uav_traj_node.py"),
                    "--trajectories",
                    LaunchConfiguration("trajectories"),
                    "--speed",
                    LaunchConfiguration("speed"),
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=["rviz2", "-d", default_rviz],
                output="screen",
                condition=IfCondition(LaunchConfiguration("start_rviz")),
            ),
        ]
    )
