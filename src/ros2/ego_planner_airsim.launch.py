"""
EGO-Planner launch 文件（配合 ProjectAirSim）
启动：ego_planner_node + traj_server

使用方法：
  ros2 launch src/ego_planner_airsim.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # FrontCamera 640x480 FOV=90 对应的内参
    # fx = fy = width / (2 * tan(fov/2)) = 640 / (2 * tan(45°)) = 320
    cx = 320.0
    cy = 240.0
    fx = 320.0
    fy = 320.0

    drone_id = '0'
    map_size_x = 40.0
    map_size_y = 40.0
    map_size_z = 5.0

    # EGO-Planner 规划节点
    ego_planner_node = Node(
        package='ego_planner',
        executable='ego_planner_node',
        name='drone_0_ego_planner_node',
        output='screen',
        remappings=[
            ('odom_world', 'drone_0_visual_slam/odom'),
            ('planning/bspline', 'drone_0_planning/bspline'),
            ('planning/data_display', 'drone_0_planning/data_display'),
            ('planning/broadcast_bspline_from_planner', '/broadcast_bspline'),
            ('planning/broadcast_bspline_to_planner', '/broadcast_bspline'),
            ('grid_map/odom', 'drone_0_visual_slam/odom'),
            ('grid_map/cloud', 'drone_0_pcl_render_node/cloud'),
            ('grid_map/pose', 'drone_0_pose'),
            ('grid_map/depth', 'drone_0_depth'),
        ],
        parameters=[
            {'fsm/flight_type': 1},
            {'fsm/thresh_replan_time': 1.0},
            {'fsm/thresh_no_replan_meter': 1.0},
            {'fsm/planning_horizon': 7.5},
            {'fsm/planning_horizen_time': 3.0},
            {'fsm/emergency_time': 1.0},
            {'fsm/realworld_experiment': False},
            {'fsm/fail_safe': True},

            {'fsm/waypoint_num': 1},
            {'fsm/waypoint0_x': 10.0},
            {'fsm/waypoint0_y': 0.0},
            {'fsm/waypoint0_z': 1.0},

            {'grid_map/resolution': 0.2},
            {'grid_map/map_size_x': map_size_x},
            {'grid_map/map_size_y': map_size_y},
            {'grid_map/map_size_z': map_size_z},
            {'grid_map/local_update_range_x': 5.5},
            {'grid_map/local_update_range_y': 5.5},
            {'grid_map/local_update_range_z': 4.5},
            {'grid_map/obstacles_inflation': 0.099},
            {'grid_map/local_map_margin': 10},
            {'grid_map/ground_height': -2.0},
            {'grid_map/cx': cx},
            {'grid_map/cy': cy},
            {'grid_map/fx': fx},
            {'grid_map/fy': fy},
            {'grid_map/use_depth_filter': True},
            {'grid_map/depth_filter_tolerance': 0.15},
            {'grid_map/depth_filter_maxdist': 4.5},
            {'grid_map/depth_filter_mindist': 0.5},
            {'grid_map/depth_filter_margin': 2},
            {'grid_map/k_depth_scaling_factor': 1000.0},
            {'grid_map/skip_pixel': 2},
            {'grid_map/p_hit': 0.65},
            {'grid_map/p_miss': 0.35},
            {'grid_map/p_min': 0.12},
            {'grid_map/p_max': 0.90},
            {'grid_map/p_occ': 0.80},
            {'grid_map/min_ray_length': 0.1},
            {'grid_map/max_ray_length': 4.5},
            {'grid_map/virtual_ceil_height': 4.5},
            {'grid_map/visualization_truncate_height': 3.5},
            {'grid_map/show_occ_time': False},
            {'grid_map/pose_type': 2},
            {'grid_map/frame_id': 'world'},

            {'manager/max_vel': 2.0},
            {'manager/max_acc': 6.0},
            {'manager/max_jerk': 4.0},
            {'manager/control_points_distance': 0.4},
            {'manager/feasibility_tolerance': 0.05},
            {'manager/planning_horizon': 7.5},
            {'manager/use_distinctive_trajs': True},
            {'manager/drone_id': 0},

            {'optimization/lambda_smooth': 1.0},
            {'optimization/lambda_collision': 0.5},
            {'optimization/lambda_feasibility': 0.1},
            {'optimization/lambda_fitness': 1.0},
            {'optimization/dist0': 0.5},
            {'optimization/swarm_clearance': 0.5},
            {'optimization/max_vel': 2.0},
            {'optimization/max_acc': 6.0},

            {'bspline/limit_vel': 2.0},
            {'bspline/limit_acc': 6.0},
            {'bspline/limit_ratio': 1.1},

            {'prediction/obj_num': 0},
            {'prediction/lambda': 1.0},
            {'prediction/predict_rate': 1.0},
        ]
    )

    # 轨迹服务节点
    traj_server_node = Node(
        package='ego_planner',
        executable='traj_server',
        name='drone_0_traj_server',
        output='screen',
        remappings=[
            ('position_cmd', 'drone_0_planning/pos_cmd'),
            ('planning/bspline', 'drone_0_planning/bspline'),
        ],
        parameters=[
            {'traj_server/time_forward': 1.0}
        ]
    )

    ld = LaunchDescription()
    ld.add_action(ego_planner_node)
    ld.add_action(traj_server_node)
    return ld
