#!/usr/bin/env python3
"""把 ProjectAirSim 离线体素地图发布到 RViz2。

默认读取 airbase 的 global 粗体素地面，并叠加五张目标局部精细图：
  /voxel_map/markers     MarkerArray：10 m 地面薄片、1 m 目标体素和地图边界；
  /voxel_map/trajectory  MarkerArray：按记录时间逐步播放的历史轨迹。

坐标转换：ProjectAirSim NED (N,E,D) → RViz ENU (E,N,-D)，fixed frame=world。
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


FRAME = 'world'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = PROJECT_ROOT / 'airbase/outputs/maps/voxel_probe/global/occupancy_packed.npz'

# 已由 probe_target_object.py 标定的五个放置对象：
# (center_north, center_east, OOB_size_x, OOB_size_y, yaw_deg, top_altitude)。
# 仅用于 RViz 真值辅助过滤，不代表体素地图具有语义识别能力。
PLACED_OBJECTS = (
    (-62.71, -16.68, 23.73, 24.73, -40.0, 6.18),
    (-167.65, -73.00, 18.84, 28.18, -40.0, 9.88),
    (-251.99, -195.52, 18.79, 28.50, -40.0, 9.00),
    (127.48, -316.15, 23.70, 28.35, -40.0, 7.21),
    (60.78, 23.31, 10.63, 22.47, -50.0, 5.94),
)


def height_color(t):
    """低处蓝、高处红的简易 jet 色表。"""
    t = min(max(float(t), 0.0), 1.0)
    r = min(max(1.5 - abs(4 * t - 3), 0.0), 1.0)
    g = min(max(1.5 - abs(4 * t - 2), 0.0), 1.0)
    b = min(max(1.5 - abs(4 * t - 1), 0.0), 1.0)
    return ColorRGBA(r=r, g=g, b=b, a=0.92)


def load_voxel_map(path, min_altitude):
    """解压体素，并返回 ENU 占用中心、分辨率和地图边界。"""
    data = np.load(path, allow_pickle=False)
    required = {'packed', 'shape', 'center_ned_m', 'resolution_m'}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f'体素文件缺少字段: {sorted(missing)}')

    shape = tuple(int(v) for v in data['shape'])
    if len(shape) != 3 or any(v <= 0 for v in shape):
        raise ValueError(f'非法体素形状: {shape}')
    ny, nz, nx = shape                       # (east, altitude, north)
    count = int(np.prod(shape))
    bitorder = str(data['bitorder']) if 'bitorder' in data.files else 'little'
    grid = np.unpackbits(data['packed'], bitorder=bitorder)[:count].reshape(shape).astype(bool)

    center = np.asarray(data['center_ned_m'], dtype=np.float64)
    resolution = float(data['resolution_m'])
    north = center[0] + (np.arange(nx) - nx // 2) * resolution
    east = center[1] + (np.arange(ny) - ny // 2) * resolution
    altitude = -center[2] + (np.arange(nz) - nz // 2) * resolution

    selected = np.argwhere(grid & (altitude[None, :, None] >= min_altitude))
    if selected.size == 0:
        raise ValueError(f'{min_altitude:g} m 以上没有占用体素')
    enu = np.column_stack((east[selected[:, 0]], north[selected[:, 2]], altitude[selected[:, 1]]))

    bounds = {
        'east_min': float(east[0] - resolution / 2),
        'east_max': float(east[-1] + resolution / 2),
        'north_min': float(north[0] - resolution / 2),
        'north_max': float(north[-1] + resolution / 2),
    }
    return enu, resolution, bounds


def filter_placed_objects(points_enu, resolution):
    """只保留五个已标定对象 OOB 周围的地面以上体素。"""
    keep = np.zeros(len(points_enu), dtype=bool)
    padding = resolution * 0.75
    for north, east, size_x, size_y, yaw_deg, top_altitude in PLACED_OBJECTS:
        delta_north = points_enu[:, 1] - north
        delta_east = points_enu[:, 0] - east
        yaw = math.radians(yaw_deg)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        local_x = cosine * delta_north + sine * delta_east
        local_y = -sine * delta_north + cosine * delta_east
        keep |= (
            (np.abs(local_x) <= size_x / 2 + padding)
            & (np.abs(local_y) <= size_y / 2 + padding)
            & (points_enu[:, 2] <= top_altitude + resolution / 2)
        )
    return points_enu[keep]


def load_timed_trajectory(directory):
    """读取 NED 轨迹和从零开始的记录时间。"""
    root = Path(directory).expanduser().resolve()
    signals_path = root / 'signals.csv'
    if signals_path.exists():
        lines = [line for line in signals_path.read_text(encoding='utf-8-sig').splitlines()
                 if line and not line.startswith('#')]
        rows = list(csv.DictReader(lines))
        if not rows:
            raise ValueError(f'轨迹 CSV 没有数据行: {signals_path}')
        poses = np.asarray([
            [float(row['n']), float(row['e']), float(row['d'])]
            for row in rows
        ], dtype=np.float64)
        times = np.asarray([float(row['t']) for row in rows], dtype=np.float64)
        times -= times[0]
    else:
        step_dir = root / 'viz_dump' if (root / 'viz_dump').is_dir() else root
        files = sorted(step_dir.glob('step_*.npz'))
        if not files:
            raise FileNotFoundError(f'轨迹目录没有 signals.csv 或 step_*.npz: {root}')
        poses_list, intervals = [], []
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                poses_list.append(data['pose'][:3])
                intervals.append(float(data['dt']))
        poses = np.stack(poses_list)
        times = np.zeros(len(files), dtype=np.float64)
        if len(files) > 1:
            times[1:] = np.cumsum(intervals[:-1])
    if not np.isfinite(poses).all() or not np.isfinite(times).all():
        raise ValueError(f'轨迹或时间包含 NaN/Inf: {directory}')
    if np.any(np.diff(times) < 0):
        raise ValueError(f'轨迹时间不是单调递增: {directory}')
    return poses, times


def load_trajectory(directory):
    """兼容旧调用，只返回轨迹位姿。"""
    return load_timed_trajectory(directory)[0]


class VoxelMapVizNode(Node):
    def __init__(self, args):
        super().__init__('voxel_map_viz')
        self.args = args
        self.points, self.resolution, self.bounds = load_voxel_map(
            Path(args.map_file).expanduser().resolve(), args.min_altitude)
        ground_mask = self.points[:, 2] <= args.ground_max_altitude
        self.ground_points = self.points[ground_mask]
        self.obstacle_points = self.points[~ground_mask]
        self.object_resolution = self.resolution

        if args.detail_map:
            detail_sets = []
            detail_resolution = None
            for detail_path in args.detail_map:
                detail_points, resolution, _ = load_voxel_map(
                    Path(detail_path).expanduser().resolve(), args.detail_min_altitude)
                if detail_resolution is None:
                    detail_resolution = resolution
                elif not math.isclose(resolution, detail_resolution):
                    raise ValueError('所有目标精细图必须使用相同分辨率')
                detail_sets.append(detail_points)
            self.obstacle_points = np.concatenate(detail_sets, axis=0)
            self.object_resolution = detail_resolution

        if args.objects_only:
            self.obstacle_points = filter_placed_objects(
                self.obstacle_points, self.object_resolution)
        timed_trajectories = [load_timed_trajectory(path) for path in args.trajectory_dir]
        self.trajectories_ned = [trajectory for trajectory, _ in timed_trajectories]
        self.trajectory_times = [times for _, times in timed_trajectories]
        self.playback_duration = max((times[-1] for times in self.trajectory_times), default=0.0)
        self.playback_started = None
        self.playback_complete = False

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.map_publisher = self.create_publisher(MarkerArray, 'voxel_map/markers', qos)
        self.trajectory_publisher = self.create_publisher(
            MarkerArray, 'voxel_map/trajectory', qos)
        self.publish_map()
        self.playback_timer = self.create_timer(1.0 / args.playback_hz, self.publish_playback)
        self.get_logger().info(
            f'已载入 {len(self.points)} 个占用体素：地面 {len(self.ground_points)}，'
            f'放置对象 {len(self.obstacle_points)}；'
            f'地面分辨率 {self.resolution:g} m，对象分辨率 {self.object_resolution:g} m；'
            f'轨迹 {len(self.trajectories_ned)} 条/'
            f'{sum(len(trajectory) for trajectory in self.trajectories_ned)} 点；'
            f'等待 RViz 后按记录时间 {args.playback_speed:g} 倍速'
            f'{"循环" if args.loop else "单次"}播放'
        )

    @staticmethod
    def base_marker(stamp, namespace, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = FRAME
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def obstacle_marker(self, stamp):
        marker = self.base_marker(stamp, 'occupied', 0, Marker.CUBE_LIST)
        marker.scale.x = marker.scale.y = marker.scale.z = self.object_resolution
        if not len(self.obstacle_points):
            return marker
        z_min = float(self.obstacle_points[:, 2].min())
        z_span = max(float(self.obstacle_points[:, 2].max() - z_min), 1e-6)
        for east, north, altitude in self.obstacle_points:
            marker.points.append(Point(x=float(east), y=float(north), z=float(altitude)))
            marker.colors.append(height_color((altitude - z_min) / z_span))
        return marker

    def ground_marker(self, stamp):
        """把粗分辨率地面压成薄片，避免 10 m 立方体遮住低矮建筑。"""
        marker = self.base_marker(stamp, 'ground', 0, Marker.CUBE_LIST)
        marker.scale.x = marker.scale.y = self.resolution
        marker.scale.z = max(0.25, self.resolution * 0.05)
        marker.color = ColorRGBA(r=0.32, g=0.42, b=0.34, a=0.48)
        for east, north, altitude in self.ground_points:
            marker.points.append(Point(x=float(east), y=float(north), z=float(altitude)))
        return marker

    def bounds_marker(self, stamp):
        marker = self.base_marker(stamp, 'bounds', 0, Marker.LINE_STRIP)
        marker.scale.x = max(0.35, self.resolution * 0.15)
        marker.color = ColorRGBA(r=0.85, g=0.85, b=0.85, a=0.9)
        b = self.bounds
        for east, north in (
            (b['east_min'], b['north_min']),
            (b['east_max'], b['north_min']),
            (b['east_max'], b['north_max']),
            (b['east_min'], b['north_max']),
            (b['east_min'], b['north_min']),
        ):
            marker.points.append(Point(x=east, y=north, z=0.15))
        return marker

    def trajectory_markers(self, stamp, playback_time):
        """把截至 playback_time 的历史 NED 轨迹转换成 ENU 动画标记。"""
        line_colors = (
            ColorRGBA(r=1.0, g=0.72, b=0.05, a=1.0),
            ColorRGBA(r=1.0, g=0.12, b=0.72, a=1.0),
        )
        markers = []
        for index, (trajectory, times) in enumerate(zip(
                self.trajectories_ned, self.trajectory_times)):
            namespace = f'trajectory_{index + 1}'
            visible_count = max(1, int(np.searchsorted(times, playback_time, side='right')))
            visible_count = min(visible_count, len(trajectory))
            line = self.base_marker(stamp, namespace, 0, Marker.LINE_STRIP)
            line.scale.x = 1.2
            line.color = line_colors[index % len(line_colors)]
            for north, east, down in trajectory[:visible_count]:
                line.points.append(Point(x=float(east), y=float(north), z=float(-down)))
            markers.append(line)

            marker_specs = [
                (1, trajectory[0], ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0), 3.0),
                (2, trajectory[visible_count - 1],
                 ColorRGBA(r=1.0, g=0.15, b=0.1, a=1.0), 3.0),
            ]
            for marker_id, pose, color, diameter in marker_specs:
                north, east, down = pose
                marker = self.base_marker(stamp, namespace, marker_id, Marker.SPHERE)
                marker.pose.position.x = float(east)
                marker.pose.position.y = float(north)
                marker.pose.position.z = float(-down)
                marker.scale.x = marker.scale.y = marker.scale.z = diameter
                marker.color = color
                markers.append(marker)
        return markers

    def publish_map(self):
        stamp = self.get_clock().now().to_msg()
        clear = self.base_marker(stamp, 'voxel_map', 0, Marker.CUBE_LIST)
        clear.action = Marker.DELETEALL
        array = MarkerArray()
        array.markers = [
            clear,
            self.ground_marker(stamp),
            self.obstacle_marker(stamp),
            self.bounds_marker(stamp),
        ]
        self.map_publisher.publish(array)

    def playback_time(self):
        """把单调时钟换算成当前循环内的轨迹记录时间。"""
        if self.playback_started is None:
            return 0.0
        elapsed = time.monotonic() - self.playback_started
        active_duration = self.playback_duration / self.args.playback_speed
        cycle_duration = (
            self.args.playback_start_delay + active_duration + self.args.playback_hold)
        phase = elapsed % cycle_duration if self.args.loop else min(elapsed, cycle_duration)
        if phase < self.args.playback_start_delay:
            return 0.0
        if phase < self.args.playback_start_delay + active_duration:
            return (phase - self.args.playback_start_delay) * self.args.playback_speed
        return self.playback_duration

    def publish_playback(self):
        if self.playback_started is None:
            if self.trajectory_publisher.get_subscription_count() == 0:
                return
            self.playback_started = time.monotonic()
            self.get_logger().info('RViz 已连接，开始轨迹播放')

        current_time = self.playback_time()
        stamp = self.get_clock().now().to_msg()
        clear = self.base_marker(stamp, 'trajectory', 0, Marker.LINE_STRIP)
        clear.action = Marker.DELETEALL
        array = MarkerArray()
        array.markers = [clear, *self.trajectory_markers(stamp, current_time)]
        self.trajectory_publisher.publish(array)
        if not self.args.loop and current_time >= self.playback_duration:
            self.playback_timer.cancel()
            if self.args.exit_after_playback:
                self.get_logger().info('轨迹单次播放完成，保留最终画面并退出节点')
                self.playback_complete = True
            else:
                self.get_logger().info('轨迹单次播放完成，保留最终画面')


def main():
    parser = argparse.ArgumentParser(description='在 RViz2 显示 ProjectAirSim 体素地图')
    parser.add_argument('--map-file', default=str(DEFAULT_MAP))
    parser.add_argument('--min-altitude', type=float, default=0.0,
                        help='不加载低于该高度的体素，默认保留地面')
    parser.add_argument('--ground-max-altitude', type=float, default=0.5,
                        help='不高于该值的体素显示为半透明地面薄片')
    parser.add_argument('--objects-only', action='store_true',
                        help='地面以上只显示五个已标定对象，隐藏机场建筑')
    parser.add_argument('--detail-map', action='append', default=[],
                        help='目标局部精细体素文件；可重复传入五次')
    parser.add_argument('--detail-min-altitude', type=float, default=0.5,
                        help='目标精细图去除地面所用的最低高度')
    parser.add_argument('--trajectory-dir', action='append', default=[],
                        help='旧运行目录或 viz_dump 目录；可重复传入多条轨迹')
    parser.add_argument('--playback-speed', type=float, default=2.0,
                        help='轨迹回放倍速，默认 2 倍速')
    parser.add_argument('--playback-hz', type=float, default=10.0,
                        help='轨迹动画发布频率，默认 10 Hz')
    parser.add_argument('--playback-start-delay', type=float, default=2.0,
                        help='每轮开始前停留秒数，默认 2 秒')
    parser.add_argument('--playback-hold', type=float, default=3.0,
                        help='每轮结束后停留秒数，默认 3 秒')
    parser.add_argument('--loop', action='store_true', help='循环播放轨迹')
    parser.add_argument('--exit-after-playback', action='store_true',
                        help='单次播放完成后退出节点，便于再次运行命令重播')
    args = parser.parse_args()
    if args.playback_speed <= 0 or args.playback_hz <= 0:
        parser.error('--playback-speed 和 --playback-hz 必须大于 0')
    if args.playback_start_delay < 0 or args.playback_hold < 0:
        parser.error('播放前后停留时间不能小于 0')

    rclpy.init()
    node = VoxelMapVizNode(args)
    try:
        while rclpy.ok() and not node.playback_complete:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
