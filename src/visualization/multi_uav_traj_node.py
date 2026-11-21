#!/usr/bin/env python3
"""ROS2 node to replay 6-UAV cooperative survey 3D ENU trajectories in RViz2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as RosPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


DRONE_COLORS_RGBA: dict[str, tuple[float, float, float, float]] = {
    "Scout-1": (0.29, 0.78, 0.96, 1.0),
    "Scout-2": (0.45, 0.88, 0.51, 1.0),
    "Scout-3": (0.76, 0.57, 0.92, 1.0),
    "Inspect-1": (1.00, 0.61, 0.22, 1.0),
    "Inspect-2": (1.00, 0.88, 0.24, 1.0),
    "Inspect-3": (1.00, 0.41, 0.37, 1.0),
}


def _color(rgba: tuple[float, float, float, float]) -> ColorRGBA:
    return ColorRGBA(r=float(rgba[0]), g=float(rgba[1]), b=float(rgba[2]), a=float(rgba[3]))


class MultiUAVTrajectoryNode(Node):
    """Publishes 6-UAV 3D trajectories, moving markers, and target beacons in `world` ENU frame."""

    def __init__(self, traj_path: Path, speed: float = 4.0, static_mode: bool = False) -> None:
        super().__init__("multi_uav_trajectory_replay")
        self.data: dict[str, Any] = json.loads(traj_path.read_text(encoding="utf-8"))
        self.speed = max(0.1, float(speed))
        self.static_mode = bool(static_mode)

        qos_static = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        qos_live = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.marker_pub = self.create_publisher(MarkerArray, "/coordination/markers", qos_static if static_mode else qos_live)
        self.path_pubs: dict[str, Any] = {}
        for dname in self.data.get("drones", {}):
            safe = dname.lower().replace("-", "_")
            self.path_pubs[dname] = self.create_publisher(RosPath, f"/coordination/paths/{safe}", qos_static if static_mode else qos_live)

        all_times = [
            float(s["t_s"])
            for info in self.data.get("drones", {}).values()
            for s in info.get("samples", [])
        ]
        self.t_min = min(all_times) if all_times else 0.0
        self.t_max = max(all_times) if all_times else 1.0
        self.play_t = self.t_max if static_mode else self.t_min
        self.last_wall = time.monotonic()
        self.pause_until: float | None = None

        self._publish_at(self.play_t)
        if not static_mode:
            self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"Loaded {len(self.data.get('drones', {}))} UAV trajectories from {traj_path} "
            f"(span={self.t_max - self.t_min:.1f}s, mode={'static' if static_mode else f'{self.speed}x'})"
        )

    def _tick(self) -> None:
        now = time.monotonic()
        dt = now - self.last_wall
        self.last_wall = now
        if self.pause_until is not None:
            if now >= self.pause_until:
                self.pause_until = None
                self.play_t = self.t_min
            return
        self.play_t += dt * self.speed
        if self.play_t >= self.t_max:
            self.play_t = self.t_max
            self._publish_at(self.play_t)
            self.pause_until = now + 2.0
            return
        self._publish_at(self.play_t)

    def _publish_at(self, t_cursor: float) -> None:
        stamp = self.get_clock().now().to_msg()
        markers: list[Marker] = []

        clear = Marker()
        clear.header.frame_id = "world"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.append(clear)

        # Home pad marker
        home_enu = self.data.get("home_enu", [0.0, 0.0, 1.0])
        home_m = Marker()
        home_m.header.frame_id = "world"
        home_m.header.stamp = stamp
        home_m.ns = "anchors"
        home_m.id = 0
        home_m.type = Marker.CYLINDER
        home_m.action = Marker.ADD
        home_m.pose.position = Point(x=float(home_enu[0]), y=float(home_enu[1]), z=1.0)
        home_m.pose.orientation.w = 1.0
        home_m.scale.x = 18.0
        home_m.scale.y = 18.0
        home_m.scale.z = 2.0
        home_m.color = _color((0.7, 0.92, 1.0, 0.65))
        markers.append(home_m)

        # Target beacons
        for idx, (tid, tinfo) in enumerate(self.data.get("targets", {}).items(), start=1):
            enu = tinfo["enu"]
            cyl = Marker()
            cyl.header.frame_id = "world"
            cyl.header.stamp = stamp
            cyl.ns = "targets"
            cyl.id = idx * 2
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position = Point(x=float(enu[0]), y=float(enu[1]), z=4.0)
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = 16.0
            cyl.scale.y = 16.0
            cyl.scale.z = 8.0
            cyl.color = _color((1.0, 0.85, 0.25, 0.7))
            markers.append(cyl)

            txt = Marker()
            txt.header.frame_id = "world"
            txt.header.stamp = stamp
            txt.ns = "targets"
            txt.id = idx * 2 + 1
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position = Point(x=float(enu[0]), y=float(enu[1]), z=35.0)
            txt.pose.orientation.w = 1.0
            txt.scale.z = 12.0
            txt.color = _color((1.0, 1.0, 1.0, 0.95))
            txt.text = tinfo.get("display_name", tid)
            markers.append(txt)

        # 6 UAV trajectories + active positions
        for idx, (dname, info) in enumerate(self.data.get("drones", {}).items(), start=1):
            samples = [s for s in info.get("samples", []) if float(s["t_s"]) <= t_cursor]
            rgba = DRONE_COLORS_RGBA.get(dname, (0.9, 0.9, 0.9, 1.0))

            path_msg = RosPath()
            path_msg.header.frame_id = "world"
            path_msg.header.stamp = stamp
            line = Marker()
            line.header.frame_id = "world"
            line.header.stamp = stamp
            line.ns = "trajectories"
            line.id = idx
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 2.2 if info.get("batch") == 2 else 1.6
            line.color = _color(rgba)

            for s in samples:
                ex, ey, ez = float(s["enu"][0]), float(s["enu"][1]), float(s["enu"][2])
                pt = Point(x=ex, y=ey, z=ez)
                line.points.append(pt)
                ps = PoseStamped()
                ps.header = path_msg.header
                ps.pose.position = pt
                ps.pose.orientation.w = 1.0
                path_msg.poses.append(ps)

            self.path_pubs[dname].publish(path_msg)
            if len(line.points) >= 2:
                markers.append(line)

            if samples:
                last = samples[-1]
                ex, ey, ez = float(last["enu"][0]), float(last["enu"][1]), float(last["enu"][2])
                sphere = Marker()
                sphere.header.frame_id = "world"
                sphere.header.stamp = stamp
                sphere.ns = "drones"
                sphere.id = idx * 2
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position = Point(x=ex, y=ey, z=ez)
                sphere.pose.orientation.w = 1.0
                sphere.scale.x = 7.0
                sphere.scale.y = 7.0
                sphere.scale.z = 7.0
                sphere.color = _color(rgba)
                markers.append(sphere)

                lbl = Marker()
                lbl.header.frame_id = "world"
                lbl.header.stamp = stamp
                lbl.ns = "drones"
                lbl.id = idx * 2 + 1
                lbl.type = Marker.TEXT_VIEW_FACING
                lbl.action = Marker.ADD
                lbl.pose.position = Point(x=ex, y=ey, z=ez + 10.0)
                lbl.pose.orientation.w = 1.0
                lbl.scale.z = 9.0
                lbl.color = _color(rgba)
                lbl.text = f"{info.get('display_name', dname)} {int(round(ez))}m"
                markers.append(lbl)

        self.marker_pub.publish(MarkerArray(markers=markers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish 6-UAV 3D trajectories for RViz2 replay.")
    parser.add_argument("--trajectories", required=True, help="Path to trajectories.json.")
    parser.add_argument("--speed", type=float, default=4.0, help="Replay speed multiplier (default: 4.0).")
    parser.add_argument("--static", action="store_true", help="Publish full static 6-UAV trajectories at once.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = MultiUAVTrajectoryNode(Path(args.trajectories), speed=args.speed, static_mode=args.static)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
