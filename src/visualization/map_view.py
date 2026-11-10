"""Publish airport occupancy grids and geometry-derived aircraft voxels."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import sys
_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                     if (p / "environment.yml").is_file() and (p / "src").is_dir())
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from common.paths import PROJECT_ROOT, project_path
from common.profiles import selected_profile
PROFILE = selected_profile()
from visualization.colors import height_color


class AirportMapNode(Node):
    def __init__(self, map_dir):
        super().__init__('airport_grid_map', namespace='/near_nose')
        root = Path(map_dir) / 'grids'
        report = json.loads((root / 'summary.json').read_text())
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.create_publisher(MarkerArray, 'map', qos)
        self.grid_publishers = []
        self.grids = []
        self.frame_publisher = StaticTransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()
        clear = self.base(stamp, '', Marker.CUBE)
        clear.action = Marker.DELETEALL
        markers = [clear]
        for info in report['maps']:
            name = info['name']
            with np.load(root / (name + '.npz')) as data:
                cells = data['occupancy']
                origin = data['origin_enu_m']
                resolution = float(data['resolution_m'])
            if not np.isin(cells, [-1, 0, 100]).all():
                raise ValueError('Invalid occupancy values: ' + name)
            msg = OccupancyGrid()
            msg.header.frame_id = 'world'
            msg.header.stamp = stamp
            msg.info.map_load_time = stamp
            msg.info.resolution = resolution
            msg.info.height, msg.info.width = cells.shape
            msg.info.origin.position = Point(x=float(origin[0]), y=float(origin[1]), z=0.0)
            msg.info.origin.orientation.w = 1.0
            msg.data = cells.reshape(-1).tolist()
            if name == 'global':
                transform = TransformStamped()
                transform.header = msg.header
                transform.child_frame_id = 'airport_grid_origin'
                transform.transform.translation.x = float(origin[0])
                transform.transform.translation.y = float(origin[1])
                transform.transform.rotation.w = 1.0
                self.frame_publisher.sendTransform(transform)
            publisher = self.create_publisher(OccupancyGrid, 'grid/' + name, qos)
            publisher.publish(msg)
            self.grid_publishers.append(publisher)
            self.grids.append(msg)
            markers.extend(self.ground(stamp, name, cells, origin, resolution,
                                       -0.10 if name == 'global' else -0.05))
        with np.load(root / 'context_voxels.npz') as data:
            context = data['centers_enu_m']
            resolution = float(data['resolution_m'])
        buildings = self.cubes(stamp, 'context_5m', context, resolution)
        buildings.color = ColorRGBA(r=0.43, g=0.56, b=0.68, a=1.0)
        markers.append(buildings)
        self.point_counts = {'context': len(context)}
        for target in report['targets']:
            name = target['name']
            with np.load(root / (name + '_voxels.npz')) as data:
                points = data['centers_enu_m']
                resolution = float(data['resolution_m'])
            marker = self.cubes(stamp, 'aircraft_' + name, points, resolution)
            for z in points[:, 2]:
                color = height_color(z / max(float(points[:, 2].max()), resolution))
                color.a = 1.0
                marker.colors.append(color)
            markers.append(marker)
            self.point_counts[name] = len(points)
        self.marker_array = MarkerArray(markers=markers)
        self.publisher.publish(self.marker_array)
        self.get_logger().info(
            f'Occupancy grids and voxels published: {self.point_counts}; ENU, frame=world')

    @staticmethod
    def base(stamp, namespace, kind):
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = 0
        marker.type = kind
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def cubes(self, stamp, namespace, points, resolution):
        marker = self.base(stamp, namespace, Marker.CUBE_LIST)
        marker.scale.x = marker.scale.y = marker.scale.z = resolution
        marker.points = [Point(x=float(e), y=float(n), z=float(z)) for e, n, z in points]
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        return marker

    def ground(self, stamp, name, cells, origin, resolution, altitude):
        """Render exact occupancy cells as row runs, without indexed texture shaders."""
        colors = {-1: (0.45, 0.45, 0.45), 0: (0.9, 0.9, 0.9), 100: (0.06, 0.06, 0.06)}
        groups = {}
        for value, rgb in colors.items():
            marker = self.base(stamp, f'grid_{name}_{value}', Marker.TRIANGLE_LIST)
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            marker.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0)
            groups[value] = marker
        for row_index, row in enumerate(cells):
            edges = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            n0 = float(origin[1] + row_index * resolution)
            n1 = n0 + resolution
            for first, last in zip(edges[:-1], edges[1:]):
                e0 = float(origin[0] + first * resolution)
                e1 = float(origin[0] + last * resolution)
                marker = groups[int(row[first])]
                marker.points.extend(Point(x=e, y=n, z=altitude) for e, n in
                                     ((e0, n0), (e1, n0), (e1, n1),
                                      (e0, n0), (e1, n1), (e0, n1)))
        return [marker for marker in groups.values() if marker.points]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default=None)
    parser.add_argument('--map-dir', default=None)
    args = parser.parse_args()
    rclpy.init()
    node = AirportMapNode(args.map_dir or project_path(PROFILE['map']['root']))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
