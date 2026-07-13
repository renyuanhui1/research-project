"""
桥接节点：同步深度图和位姿，统一时间戳后转发给 EGO-Planner
"""

import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class PoseToOdomNode(Node):
    def __init__(self):
        super().__init__('pose_to_odom_node')

        self.latest_pose = None
        self.origin_position = None
        self.ego_start_z = 1.0

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/Sim/SceneDroneSensors/robots/Drone1/actual_pose',
            self.pose_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/Sim/SceneDroneSensors/robots/Drone1/sensors/FrontCamera/depth_camera/image',
            self.depth_callback,
            qos_profile_sensor_data
        )

        self.odom_pub = self.create_publisher(Odometry, 'drone_0_visual_slam/odom', 10)
        self.depth_pub = self.create_publisher(Image, 'drone_0_depth', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'drone_0_pose', 10)

    def pose_callback(self, msg: PoseStamped):
        self.latest_pose = msg

    def depth_callback(self, msg: Image):
        if self.latest_pose is None:
            return

        # EGO-Planner uses message_filters on depth + pose/odom. Publish all
        # derived messages from one depth frame with exactly the same stamp.
        stamp = msg.header.stamp

        pose_out = PoseStamped()
        pose_out.header.stamp = stamp
        pose_out.header.frame_id = 'world'
        pose_out.pose = copy.deepcopy(self.latest_pose.pose)

        if self.origin_position is None:
            self.origin_position = copy.deepcopy(self.latest_pose.pose.position)
            self.get_logger().info(
                'Set EGO local origin from UE pose: '
                f'x={self.origin_position.x:.3f}, '
                f'y={self.origin_position.y:.3f}, '
                f'z={self.origin_position.z:.3f}'
            )

        pose_out.pose.position.x = self.latest_pose.pose.position.x - self.origin_position.x
        pose_out.pose.position.y = self.latest_pose.pose.position.y - self.origin_position.y
        pose_out.pose.position.z = (
            self.latest_pose.pose.position.z - self.origin_position.z + self.ego_start_z
        )

        odom = Odometry()
        odom.header = copy.deepcopy(pose_out.header)
        odom.child_frame_id = 'drone_0'
        odom.pose.pose = copy.deepcopy(pose_out.pose)

        depth_out = copy.deepcopy(msg)
        depth_out.header.stamp = stamp
        depth_out.header.frame_id = 'drone_0_front_camera'

        self.odom_pub.publish(odom)
        self.pose_pub.publish(pose_out)
        self.depth_pub.publish(depth_out)


def main(args=None):
    rclpy.init(args=args)
    node = PoseToOdomNode()
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
