import asyncio
import copy
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

import projectairsim
from projectairsim import Drone
from projectairsim.drone import YawControlMode
from projectairsim.utils import projectairsim_log
from projectairsim_ros2 import ROS2Node
from projectairsim_rosbridge import ProjectAirSimROSBridge

SCENE_CONFIG = "scene_drone_sensors.jsonc"

# 控制参数（先用推荐值，跑通后按实测调）
TAKEOFF_HEIGHT_M = 10.0     # 起飞爬升的真实相对高度（m），可调
EGO_START_Z = 2.0           # EGO 本地系高度偏移（吸收 waypoint 写死的 1.0m），可调
CTRL_KP = 1.0               # 位置误差比例增益，超调就降到 0.6
CTRL_MAX_SPEED = 2.5        # 合速度限幅（m/s），EGO max_vel=2.0 + 前馈余量
CTRL_DEADBAND_M = 0.05      # 位置误差死区（m），抑制悬停抖动
CTRL_TIMEOUT_S = 0.5        # pos_cmd 断流超时（s），超时转 hover


class EgoPlannerBridge(Node):
    def __init__(self):
        super().__init__("ego_planner_bridge")
        self.latest_pose = None
        self.origin_position = None
        self.ego_start_z = EGO_START_Z

        # 轨迹跟踪状态（回调单写、主循环单读，GIL 下无需锁）
        self.latest_cmd = None
        self.cmd_stamp = None
        self.tracking = False

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/Sim/SceneDroneSensors/robots/Drone1/actual_pose",
            self.pose_callback,
            10,
        )
        self.depth_sub = self.create_subscription(
            Image,
            "/Sim/SceneDroneSensors/robots/Drone1/sensors/FrontCamera/depth_camera/image",
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.pos_cmd_sub = self.create_subscription(
            PositionCommand,
            "drone_0_planning/pos_cmd",
            self.pos_cmd_callback,
            10,
        )

        self.odom_pub = self.create_publisher(Odometry, "drone_0_visual_slam/odom", 10)
        self.depth_pub = self.create_publisher(Image, "drone_0_depth", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "drone_0_pose", 10)

    def pose_callback(self, msg):
        self.latest_pose = msg
        pose_out = self.to_ego_pose(msg, msg.header.stamp)
        odom = Odometry()
        odom.header = copy.deepcopy(pose_out.header)
        odom.child_frame_id = "drone_0"
        odom.pose.pose = copy.deepcopy(pose_out.pose)
        self.odom_pub.publish(odom)

    def depth_callback(self, msg):
        if self.latest_pose is None:
            return

        stamp = msg.header.stamp
        pose_out = self.to_ego_pose(self.latest_pose, stamp)

        odom = Odometry()
        odom.header = copy.deepcopy(pose_out.header)
        odom.child_frame_id = "drone_0"
        odom.pose.pose = copy.deepcopy(pose_out.pose)

        depth_out = copy.deepcopy(msg)
        depth_out.header.stamp = stamp
        depth_out.header.frame_id = "drone_0_front_camera"

        self.odom_pub.publish(odom)
        self.pose_pub.publish(pose_out)
        self.depth_pub.publish(depth_out)

    def to_ego_pose(self, msg, stamp):
        pose_out = PoseStamped()
        pose_out.header.stamp = stamp
        pose_out.header.frame_id = "world"
        pose_out.pose = copy.deepcopy(msg.pose)

        if self.origin_position is None:
            self.origin_position = copy.deepcopy(msg.pose.position)
            self.get_logger().info(
                "Set EGO local origin from UE pose: "
                f"x={self.origin_position.x:.3f}, "
                f"y={self.origin_position.y:.3f}, "
                f"z={self.origin_position.z:.3f}"
            )

        pose_out.pose.position.x = msg.pose.position.x - self.origin_position.x
        pose_out.pose.position.y = msg.pose.position.y - self.origin_position.y
        pose_out.pose.position.z = msg.pose.position.z - self.origin_position.z + self.ego_start_z
        return pose_out

    def pos_cmd_callback(self, msg):
        # 只存数据，不计算、不 await（同步回调线程）
        self.latest_cmd = msg
        self.cmd_stamp = time.time()
        self.tracking = True

    def compute_ned_velocity(self):
        """把最新 pos_cmd 转成 NED 速度+yaw 指令。断流返回 None（主循环转 hover）。"""
        if self.latest_cmd is None or self.cmd_stamp is None:
            return None
        if time.time() - self.cmd_stamp > CTRL_TIMEOUT_S:
            return None
        if self.latest_pose is None:
            return None

        cmd = self.latest_cmd
        cur = self.to_ego_pose(self.latest_pose, self.latest_pose.header.stamp)
        p = cur.pose.position

        # 控制律（EGO 本地系）：v = 前馈 + Kp * 位置误差，误差带死区
        def axis(des, cur_val, v_ff):
            err = des - cur_val
            if abs(err) < CTRL_DEADBAND_M:
                err = 0.0
            return v_ff + CTRL_KP * err

        v_ex = axis(cmd.position.x, p.x, cmd.velocity.x)
        v_ey = axis(cmd.position.y, p.y, cmd.velocity.y)
        v_ez = axis(cmd.position.z, p.z, cmd.velocity.z)

        # EGO 本地系（ROS 右手系）→ NED 速度（速度是相对量，不受 origin/ego_start_z 平移影响）
        v_north = v_ex
        v_east = -v_ey
        v_down = -v_ez

        # 合速度限幅
        speed = math.sqrt(v_north**2 + v_east**2 + v_down**2)
        if speed > CTRL_MAX_SPEED:
            k = CTRL_MAX_SPEED / speed
            v_north, v_east, v_down = v_north * k, v_east * k, v_down * k

        # yaw：NED 与 ROS 仅 Z 轴反向 → 旋转正方向相反 → 纯取负
        yaw_ned = -cmd.yaw
        return v_north, v_east, v_down, yaw_ned


async def main(ros_node, ego_bridge_node, drone):
    hovering = False
    try:
        while ros_node.spin_once():
            rclpy.spin_once(ego_bridge_node, timeout_sec=0.0)
            if ego_bridge_node.tracking:
                cmd = ego_bridge_node.compute_ned_velocity()
                if cmd is None:
                    if not hovering:           # 只在刚断流时触发一次 hover
                        await drone.hover_async()
                        hovering = True
                else:
                    vn, ve, vd, yaw = cmd
                    # 不 await 命令完成：duration=0.1 是断流自停的看门狗，
                    # 新指令每 0.02s 覆盖旧的，await 完成会把循环拖到 ~8Hz。
                    await drone.move_by_velocity_async(
                        vn, ve, vd, duration=0.1,
                        yaw_control_mode=YawControlMode.MaxDegreeOfFreedom,
                        yaw_is_rate=False, yaw=yaw,
                    )
                    hovering = False
            await asyncio.sleep(0.02)          # 50Hz
    except KeyboardInterrupt:
        pass


async def takeoff_and_hover(drone):
    drone.enable_api_control()
    drone.arm()
    projectairsim_log().info("Taking off Drone1...")
    task = await drone.takeoff_async()
    await task
    projectairsim_log().info(f"Climbing Drone1 to ~{TAKEOFF_HEIGHT_M}m...")
    climb_speed = 2.0
    task = await drone.move_by_velocity_async(
        v_north=0.0,
        v_east=0.0,
        v_down=-climb_speed,
        duration=TAKEOFF_HEIGHT_M / climb_speed,
    )
    await task
    task = await drone.hover_async()
    await task
    projectairsim_log().info("Drone1 takeoff complete and hovering.")


if __name__ == "__main__":
    rclpy.init()

    # Connect to ProjectAirSim
    client = projectairsim.ProjectAirSimClient(address="172.21.192.1")
    client.connect()

    # Load scene
    world = projectairsim.World(
        client,
        SCENE_CONFIG,
        sim_config_path="sim_config/",
        delay_after_load_sec=2,
    )

    # Create ROS2 node and bridge
    ros_node = ROS2Node(name="projectairsim", anonymous=True)
    bridge = ProjectAirSimROSBridge(
        ros_node=ros_node,
        client=client,
        sim_config_path="sim_config/",
    )
    drone = Drone(client, world, "Drone1")
    ego_bridge_node = EgoPlannerBridge()

    projectairsim_log().info("ROS2 Bridge and EGO adapter ready. Press Ctrl+C to stop.")

    # Spin
    try:
        asyncio.run(takeoff_and_hover(drone))
        asyncio.run(main(ros_node, ego_bridge_node, drone))
    except KeyboardInterrupt:
        pass

    ego_bridge_node.destroy_node()
    rclpy.try_shutdown()
    client.disconnect()
    print("Disconnected.")
