"""
按 COLMAP 轨迹飞行，只指定起点，统一 SCALE 版本。
COLMAP 相机坐标系 (X右, Y下, Z前) → NED (无人机朝东飞)
  COLMAP +Z (前) → NED +East (+y)
  COLMAP +X (右) → NED -North (-x, 朝东飞时右边是南)
  COLMAP +Y (下) → NED -Down (-z)
"""

import asyncio
import csv
from pathlib import Path

import numpy as np

import projectairsim
from projectairsim import Drone, World
from projectairsim.utils import projectairsim_log


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_CONFIG = "scene_drone_sensors.jsonc"
CSV_PATH = PROJECT_ROOT / "data/run-14-trajectory.csv"

# NED_START = np.array([-423.10, -460.70, -11.80])
NED_START = np.array([-425.10, -472.70, -8.80])

# COLMAP 单位 → 米。三轴统一缩放。
# SCALE = 1.6
SCALE = 2.0

VELOCITY = 2.0


def load_colmap_positions(csv_path):
    positions = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.array(positions)


def colmap_to_ned_offset(colmap_delta):
    """
    Convert COLMAP frame relative offset to NED offset.
    Drone is facing East (yaw=90 deg).
    COLMAP cam: +X right, +Y down, +Z forward
    Drone-East NED: forward=+East(+y), right=-North(-x), down=+Down(+z)
    Note: COLMAP Y axis is flipped here because the recovered trajectory
    actually moves upward (camera ascends), so +Y_colmap → -Down (up).
    """
    cx, cy, cz = colmap_delta
    n_north = -cx
    n_east = cz
    n_down = -cy
    return np.array([n_north, n_east, n_down])


async def main():
    client = projectairsim.ProjectAirSimClient(address="172.21.192.1")
    try:
        client.connect()
        world = World(client, SCENE_CONFIG, sim_config_path="sim_config/", delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")

        drone.enable_api_control()
        drone.arm()

        projectairsim_log().info("Taking off")
        task = await drone.takeoff_async()
        await task

        projectairsim_log().info("Rotating 90 degrees to the right")
        task = await drone.rotate_to_yaw_async(yaw=np.deg2rad(90.0))
        await task

        projectairsim_log().info(f"Moving to start position {NED_START}")
        task = await drone.move_to_position_async(
            north=float(NED_START[0]),
            east=float(NED_START[1]),
            down=float(NED_START[2]),
            velocity=VELOCITY,
        )
        await task

        colmap_pts = load_colmap_positions(CSV_PATH)
        c_start = colmap_pts[0]

        path = []
        for p in colmap_pts:
            delta = (p - c_start) * SCALE
            ned_offset = colmap_to_ned_offset(delta)
            ned_point = NED_START + ned_offset
            path.append((float(ned_point[0]), float(ned_point[1]), float(ned_point[2])))

        projectairsim_log().info(
            f"COLMAP raw end offset (X,Y,Z) = {(colmap_pts[-1] - c_start).tolist()}"
        )
        projectairsim_log().info(
            f"Final NED point = {path[-1]} (start was {NED_START.tolist()})"
        )
        projectairsim_log().info(f"Flying {len(path)} waypoints, scale={SCALE}")

        task = await drone.move_on_path_async(path, VELOCITY)
        await task
        projectairsim_log().info("Trajectory complete.")

    except Exception as err:
        projectairsim_log().error(f"Exception: {err}", exc_info=True)
    finally:
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
