"""
让无人机执行 COLMAP 轨迹

CSV: image_id, image_name, camera_id, x, y, z, qw, qx, qy, qz, ...
对齐方法：用首末两点把 COLMAP 轨迹旋转+缩放到 NED 起点和终点。
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
CSV_PATH = PROJECT_ROOT / "data/camera_trajectory.csv"

# NED meters (UE cm / 100, with z negated for NED)
NED_START = np.array([-423.10, -460.70, -11.80])
NED_END = np.array([-406.30, -450.10, -11.80])

VELOCITY = 2.0


def load_colmap_positions(csv_path):
    positions = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return np.array(positions)


def align_trajectory(colmap_pts, ned_start, ned_end):
    """Align COLMAP trajectory so first/last points map to ned_start/ned_end."""
    c_start = colmap_pts[0]
    c_end = colmap_pts[-1]
    c_vec = c_end - c_start
    n_vec = ned_end - ned_start

    c_len = np.linalg.norm(c_vec)
    n_len = np.linalg.norm(n_vec)
    scale = n_len / c_len if c_len > 1e-6 else 1.0

    # Rotation that aligns c_vec direction to n_vec direction
    c_dir = c_vec / c_len if c_len > 1e-6 else np.array([1.0, 0.0, 0.0])
    n_dir = n_vec / n_len if n_len > 1e-6 else np.array([1.0, 0.0, 0.0])

    v = np.cross(c_dir, n_dir)
    s = np.linalg.norm(v)
    cos_t = np.dot(c_dir, n_dir)

    if s < 1e-8:
        R = np.eye(3) if cos_t > 0 else -np.eye(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - cos_t) / (s * s))

    aligned = []
    for p in colmap_pts:
        rel = (p - c_start) * scale
        rotated = R @ rel
        aligned.append(ned_start + rotated)
    return np.array(aligned)


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

        projectairsim_log().info("Rotating 90 degrees to the right (yaw)")
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
        ned_pts = align_trajectory(colmap_pts, NED_START, NED_END)
        path = [(float(p[0]), float(p[1]), float(p[2])) for p in ned_pts]

        projectairsim_log().info(f"Loaded {len(path)} waypoints. Flying trajectory.")
        task = await drone.move_on_path_async(path, VELOCITY)
        await task
        projectairsim_log().info("Trajectory complete.")

    except Exception as err:
        projectairsim_log().error(f"Exception: {err}", exc_info=True)
    finally:
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
