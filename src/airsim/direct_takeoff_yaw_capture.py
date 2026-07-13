"""Take off, yaw right 90 degrees, then save Chase and FrontCamera images.

This script uses Project AirSim's built-in simple-flight controller, not PX4.
"""

import argparse
import asyncio
from pathlib import Path
import time

import cv2
import numpy as np
import projectairsim
from projectairsim import Drone, World
from projectairsim.types import ImageType
from projectairsim.utils import unpack_image


DEFAULT_ADDRESS = "172.21.192.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/captures/direct_yaw90"
DEFAULT_SCENE = "my_scene.jsonc"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use simple-flight to take off, and save "
            "Chase/FrontCamera RGB images."
        )
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--altitude",
        type=float,
        default=6.0,
        help="Target flight altitude in meters above takeoff point.",
    )
    parser.add_argument(
        "--altitude-speed",
        type=float,
        default=2.0,
        help="Speed used when moving to the requested altitude.",
    )
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--no-land", action="store_true")
    return parser.parse_args()


def next_frame_index(*directories: Path) -> int:
    indices = []
    for directory in directories:
        for file_path in directory.glob("image_*.png"):
            try:
                indices.append(int(file_path.stem.split("_", 1)[1]))
            except ValueError:
                pass
    return max(indices, default=-1) + 1


def verify_rgb_camera(drone: Drone, camera_id: str):
    if camera_id not in drone.sensors:
        raise RuntimeError(f"Camera '{camera_id}' is not configured in this scene.")
    if "scene_camera" not in drone.sensors[camera_id]:
        raise RuntimeError(f"Camera '{camera_id}' does not have RGB scene capture enabled.")


def save_camera_image(drone: Drone, camera_id: str, file_path: Path):
    images = drone.get_images(camera_id, [ImageType.SCENE])
    image = images.get(ImageType.SCENE)
    if image is None:
        raise RuntimeError(f"Camera '{camera_id}' did not return an RGB image.")
    img = unpack_image(image)
    if not cv2.imwrite(str(file_path), img):
        raise RuntimeError(f"Failed to save image: {file_path}")


def require_success(action_name: str, result):
    print(f"{action_name}: {result}")
    if result is not True:
        raise RuntimeError(f"{action_name} failed: {result}")


def get_current_ned_position(drone: Drone):
    kinematics = drone.get_ground_truth_kinematics()
    position = kinematics["pose"]["position"]
    return float(position["x"]), float(position["y"]), float(position["z"])


async def main(args):
    if args.frames <= 0:
        raise ValueError("--frames must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.altitude <= 0:
        raise ValueError("--altitude must be positive.")
    if args.altitude_speed <= 0:
        raise ValueError("--altitude-speed must be positive.")
    if args.settle_sec < 0:
        raise ValueError("--settle-sec cannot be negative.")

    output_dir = args.output_dir.expanduser().resolve()
    chase_dir = output_dir / "imgs_chase"
    front_dir = output_dir / "imgs_front"
    chase_dir.mkdir(parents=True, exist_ok=True)
    front_dir.mkdir(parents=True, exist_ok=True)

    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None
    api_control_enabled = False

    try:
        client.connect()
        world = World(
            client,
            args.scene,
            sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
            delay_after_load_sec=2,
        )
        drone = Drone(client, world, "Drone1")

        verify_rgb_camera(drone, "Chase")
        verify_rgb_camera(drone, "FrontCamera")

        print("Enabling API control...")
        require_success("enable_api_control", drone.enable_api_control())
        api_control_enabled = True

        print("Arming...")
        require_success("arm", drone.arm())

        north, east, cur_z = get_current_ned_position(drone)
        print(f"Moving to altitude {args.altitude:.2f} m above current position...")
        altitude_task = await drone.move_to_position_async(
            north=north,
            east=east,
            down=cur_z - args.altitude,
            velocity=args.altitude_speed,
        )
        await asyncio.wait_for(altitude_task, timeout=30.0)


        if args.settle_sec > 0:
            await asyncio.sleep(args.settle_sec)

        frame_index = next_frame_index(chase_dir, front_dir)
        interval = 1.0 / args.fps
        print(f"Saving {args.frames} image pair(s) to {output_dir}")

        for _ in range(args.frames):
            started = time.monotonic()
            chase_path = chase_dir / f"image_{frame_index}.png"
            front_path = front_dir / f"image_{frame_index}.png"

            save_camera_image(drone, "Chase", chase_path)
            save_camera_image(drone, "FrontCamera", front_path)
            print(f"Saved: {chase_path}")
            print(f"Saved: {front_path}")

            frame_index += 1
            delay = interval - (time.monotonic() - started)
            if delay > 0:
                await asyncio.sleep(delay)

    finally:
        if api_control_enabled and drone is not None and not args.no_land:
            try:
                print("Landing...")
                land_task = await drone.land_async()
                await land_task
                drone.disarm()
                drone.disable_api_control()
            except Exception as err:
                print(f"Unable to land or release control cleanly: {err}")
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
