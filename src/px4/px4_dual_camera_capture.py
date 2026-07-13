"""Record PX4 drone chase and onboard-front RGB image pairs from Project AirSim."""

import argparse
import asyncio
from pathlib import Path
import time

import cv2
import projectairsim
from projectairsim import Drone, World
from projectairsim.types import ImageType
from projectairsim.utils import unpack_image


DEFAULT_ADDRESS = "172.21.192.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/captures/px4_dual_camera"
DEFAULT_SCENE = "scene_px4_sitl_wsl2.jsonc"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save tracking and front-camera RGB frames from a PX4 scene."
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to record; 0 records until Ctrl+C.",
    )
    parser.add_argument(
        "--takeoff",
        action="store_true",
        help="Use PX4 API to arm and take off before waiting to record.",
    )
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="Begin saving as soon as the cameras are available; default waits for Enter.",
    )
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


def save_camera_image(drone: Drone, camera_id: str, file_path: Path):
    images = drone.get_images(camera_id, [ImageType.SCENE])
    image = images.get(ImageType.SCENE)
    if image is None:
        raise RuntimeError(f"Camera '{camera_id}' did not return an RGB image.")
    if not cv2.imwrite(str(file_path), unpack_image(image)):
        raise RuntimeError(f"Failed to save image: {file_path}")


async def main(args):
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.duration < 0:
        raise ValueError("--duration cannot be negative.")

    output_dir = args.output_dir.expanduser().resolve()
    third_view_dir = output_dir / "imgs_third_view"
    wrist_view_dir = output_dir / "imgs_wrist"
    third_view_dir.mkdir(parents=True, exist_ok=True)
    wrist_view_dir.mkdir(parents=True, exist_ok=True)

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

        for camera_id in ("Chase", "FrontCamera"):
            if (
                camera_id not in drone.sensors
                or "scene_camera" not in drone.sensors[camera_id]
            ):
                raise RuntimeError(f"RGB capture is unavailable for '{camera_id}'.")

        if args.takeoff:
            drone.enable_api_control()
            api_control_enabled = True
            drone.arm()
            takeoff_task = await drone.takeoff_async()
            await takeoff_task

        if not args.start_immediately:
            print("Ready. Take off, move to the starting position, and prepare the path.")
            await asyncio.to_thread(input, "Press Enter to begin image recording... ")

        frame_index = next_frame_index(third_view_dir, wrist_view_dir)
        interval = 1.0 / args.fps
        deadline = None if args.duration == 0 else time.monotonic() + args.duration
        print(f"Recording to {output_dir}. Press Ctrl+C to stop.")

        while deadline is None or time.monotonic() < deadline:
            started = time.monotonic()
            save_camera_image(
                drone, "Chase", third_view_dir / f"image_{frame_index}.png"
            )
            save_camera_image(
                drone, "FrontCamera", wrist_view_dir / f"image_{frame_index}.png"
            )
            frame_index += 1
            delay = interval - (time.monotonic() - started)
            if delay > 0:
                await asyncio.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        if api_control_enabled and drone is not None:
            try:
                land_task = await drone.land_async()
                await land_task
                drone.disarm()
                drone.disable_api_control()
            except Exception as err:
                print(f"Unable to land or release PX4 control cleanly: {err}")
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
