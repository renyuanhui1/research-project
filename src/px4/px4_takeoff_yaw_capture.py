"""Take off, yaw right 90 degrees, then save Chase and FrontCamera RGB images."""

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/captures/px4_yaw90"
DEFAULT_SCENE = "scene_px4_sitl_wsl2.jsonc"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Connect to Project AirSim, arm PX4, take off, yaw right 90 degrees, "
            "and save images from Chase and FrontCamera."
        )
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", type=int, default=1, help="Number of image pairs to save.")
    parser.add_argument("--fps", type=float, default=2.0, help="Capture rate when frames > 1.")
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=1.0,
        help="Seconds to wait after yaw before capturing.",
    )
    parser.add_argument(
        "--px4-wait-sec",
        type=float,
        default=120.0,
        help="Seconds to wait for PX4 to connect and become ready.",
    )
    parser.add_argument(
        "--no-land",
        action="store_true",
        help="Leave the drone flying instead of landing after capture.",
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

    image_array = unpack_image(image)
    if not cv2.imwrite(str(file_path), image_array):
        raise RuntimeError(f"Failed to save image: {file_path}")


def verify_rgb_camera(drone: Drone, camera_id: str):
    if camera_id not in drone.sensors:
        raise RuntimeError(f"Camera '{camera_id}' is not configured in this scene.")
    if "scene_camera" not in drone.sensors[camera_id]:
        raise RuntimeError(f"Camera '{camera_id}' does not have RGB scene capture enabled.")


def is_vehicle_not_connected_error(err: Exception) -> bool:
    message = str(err)
    return (
        "no vehicle is connected" in message
        or "vehicle is not responding" in message
    )


async def wait_for_px4_ready(drone: Drone, timeout_sec: float):
    deadline = time.monotonic() + timeout_sec
    attempt = 1

    print("Waiting for PX4 vehicle connection...")
    print("PX4 console should show: Simulator connected on TCP port 4560")
    print("PX4 console should later show: home_set")

    while True:
        try:
            ready_state = drone.get_ready_state()
            can_arm = drone.can_arm()
            print(f"PX4 ready_state={ready_state}, can_arm={can_arm}")
            if can_arm:
                return
        except RuntimeError as err:
            if not is_vehicle_not_connected_error(err):
                raise
            print(f"PX4 not connected yet, attempt {attempt}: {err}")

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for PX4. Start PX4 with:\n"
                "  export PX4_SIM_HOST_ADDR=172.21.192.1\n"
                "  cd ~/PX4-Autopilot\n"
                "  make px4_sitl none_iris\n"
                "Then wait for 'Simulator connected on TCP port 4560' and 'home_set'."
            )

        attempt += 1
        await asyncio.sleep(2.0)


def require_success(action_name: str, result):
    print(f"{action_name}: {result}")
    if result is not True:
        raise RuntimeError(f"{action_name} failed: {result}")


async def main(args):
    if args.frames <= 0:
        raise ValueError("--frames must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.settle_sec < 0:
        raise ValueError("--settle-sec cannot be negative.")
    if args.px4_wait_sec <= 0:
        raise ValueError("--px4-wait-sec must be positive.")

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

        await wait_for_px4_ready(drone, args.px4_wait_sec)

        print("Enabling API control...")
        require_success("enable_api_control", drone.enable_api_control())
        api_control_enabled = True

        print("Arming...")
        require_success("arm", drone.arm())

        print("Taking off...")
        takeoff_task = await drone.takeoff_async()
        await takeoff_task

        print("Yawing right 90 degrees...")
        yaw_task = await drone.rotate_to_yaw_async(yaw=np.deg2rad(90.0))
        await yaw_task

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
            print(f"Saved: {chase_path} and {front_path}")

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
                print(f"Unable to land or release PX4 control cleanly: {err}")
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
