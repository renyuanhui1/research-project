"""Verify the coastal single-drone scene by recording RGB and depth at a saved coastal inspection viewpoint."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from projectairsim import Drone, ProjectAirSimClient, World
from projectairsim.types import Pose, Quaternion, Vector3

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))
from common.runtime import configure_client_log
from common.sensors import decode_image
from capture.collect_part_views import decode_rpc_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--view", choices=("grounded-vessel", "fish-market"), default="grounded-vessel"
    )
    parser.add_argument(
        "--market-distance-m",
        type=float,
        help="Horizontal standoff on the market facade normal",
    )
    args = parser.parse_args()
    if args.market_distance_m is not None and (
        args.view != "fish-market" or not 5 <= args.market_distance_m <= 65
    ):
        parser.error(
            "--market-distance-m requires --view fish-market and a distance from 5 to 65 m"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    configure_client_log(args.out)
    client = ProjectAirSimClient(args.address)
    try:
        client.connect()
        client.socket_services.recv_timeout = 45000
        world = World(
            client,
            "scene_coastal_harbor.jsonc",
            sim_config_path=str(root / "configs/simulation"),
            delay_after_load_sec=2,
        )
        drone = Drone(client, world, "Drone1")
        # Static validation view toward the grounded tug on the eastern tidal flat.
        viewpoint = [312.6, -224.3, -10.3]
        yaw = math.radians(60)
        if args.view == "fish-market":
            stations = root / "data/metadata/coastal_harbor/market-stations.json"
            station = json.loads(stations.read_text())["Building_071"]
            x, y, z = station["eye"]
            viewpoint = [x, y, -z]
            tx, ty, _ = station["aim"]
            if args.market_distance_m is not None:
                dist = args.market_distance_m
                nx, ny = station["front_normal"]
                x, y = tx + nx * dist, ty + ny * dist
                viewpoint = [
                    x,
                    y,
                    -(station["aim"][2] + dist * math.tan(math.radians(30)) + 0.3),
                ]
            expected_depth = math.hypot(tx - x, ty - y) / math.cos(math.radians(30))
            yaw = math.atan2(ty - y, tx - x)
        pose = Pose(
            {
                "translation": Vector3(dict(zip(("x", "y", "z"), viewpoint))),
                "rotation": Quaternion(
                    {"w": math.cos(yaw / 2), "x": 0, "y": 0, "z": math.sin(yaw / 2)}
                ),
                "frame_id": "DEFAULT_ID",
            }
        )
        if args.view == "fish-market":
            initial = drone.get_ground_truth_kinematics()["pose"]["position"]
            for x, y in ((initial["x"], initial["y"]), tuple(viewpoint[:2])):
                elevated = Pose(
                    {
                        "translation": Vector3({"x": x, "y": y, "z": -65}),
                        "rotation": pose.rotation,
                        "frame_id": "DEFAULT_ID",
                    }
                )
                if not drone.set_pose(elevated, reset_kinematics=True):
                    raise RuntimeError("Elevated camera staging pose rejected")
                time.sleep(0.55)
                actual = drone.get_ground_truth_kinematics()["pose"]["position"]
                if math.dist([actual[k] for k in "xyz"], [x, y, -65]) > 1.5:
                    raise RuntimeError(
                        f"Did not reach elevated camera station: {actual}"
                    )
        timestamps = []
        deadline = time.monotonic() + 150
        warmup_frames = 0
        timeouts = 0
        while len(timestamps) < 3:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for {args.view} RGB-D")
            if not drone.set_pose(pose, reset_kinematics=True):
                raise RuntimeError("Camera viewpoint rejected")
            time.sleep(0.75)
            try:
                frames = drone.get_images("FrontCamera", [0, 2])
            except RuntimeError as exc:
                if "Timeout waiting for new captured_images" not in str(exc):
                    raise
                timeouts += 1
                continue
            color, info = decode_image(frames[0])
            depth = decode_rpc_depth(frames[2]).astype(np.float32)
            valid = np.isfinite(depth) & (depth > 0)
            assert color.shape == (1080, 1920, 3) and depth.shape == (360, 640)
            assert color.std() > 4 and valid.mean() > 0.25
            # At this fixed verification pose, the red antifouling band distinguishes loaded
            # hull paint from UE's temporary monochrome shader-compilation placeholder.
            crop = color[
                color.shape[0] // 4 : 3 * color.shape[0] // 4,
                color.shape[1] // 4 : 3 * color.shape[1] // 4,
            ]
            h, s, v = cv2.split(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV))
            hull_red = float((((h < 12) | (h > 168)) & (s > 55) & (v > 35)).mean())
            stamp = [int(frames[t]["time_stamp"]) for t in (0, 2)]
            if args.view == "fish-market":
                patch = depth[174:186, 314:326]
                usable = patch[np.isfinite(patch) & (patch > 0)]
                center_depth = float(np.median(usable)) if usable.size else 0.0
                ready = (
                    expected_depth * 0.76 < center_depth < expected_depth * 1.25
                    and float((s > 45).mean()) > 0.025
                )
                view_check = {"center_depth_m": center_depth}
            else:
                ready = hull_red >= 0.004
                view_check = {"red_hull_fraction_center": hull_red}
            if not ready and args.view == "fish-market" and warmup_frames % 10 == 0:
                print("Market camera warmup", view_check, flush=True)
                cv2.imwrite(str(args.out / "warmup.jpg"), color[:, :, ::-1])
            if not ready:
                warmup_frames += 1
                timestamps.clear()
                continue
            if args.view == "fish-market":
                expected_camera = [viewpoint[0], viewpoint[1], viewpoint[2] + 0.3]
                if (
                    math.dist(expected_camera, [frames[0]["pos_" + k] for k in "xyz"])
                    > 0.8
                ):
                    raise RuntimeError(
                        "RGB camera missed its designated fish-market station"
                    )
            if stamp[0] != stamp[1]:
                continue
            if not timestamps or all(t > p for t, p in zip(stamp, timestamps[-1])):
                timestamps.append(stamp)
        assert all(
            b[0] > a[0] and b[1] > a[1] for a, b in zip(timestamps, timestamps[1:])
        )
        cv2.imwrite(str(args.out / "front-rgb.jpg"), color[:, :, ::-1])
        np.savez_compressed(
            args.out / "front-depth.npz", depth_m=depth.astype(np.float16)
        )
        report = {
            "scene": world.parent_topic,
            "actors": world.list_actors(),
            "camera": "FrontCamera",
            "timestamps": timestamps,
            "warmup_frames": warmup_frames,
            "capture_timeouts": timeouts,
            "view": args.view,
            **view_check,
            "rgb_shape": color.shape,
            "depth_shape": depth.shape,
            "fraction_positive_finite": round(float(valid.mean()), 4),
            "median_depth_m": round(float(np.median(depth[valid])), 3),
            "viewpoint_ned_m": viewpoint,
            "rgb_camera_ned_m": [frames[0]["pos_" + k] for k in "xyz"],
            "rgb_camera_rotation_xyzw": [frames[0]["rot_" + k] for k in "xyzw"],
            "capture_pose": drone.get_ground_truth_kinematics()["pose"],
            "rgb_encoding": info["encoding"],
            "depth_encoding": frames[2]["encoding"],
            "method": "three static set-pose camera checks",
        }
        (args.out / "check.json").write_text(
            json.dumps(report, indent=2), encoding="utf8"
        )
        print(json.dumps(report, indent=2))
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
