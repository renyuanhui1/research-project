"""生成 airbase 场景体素地图并保存诊断图，不控制无人机。

默认先做全局粗地图（2 km x 2 km x 100 m，10 m 体素）：
    python src/airsim/probe/probe_voxel_map.py --profile global

目标附近精细地图（300 m x 300 m x 100 m，2 m 体素）：
    python src/airsim/probe/probe_voxel_map.py --profile local

坐标约定：输入中心为 NED；输出图像上方为北、右侧为东。
ProjectAirSim 返回的一维索引为 i + nx * (k + nz * j)，因此内部数组形状
统一保存为 (east, altitude, north)。
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import projectairsim
from projectairsim import World
from projectairsim.types import Pose, Quaternion, Vector3


AIRBASE_ROOT = next(
    path for path in Path(__file__).resolve().parents
    if (path / "sim_config").is_dir()
)
DEFAULT_CONFIG_DIR = AIRBASE_ROOT / "sim_config"
DEFAULT_OUTPUT_ROOT = AIRBASE_ROOT / "outputs/maps/voxel_probe"

PROFILES = {
    "global": {
        "center_ned": (0.0, 0.0, -50.0),
        "size_m": (2000.0, 2000.0, 100.0),
        "resolution_m": 10.0,
        "slice_altitudes_m": (0.0, 10.0, 20.0, 30.0, 50.0, 80.0),
        "projection_min_altitude_m": 5.0,
    },
    "local": {
        "center_ned": (-62.71, -16.68, -50.0),
        "size_m": (300.0, 300.0, 100.0),
        "resolution_m": 2.0,
        "slice_altitudes_m": (0.0, 2.0, 6.0, 10.0, 20.0, 30.0, 50.0, 80.0),
        "projection_min_altitude_m": 2.0,
    },
}


def make_pose(center_ned):
    """构造 ProjectAirSim 所需的 NED 中心位姿。"""
    north, east, down = center_ned
    return Pose({
        "translation": Vector3({"x": north, "y": east, "z": down}),
        "rotation": Quaternion({"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}),
        "frame_id": "DEFAULT_ID",
    })


def cell_centers(center_ned, shape, resolution_m):
    """返回 north/east/altitude 三个轴的体素中心坐标。"""
    ny, nz, nx = shape
    north = center_ned[0] + (np.arange(nx) - nx // 2) * resolution_m
    east = center_ned[1] + (np.arange(ny) - ny // 2) * resolution_m
    altitude = -center_ned[2] + (np.arange(nz) - nz // 2) * resolution_m
    return north, east, altitude


def as_map_image(east_north_mask):
    """(east, north) 数组转为显示图：上北、右东。"""
    return east_north_mask.T[::-1, :]


def resize_for_view(image, max_side=1200):
    scale = min(max_side / max(image.shape[:2]), 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )


def add_map_labels(image, title, north, east):
    canvas = cv2.copyMakeBorder(image, 52, 36, 75, 18, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (12, 28), font, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "N +", (8, 72), font, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "E +", (canvas.shape[1] - 62, canvas.shape[0] - 10),
                font, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"N [{north[0]:.0f}, {north[-1]:.0f}] m",
                (84, canvas.shape[0] - 10), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"E [{east[0]:.0f}, {east[-1]:.0f}] m",
                (8, canvas.shape[0] - 29), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def save_slice_montage(grid, north, east, altitude, requested_altitudes, output_path):
    panels = []
    for requested in requested_altitudes:
        k = int(np.argmin(np.abs(altitude - requested)))
        occupied = as_map_image(grid[:, k, :])
        panel = np.full((*occupied.shape, 3), 245, dtype=np.uint8)
        panel[occupied] = (30, 30, 30)
        panel = resize_for_view(panel, max_side=480)
        panel = add_map_labels(
            panel,
            f"altitude {altitude[k]:.1f} m | occupied {int(occupied.sum())}",
            north,
            east,
        )
        panels.append(panel)

    columns = 2
    rows = math.ceil(len(panels) / columns)
    height = max(panel.shape[0] for panel in panels)
    width = max(panel.shape[1] for panel in panels)
    montage = np.full((rows * height, columns * width, 3), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        montage[row * height:row * height + panel.shape[0],
                column * width:column * width + panel.shape[1]] = panel
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"无法写入图像: {output_path}")


def save_obstacle_projection(grid, north, east, altitude, min_altitude_m, output_path):
    """保存每个水平位置的最高占用高度；无障碍处为白色。"""
    selected = np.flatnonzero(altitude >= min_altitude_m)
    if selected.size == 0:
        raise ValueError("投影最低高度超出体素地图范围")

    subgrid = grid[:, selected, :]
    has_obstacle = subgrid.any(axis=1)
    heights = np.full(has_obstacle.shape, np.nan, dtype=np.float32)
    for local_k, world_k in enumerate(selected):
        heights[subgrid[:, local_k, :]] = altitude[world_k]

    map_heights = as_map_image(heights)
    map_mask = as_map_image(has_obstacle)
    valid = map_heights[map_mask]
    if valid.size:
        low, high = float(valid.min()), float(valid.max())
        span = max(high - low, 1e-6)
        normalized = np.nan_to_num((map_heights - low) / span, nan=0.0)
        colors = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    else:
        low = high = float("nan")
        colors = np.zeros((*map_heights.shape, 3), dtype=np.uint8)
    colors[~map_mask] = (245, 245, 245)
    colors = resize_for_view(colors)
    title = f"highest occupied altitude >= {min_altitude_m:.1f} m"
    if valid.size:
        title += f" | range [{low:.1f}, {high:.1f}] m"
    colors = add_map_labels(colors, title, north, east)
    if not cv2.imwrite(str(output_path), colors):
        raise RuntimeError(f"无法写入图像: {output_path}")


def write_slice_summary(grid, altitude, output_path):
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["altitude_m", "occupied_voxels", "horizontal_coverage_percent"])
        horizontal_cells = grid.shape[0] * grid.shape[2]
        for k, value in enumerate(altitude):
            occupied = int(grid[:, k, :].sum())
            writer.writerow([f"{value:.3f}", occupied, f"{100.0 * occupied / horizontal_cells:.6f}"])


def target_neighborhood_check(grid, north, east, altitude, target_ned, radius_cells=1):
    target = (target_ned[0], target_ned[1], -target_ned[2])
    i = int(np.argmin(np.abs(north - target[0])))
    j = int(np.argmin(np.abs(east - target[1])))
    k = int(np.argmin(np.abs(altitude - target[2])))
    x0, x1 = max(0, i - radius_cells), min(len(north), i + radius_cells + 1)
    y0, y1 = max(0, j - radius_cells), min(len(east), j + radius_cells + 1)
    z0, z1 = max(0, k - radius_cells), min(len(altitude), k + radius_cells + 1)
    return {
        "target_ned_m": list(target_ned),
        "nearest_cell_ned_m": [float(north[i]), float(east[j]), float(-altitude[k])],
        "nearest_cell_occupied": bool(grid[j, k, i]),
        "occupied_in_3x3x3": int(grid[y0:y1, z0:z1, x0:x1].sum()),
    }


def main(args):
    profile = PROFILES[args.profile]
    center_ned = tuple(args.center_ned or profile["center_ned"])
    size_m = tuple(args.size_m or profile["size_m"])
    resolution_m = float(args.resolution_m or profile["resolution_m"])

    counts = tuple(int(round(length / resolution_m)) for length in size_m)
    if any(not math.isclose(count * resolution_m, length, abs_tol=1e-6)
           for count, length in zip(counts, size_m)):
        raise ValueError("长、宽、高必须分别是分辨率的整数倍")
    nx, ny, nz = counts
    voxel_count = nx * ny * nz
    if voxel_count > args.max_voxels:
        raise ValueError(
            f"请求 {voxel_count:,} 个体素，超过 --max-voxels={args.max_voxels:,}；"
            "请提高分辨率或缩小地图"
        )

    output_dir = (args.output_dir or DEFAULT_OUTPUT_ROOT / args.profile).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空: {output_dir}；如需覆盖请加 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"配置: profile={args.profile}, center NED={center_ned}")
    print(f"范围: N={size_m[0]:.1f}m, E={size_m[1]:.1f}m, Z={size_m[2]:.1f}m")
    print(f"分辨率: {resolution_m:.1f}m；体素: {nx} x {ny} x {nz} = {voxel_count:,}")

    client = projectairsim.ProjectAirSimClient(address=args.address)
    started = time.monotonic()
    try:
        client.connect()
        print("已连接 ProjectAirSim")
        world = World(
            client,
            args.scene,
            sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
            delay_after_load_sec=2,
        )
        print(f"场景已加载: {args.scene}；开始计算体素地图……")
        flat = world.create_voxel_grid(
            make_pose(center_ned),
            size_m[0],
            size_m[1],
            size_m[2],
            resolution_m,
            actors_to_ignore=args.ignore_actor,
        )
    finally:
        client.disconnect()
        print("已断开连接")

    elapsed = time.monotonic() - started
    if len(flat) != voxel_count:
        raise RuntimeError(f"服务端返回 {len(flat):,} 个体素，预期 {voxel_count:,}")

    grid = np.asarray(flat, dtype=np.bool_).reshape(ny, nz, nx)
    north, east, altitude = cell_centers(center_ned, grid.shape, resolution_m)
    occupied_count = int(grid.sum())
    target_check = target_neighborhood_check(grid, north, east, altitude, tuple(args.target_ned))

    np.savez_compressed(
        output_dir / "occupancy_packed.npz",
        packed=np.packbits(grid.reshape(-1), bitorder="little"),
        shape=np.asarray(grid.shape, dtype=np.int32),
        center_ned_m=np.asarray(center_ned, dtype=np.float64),
        size_m=np.asarray(size_m, dtype=np.float64),
        resolution_m=np.asarray(resolution_m, dtype=np.float64),
        bitorder=np.asarray("little"),
    )
    write_slice_summary(grid, altitude, output_dir / "slice_summary.csv")
    save_slice_montage(
        grid, north, east, altitude,
        profile["slice_altitudes_m"],
        output_dir / "altitude_slices.png",
    )
    save_obstacle_projection(
        grid, north, east, altitude,
        profile["projection_min_altitude_m"],
        output_dir / "obstacle_projection.png",
    )

    metadata = {
        "scene": args.scene,
        "profile": args.profile,
        "coordinate_frame": "NED; image top=north+, image right=east+",
        "array_layout": "(east_index, altitude_index, north_index)",
        "center_ned_m": list(center_ned),
        "size_m": list(size_m),
        "resolution_m": resolution_m,
        "shape_east_altitude_north": list(grid.shape),
        "voxel_count": voxel_count,
        "occupied_voxel_count": occupied_count,
        "occupied_percent": 100.0 * occupied_count / voxel_count,
        "generation_seconds": elapsed,
        "ignored_actors": args.ignore_actor,
        "target_sanity_check": target_check,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"完成: 占用 {occupied_count:,}/{voxel_count:,} ({metadata['occupied_percent']:.4f}%)")
    print(f"目标附近 3x3x3 占用数: {target_check['occupied_in_3x3x3']}")
    print(f"耗时: {elapsed:.1f}s")
    print(f"输出: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="生成 airbase 场景体素地图和诊断图")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="global")
    parser.add_argument("--address", default="172.21.192.1")
    parser.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--scene", default="scene_airbase.jsonc")
    parser.add_argument("--center-ned", type=float, nargs=3, metavar=("N", "E", "D"))
    parser.add_argument("--size-m", type=float, nargs=3, metavar=("N", "E", "Z"))
    parser.add_argument("--resolution-m", type=float)
    parser.add_argument("--target-ned", type=float, nargs=3,
                        default=(-62.71, -16.68, -2.94), metavar=("N", "E", "D"))
    parser.add_argument("--ignore-actor", action="append", default=["Drone1"])
    parser.add_argument("--max-voxels", type=int, default=10_000_000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
