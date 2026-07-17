"""把 --map 落盘的点云(step_*.npz 里的 pts/cols)离线重建成连续表面网格。

流程：读一个 dump 目录所有 step_*.npz → 合并世界点云(NED→ENU, z朝上) → 体素降采样
+ 统计离群点剔除 → 估法向 → 表面重建(泊松/球旋转) → 存 .ply(点云 + 网格)。
rviz 只能显示点，连续曲面得靠这种离线重建(Meshlab/Open3D 里看)。

依赖：pip install open3d   （纯 CPU 可跑，不碰仿真/不吃 GPU）

用法：
  python src/mpc/pointcloud_to_mesh.py --dump-dir ~/mnt/server_runs/map02
  # 默认输出 outputs/meshes/{cloud.ply, mesh.ply}
  # --method bpa 用球旋转(开放场景不易过度封闭); poisson(默认)更平滑但会补洞
  # --show 直接弹窗预览(需要本地有显示)
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "meshes"


def load_cloud(dump_dir):
    """读所有 step_*.npz 的 pts(世界NED)/cols，合并并转 ENU(z朝上)。"""
    files = sorted(Path(dump_dir).glob("step_*.npz"))
    if not files:
        raise SystemExit(f"没有 step_*.npz: {dump_dir}")
    P, C = [], []
    for f in files:
        d = np.load(f)
        if "pts" not in d.files:            # 非 --map 的 dump 没有点云
            continue
        P.append(np.asarray(d["pts"], np.float64))
        C.append(np.asarray(d["cols"], np.float64))
    if not P:
        raise SystemExit(f"这些 npz 里没有 pts（跑闭环时没加 --map？）: {dump_dir}")
    pts = np.concatenate(P, 0)
    cols = np.concatenate(C, 0) / 255.0
    # NED(北,东,下) → ENU(东,北,上)，与 rviz/plot 一致，且让 z 朝上(泊松需一致的上方向)
    enu = np.stack([pts[:, 1], pts[:, 0], -pts[:, 2]], axis=1)
    return enu, cols


def build_pcd(enu, cols, voxel):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(enu)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    n0 = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"点数: {n0} → 降采样+去噪后 {len(pcd.points)}")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def recon_poisson(pcd, depth, density_thresh):
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    # 剔除低密度顶点(泊松在没数据处会"吹气球"补面，切掉这些幻影)
    keep = dens > np.quantile(dens, density_thresh)
    mesh.remove_vertices_by_mask(~keep)
    return mesh


def recon_bpa(pcd):
    d = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg = float(d.mean())
    radii = o3d.utility.DoubleVector([avg * 1.5, avg * 3.0, avg * 6.0])
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="含 step_*.npz 的目录(--map 跑出来的)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--voxel", type=float, default=0.05, help="体素降采样尺寸(米)")
    ap.add_argument("--method", choices=["poisson", "bpa"], default="poisson",
                    help="poisson=平滑补洞; bpa=球旋转,开放场景不易过度封闭")
    ap.add_argument("--depth", type=int, default=9, help="泊松八叉树深度(越大越细也越慢)")
    ap.add_argument("--density-thresh", type=float, default=0.05,
                    help="泊松低密度顶点剔除分位(切掉补出来的幻影面)")
    ap.add_argument("--show", action="store_true", help="弹窗预览(需本地显示)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    enu, cols = load_cloud(args.dump_dir)
    pcd = build_pcd(enu, cols, args.voxel)
    o3d.io.write_point_cloud(str(out / "cloud.ply"), pcd)
    print(f"已存点云: {out / 'cloud.ply'}")

    mesh = recon_poisson(pcd, args.depth, args.density_thresh) if args.method == "poisson" \
        else recon_bpa(pcd)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)
    print(f"已存网格: {out / 'mesh.ply'}  "
          f"({len(mesh.vertices)} 顶点, {len(mesh.triangles)} 面, method={args.method})")

    if args.show:
        o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
