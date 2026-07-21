"""深度 → 世界系点云：给闭环实时建图用。

深度流实测格式(见 probe_depth.py)：encoding=16UC1，uint16 单通道，单位=毫米，
天空/超量程为哨兵值 ~65504。这里负责：
  1) pub/sub 缓存最新深度帧（和 collect_episode 的 image_callback 同套路）；
  2) decode_depth: uint16 → 米 + 有效掩码（丢 0/天空/超量程）；
  3) backproject: 深度(perspective=沿射线距离) + 内参(fov90→fx=fy=w/2) → 相机系 3D 点 + 颜色；
  4) cam_to_world_ned: 相机系 → 机体 FRD(含 0.5m 前向安装偏移) → 世界 NED。

坐标约定：机体 FRD(x前 y右 z下)，世界 NED。光学系(x右 y下 z前)→机体 = (z,x,y)。
"""
import threading

import numpy as np

SKY_RAW = 65000        # >= 此原始值视为天空/超量程，丢弃（实测哨兵 65504）
CAM_OFFSET_BODY = np.array([0.5, 0.0, 0.0], np.float32)  # 相机相对机体前向 0.5m（config origin）

_lock = threading.Lock()
_depth_latest = None


def depth_callback(topic, msg):
    global _depth_latest
    with _lock:
        _depth_latest = msg


def get_depth():
    with _lock:
        return _depth_latest


def reset_depth():
    global _depth_latest
    with _lock:
        _depth_latest = None


def decode_depth(msg):
    """16UC1 深度 msg → (depth_m HxW float32, valid HxW bool)。单位毫米→米。"""
    h, w = msg["height"], msg["width"]
    data = msg["data"]
    raw = (np.frombuffer(data, dtype=np.uint16)
           if isinstance(data, (bytes, bytearray, memoryview))
           else np.asarray(data, dtype=np.uint16)).reshape(h, w)
    depth_m = raw.astype(np.float32) / 1000.0
    valid = (raw > 0) & (raw < SKY_RAW)
    return depth_m, valid


def backproject(depth_m, valid, rgb_full, stride, max_range, fov_deg=90.0):
    """深度 → 相机系点云。返回 (pts_cam Nx3 float32, colors Nx3 uint8)。

    perspective 深度 = 光心到点的射线距离，故 point = depth * 单位射线方向。
    rgb_full: 与深度同分辨率同 FOV 的整帧 RGB(HxWx3)，按同像素采色。
    """
    h, w = depth_m.shape
    fx = fy = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0

    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)                      # (Hs, Ws)
    d = depth_m[vv, uu]
    m = valid[vv, uu] & (d <= max_range)
    uu, vv, d = uu[m], vv[m], d[m]

    x = (uu - cx) / fx                                # 光学系 x(右)
    y = (vv - cy) / fy                                # 光学系 y(下)
    dirs = np.stack([x, y, np.ones_like(x)], axis=1)  # (N,3) 光学 [右,下,前]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts_opt = dirs * d[:, None]                       # perspective：沿射线

    # 光学(右,下,前) → 机体 FRD(前,右,下) = (z,x,y)，再加相机安装偏移
    pts_cam = pts_opt[:, [2, 0, 1]] + CAM_OFFSET_BODY
    colors = rgb_full[vv, uu].astype(np.uint8)
    return pts_cam.astype(np.float32), colors, uu, vv


def _quat_to_R(q):
    """机体→世界 旋转矩阵，q=(w,x,y,z)。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], np.float32)


def cam_to_world_ned(pts_cam, pose7):
    """机体 FRD 点 → 世界 NED 点。pose7 = [pos_ned(3), quat_wxyz(4)]。"""
    pos = np.asarray(pose7[:3], np.float32)
    R = _quat_to_R(np.asarray(pose7[3:7], np.float32))
    return (pts_cam @ R.T) + pos


def frame_to_world(depth_msg, rgb_full, pose7, stride, max_range, fov_deg=90.0):
    """一步到位：深度帧 + 整帧RGB + 位姿 → (世界NED点 Nx3 float32, 颜色 Nx3 uint8)。"""
    depth_m, valid = decode_depth(depth_msg)
    pts_cam, colors, _, _ = backproject(depth_m, valid, rgb_full, stride, max_range, fov_deg)
    return cam_to_world_ned(pts_cam, pose7), colors


def frame_to_world_tagged(depth_msg, rgb_full, pose7, stride, max_range,
                          sim_grid, grid, tgt_thresh, fov_deg=90.0):
    """同 frame_to_world，但额外用指纹响应图 sim_grid(grid×grid) 给每个点打"是否目标"标签。

    每点像素 (u,v) → 归一化 → 对应 patch → sim 响应；>tgt_thresh 即判为目标点。
    返回 (世界NED点 Nx3, 颜色 Nx3, is_target 布尔 N)。
    """
    depth_m, valid = decode_depth(depth_msg)
    pts_cam, colors, uu, vv = backproject(depth_m, valid, rgb_full, stride, max_range, fov_deg)
    pts_world = cam_to_world_ned(pts_cam, pose7)
    h, w = depth_m.shape
    col = np.clip((uu.astype(np.float64) / w * grid).astype(int), 0, grid - 1)
    row = np.clip((vv.astype(np.float64) / h * grid).astype(int), 0, grid - 1)
    is_target = sim_grid[row, col] > tgt_thresh
    return pts_world, colors, is_target
