#!/usr/bin/env python3
"""闭环规划过程的 rviz 可视化节点。

读取 plan_closed_loop.py --viz-dump 落盘的 step_*.npz，发布到 rviz：
  - /plan/markers  MarkerArray：候选动作轨迹束（按 cost 红→绿染色）、最优序列（粗绿线）、
                   无人机（蓝球）、目标圆环（绿圈，位置用 --goal-ned/--goal-alt 指定）
  - /plan/traj     Path：无人机已飞路径
  - /plan/view     Image：机载画面 + 指纹响应热力图叠加（红=响应强）——"它看到了什么"

坐标：NED → rviz (x=N, y=E, z=-D)，fixed frame = world。
候选轨迹由动作序列（NED 速度指令）从当前位置积分得到（近似，不含动力学）。

用法：
  实时：闭环加 --viz-dump outputs/runs/mppi/run01，另开终端跑本节点（会跟播新文件）
  回放：闭环跑完后再跑本节点即可，--rate 控制回放速度
  ros2 launch 一步到位：ros2 launch src/mpc/plan_viz.launch.py  (dump_dir:=...)
"""

import argparse
import time
from pathlib import Path as FsPath

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

FRAME = 'world'
PROJECT_ROOT = FsPath(__file__).resolve().parents[2]


def ned_to_viz(p):
    """NED → ENU (x=东, y=北, z=上)。注意不能用 (N,E,-D)：那是左手系，画面会左右镜像。"""
    return float(p[1]), float(p[0]), float(-p[2])


def jet(t):
    """t∈[0,1] → (r,g,b) jet 彩虹：低=蓝，中=绿黄，高=红。给体素按高度上色。"""
    t = min(max(t, 0.0), 1.0)
    r = min(max(1.5 - abs(4 * t - 3), 0.0), 1.0)
    g = min(max(1.5 - abs(4 * t - 2), 0.0), 1.0)
    b = min(max(1.5 - abs(4 * t - 1), 0.0), 1.0)
    return r, g, b


class PlanVizNode(Node):
    def __init__(self, args):
        super().__init__('plan_viz')
        self.args = args
        self.dump = FsPath(args.dump_dir)
        self.seen = set()

        self.marker_pub = self.create_publisher(MarkerArray, 'plan/markers', 10)
        self.path_pub = self.create_publisher(Path, 'plan/traj', 10)
        self.img_pub = self.create_publisher(Image, 'plan/view', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, 'plan/cloud', 1)
        self.voxel_pub = self.create_publisher(Marker, 'plan/voxels', 1)
        self.target_pub = self.create_publisher(Marker, 'plan/target_voxels', 1)

        self.traj = Path()
        self.traj.header.frame_id = FRAME

        # 实时建图：累加各步点云，按体素去重（键=量化坐标），避免无限膨胀
        self.cx, self.cy, self.cz, self.crgb = [], [], [], []
        self.vox = set()
        self.tx, self.ty, self.tz, self.tvox = [], [], [], set()  # 目标高亮体素(ENU)

        self.timer = self.create_timer(1.0 / args.rate, self.tick)
        self.get_logger().info(f'监听 {self.dump}（{args.rate} 步/秒）')

    def tick(self):
        files = sorted(self.dump.glob('step_*.npz'))
        for f in files:
            if f.name in self.seen:
                continue
            self.seen.add(f.name)
            try:
                d = np.load(f)
            except Exception:  # 写入未完成，下个周期再试
                self.seen.discard(f.name)
                return
            self.publish_step(d)
            return  # 每个周期只播一步，rate 即回放速度

    def publish_step(self, d):
        now = self.get_clock().now().to_msg()
        pos = d['pose'][:3]
        dt = float(d['dt']) if 'dt' in d.files else 0.1

        # 路径
        ps = PoseStamped()
        ps.header.frame_id = FRAME
        ps.header.stamp = now
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = ned_to_viz(pos)
        self.traj.poses.append(ps)
        self.traj.header.stamp = now
        self.path_pub.publish(self.traj)

        arr = MarkerArray()
        arr.markers.append(self.drone_marker(now, pos))
        if 'samp_acts' in d.files:          # 规划(闭环)dump：画目标环/采样束/最优线/dist
            arr.markers.append(self.goal_ring(now))
            arr.markers.append(self.samples_marker(now, pos, d['samp_acts'], d['samp_cost'], dt))
            arr.markers.append(self.best_marker(now, pos, d['act_best'], dt))
            arr.markers.append(self.text_marker(now, pos,
                f"step {int(d['step'])}  dist={float(d['dist']):.3f}  best={float(d['best']):.3f}"))
        else:                               # 纯建图 dump：只标无人机 + 步数
            arr.markers.append(self.text_marker(now, pos, f"step {int(d['step'])}  (mapping)"))
        self.marker_pub.publish(arr)

        if 'rgb' in d.files:
            self.img_pub.publish(self.view_image(now, d['rgb'], d['sim'], int(d['grid'])))

        if 'pts' in d.files:            # 实时建图：本步点云累加后重发整片
            self.add_cloud(np.asarray(d['pts']), np.asarray(d['cols']))
            if self.cx:
                self.cloud_pub.publish(self.make_cloud(now))
                self.voxel_pub.publish(self.make_voxels(now))
        if 'tgt_pts' in d.files:        # 目标高亮：把目标点累加成醒目方块
            self.add_target(np.asarray(d['tgt_pts']))
        if self.tx:
            self.target_pub.publish(self.make_target_voxels(now))

    # ---------- markers ----------

    def _base(self, stamp, ns, mid, mtype):
        m = Marker()
        m.header.frame_id = FRAME
        m.header.stamp = stamp
        m.ns, m.id, m.type, m.action = ns, int(mid), mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    @staticmethod
    def integrate(pos, acts, dt):
        """动作序列(H,4: vN,vE,vD,yaw_rate) → 轨迹点 (H+1,3) NED。"""
        v = acts[:, :3]
        pts = np.vstack([pos[None], pos[None] + np.cumsum(v * dt, axis=0)])
        return pts

    def samples_marker(self, stamp, pos, samp_acts, samp_cost, dt):
        m = self._base(stamp, 'samples', 0, Marker.LINE_LIST)
        m.scale.x = 0.02
        c = np.asarray(samp_cost, dtype=np.float64)
        cn = (c - c.min()) / (c.max() - c.min() + 1e-9)   # 0=最好 1=最差
        for k in range(samp_acts.shape[0]):
            pts = self.integrate(pos, samp_acts[k], dt)
            col = ColorRGBA(r=float(cn[k]), g=float(1 - cn[k]), b=0.1, a=0.35)
            for i in range(len(pts) - 1):
                for p in (pts[i], pts[i + 1]):
                    x, y, z = ned_to_viz(p)
                    m.points.append(Point(x=x, y=y, z=z))
                    m.colors.append(col)
        return m

    def best_marker(self, stamp, pos, act_best, dt):
        m = self._base(stamp, 'best', 0, Marker.LINE_STRIP)
        m.scale.x = 0.08
        m.color = ColorRGBA(r=0.1, g=1.0, b=0.3, a=1.0)
        for p in self.integrate(pos, act_best, dt):
            x, y, z = ned_to_viz(p)
            m.points.append(Point(x=x, y=y, z=z))
        return m

    def drone_marker(self, stamp, pos):
        m = self._base(stamp, 'drone', 0, Marker.SPHERE)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = ned_to_viz(pos)
        m.scale.x = m.scale.y = m.scale.z = 0.6
        m.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=1.0)
        return m

    def goal_ring(self, stamp):
        """目标圆环：竖直平面(法线朝 -N)上的圆。"""
        gn, ge = self.args.goal_ned
        alt, r = self.args.goal_alt, self.args.goal_radius
        m = self._base(stamp, 'goal', 0, Marker.LINE_STRIP)
        m.scale.x = 0.15
        m.color = ColorRGBA(r=0.2, g=0.9, b=0.3, a=0.9)
        # ENU：环平面法线朝北(+y)，圆在 x(东)-z(上) 平面展开
        for th in np.linspace(0, 2 * np.pi, 40):
            m.points.append(Point(x=float(ge + r * np.cos(th)), y=float(gn),
                                  z=float(alt + r * np.sin(th))))
        return m

    def text_marker(self, stamp, pos, text):
        m = self._base(stamp, 'info', 0, Marker.TEXT_VIEW_FACING)
        x, y, z = ned_to_viz(pos)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z + 1.5
        m.scale.z = 0.6
        m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
        m.text = text
        return m

    # ---------- 机载视图 + 响应热力图 ----------

    def view_image(self, stamp, rgb, sim, grid):
        img = np.asarray(rgb, dtype=np.float32)
        h, w = img.shape[:2]
        if sim.size == grid * grid:
            heat = sim.reshape(grid, grid).astype(np.float32)
            # 绝对刻度：<0.2(mass 阈值)不显色，0.6 全红。
            # 不用 min-max：全图弱噪声会被拉成满屏红，造成"到处都匹配"的假象。
            heat = np.clip((heat - 0.2) / 0.4, 0.0, 1.0)
            heat = np.kron(heat, np.ones((h // grid, w // grid), np.float32))  # 上采样
            # 红色叠加：响应越强越红
            img[..., 0] = img[..., 0] * (1 - 0.6 * heat) + 255 * 0.6 * heat
            img[..., 1] *= (1 - 0.5 * heat)
            img[..., 2] *= (1 - 0.5 * heat)
        msg = Image()
        msg.header.frame_id = FRAME
        msg.header.stamp = stamp
        msg.height, msg.width = h, w
        msg.encoding = 'rgb8'
        msg.step = w * 3
        msg.data = img.clip(0, 255).astype(np.uint8).tobytes()
        return msg

    # ---------- 实时建图点云 ----------

    def add_cloud(self, pts_ned, cols):
        """把本步世界NED点(Nx3)转 ENU、体素去重后追加。cols: Nx3 uint8。向量化,支持高密度。"""
        if pts_ned.size == 0:
            return
        # NED → ENU：x=东(E)=p[1], y=北(N)=p[0], z=上=-D=-p[2]（与 ned_to_viz 一致）
        enu = np.stack([pts_ned[:, 1], pts_ned[:, 0], -pts_ned[:, 2]], axis=1)
        keys = np.floor(enu / self.args.map_voxel).astype(np.int64)
        off = 1 << 20                                    # 平移保证非负; 每维 21bit 打包进 int64
        packed = (((keys[:, 0] + off) << 42) |
                  ((keys[:, 1] + off) << 21) | (keys[:, 2] + off))
        uniq, idx = np.unique(packed, return_index=True)  # 本帧内先去重
        novel = np.fromiter((p not in self.vox for p in uniq.tolist()),
                            dtype=bool, count=len(uniq))   # 只对帧内唯一键查历史
        if not novel.any():
            return
        sel = idx[novel]
        self.vox.update(uniq[novel].tolist())
        rgb = (cols[:, 0].astype(np.uint32) << 16 |
               cols[:, 1].astype(np.uint32) << 8 | cols[:, 2].astype(np.uint32))
        self.cx.extend(enu[sel, 0].tolist()); self.cy.extend(enu[sel, 1].tolist())
        self.cz.extend(enu[sel, 2].tolist()); self.crgb.extend(rgb[sel].tolist())

    def make_cloud(self, stamp):
        """累加的点 → PointCloud2(带 rgb)。rgb 按 rviz 惯例打包进 float32 字段。"""
        n = len(self.cx)
        arr = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
        arr['x'] = self.cx; arr['y'] = self.cy; arr['z'] = self.cz; arr['rgb'] = self.crgb
        msg = PointCloud2()
        msg.header.frame_id = FRAME
        msg.header.stamp = stamp
        msg.height, msg.width = 1, n
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n
        msg.is_dense = True
        msg.data = arr.tobytes()
        return msg

    def make_voxels(self, stamp):
        """累加的体素 → CUBE_LIST 方块地图(像#14)，格心对齐、按高度 jet 上色。"""
        v = self.args.map_voxel
        m = self._base(stamp, 'voxels', 0, Marker.CUBE_LIST)
        m.scale.x = m.scale.y = m.scale.z = v
        cz = np.asarray(self.cz)
        zmin, span = float(cz.min()), max(float(cz.max() - cz.min()), 1e-3)
        for x, y, z in zip(self.cx, self.cy, self.cz):
            m.points.append(Point(x=float((np.floor(x / v) + 0.5) * v),
                                  y=float((np.floor(y / v) + 0.5) * v),
                                  z=float((np.floor(z / v) + 0.5) * v)))
            r, g, b = jet((z - zmin) / span)
            m.colors.append(ColorRGBA(r=r, g=g, b=b, a=1.0))
        return m

    def add_target(self, pts_ned):
        """目标点(世界NED)→ENU、体素去重累加(单独一套，用于高亮)。"""
        if pts_ned.size == 0:
            return
        enu = np.stack([pts_ned[:, 1], pts_ned[:, 0], -pts_ned[:, 2]], axis=1)
        keys = np.floor(enu / self.args.map_voxel).astype(np.int64)
        off = 1 << 20
        packed = (((keys[:, 0] + off) << 42) | ((keys[:, 1] + off) << 21) | (keys[:, 2] + off))
        uniq, idx = np.unique(packed, return_index=True)
        novel = np.fromiter((p not in self.tvox for p in uniq.tolist()), dtype=bool, count=len(uniq))
        if not novel.any():
            return
        sel = idx[novel]
        self.tvox.update(uniq[novel].tolist())
        self.tx.extend(enu[sel, 0].tolist()); self.ty.extend(enu[sel, 1].tolist())
        self.tz.extend(enu[sel, 2].tolist())

    def make_target_voxels(self, stamp):
        """目标高亮方块：醒目品红、略大于环境体素，叠在地图上一眼可辨哪坨是目标。"""
        v = self.args.map_voxel
        m = self._base(stamp, 'target', 0, Marker.CUBE_LIST)
        m.scale.x = m.scale.y = m.scale.z = v * 1.2
        m.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)
        for x, y, z in zip(self.tx, self.ty, self.tz):
            m.points.append(Point(x=float((np.floor(x / v) + 0.5) * v),
                                  y=float((np.floor(y / v) + 0.5) * v),
                                  z=float((np.floor(z / v) + 0.5) * v)))
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-dir', default=str(PROJECT_ROOT / 'outputs/runs/mppi/run01'))
    ap.add_argument('--rate', type=float, default=5.0, help='回放速度（步/秒）')
    ap.add_argument('--goal-ned', type=float, nargs=2, default=[42.3, 7.5])
    ap.add_argument('--goal-alt', type=float, default=14.0)
    ap.add_argument('--goal-radius', type=float, default=4.0)
    ap.add_argument('--map-voxel', type=float, default=0.15,
                    help='建图点云体素去重尺寸(米)，越大越稀越省')
    args = ap.parse_args()

    rclpy.init()
    node = PlanVizNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
