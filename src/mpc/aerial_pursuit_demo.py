#!/usr/bin/env python3
"""
MPPI 空中跟踪演示：无人机悬停接近一个移动的地面目标。

场景：地面目标沿一条已知的 8 字曲线（Lissajous）在地面 (z=0) 上移动，
一架无人机（3D 质点，二阶积分：位置+速度状态，加速度控制）用 MPPI
持续飞到目标正上方固定高度悬停跟踪。

MPPI 每步：采样 K 条加速度序列 → 前向 rollout 无人机轨迹 →
按"到期望悬停点的距离 + 控制能耗"代价加权 → 得最优加速度并执行第一步。
期望悬停点 = 对地面目标做匀速外推预测后的位置 + 悬停高度偏移，
体现 MPC 的"预测目标未来 + 滚动重规划"。

rviz 可视化（fixed frame = world）：
  - 半透明地面、移动的地面目标（红盒）、期望悬停点（黄球）
  - 无人机（蓝球）+ 一条竖直虚线连到地面投影，便于在 3D 里感知高度
  - 采样轨迹束（灰细线）、最优轨迹（绿粗线）
  - 无人机飞行路径 / 地面目标轨迹（两条 Path）

运行：
  ros2 launch src/mpc/aerial_pursuit.launch.py
或单独：
  python3 src/mpc/aerial_pursuit_demo.py
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


FRAME = 'world'


class MPPIAerial:
    """3D 二阶积分无人机的 MPPI 控制器。

    状态 x = [px, py, pz, vx, vy, vz]，控制 u = [ax, ay, az]。
    """

    def __init__(self):
        # ---- MPPI 超参数 ----
        self.K = 500          # 采样数
        self.T = 25           # 预测步数
        self.dt = 0.1         # 步长 (s)
        self.lam = 1.0        # 温度 lambda
        self.sigma = 2.0      # 加速度采样噪声标准差 (m/s^2)

        # 限幅
        self.a_max = 4.0      # 最大加速度
        self.v_max = 4.0      # 最大速度

        # 代价权重
        self.w_track = 3.0    # 过程：到期望悬停点的距离
        self.w_term = 30.0    # 终端：末端到期望点的距离
        self.w_ctrl = 0.02    # 控制能耗

        # 名义控制序列 U: (T, 3)
        self.U = np.zeros((self.T, 3))

    def _rollout(self, state, A):
        """向量化 rollout。A: (K, T, 3) -> 位置轨迹 (K, T+1, 3)。"""
        K = A.shape[0]
        p = np.tile(state[:3].astype(float), (K, 1))
        v = np.tile(state[3:].astype(float), (K, 1))
        pos = np.empty((K, self.T + 1, 3))
        pos[:, 0] = p
        for t in range(self.T):
            v = v + A[:, t] * self.dt
            # 限速
            sp = np.linalg.norm(v, axis=1, keepdims=True)
            scale = np.minimum(1.0, self.v_max / (sp + 1e-9))
            v = v * scale
            p = p + v * self.dt
            pos[:, t + 1] = p
        return pos

    def _cost(self, pos, ref):
        """代价。pos: (K,T+1,3)，ref: (T,3) 每步的期望悬停点。"""
        pts = pos[:, 1:]                                   # (K,T,3)
        d = np.linalg.norm(pts - ref[None], axis=2)        # (K,T)
        cost = self.w_track * d.sum(axis=1)
        cost += self.w_term * d[:, -1]
        return cost

    def step(self, state, ref):
        """跑一次 MPPI。ref:(T,3) 期望悬停点序列。返回 (u, 采样轨迹, 最优轨迹)。"""
        noise = np.random.randn(self.K, self.T, 3) * self.sigma
        A = self.U[None] + noise
        # 加速度限幅（按模长）
        amag = np.linalg.norm(A, axis=2, keepdims=True)
        A = A * np.minimum(1.0, self.a_max / (amag + 1e-9))

        pos = self._rollout(state, A)
        cost = self._cost(pos, ref)
        cost += self.w_ctrl * (A ** 2).sum(axis=(1, 2))

        beta = cost.min()
        w = np.exp(-(cost - beta) / self.lam)
        w /= w.sum() + 1e-9

        self.U = np.einsum('k,ktu->tu', w, A)

        u = self.U[0].copy()
        best = self._rollout(state, self.U[None])[0]

        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0
        return u, pos, best


def target_state(t):
    """地面目标在时刻 t 的位置和速度（8 字 Lissajous 曲线，z=0）。"""
    ax, ay = 6.0, 4.0
    w1, w2 = 0.25, 0.5
    px = ax * np.sin(w1 * t)
    py = ay * np.sin(w2 * t)
    vx = ax * w1 * np.cos(w1 * t)
    vy = ay * w2 * np.cos(w2 * t)
    return np.array([px, py, 0.0]), np.array([vx, vy, 0.0])


class AerialPursuitNode(Node):
    def __init__(self):
        super().__init__('aerial_pursuit_demo')

        self.mppi = MPPIAerial()
        self.hover_h = 2.5                       # 悬停在目标上方的高度
        self.sim_t = 0.0

        # 无人机初始状态：远处高空（着重体现从高空下降接近地面目标）
        self.state = np.array([-10.0, 10.0, 18.0, 0.0, 0.0, 0.0])

        self.drone_path = Path(); self.drone_path.header.frame_id = FRAME
        self.target_path = Path(); self.target_path.header.frame_id = FRAME

        self.marker_pub = self.create_publisher(MarkerArray, 'aerial/markers', 10)
        self.drone_path_pub = self.create_publisher(Path, 'aerial/drone_path', 10)
        self.target_path_pub = self.create_publisher(Path, 'aerial/target_path', 10)

        self.timer = self.create_timer(0.1, self.update)   # 10 Hz
        self.get_logger().info('Aerial pursuit demo 启动，rviz fixed frame 设为 world')

    def _ref_sequence(self):
        """基于当前观测的目标位置+速度，匀速外推出未来 T 步的期望悬停点。"""
        p0, v0 = target_state(self.sim_t)
        offset = np.array([0.0, 0.0, self.hover_h])
        ref = np.empty((self.mppi.T, 3))
        for k in range(self.mppi.T):
            tk = (k + 1) * self.mppi.dt
            ref[k] = p0 + v0 * tk + offset       # 匀速预测 + 悬停高度
        return ref, p0

    def update(self):
        now = self.get_clock().now().to_msg()

        ref, tgt_pos = self._ref_sequence()
        u, samples, best = self.mppi.step(self.state, ref)

        # 二阶积分推进无人机真实状态
        v = self.state[3:] + u * self.mppi.dt
        sp = np.linalg.norm(v)
        if sp > self.mppi.v_max:
            v *= self.mppi.v_max / sp
        self.state[3:] = v
        self.state[:3] += v * self.mppi.dt

        self.sim_t += self.mppi.dt

        self._append_path(self.drone_path, now, self.state[:3])
        self._append_path(self.target_path, now, tgt_pos)

        self.drone_path.header.stamp = now
        self.target_path.header.stamp = now
        self.drone_path_pub.publish(self.drone_path)
        self.target_path_pub.publish(self.target_path)
        self.publish_markers(now, samples, best, tgt_pos, ref[0])

    def _append_path(self, path, stamp, p):
        ps = PoseStamped()
        ps.header.frame_id = FRAME
        ps.header.stamp = stamp
        ps.pose.position.x = float(p[0])
        ps.pose.position.y = float(p[1])
        ps.pose.position.z = float(p[2])
        path.poses.append(ps)
        if len(path.poses) > 800:
            path.poses[:] = path.poses[-800:]

    # ---------- 可视化 ----------

    def publish_markers(self, stamp, samples, best, tgt_pos, hover_pt):
        arr = MarkerArray()

        # 半透明地面
        g = self._base(stamp, 'ground', 0, Marker.CUBE)
        g.pose.position.z = -0.02
        g.scale.x = g.scale.y = 20.0
        g.scale.z = 0.02
        g.color = ColorRGBA(r=0.3, g=0.3, b=0.35, a=0.25)
        arr.markers.append(g)

        # 地面目标（红盒）
        t = self._base(stamp, 'target', 0, Marker.CUBE)
        t.pose.position.x = float(tgt_pos[0])
        t.pose.position.y = float(tgt_pos[1])
        t.pose.position.z = 0.2
        t.scale.x = t.scale.y = t.scale.z = 0.6
        t.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
        arr.markers.append(t)

        # 期望悬停点（黄球）
        h = self._base(stamp, 'hover', 0, Marker.SPHERE)
        h.pose.position.x = float(hover_pt[0])
        h.pose.position.y = float(hover_pt[1])
        h.pose.position.z = float(hover_pt[2])
        h.scale.x = h.scale.y = h.scale.z = 0.35
        h.color = ColorRGBA(r=1.0, g=0.9, b=0.1, a=0.9)
        arr.markers.append(h)

        # 无人机（蓝球）
        d = self._base(stamp, 'drone', 0, Marker.SPHERE)
        d.pose.position.x = float(self.state[0])
        d.pose.position.y = float(self.state[1])
        d.pose.position.z = float(self.state[2])
        d.scale.x = d.scale.y = d.scale.z = 0.5
        d.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=1.0)
        arr.markers.append(d)

        # 无人机到地面投影的竖直虚线（帮助在 3D 里判断高度）
        line = self._base(stamp, 'altitude', 0, Marker.LINE_LIST)
        line.scale.x = 0.02
        line.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.5)
        line.points = [
            Point(x=float(self.state[0]), y=float(self.state[1]), z=float(self.state[2])),
            Point(x=float(self.state[0]), y=float(self.state[1]), z=0.0),
        ]
        arr.markers.append(line)

        # 采样轨迹束（灰细线）
        s = self._base(stamp, 'samples', 0, Marker.LINE_LIST)
        s.scale.x = 0.01
        s.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.2)
        n_show = min(100, samples.shape[0])
        for k in range(n_show):
            tr = samples[k]
            for i in range(tr.shape[0] - 1):
                s.points.append(Point(x=float(tr[i, 0]), y=float(tr[i, 1]), z=float(tr[i, 2])))
                s.points.append(Point(x=float(tr[i + 1, 0]), y=float(tr[i + 1, 1]), z=float(tr[i + 1, 2])))
        arr.markers.append(s)

        # 最优轨迹（绿粗线）
        b = self._base(stamp, 'best', 0, Marker.LINE_STRIP)
        b.scale.x = 0.06
        b.color = ColorRGBA(r=0.1, g=1.0, b=0.2, a=0.9)
        b.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in best]
        arr.markers.append(b)

        self.marker_pub.publish(arr)

    def _base(self, stamp, ns, mid, mtype):
        m = Marker()
        m.header.frame_id = FRAME
        m.header.stamp = stamp
        m.ns = ns
        m.id = int(mid)
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m


def main():
    rclpy.init()
    node = AerialPursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
