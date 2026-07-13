#!/usr/bin/env python3
"""
MPPI（Model Predictive Path Integral）演示节点。

场景：一个差速小车（unicycle）在 2D 平面上从起点导航到目标点，
途中绕开若干圆形障碍。控制器用 MPPI：每步采样 K 条控制序列，
前向 rollout，按代价做指数加权平均，得到最优控制并执行第一步。

在 rviz 里可视化（fixed frame = world）：
  - 障碍物（绿色圆柱）、目标点（蓝色球）、小车（红色箭头，带朝向）
  - 采样出的候选轨迹束（灰色细线，MPPI 最直观的部分）
  - 加权最优轨迹（绿色粗线）
  - 已走过的路径（nav_msgs/Path）

运行：
  ros2 launch src/mpc/mppi_demo.launch.py
或单独：
  python3 src/mpc/mppi_demo.py
"""

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


FRAME = 'world'


class MPPI:
    """针对 unicycle 模型的极简 MPPI 控制器。

    状态 x = [px, py, theta]，控制 u = [v, omega]。
    """

    def __init__(self, obstacles, goal):
        # ---- MPPI 超参数 ----
        self.K = 400          # 采样数
        self.T = 30           # 预测步数（horizon）
        self.dt = 0.1         # 步长 (s)
        self.lam = 1.0        # 温度 lambda，越小越"贪心"
        self.sigma = np.array([0.4, 0.6])   # [v, omega] 采样噪声标准差

        # 控制限幅（调小一点，运动更平缓、便于录屏）
        self.v_max = 1.2
        self.omega_max = 2.0

        # 代价权重
        self.w_goal = 2.0     # 距目标的过程代价
        self.w_term = 20.0    # 距目标的终端代价
        self.w_obs = 1000.0   # 撞障碍惩罚
        self.robot_radius = 0.3

        self.obstacles = obstacles     # (M, 3): px, py, radius
        self.goal = np.asarray(goal, dtype=float)

        # 名义控制序列 U: (T, 2)，滚动优化时复用
        self.U = np.zeros((self.T, 2))

    def _rollout(self, state, V):
        """向量化 rollout。V: (K, T, 2) -> trajs: (K, T+1, 2) 只取 xy。"""
        K = V.shape[0]
        x = np.tile(state.astype(float), (K, 1))    # (K, 3)
        trajs = np.empty((K, self.T + 1, 2))
        trajs[:, 0] = x[:, :2]
        for t in range(self.T):
            v = V[:, t, 0]
            omega = V[:, t, 1]
            x[:, 0] += v * np.cos(x[:, 2]) * self.dt
            x[:, 1] += v * np.sin(x[:, 2]) * self.dt
            x[:, 2] += omega * self.dt
            trajs[:, t + 1] = x[:, :2]
        return trajs

    def _cost(self, trajs):
        """对每条轨迹算总代价。trajs: (K, T+1, 2) -> (K,)。"""
        pts = trajs[:, 1:]                          # (K, T, 2)，跳过起点
        # 距目标代价
        d_goal = np.linalg.norm(pts - self.goal, axis=2)   # (K, T)
        cost = self.w_goal * d_goal.sum(axis=1)
        # 终端代价
        cost += self.w_term * d_goal[:, -1]
        # 障碍代价：任意步进入 (障碍半径 + 车半径) 记一次大惩罚
        for ox, oy, orad in self.obstacles:
            d = np.linalg.norm(pts - np.array([ox, oy]), axis=2)   # (K, T)
            hit = d < (orad + self.robot_radius)
            cost += self.w_obs * hit.sum(axis=1)
        return cost

    def step(self, state):
        """跑一次 MPPI，返回 (要执行的控制 u, 采样轨迹束, 最优轨迹)。"""
        # 采样噪声并叠加到名义序列上
        noise = np.random.randn(self.K, self.T, 2) * self.sigma
        V = self.U[None] + noise
        V[:, :, 0] = np.clip(V[:, :, 0], -self.v_max, self.v_max)
        V[:, :, 1] = np.clip(V[:, :, 1], -self.omega_max, self.omega_max)

        trajs = self._rollout(state, V)
        cost = self._cost(trajs)

        # 指数加权：w = exp(-1/lam * (S - min S))，再归一化
        beta = cost.min()
        w = np.exp(-(cost - beta) / self.lam)
        w /= w.sum() + 1e-9

        # 更新名义控制序列
        self.U = np.einsum('k,ktu->tu', w, V)

        u = self.U[0].copy()

        # 最优轨迹：用更新后的 U 从当前状态 rollout 一条
        best = self._rollout(state, self.U[None])[0]

        # 滚动：把序列前移一格，尾部补零
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        return u, trajs, best


class MPPIDemoNode(Node):
    def __init__(self):
        super().__init__('mppi_demo')

        # 场景：起点、障碍阵、巡回目标点
        self.state = np.array([-6.0, -6.0, 0.0])   # px, py, theta
        # 一片障碍林（px, py, radius），构成需要来回穿插的走廊
        self.obstacles = np.array([
            [-4.0, -2.0, 0.7],
            [-2.0,  1.0, 0.9],
            [-3.5,  3.5, 0.7],
            [ 0.0, -1.0, 1.0],
            [ 0.5,  3.0, 0.8],
            [-1.0, -4.0, 0.7],
            [ 2.5,  0.5, 0.9],
            [ 3.5, -3.0, 0.8],
            [ 4.5,  3.0, 0.9],
            [ 2.0,  5.0, 0.7],
        ])

        # 依次巡回的目标点（到一个就去下一个，循环往复，录屏更耐看）
        self.waypoints = np.array([
            [ 6.0,  6.0],
            [-6.0,  6.0],
            [ 6.0, -6.0],
            [-6.0, -6.0],
        ], dtype=float)
        self.wp_idx = 0
        goal = self.waypoints[self.wp_idx]

        self.mppi = MPPI(self.obstacles, goal)
        self.goal = np.array(goal, dtype=float)
        self.goal_tol = 0.3

        self.path = Path()
        self.path.header.frame_id = FRAME

        self.marker_pub = self.create_publisher(MarkerArray, 'mppi/markers', 10)
        self.path_pub = self.create_publisher(Path, 'mppi/path', 10)

        # 控制/可视化频率 10 Hz（配合调小的 v_max，运动平缓便于录屏）
        self.timer = self.create_timer(0.1, self.update)
        self.get_logger().info('MPPI demo 启动，rviz fixed frame 设为 world')

    def update(self):
        now = self.get_clock().now().to_msg()

        u, trajs, best = self.mppi.step(self.state)

        # 用第一步控制推进真实状态
        v, omega = u
        self.state[0] += v * np.cos(self.state[2]) * self.mppi.dt
        self.state[1] += v * np.sin(self.state[2]) * self.mppi.dt
        self.state[2] += omega * self.mppi.dt

        # 记录路径
        ps = PoseStamped()
        ps.header.frame_id = FRAME
        ps.header.stamp = now
        ps.pose.position.x = float(self.state[0])
        ps.pose.position.y = float(self.state[1])
        self.path.poses.append(ps)
        # 巡回不停，路径无限增长会拖慢渲染，只保留最近一段拖尾
        if len(self.path.poses) > 800:
            self.path.poses = self.path.poses[-800:]

        # 到达当前目标就切换到下一个（循环巡回）
        if np.linalg.norm(self.state[:2] - self.goal) < self.goal_tol:
            self.wp_idx = (self.wp_idx + 1) % len(self.waypoints)
            self.goal = self.waypoints[self.wp_idx].copy()
            self.mppi.goal = self.goal
            self.get_logger().info(f'到达，切换到目标 {self.wp_idx}: {self.goal}')

        self.path.header.stamp = now
        self.path_pub.publish(self.path)
        self.publish_markers(now, trajs, best)

    # ---------- 可视化 ----------

    def publish_markers(self, stamp, trajs, best):
        arr = MarkerArray()

        # 障碍物（绿色圆柱）
        for i, (ox, oy, orad) in enumerate(self.obstacles):
            m = self._base_marker(stamp, 'obstacles', i, Marker.CYLINDER)
            m.pose.position.x = float(ox)
            m.pose.position.y = float(oy)
            m.pose.position.z = 0.5
            m.scale.x = m.scale.y = float(orad * 2)
            m.scale.z = 1.0
            m.color = ColorRGBA(r=0.2, g=0.8, b=0.3, a=0.6)
            arr.markers.append(m)

        # 目标点（蓝色球）
        g = self._base_marker(stamp, 'goal', 0, Marker.SPHERE)
        g.pose.position.x = float(self.goal[0])
        g.pose.position.y = float(self.goal[1])
        g.pose.position.z = 0.2
        g.scale.x = g.scale.y = g.scale.z = 0.5
        g.color = ColorRGBA(r=0.1, g=0.3, b=1.0, a=0.9)
        arr.markers.append(g)

        # 小车（红色箭头，表示朝向）
        r = self._base_marker(stamp, 'robot', 0, Marker.ARROW)
        r.scale.x = 0.15   # 轴直径
        r.scale.y = 0.3    # 箭头直径
        r.scale.z = 0.3    # 箭头长度
        r.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
        px, py, th = self.state
        tip = 0.6
        r.points = [
            Point(x=float(px), y=float(py), z=0.2),
            Point(x=float(px + tip * np.cos(th)),
                  y=float(py + tip * np.sin(th)), z=0.2),
        ]
        arr.markers.append(r)

        # 采样轨迹束（灰色细线，LINE_LIST 一个 marker 装下）
        s = self._base_marker(stamp, 'samples', 0, Marker.LINE_LIST)
        s.scale.x = 0.01
        s.color = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.25)
        n_show = min(120, trajs.shape[0])
        for k in range(n_show):
            traj = trajs[k]
            for t in range(traj.shape[0] - 1):
                s.points.append(Point(x=float(traj[t, 0]), y=float(traj[t, 1]), z=0.1))
                s.points.append(Point(x=float(traj[t + 1, 0]), y=float(traj[t + 1, 1]), z=0.1))
        arr.markers.append(s)

        # 最优轨迹（绿色粗线）
        b = self._base_marker(stamp, 'best', 0, Marker.LINE_STRIP)
        b.scale.x = 0.06
        b.color = ColorRGBA(r=0.1, g=1.0, b=0.2, a=0.9)
        b.points = [Point(x=float(p[0]), y=float(p[1]), z=0.15) for p in best]
        arr.markers.append(b)

        self.marker_pub.publish(arr)

    def _base_marker(self, stamp, ns, mid, mtype):
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
    node = MPPIDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
