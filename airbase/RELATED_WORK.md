# 相关工作调研 —— 抵近目标并停在阈值距离

**调研日期**：2026-09-15　**工具**：`gh` CLI（GitHub 搜索 API）
**调研问题**：单目/深度相机抵近目标、停在阈值距离；用模板指定目标部位；换目标能否直接飞。

> 本文是**外部参考**，不是本项目结论。每条都附 star 数与最近提交时间，用于判断活跃度。

---

## 一、最相关：PairUAV（"最后一米"精调）

| | |
|---|---|
| 仓库 | [`YaxuanLi-cn/PairUAV`](https://github.com/YaxuanLi-cn/PairUAV) |
| star / 最近提交 | 54 / 2026-07-07 |
| license | **无**（法律上不能直接用其代码） |
| 论文 | arXiv 2607.04352，*Last-Meter Precision Navigation for UAVs: A Diffusion-Refined Aerial Visual Servoing Approach* |
| 出处 | ACM MM 2026 Workshop (UAVM) Challenge 的官方 baseline |

**为什么相关**：它解决的问题和本项目**同一个位置**——"从粗定位到最后一米精调"。
技术路线也重叠：`models/dino_resnet` + `controlnet`，baseline 用 SuperGlue 做匹配、视觉伺服。

**和本项目的差别**：它做的是**着陆/投递**（地面目标），本项目是**目标表面部位**；
它有 diffusion 精调阶段，本项目是单步指纹伺服。

**建议**：值得读它的 `baseline/train.py` 与 `step2_refine/`，看它怎么组织"伺服 + 学习式精调"两段式。
（要的话我可以拉下来读。）

---

## 二、其余相关仓库（按相关度排序）

| 仓库 | star / 提交 | 为什么相关 |
|---|---|---|
| [`SpaceTime-Vision-Robotics-Laboratory/nser-ibvs-drone`](https://github.com/SpaceTime-Vision-Robotics-Laboratory/nser-ibvs-drone) | 17 / 2026-02 | **NSER-IBVS**：数值稳定的高效 IBVS + 知识蒸馏。直击"图像雅可比不稳定"这个 IBVS 通病 |
| [`natesimon/MonoNav`](https://github.com/natesimon/MonoNav) | 145 / 2024-02 | **单目深度估计 + 重建**做 MAV 导航。单目测距路线的代表，对应本项目"PX4 路径没有深度相机"的缺口 |
| [`carlo98/precision_landing_shaping_RL`](https://github.com/carlo98/precision_landing_shaping_RL) | 45 / 2023-12 | 用 **RL + reward shaping** 做精确着陆（Gazebo）。对应我们讨论过的"在线 RL"路线（备选方案 5） |
| [`Kenil16/master_project`](https://github.com/Kenil16/master_project) | 70 / 2021-05 | 视觉导航 + 精确着陆，ROS/PX4 全栈。老实说有点旧，但工程结构可参考 |
| [`swarm-subnet/swarm-interceptorV1`](https://github.com/swarm-subnet/swarm-interceptorV1) | 8 / 2026-08 | YOLO 检测 + Kalman 跟踪 + 视觉伺服的拦截机。工程链路完整（Tello） |
| [`cweekiat/ibvs`](https://github.com/cweekiat/ibvs) | 9 / 2023-12 | 无人机的图像视觉伺服，实现较干净 |
| [`omkarUAV/IRP-contact-based-ultrasonic-testing-through-drones-for-aircraft-inspection`](https://github.com/omkarUAV/IRP-contact-based-ultrasonic-testing-through-drones-for-aircraft-inspection) | 0 / **2026-09-01** | **接触式**超声检测飞机——"无人机贴到飞机表面"这个动作本身。刚建，无 star，但方向对口 |
| [`Aximiotis/Visual-Servoing`](https://github.com/Aximiotis/Visual-Servoing) | 10 / 2025-09 | YOLO + 视觉伺服的无人机自主驾驶 |

---

## 三、没搜到的（= 可能的空白）

这几条用多组关键词搜过，**没有命中对口仓库**：

1. **模板条件的 / 跨目标泛化的视觉伺服** —— 即"给一张新目标的部位模板就能飞过去"。
   搜过 `few-shot visual servoing`、`one-shot visual servoing`、`template matching visual servoing`，
   只有通用的模板匹配/跟踪，没有"换目标不重训"的伺服工作。
   → **这正好是本项目 §2026-09-15 讨论的那个开放问题，看起来确实是空白。**
2. **航空器表面「部位」抵近** —— 搜到的是"检测缺陷"（裂纹/腐蚀）与"整机巡检"，
   没有"指定机头/机翼/尾翼并飞过去"的。
3. **停在阈值距离的判据本身** —— 没有人专门做这个；文献里它被当作巡检/着陆任务的附属。

---

## 四、搜索边界（诚实说明）

- 用过的关键词：`visual servoing UAV/drone`、`aerial visual servoing`、`IBVS drone`、
  `"last-meter" UAV`、`"precision landing" drone`、`"visual servoing" drone`、
  `aircraft inspection drone`、`monocular depth estimation UAV navigation`、
  `monocular object distance estimation`、`DINOv2 visual servoing`、`few-shot visual servoing`、
  `aerial manipulation contact UAV`、`"visual servoing" "foundation model"`。
- **`gh search repos` 多词查询是 AND 匹配**，词一多（4 个以上）基本返回空。
  有效的做法是用**引号包住短语**、或走底层 API。这是本次踩到的坑，记下来。
- GitHub 代码搜索（`gh search code`）用的是**旧版引擎**，与网页版结果不一致，不支持正则；
  本次没深挖代码层。
- 只搜了 GitHub。arXiv / Google Scholar 上的论文没系统查（除顺着仓库 README 找到的）。
