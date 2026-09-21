# airbase 场景（空中接触地面飞机目标）

新场景：机场卫星图 + 放入的 **3D 飞机模型**（卫星图里烘焙的飞机是背景干扰，不碰）。
任务：无人机从空中斜下滑近、接触 3D 目标飞机的不同部位（机头/机翼/机尾）。
纯视觉（目标坐标脏、不可稳定还原），沿用 DINOv2 潜空间世界模型 + 目标条件 cost + MPPI 主线。
方法与老场景（绿门）一致，见 `../docs/ARCHITECTURE.md`；本文件夹是**本场景自包含的一套**，老场景不受影响。

## 目录

```
airbase/
  src/airsim/            # 复制自 ../src/airsim 的流水线脚本(可针对本场景改) + tools/
  sim_config/            # 本场景配置(已针对性修改)
  weights/               # 本场景重训的 predictor/value(待训);
                         # dinov2_vits14_pretrain.pth 软链→共享冻结 DINO 权重
  outputs/               # 本场景所有过程产物(数据集/特征/评估/runs/模板, 边跑边生成)
  dinov2 -> ../dinov2    # 软链: 共享冻结 DINO 仓(hubconf)
```

**关键：脚本保持"场景根下第 2 层"(`src/airsim/`)，故 `PROJECT_ROOT=parents[2]` 自动指到本文件夹，
默认参数全落进 `airbase/`——路径代码无需改。** 直接 `python airbase/src/airsim/<组>/xxx.py` 即可（08-01 按流程分成 wm/ loop/ px4/ probe/，
脚本开头的引导向上找 `sim_config/` 定位本文件夹，与层级无关）。

## 已做的配置改动（相对老场景）

- `sim_config/scene_airbase.jsonc`：`id` 为 `SceneAirbase`；Simple Flight 出生点为 NED
  `(0,0,-1.0)`，比跑道面高 1 m，避免原 `z=0` 与地面碰撞体穿透。PX4 场景仍保持原点不变。
- `sim_config/robot_quadrotor_fastphysics_sensors.jsonc`：FrontCamera **下倾 pitch=-50°**（斜下滑近，
  较陡以对齐从高空滑向地面目标的滑降角→目标全程居中；斜视动力学信息量更利于世界模型）。
  配合 **缩小飞机(~20m)**；垂直方向靠伺服 **cy 闭环(视线角控制)** 沿视线接触，不再开环匀速降
  （这才是"落在目标前面"的真正修法）。（曾试 -90° 纯俯视，代码留作备选 `--nadir`。）
  **开发顺序**：先用纯伺服从 50m 斜下接触打通（验证指纹信号 + 采训练数据 + 当基线），
  再训世界模型（课题主角）。最终飞行时"世界模型全程 vs 远段规划+终端伺服"的空间分段是后话，暂不定。

## 进展与待办

**实验记录见 [`EXPERIMENTS.md`](EXPERIMENTS.md)（本场景独立台账，不与老场景混用）。**
**PX4 联调怎么开机见 [`RUN_PX4.md`](RUN_PX4.md)（启动命令 / 失败查表 / 换设备迁移 checklist）。**

已完成（2026-07-22）：
- 相机 -50°、缩小飞机(~20m)、纯视觉伺服 + **cy 视线角闭环**打通斜下接近；横向对准可靠。
- 离线信号单调(ρ +0.81/−0.85/−0.93)；但指纹**只分"飞机vs背景"、不分部位**（整机全红）。

已完成（2026-07-24）：
- **伺服转 PX4 首飞跑通**：UE + PX4 SITL + MAVSDK offboard 三方联调，40m→30m 冒烟测试一次通过；
  NED 原点对齐 / offboard 频率 / 目标可见性三大风险全部清掉，控制律在 PX4 上与 simpleflight 表现一致。

待办（按顺序）：
0. **PX4 完整接触**：冒烟已过（`--stop-alt 30`），下一跑放开到 `--stop-alt 7` 验真接触。
1. **修伺服接触**：机头模板 + `--stop-alt` 判高度接触（治"停太高"）；加"低空自动减速"（治红星"冲过头"）。
   注：场景中各模型尺寸不一，`mass` 阈值不通用，故减速不应依赖 mass。
2. **重训世界模型**：用 -50° 接近数据走脚本 1→6（collect→dino→predictor），接 MPPI 闭环（远段），伺服作基线对照。
3. **部位可分（可选）**：整机全红下部位不可分；如需分，走"聚焦小模板 + 峰值匹配"或"近距离才分"的两段式，另开验证。
4. 目标函数：指纹拿 baseline；价值函数 V 作泛化跟进（须配对新 predictor 重训）。

## 闭环任务架构（暂定：两段式制导）

跑完整任务做闭环验证时，**世界模型和伺服在空间上分工**（制导领域标准的 midcourse + terminal homing）：

| 阶段 | 高度/条件 | 谁负责 | 为什么 |
|---|---|---|---|
| **远/中段** | 从 50m（或更高）到接触前 ~15–20m | **世界模型 + MPPI** | 目标还小、需往前看规划下降路径——世界模型主场，也是课题主角 |
| **终端** | 目标充满画面（~15m 以下）到接触 | **伺服 + cy 闭环** | 贴近了无甚可规划，目标又大又清晰，要反应快、精确对准接触 |

- **切换判据**：目标在画面里的 mass/尺寸超阈值（或降到某高度）→ 世界模型交棒伺服。
- **为什么不反过来**（伺服远→世界模型近）：伺服要目标清晰可见才好使，远处目标太小锁不稳；
  世界模型在贴脸接触时几乎没有可规划的东西。故"能往前看的放远段、反应快的放终端"。
- **与开发顺序区分**：开发上是**先伺服后世界模型**（伺服先打通 + 采数据 + 当基线）；
  上面说的是**最终飞行时**的空间分工，两者是两个轴，别混。
- 暂定，待伺服/世界模型都跑通后据实际再敲定切换判据与边界。

### 终端伺服 v2（2026-08-04 落地，08-04/05 已实飞 8 次，复盘见台账 §2026-08-06）

`loop/servo_closed_loop.py` 保持“DINO 模板给部位方向、深度给接触距离”的最小架构，不把世界模型
或目标真值坐标混进方案 A：

```
RGB(新帧驱动) → DINO 指纹 → EMA(cx/cy/mass/peak) → 机体系视线角控制
                         │
depth + 目标 Actor 分割 ─┴→ 目标主体内深度 → 距离减速 / 连续多帧接触确认
```

可选粗到细原型：显式传 `--template-roi` 后，远段仍用全模板；可信目标深度连续进入
`--roi-switch-dist`（默认 50 m）后一次性切到 ROI，并重置方向/距离 EMA。默认不传 ROI 时
完全保持原控制路径。`run.h5/control` 的 `roi_active` 记录每帧实际使用的原型。

- **连续控制**：相机 10Hz 新帧驱动；速度任务只提交、不等待结束，保活时间覆盖下一帧与 DINO 推理，
  避免“移动一拍—推理时停住”的控制空档。
- **失锁保护**：低置信度时不在噪声上偏航；短暂失锁只沿初始航向低速直行，持续失锁则悬停。
- **深度安全边界**：只有指纹可信、RGB/depth/seg 时间差合格、且采样点属于目标 Actor 掩膜时，
  深度才可进入减速与接触判据；接触须由多张独立深度帧连续确认。
- **分割的角色**：当前 UE 分割是整架飞机 Actor 级，只用于拒绝地面/背景深度，**不负责区分机头、
  机翼、尾翼、驾驶舱**。部位方向仍来自 DINO 模板，因此 v2 改善控制与安全性，不宣称解决部位可分性。
- **部署边界**：若真机没有可部署的分割模型，运行时用 `--no-seg-depth` 退化为指纹点深度；UE 真值
  分割只能作为仿真控制消融、评价或教师标签，不能包装成真机可用输入。
- **运行记录**：`run.h5/control` 与 `signals.csv` 同步保存原始/滤波信号、目标深度、锁定/分割状态、
  真实控制周期和接触连续计数；是否命中具体部位仍统一交给 `loop/eval_run.py` 事后评价。
- **部位定位离线评价**：`loop/eval_target_localization.py` 读取保存的 RGB/位姿，用目标 OOB
  投影出的部位锚点作为仅供评价的真值，比较当前“全模板平均原型”和指定模板 ROI 原型。
  评价真值不进入在线控制器；ROI 使用模板归一化坐标，换模板时必须重新指定。

### 三机多目标协同（2026-09-16）

`loop/multi_servo.py` 在一个 ProjectAirSim 客户端和一个 World 内控制 Drone1/2/3；任务由
`mission_config/three_aircraft.jsonc` 配置，默认分别分配 `StaticMeshActor_22/18/16` 三架飞机。
已从场景删除的汽车 `StaticMeshActor_24` 不进入任务和 RViz 精细地图。

```
三路 RGB/depth/seg → 三个独立 Mailbox ─┐
                                      ├→ 共享 DINO 批量前向
每机模板/统计/EMA/停止状态 ────────────┘          │
                                                   ↓
同区起飞 → 30m 等高等待点 → 交叉区预约通行 → 分层目标航路
                                              ↓
                              各机视觉终端控制 → 间距安全仲裁
```

- **协调中段**：三机在同一起飞区内沿 East 仅错开 2 m，先到 30 m 协调高度和三个等待点；
  三条入口→出口线段在 `(-15,-10)` 交叉。中央预约器按优先级一次放行一架，未放行无人机
  显示 `WAIT` 并悬停，已离开者显示 `CLEAR`。
- **目标中段**：离开交叉区后，三机并行进入 36/40/44 m 高度层和各自视觉接近点；
  Drone1 距 Actor_22 40 m，Drone2/3 距各自目标 60 m。
- **终端**：到达接近点后只用各自 RGB 指纹给方向、目标 Actor 分割深度给减速与停止；每机拥有
  独立 EMA、失锁、接触计数和 `TRACK/ARRIVED/FAILED` 状态，一机失败不阻塞另外两机。
  通用停止距离为 12 m；Actor_16 因近距时先从画面下缘离开，在 Drone3 任务项覆盖为 14 m，
  并要求两张独立新深度帧确认；Drone1/2 仍要求三张。
- **协同安全**：当前位置小于 5 m 时三机悬停；1 s 预测间距小于 8 m 时低优先级无人机悬停。
  到达或失败的无人机持续零速保位，直到整个任务结束。
- **共享计算**：DINO 权重只加载一次；每轮把当前有新帧的无人机组成批次，各自使用任务配置的
  标准化统计与模板原型。默认三机暂用同一 `整机.jpg` 做链路验证，正式目标区分需补采独立模板。
- **输出**：`outputs/runs/multi_servo/three_aircraft_<时间>/Drone{1,2,3}/` 各自保存
  `coordination.csv + coordination_dump/` 和视觉阶段的 `run.h5 + signals.csv + viz_dump/`；
  根目录 `mission_summary.json` 记录通行顺序、等待时间及全程/交叉阶段最小机间距。
- **回放**：`src/mpc/multi_viz.launch.py` 启动三套带命名空间的回放节点和一套静态地图；青/黄/洋红
  分别表示 Drone1/2/3，先播放协调阶段，再连续播放视觉阶段；交叉区显示橙色圆环和
  `WAIT/CROSS/CLEAR` 状态，不提前显示终点。三路 `/drone{1,2,3}/plan/view` 同步显示各机
  RGB 与 DINO 响应热力图叠加画面。
- **观察视角**：`Chase` 恢复 airbase 初始近距离姿态（机后 3 m、机上 1 m、俯视 11.46°）。
  `FrontCamera` 仅 RGB 开启 streaming，depth/seg 保持 capture、关闭 streaming；Unreal viewport
  按 `Tab` 在每架机的 Chase 和第一人称 RGB 间切换，不会进入深度/分割。三机模型、轨迹和
  三路 Camera+DINO 画面的同步观察使用上述 RViz 回放，因为 Unreal 同时只能显示一路流。
  实飞时给 `loop/multi_servo.py` 加 `--show-cameras`，可额外打开 Drone1/2/3 三路第三人称
  Chase RGB 的实时横向三联窗口；标题栏同步显示各机当前的协调阶段或任务状态。Chase raw
  capture 以 10 FPS 运行，FrontCamera 仍独立用于视觉控制和 Tab 第一人称切换。
- **交叉航路实飞验证**：`outputs/runs/multi_servo/three_aircraft_0916_192211` 三机均以
  `ARRIVED/contact` 结束，分别保存 159/216/202 帧；中心最近距离为 11.42/7.94/15.57 m。
  放行顺序为 Drone1→Drone2→Drone3，等待 0/5.74/11.44 s，交叉阶段最小机间距 8.14 m。
  全程最小值 1.60 m 出现在初始 2 m 间隔的同区出生阶段，不代表交叉区间距。

## 脚本清单（`src/airsim/`）

> 🆕 = airbase 新增/特有；其余为从老场景复制、可针对本场景改。按流水线阶段分组。

**数据采集**
- `decode_check.py` — 脚本1：相机帧解码验证（数据闭环第一步）
- `collect_episode.py` — 脚本2：单条 episode 采集（飞预设轨迹，同步记 图像/动作/位姿/时间 → HDF5）
- `replay_check.py` — 脚本3：采集质量验证（放成带标注 mp4 肉眼查对齐）
- `batch_collect.py` — 脚本4：批量采多条 episode
- `consolidate.py` — 整理：合并多目录、丢 3Hz 废帧、裁冻结尾 → 干净数据集
- `record_approach.py` — 录"直线飞近目标"接近 episode（斜视方案用；供目标函数标定）
- `grab_nadir_frames.py` 🆕 — 俯视：悬到目标正上方垂直下降逐高度抓帧，一次产出 stats-episode(h5) + 各高度 png（供裁"俯视整机"模板；50m 飞机须≥40m 高才拍全）

**探针/标定（airbase 特有）**
- `probe_image_msg.py` 🆕 — 抓一帧原始相机消息，排查解码条纹/行跨距
- `probe_spawn.py` 🆕 — 读回真实出生位姿（新场景 UE actor 非 1 缩放，标定出生点用）

**特征 + 世界模型（服务器训）**
- `extract_dino_features.py` — 脚本5：冻结 DINOv2 抽 latent（训练前一次性预处理）
- `train_predictor.py` — 脚本6：训世界模型核心 (z_t,a_t)→z_{t+1}
- `verify_predictor.py` — 验世界模型是否学到动力学（vs identity 基线）
- `train_value.py` — 训"时间距离"价值函数 V(z,z_goal)≈还差几步

**目标函数 / 生死判据（离线诊断，不连仿真）**
- `check_target_signal.py` 🆕 — 解耦生死判据：指纹 cost 有没有信号（不依赖 predictor，本场景当前主力）
- `eval_target_localization.py` 🆕 — 用闭环保存帧比较全模板/ROI 原型的部位投影定位误差（纯离线）
- `check_cost_monotonic.py` — 指纹 cost vs 旧整图 cost 谁随接近单调下降
- `check_target_cost.py` — 逐帧给 target 目标函数打分
- `check_target_action_ranking.py` — target cost+WM 会不会把合理动作排前
- `check_lateral_reliability.py` — 横向乱漂是调参能救还是 WM 预测本身噪声

**规划 / 闭环控制**
- `plan_mppi.py` — 脚本7-A：离线潜空间 MPPI 验证（不连仿真）
- `plan_closed_loop.py` — 脚本7-B：仿真闭环 MPPI 规划（世界模型 rollout + 指纹目标）；`--visual-servo`/`--handoff`/`--log-csv`
- `servo_closed_loop.py` 🆕 — 方案A：纯指纹视觉伺服闭环（不用世界模型）；v2 用 10Hz 连续指令、
  EMA/失锁保护、目标分割约束深度、距离减速和连续接触确认；`--nadir` 为俯视备选模式，运行自动落盘。
- `multi_servo.py` 🆕 — 三机多目标调度：分层中段、共享 DINO 批推理、独立终端伺服、间距仲裁与分机日志。

**可视化 / 建图**
- `dump_servo_viz.py` 🆕 — 把伺服 run.h5 离线转成 rviz 可播 npz（轨迹+指纹热力图+
  `step/mass/peak` 标签，不用重飞）
- `depth_map.py` — 深度→世界系点云，给实时建图用
- `probe_voxel_map.py` → `../src/mpc/voxel_map_viz_node.py` — ProjectAirSim 碰撞体素离线落盘后，
  将地面薄片和场景障碍分层、按 NED→ENU 发布静态 `/voxel_map/markers`；默认显示 2 km 全局图，
  `voxel_map_viz.launch.py` 默认用四架已标定飞机的 OOB 真值区域隐藏机场建筑（仅辅助显示，非语义识别），
  将 10 m 全局地面与四张 `60×60×20 m / 1 m` 目标局部图叠加。推荐先单独运行
  `rviz2 -d /home/hui/my/src/mpc/voxel_map_viz.rviz`，再重复运行
  `ros2 launch src/mpc/voxel_map_viz.launch.py` 播放轨迹；播放节点完成后自动退出，RViz 保持打开。
  RViz 连接 `/voxel_map/trajectory` 后，默认按真实记录时间以 2 倍速单次播放
  `机头_0805_103801`（黄线，500点）和 `红星_0722_194121`（洋红线，100点）两条历史轨迹；
  绿球使用每条数据的首帧位姿，红球跟随当前轨迹末端移动，播放结束后保留最终画面。
  `--trajectory-dir` 可重复传入；launch 参数 `start_rviz:=true` 可恢复同时启动 RViz，
  节点显式传入 `--loop` 才会循环播放。

## 与老场景共享/不共享

- **不复制、软链共享**：冻结 DINO 仓 `dinov2/` 和权重 `dinov2_vits14_pretrain.pth`。
- **完全不动**：`../sim_config/`、`../weights/`、`../outputs/`、`../src/`。
- **rviz 可视化**：闭环回放使用共享的 `../src/mpc/plan_viz*`；离线体素地图使用
  `../src/mpc/voxel_map_viz*`。`plan_viz.launch.py` 会同时启动静态体素地图发布器，将 10 m
  全局地面和四张 1 m 飞机精细图叠加到闭环轨迹，并显示放大的四旋翼、绿色起点和 0.35 m
  宽轨迹；不提前显示尚未飞到的终点。两者均为本地 ROS2，无需 colcon 构建。
