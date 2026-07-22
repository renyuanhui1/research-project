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
默认参数全落进 `airbase/`——路径代码无需改。** 直接 `python airbase/src/airsim/xxx.py` 即可。

## 已做的配置改动（相对老场景）

- `sim_config/scene_airbase.jsonc`：`id` 改为 `SceneAirbase`；出生点 `xyz` **仍是老场景遗留值，待重设**(见文件 TODO)。
- `sim_config/robot_quadrotor_fastphysics_sensors.jsonc`：FrontCamera **下倾 pitch=-50°**（斜下滑近，
  较陡以对齐从高空滑向地面目标的滑降角→目标全程居中；斜视动力学信息量更利于世界模型）。
  配合 **缩小飞机(~20m)**；垂直方向靠伺服 **cy 闭环(视线角控制)** 沿视线接触，不再开环匀速降
  （这才是"落在目标前面"的真正修法）。（曾试 -90° 纯俯视，代码留作备选 `--nadir`。）
  **开发顺序**：先用纯伺服从 50m 斜下接触打通（验证指纹信号 + 采训练数据 + 当基线），
  再训世界模型（课题主角）。最终飞行时"世界模型全程 vs 远段规划+终端伺服"的空间分段是后话，暂不定。

## 进展与待办

**实验记录见 [`EXPERIMENTS.md`](EXPERIMENTS.md)（本场景独立台账，不与老场景混用）。**

已完成（2026-07-22）：
- 相机 -50°、缩小飞机(~20m)、纯视觉伺服 + **cy 视线角闭环**打通斜下接近；横向对准可靠。
- 离线信号单调(ρ +0.81/−0.85/−0.93)；但指纹**只分"飞机vs背景"、不分部位**（整机全红）。

待办（按顺序）：
1. **修伺服接触**：机头模板 + `--stop-alt` 判高度接触（治"停太高"）；加"低空自动减速"（治红星"冲过头"）。
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
- `check_cost_monotonic.py` — 指纹 cost vs 旧整图 cost 谁随接近单调下降
- `check_target_cost.py` — 逐帧给 target 目标函数打分
- `check_target_action_ranking.py` — target cost+WM 会不会把合理动作排前
- `check_lateral_reliability.py` — 横向乱漂是调参能救还是 WM 预测本身噪声

**规划 / 闭环控制**
- `plan_mppi.py` — 脚本7-A：离线潜空间 MPPI 验证（不连仿真）
- `plan_closed_loop.py` — 脚本7-B：仿真闭环 MPPI 规划（世界模型 rollout + 指纹目标）；`--visual-servo`/`--handoff`/`--log-csv`
- `servo_closed_loop.py` 🆕 — 方案A：纯指纹视觉伺服闭环（不用世界模型）；`--nadir` 俯视模式(主用)：开环到目标上空→cx/cy 双闭环对中→居中才降；跑完自动出 `viz_dump/`

**可视化 / 建图**
- `dump_servo_viz.py` 🆕 — 把伺服 run.h5 离线转成 rviz 可播 npz（轨迹+指纹热力图，不用重飞）
- `depth_map.py` — 深度→世界系点云，给实时建图用

## 与老场景共享/不共享

- **不复制、软链共享**：冻结 DINO 仓 `dinov2/` 和权重 `dinov2_vits14_pretrain.pth`。
- **完全不动**：`../sim_config/`、`../weights/`、`../outputs/`、`../src/`。
- **rviz 可视化**：仍用共享的 `../src/mpc/plan_viz*`（本地 ROS2，靠 `--dump-dir` 读，无需 fork）。
