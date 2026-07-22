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
- `sim_config/robot_quadrotor_fastphysics_sensors.jsonc`：FrontCamera 由前视改 **下倾 pitch=-35°**
  （空中俯冲时前视会丢机腹下方目标；-35° 兼顾"看得见目标 + 保留前进光流"，可调）。

## 待办（按顺序）

1. **定出生点**：把 drone 摆到某架 3D 目标飞机上方、相机能看到目标的位置，写进 `scene_airbase.jsonc`。
2. **离线生死判据**（少量帧即可，不训练）：给一架目标飞机做模板，跑
   `check_cost_monotonic.py` / `check_target_cost.py`，验证 ① 斜下滑近时指纹 cost 单调；
   ② 3D 目标 vs 贴图飞机、机头/机翼/机尾**部位间可分**。不可分则先改 cost/模板，别急着采集重训。
3. 可分成立 → 按"斜下滑近"重设采集轨迹，走脚本 1→6（collect→dino→predictor）重训世界模型。
4. 目标函数：先指纹(每部位一张模板, 两段式 handoff) 拿 baseline；价值函数 V 作泛化跟进(须配对新 predictor 重训)。

## 脚本清单（`src/airsim/`）

> 🆕 = airbase 新增/特有；其余为从老场景复制、可针对本场景改。按流水线阶段分组。

**数据采集**
- `decode_check.py` — 脚本1：相机帧解码验证（数据闭环第一步）
- `collect_episode.py` — 脚本2：单条 episode 采集（飞预设轨迹，同步记 图像/动作/位姿/时间 → HDF5）
- `replay_check.py` — 脚本3：采集质量验证（放成带标注 mp4 肉眼查对齐）
- `batch_collect.py` — 脚本4：批量采多条 episode
- `consolidate.py` — 整理：合并多目录、丢 3Hz 废帧、裁冻结尾 → 干净数据集
- `record_approach.py` — 录"直线飞近目标"接近 episode（供目标函数标定；两条 tgt1 数据即出自此）

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
- `servo_closed_loop.py` 🆕 — 方案A：纯指纹视觉伺服闭环（不用世界模型，实时 center/mass 驱动飞向目标）；跑完自动出 `viz_dump/`

**可视化 / 建图**
- `dump_servo_viz.py` 🆕 — 把伺服 run.h5 离线转成 rviz 可播 npz（轨迹+指纹热力图，不用重飞）
- `depth_map.py` — 深度→世界系点云，给实时建图用

## 与老场景共享/不共享

- **不复制、软链共享**：冻结 DINO 仓 `dinov2/` 和权重 `dinov2_vits14_pretrain.pth`。
- **完全不动**：`../sim_config/`、`../weights/`、`../outputs/`、`../src/`。
- **rviz 可视化**：仍用共享的 `../src/mpc/plan_viz*`（本地 ROS2，靠 `--dump-dir` 读，无需 fork）。
