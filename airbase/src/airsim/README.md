# airbase 脚本索引

这个目录有 **29 个脚本**，平铺在一起。本文件是导航图：每个脚本干什么、**现在还能不能跑**、
默认连哪个场景。写于 2026-07-30。

---

## ⚠️ 先看这条：跑脚本前确认自己在 `airbase/` 下

`~/my/src/airsim/` 和 `~/my/airbase/src/airsim/` 有**十几个同名脚本**，内容已经分叉。
在 `~/my` 下敲 `python src/airsim/xxx.py` 会命中**老场景（绿门）**那份。

```bash
cd ~/my/airbase && python src/airsim/collect_episode.py ...   # ✅ 推荐
python airbase/src/airsim/collect_episode.py ...              # ✅ 也行
python src/airsim/collect_episode.py ...                      # ❌ 在 ~/my 下 = 老场景脚本
```

**为什么危险**：老脚本会正常启动、连上仿真、按老场景参数飞（起点在出生点 ±3m、200 步、
世界系动作），产出一堆看起来正常但完全没用的数据，还会写进 `~/my/outputs/` 污染老场景数据集。
2026-07-31 实际踩到过一次，只是因为新脚本改了参数名（`--band/--num`）才被 argparse 拦住。
这与 07-30 记录的"两个同名 `EXPERIMENTS.md`"是同一类坑，但更隐蔽——**那个会报错，这个不会**。

状态标记：

| 记号 | 含义 |
|---|---|
| 🟢 | 在用，且 `airbase/outputs/` 下有它产出的东西 |
| 🟡 | 能跑，但当前不适用 / 有坑（见备注） |
| 🔴 | **在 airbase 跑不了** —— 缺前置条件（见文末「当前阻塞」） |

---

## 跑任何脚本之前必须知道的五件事

**1. 两个场景，各自绑不同的机器人配置** —— 这个搞错会让你怀疑人生（相机、传感器全变）：

| 场景 | 机器人配置 | 飞控 | 传感器 |
|---|---|---|---|
| `scene_airbase.jsonc` | `robot_quadrotor_fastphysics_sensors.jsonc` | Simple Flight | FrontCamera(RGB+深度+**分割**)、lidar、GPS… |
| `scene_airbase_px4.jsonc` | `robot_quadrotor_px4_airbase.jsonc` | **PX4 SITL** | FrontCamera(仅 RGB)、Chase、DownCamera |

两个 FrontCamera 都是 **pitch −50°、FOV 90°、960×540（16:9）**。

**2. UE 必须在前台 Play**，否则采集帧率掉到 ~3Hz。

**3. 跑 PX4 那两个脚本前**：每次重跑都要重起 PX4 **并且**在 UE 里 End PIE 再 Play，
否则卡在"连接 PX4"。详见 `../../RUN_PX4.md`。

**4. 环境**：`conda activate airsim`。

**5. 路径约定**：27 个脚本用 `PROJECT_ROOT = Path(__file__).resolve().parents[2]` 定位
`airbase/`，并且互相按同目录 import（`batch_collect → collect_episode`、
`servo_closed_loop_px4 → servo_closed_loop`、7 个脚本 → `extract_dino_features` …）。
**所以不要把脚本挪进子目录** —— `parents[2]` 会指错，所有默认路径和导入一起失效。

---

## 探针 / 标定（一次性，查完就完）

都是只读或轻量飞行，用来把"不知道的事实"钉死。

| 脚本 | 状态 | 干什么 | 场景 |
|---|---|---|---|
| `probe_spawn.py` | 🟢 | 读回真实出生位姿 / 触地测地面基准（UE 的 cm 不能想当然 /100） | airbase |
| `probe_target_object.py` | 🟢 | **目标飞机叫什么、在哪、朝哪、多大**。按坐标认人，不靠硬编码名字 | airbase |
| `probe_segmentation.py` | 🟢 | 分割相机能力探测 + 名字/ID 映射（只读，不起飞） | airbase |
| `probe_depth_seg_flight.py` | 🟢 | 爬到高空拍地面目标，验证深度=米制、分割能区分不同飞机 | airbase |
| `probe_image_msg.py` | 🟡 | 抓一帧原始消息打印字段 + 行跨距自检 | airbase |
| `px4_yaw_probe.py` | 🟢 | 隔离测试 PX4 offboard 下航向保不住是谁的锅（A~G 七段） | **airbase_px4** |

**已由它们钉死的事实**（细节在 `../../EXPERIMENTS.md`）：
- 出生点 NED `(0,0,0)`，机头朝北
- 目标飞机 `StaticMeshActor_22`，OOB 中心 `(-62.71,-16.68,-2.94)`，尺寸 23.73×24.73×6.48m，
  **顶部在 6.18m 高度**（不是 1.4m！）
- 深度是 **米制整数**（量化 1m），`65504` 是天空哨兵
- 分割默认 ID 全撞车（都是 83），**必须**用 `set_segmentation_id_by_name` 显式指定
- 场景里还有 3 架飞机 + 1 辆车；滑行道上另有一架**被烘进地形网格**的假飞机（查不到、分割不出，但相机里看得见）

---

## 采集

| 脚本 | 状态 | 干什么 | 场景 |
|---|---|---|---|
| `decode_check.py` | 🟢 | 解码验证（数据闭环第一步）。默认场景已修为 `scene_airbase.jsonc`（07-31） | airbase |
| `collect_episode.py` | 🟢 | **近场采集（07-31 重写）**：以目标为心的球内随机站位 → 机体系动作 → 同步存 RGB+深度+分割 → 一次起飞串 `--num` 条；目标 OOB 写进 h5 attrs 供 evaluator 离线补算 | airbase |
| `batch_collect.py` | 🔴 | **已废弃**：老场景遗留，功能被 `collect_episode.py --num N` 取代；依赖已删除的 `TRAJECTORY`，**import 即报错**（有意的安全失败） | — |
| `record_approach.py` | 🟢 | 录"直线飞向目标"的完整接近段（`recordings/approach/` 那 3 条就是它产的） | airbase |
| `grab_nadir_frames.py` | 🟡 | 正俯视垂直下降逐高度抓帧（裁俯视整机模板用）。**没跑过**，产物 `airbase_tgt1_nadir.h5` 不存在 | airbase |
| `replay_check.py` | 🟡 | 采集质量验证（回放 HDF5 看动作/图像对齐） | — |
| `consolidate.py` | 🟡 | 一次性：合并多个采集目录 → 干净数据集 | — |

**上面三条 🟡→🟢 的由来（07-31 近场改造）**：老版三个硬伤——起点只在出生点周围抖 ±3m
（目标在 65m 外根本不在画面里）、200 步全程绕着出生点、动作记的是世界 NED 速度
（同一向量在不同航向下含义不同，覆盖不了 360° 切入方位）——已随重写一并解决。
详见 `../../EXPERIMENTS.md` §2026-07-31。

跑法：`python collect_episode.py --num 20`（一次起飞串 20 条，约 10 分钟）。
站位采样已离线验证 **5000/5000 目标都在画面内**，r 中位 16.6m、<15m 占 41%。

**`airbase/outputs/datasets/` 目前仍是空的 —— 采集尚未实跑。**

---

## 训练（都在服务器上跑，本地不建 wm 环境）

| 脚本 | 状态 | 干什么 |
|---|---|---|
| `extract_dino_features.py` | 🟢 | 冻结 DINOv2 抽 latent（训练前一次性预处理）。**7 个脚本 import 它** |
| `train_predictor.py` | 🔴 | 训 action-conditioned latent predictor = 世界模型 |
| `train_value.py` | 🔴 | 训"时间距离"价值函数 V(z, z_goal) |
| `verify_predictor.py` | 🔴 | 验证世界模型在验证集上有没有学到动力学 |

---

## 离线验证（cost / 信号，不连仿真）

| 脚本 | 状态 | 干什么 |
|---|---|---|
| `check_target_signal.py` | 🟢 | **生死判据**：指纹 cost 到底有没有信号（不需要 predictor） |
| `check_correspondence_signal.py` | 🟢 | 远段 cost 读出方式对比：patch 相似度 vs 几何对应 |
| `check_target_cost.py` | 🔴 | 检查 target-conditioned 目标函数 |
| `check_target_action_ranking.py` | 🔴 | target cost 在世界模型 rollout 下是否偏好合理动作 |
| `check_cost_monotonic.py` | 🔴 | 指纹 cost vs 旧整图 cost 的单调性对比 |
| `check_lateral_reliability.py` | 🔴 | 世界模型对横向/垂直运动的预测是否可靠 |

⚠️ 前两个用的 Spearman 曾因**并列秩**和**标准化泄漏**虚报过相关性，已修（`_rank()` +
`--stats-episode`）。见 `../../EXPERIMENTS.md` §2026-07-29。

---

## 闭环

| 脚本 | 状态 | 干什么 | 场景 |
|---|---|---|---|
| `servo_closed_loop_px4.py` | 🟢 | **当前主力**：指纹视觉伺服闭环 PX4 版（消融的 A 条件）。`runs/servo/px4_*` 都是它产的 | **airbase_px4** |
| `servo_closed_loop.py` | 🟡 | 同上的 Simple Flight 版。`Fingerprint`/`save_run` 被 PX4 版复用 | airbase |
| `plan_mppi.py` | 🔴 | 离线潜空间 MPPI 验证（不连仿真） | — |
| `plan_closed_loop.py` | 🔴 | 仿真闭环 MPPI 规划（消融的 B 条件） | — |

**PX4 伺服三个坑**（都踩过）：
- 甩头 ~300°：根因是**交接前航向没停稳** → `--yaw-settle`（默认 2s）
- `--v-down 1.6` 必须显式传，默认 1.2 降得不够
- **`--stop-alt` 会关掉接触判据** → 要接触就别传它

---

## 工具

| 脚本 | 状态 | 干什么 |
|---|---|---|
| `depth_map.py` | 🟡 | 深度 → 世界系点云（给 `plan_closed_loop --map` 建图用）。**只被它调用，而 `--map` 从没启用过**（2754 个 viz_dump npz 里 0 个含点云） |
| `dump_servo_viz.py` | 🟢 | 把伺服 `run.h5` 离线转成 rviz 可播的 npz（轨迹 + 指纹热力图） |

---

## 当前阻塞：为什么 🔴 那一片跑不了

**`airbase/weights/` 里没有 `predictor_h5.pt`** —— 只有 DINO 权重
（`dinov2_vits14_pretrain.pth` 软链 + `dinov3_vits16_pretrain...pth`）。

所以世界模型是**没实现，不是没跑**。要解锁那 8 个 🔴，顺序是：

```
改采集脚本(球内站位 + 机体系动作 + 近场模板)
  → 在 25m 球内采 ~100~300 条短轨迹(每个目标一轮)
  → 服务器上训 predictor
  → 才能跑 plan_* 和 check_target_* / check_*_reliability
```

---

## 和 `src/airsim/`（老场景）的关系

airbase 是从老场景**镜像**出来的，所以两边同名文件很多：

- **12 个逐字节相同**：`batch_collect` `check_cost_monotonic` `check_lateral_reliability`
  `check_target_action_ranking` `check_target_cost` `consolidate` `extract_dino_features`
  `plan_mppi` `replay_check` `train_predictor` `train_value` `verify_predictor`
- **5 个已分叉**：`collect_episode`(3行) `decode_check`(23行) `depth_map`(14行)
  `plan_closed_loop`(41行) `record_approach`(44行)
- **12 个只有 airbase 有**：全部 `probe_*`、`servo_*`、`check_target_signal`、
  `check_correspondence_signal`、`dump_servo_viz`、`grab_nadir_frames`、`px4_yaw_probe`

⚠️ **这个重复咬过一次**：`depth_map.py` 的 `/1000` 单位 bug 两边都有，2026-07-30 只修了
airbase 这份。**改 airbase 的文件时，别顺手去改 `src/`；改 `src/` 的时候要想到 airbase。**
airbase 是活的主线，`src/airsim/` 只读。
（`src/mpc/plan_viz_node.py` 例外 —— 它是**活的**，airbase 的 rviz 可视化还在用它。）
