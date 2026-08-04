# airbase 脚本索引

**34 个脚本，按「一条完整流程」分成 4 个目录**（2026-08-01 重组，原先 30 个平铺在一起）。
本文件是导航图：每个脚本干什么、**现在还能不能跑**、默认连哪个场景。

```
src/airsim/
├── wm/      世界模型流水线：采集 → 抽特征 → 训练 → 评估训练效果        (14)
├── loop/    闭环：规划/伺服 → 信号验证 → 运行评价 → 可视化            (13)
├── px4/     PX4 部署                                                 (2)
└── probe/   场景标定探针（一次性，查完就完）                          (5)
```

分组按**流程**不按类型：打开一个目录就是一条完整的路，不用在几个目录间跳。
所以 `eval_run`（评价一次闭环运行）在 `loop/`，而 `eval_wm`/`eval_value_fn`（评价训练出的模型）在 `wm/` ——
评估脚本跟着它所服务的流程走。

---

## ⚠️ 先看这条：跑脚本前确认自己在 `airbase/` 下

`~/my/src/airsim/` 和 `~/my/airbase/src/airsim/` 有**十几个同名脚本**，内容已经分叉。
在 `~/my` 下敲 `python src/airsim/xxx.py` 会命中**老场景（绿门）**那份。

```bash
cd ~/my/airbase && python src/airsim/wm/collect_episode.py ...   # ✅ 推荐
python airbase/src/airsim/wm/collect_episode.py ...              # ✅ 也行
python src/airsim/collect_episode.py ...                         # ❌ 在 ~/my 下 = 老场景脚本
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

**5. 路径约定（08-01 已改，不再限制目录层级）**：每个脚本开头有一段自足的引导：

```python
_AIRBASE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "sim_config").is_dir())
_AIRBASE_SRC = _AIRBASE_ROOT / "src/airsim"
sys.path[:0] = [str(_AIRBASE_SRC)] + [各流程目录]      # 跨分组 import
```

**向上找 `sim_config/` 这个标记**来定位 `airbase/`，与脚本放在第几层无关；同时把 4 个流程目录
都挂上 `sys.path`，所以跨组 import 照常работа（`px4/servo_closed_loop_px4` →
`loop/servo_closed_loop` → `wm/decode_check` 这条链实测通过）。

> 原先写死 `parents[2]`，脚本一挪进子目录就指错，本文件为此专门警告过「不要把脚本挪进子目录」。
> 08-01 按流程重组时一并根治 —— **现在目录随便怎么调整都不会坏**。
> 重组后逐个验证：34 个脚本 33 个可正常加载，唯一"失败"的 `wm/batch_collect.py`
> 正是它自己设计的废弃保护（import 即报错）。

---

## 探针 / 标定（一次性，查完就完）

都是只读或轻量飞行，用来把"不知道的事实"钉死。

| 脚本 | 状态 | 干什么 | 场景 |
|---|---|---|---|
| `probe/probe_spawn.py` | 🟢 | 读回真实出生位姿 / 触地测地面基准（UE 的 cm 不能想当然 /100） | airbase |
| `probe/probe_target_object.py` | 🟢 | **目标飞机叫什么、在哪、朝哪、多大**。按坐标认人，不靠硬编码名字 | airbase |
| `probe/probe_segmentation.py` | 🟢 | 分割相机能力探测 + 名字/ID 映射（只读，不起飞） | airbase |
| `probe/probe_depth_seg_flight.py` | 🟢 | 爬到高空拍地面目标，验证深度=米制、分割能区分不同飞机 | airbase |
| `probe/probe_image_msg.py` | 🟡 | 抓一帧原始消息打印字段 + 行跨距自检 | airbase |
| `px4/px4_yaw_probe.py` | 🟢 | 隔离测试 PX4 offboard 下航向保不住是谁的锅（A~G 七段） | **airbase_px4** |

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
| `wm/decode_check.py` | 🟢 | 解码验证（数据闭环第一步）。默认场景已修为 `scene_airbase.jsonc`（07-31） | airbase |
| `wm/collect_episode.py` | 🟢 | **近场采集（07-31 重写）**：以目标为心的球内随机站位 → 机体系动作 → 同步存 RGB+深度+分割 → 一次起飞串 `--num` 条；目标 OOB 写进 h5 attrs 供 evaluator 离线补算 | airbase |
| `wm/batch_collect.py` | 🔴 | **已废弃**：老场景遗留，功能被 `collect_episode.py --num N` 取代；依赖已删除的 `TRAJECTORY`，**import 即报错**（有意的安全失败） | — |
| `wm/record_approach.py` | 🟢 | 录"直线飞向目标"的完整接近段（`recordings/approach/` 那 3 条就是它产的） | airbase |
| `wm/grab_nadir_frames.py` | 🟡 | 正俯视垂直下降逐高度抓帧（裁俯视整机模板用）。**没跑过**，产物 `airbase_tgt1_nadir.h5` 不存在 | airbase |
| `wm/replay_check.py` | 🟡 | 采集质量验证（回放 HDF5 看动作/图像对齐） | — |
| `wm/consolidate.py` | 🟡 | 一次性：合并多个采集目录 → 干净数据集 | — |

**上面三条 🟡→🟢 的由来（07-31 近场改造）**：老版三个硬伤——起点只在出生点周围抖 ±3m
（目标在 65m 外根本不在画面里）、200 步全程绕着出生点、动作记的是世界 NED 速度
（同一向量在不同航向下含义不同，覆盖不了 360° 切入方位）——已随重写一并解决。
详见 `../../EXPERIMENTS.md` §2026-07-31。

跑法：`python src/airsim/wm/collect_episode.py --band near --num 30 --output-dir outputs/datasets/xxx`
（一次起飞串 30 条，约 15 分钟）。

**08-01 又改了一次动作采样**：`TEMPLATES`（一条 episode 一个模板）→ `SEG_POOL`（10 个基元）+
`build_random_traj()`（一条随机串 8 段 × 13 步）。原因是模板采集训出来的世界模型
**动作输入等于没接**（zero-act≈1.00）—— 动作在 episode 内几乎不变、与视觉上下文绑死，
模型"看图猜模板"就能绕开动作。段长 13 步的下界由飞控速度环决定（~11 步才跟上指令）。

已采：`outputs/datasets/` 下 `nearfield` 40 / `randseg` 60 / **`wm300` 300**（主数据集）。
`tools` 目录已取消，`interleave_dataset.py` 挪到 `wm/`——按(目标×距离带)轮转重编号，
让 held-out 跨目标跨带（`split_episodes` 与 `eval_wm` 都取"排序后最后 N 条"）。

---

## 训练（都在服务器上跑，本地不建 wm 环境）

| 脚本 | 状态 | 干什么 |
|---|---|---|
| `wm/extract_dino_features.py` | 🟢 | 冻结 DINOv2 抽 latent（训练前一次性预处理）。**7 个脚本 import 它** |
| `wm/train_predictor.py` | 🟢 | 训 action-conditioned latent predictor = 世界模型。**08-01 加 `--action-hist`**（每步拼最近 k 条指令，k=8 为主力）、**按 val 存最优检查点**、`--val-every` |
| `wm/train_value.py` | 🟡 | 训"时间距离"价值函数 V(z, z_goal)。airbase 尚未训 |
| `wm/verify_predictor.py` | 🟡 | 单步 vs identity。**弱判据** —— 单步画面几乎不动、identity 天然占优，老场景通过的模型在 h1 也是输的。判死活看下面的 `eval_wm` |
| `wm/eval_wm.py` | 🟢 | **判死活的三条判据**（多步 rollout）：`model/identity`(学到运动)、`model/zero-act`(真听命令)、`model/swap`(动作特异)，`<1` 为过 |
| `wm/eval_value_fn.py` | 🔴 | 价值函数评估（airbase 还没有 value 权重） |
| `wm/interleave_dataset.py` | 🟢 | 按(目标×距离带)轮转重编号，使 held-out 跨目标跨带。默认 dry-run |

⚠️ **`extract_dino_features` 只按文件名判"已存在"**。不同数据集若编号重叠，会静默复用旧 latent →
**图像与动作错配且不报错**。规避方式：**一个数据集一个 latent 目录**（`--output-dir`）。

⚠️ **`torch.save` 没有覆盖保护**（与 `save_hdf5` 的 `FileExistsError` 不同），`--ckpt` 写错会静默盖掉已有权重。

---

## 离线验证（cost / 信号，不连仿真）

| 脚本 | 状态 | 干什么 |
|---|---|---|
| `loop/check_target_signal.py` | 🟢 | **生死判据**：指纹 cost 到底有没有信号（不需要 predictor） |
| `loop/check_correspondence_signal.py` | 🟢 | 远段 cost 读出方式对比：patch 相似度 vs 几何对应 |
| `loop/check_target_cost.py` | 🟡 | 检查 target-conditioned 目标函数（**前置已具备**：predictor 权重已有，尚未在 airbase 实跑） |
| `loop/check_target_action_ranking.py` | 🟡 | target cost 在世界模型 rollout 下是否偏好合理动作（同上） |
| `loop/check_cost_monotonic.py` | 🟡 | 指纹 cost vs 旧整图 cost 的单调性对比（同上） |
| `loop/check_lateral_reliability.py` | 🟡 | 世界模型对横向/垂直运动的预测是否可靠（同上） |
| `loop/eval_run.py` | 🟢 | **闭环运行的特权评价器**（L0 健全性 + L1 任务判据，纯离线）。`run.h5` 的 `reached` 混淆高度停止与 mass 停止，不能当成功判据 |
| `loop/eval_mppi_dir.py` | 🔴 | 批量评估 MPPI 运行目录（airbase 还没跑过 MPPI） |

⚠️ 前两个用的 Spearman 曾因**并列秩**和**标准化泄漏**虚报过相关性，已修（`_rank()` +
`--stats-episode`）。见 `../../EXPERIMENTS.md` §2026-07-29。

---

## 闭环

| 脚本 | 状态 | 干什么 | 场景 |
|---|---|---|---|
| `px4/servo_closed_loop_px4.py` | 🟢 | **当前主力**：指纹视觉伺服闭环 PX4 版（消融的 A 条件）。`runs/servo/px4_*` 都是它产的 | **airbase_px4** |
| `loop/servo_closed_loop.py` | 🟡 | 同上的 Simple Flight 版。`Fingerprint`/`save_run` 被 PX4 版复用 | airbase |
| `loop/plan_mppi.py` | 🟡 | 离线潜空间 MPPI 验证（不连仿真）。**前置已具备**，尚未在 airbase 实跑 | — |
| `loop/plan_closed_loop.py` | 🟡 | 仿真闭环 MPPI 规划（消融的 B 条件）。同上 | — |

⚠️ **A 与 B 跑在两条控制栈上**：`px4/servo_closed_loop_px4.py` 走 MAVSDK offboard，
`loop/plan_closed_loop.py` 走 ProjectAirSim `move_by_velocity_async`（Simple Flight）。
**不可公平比较** —— 做消融前必须先统一（建议 A/B/C 都先走 Simple Flight，跑通再整体接 PX4）。

**PX4 伺服三个坑**（都踩过）：
- 甩头 ~300°：根因是**交接前航向没停稳** → `--yaw-settle`（默认 2s）
- `--v-down 1.6` 必须显式传，默认 1.2 降得不够
- **`--stop-alt` 会关掉接触判据** → 要接触就别传它

---

## 工具

| 脚本 | 状态 | 干什么 |
|---|---|---|
| `loop/depth_map.py` | 🟡 | 深度 → 世界系点云（给 `plan_closed_loop --map` 建图用）。**只被它调用，而 `--map` 从没启用过**（2754 个 viz_dump npz 里 0 个含点云） |
| `loop/dump_servo_viz.py` | 🟢 | 把伺服 `run.h5` 离线转成 rviz 可播的 npz（轨迹 + 指纹热力图） |

---

## 当前阻塞：**已解除**（08-01）

原文说"`airbase/weights/` 里没有 `predictor_h5.pt`，世界模型是没实现不是没跑"。现在有了：

| 权重 | 数据 | 三判据 (held-out, h3~h30) |
|---|---|---|
| `predictor_tmpl40.pt` | 模板采集 40 条 | zero-act ≈1.00、swap ≈1.03（**动作输入等于没接**） |
| `predictor_h5.pt` | 随机分段 60 条 | zero-act 0.76~0.96、swap 0.91~0.98 |
| `predictor_wm300_h4.pt` | 300 条 + 动作历史 4 | zero-act 0.71~0.81、swap 0.75~0.78 |
| `predictor_wm300_h8.pt` | 300 条 + 动作历史 8（60 轮，未收敛） | identity 0.51~0.56 / zero-act 0.66~0.74 / swap 0.71~0.70 |
| **`predictor_wm300_h8e150.pt`** | 同上训满 150 轮 | **identity 0.48~0.55 / zero-act 0.60~0.66 / swap 0.65~0.62 ← 主力，swap 追平老场景** |

数据：`outputs/datasets/` 下三份（`nearfield` 40 / `randseg` 60 / **`wm300` 300**，
5 目标 × 2 距离带 × 30）。**三份动作结构不同，不可混用**。

所以 `loop/plan_*`、`loop/check_target_*` 这些原先 🔴 的脚本，前置条件已经具备。
详细经过（两轮阴性结果 + 根因定位）见 `../../EXPERIMENTS.md` §2026-08-01。

---

## 和 `src/airsim/`（老场景）的关系

airbase 是从老场景**镜像**出来的，所以两边同名文件很多。

**08-01 重组后目录结构已经不同了**：airbase 按流程分成 `wm/ loop/ px4/ probe/`，
老场景 `src/airsim/` 仍是平铺。文件名相同但路径不同，反而降低了误跑的概率
（`~/my` 下敲 `python src/airsim/collect_episode.py` 现在只会命中老场景那份，
airbase 的在 `airbase/src/airsim/wm/` 下）。

分叉状况（截至 07-30 统计，重组只改了位置和路径引导，未改逻辑）：

- **12 个曾逐字节相同**：`batch_collect` `check_cost_monotonic` `check_lateral_reliability`
  `check_target_action_ranking` `check_target_cost` `consolidate` `extract_dino_features`
  `plan_mppi` `replay_check` `train_predictor` `train_value` `verify_predictor`
  （其中 `train_predictor`/`verify_predictor` 已因 08-01 的动作历史改动而分叉）
- **5 个已分叉**：`collect_episode`(3行→08-01 大改) `decode_check`(23行) `depth_map`(14行)
  `plan_closed_loop`(41行) `record_approach`(44行)
- **只有 airbase 有**：全部 `probe_*`、`servo_*`、`check_target_signal`、
  `check_correspondence_signal`、`dump_servo_viz`、`grab_nadir_frames`、`px4_yaw_probe`、
  `eval_run`、`interleave_dataset`

⚠️ **这个重复咬过一次**：`loop/depth_map.py` 的 `/1000` 单位 bug 两边都有，2026-07-30 只修了
airbase 这份。**改 airbase 的文件时，别顺手去改 `src/`；改 `src/` 的时候要想到 airbase。**
airbase 是活的主线，`src/airsim/` 只读。
（`src/mpc/plan_viz_node.py` 例外 —— 它是**活的**，airbase 的 rviz 可视化还在用它。）
