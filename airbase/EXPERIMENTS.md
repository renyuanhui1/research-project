# airbase 实验台账（视觉伺服 / 世界模型 接触地面飞机）

> 本场景自包含的实验记录，**不与老场景（绿门）`../docs/EXPERIMENTS.md` 混用**。
> 方法总览见 `README.md`；架构见 `README.md` 的「闭环任务架构（两段式制导）」。

---

## §2026-07-22　纯视觉伺服 斜下接触（相机 -50° + cy 视线角闭环）

### 设置
- 场景：缩小飞机（~20m，顶面≈6m）+ 相机下倾 **-50°**（对齐从高空滑向地面的滑降角，目标全程居中）。
- 方法：**纯指纹视觉伺服**（不用世界模型）。`--face-ned` 初始一次性对准 → 之后纯视觉；
  `cx→yaw`（横向对准）、`cy→下降率`（**视线角闭环**，让目标保持画面中心、沿视线接触）。
- 起点 50m，`--v-down 2.0`（滑降角 ~34°）。落盘 `signals.csv`/`run.h5`/`viz_dump/`。

### 离线信号（`check_target_signal.py`，`airbase_tgt1_100m`）
- cost/peak/mass vs 真实距离**单调**：Spearman **+0.81 / −0.85 / −0.93**。
- 结论：信号沿接近轨迹**有梯度**。**但这是必要非充分**——只测了"沿好路靠近时单调"，
  **没测**判别性（干扰物）、部位可分、闭环控制。实跑踩的坑都在这后半张考卷。

### 指纹判别性（rviz 热力图 + 多模板）
- 整机/机头模板热力图**整机全红**（`plan_viz_node` 用绝对刻度 sim≥0.6 才红，非 min-max 假象）
  → 指纹只分"**飞机 vs 背景**"，**不分部位**。
- 根因：`proto = 模板所有 patch 求平均` → 塌成"通用飞机表面特征"；远距离整机才占几个 patch。
- 远处车辆、跑道标线**易被误标红**：DINO 语义特征相似，**非 UE 渲染问题**，换工具同样存在。

### 闭环两条对照（50m 起，-50°，v_down 2.0）
| 模板 | 结果 | 末位置/高度 | 问题 |
|---|---|---|---|
| **机头** | 位置最好，居中填满画面 | (-60.4,-14.4)，**14.4m** | `mass=0.20` 提前停（整机模板 mass 早饱和），**未接触** |
| **红星** | 判别性看似干净（mass 全程低） | (-74.8,-22.6)，4.4m | 因 mass 低→不减速→**水平冲过头 ~11m 脱靶**，末端降到目标后方空地、误匹配跑道标线 |

- 终点帧证据：机头末帧飞机居中填满；红星末帧几乎全是空地、飞机只剩边缘。
- **横向对准全程可靠**（`cx≈0`）；cy 视线角闭环**修好了旧"落在目标前面"的根因**（旧版开环匀速降→目标沉出画面）。

### 待修点（下一步）
1. **停太高（机头）** → 用 `--stop-alt`（本次新增，按高度判接触，弃早饱和+噪声的 mass 判停）。
   机头 + `--stop-alt 7` 大概率可直接接触。
2. **水平冲过头（红星）** → 减速不能只靠 mass；拟加**"低空自动减速前进"**（不依赖 mass，低空即减速），
   任何模板都不会冲过头。
3. 部位可分性（机头 vs 尾翼）**目前不成立**（整机全红）；如需分部位，走"更聚焦小模板 + 峰值/top-k 代替质心平均"或"近距离才分部位"的两段式，另开验证。

### 产物
- `outputs/runs/servo/{整机,机头,红星,驾驶舱,机头2}_0722_*/`：`signals.csv` + `run.h5` + `viz_dump/`
- 各 run 下 `replay.mp4`（带标注：绿十字=画面中心、红圈=指纹质心 cx/cy）
- `outputs/recordings/approach/airbase_tgt1_50m.h5`（-50° 接近录像，当 stats-episode + 模板源）

---

## §2026-07-23　伺服转 PX4（方案B: MAVSDK 直控，代码落地）

### 架构
- 飞控从 AirSim simpleflight 换成 **PX4**；`ProjectAirSim` 只剩「渲染 + 出 FrontCamera(-50°)」；
  起飞/位姿/下发全部走 **MAVSDK offboard 直连 PX4(14540)**（选方案B 而非经 AirSim 转发，因为将来能无缝上真机 Jetson）。
- 感知（DINO 指纹）与控制律**平台无关、一字不动**：新脚本直接 import 老脚本的 `Fingerprint`/`save_run`，
  控制律在新脚本放机体系副本（`compute_body_cmd`），已数值验证与老脚本世界系控制律**逐项零误差等价**。
- 下发映射：机体系 `set_velocity_body(forward, right, down, yawspeed°/s)`，正好省掉老代码 vf→vn/ve 的 yaw 旋转；
  `yaw_rate` rad/s → deg/s（×180/π）。取位姿：MAVSDK `position_velocity_ned` + `attitude_euler`（后台任务刷最新值）。

### 产物
- `src/airsim/servo_closed_loop_px4.py`（新，老脚本 `servo_closed_loop.py` 未动）。
- `sim_config/robot_quadrotor_px4_airbase.jsonc`（复制自顶层 px4 config，仅 FrontCamera 俯仰 0→-50°）。
- `sim_config/scene_airbase_px4.jsonc`（PX4 场景，出生点 000 与 airbase 一致→PX4 的 EKF home=目标坐标系原点）。
- 依赖：`airsim` conda 环境已 `pip install mavsdk`（纯新增 grpcio+protobuf+mavsdk，零降级）。

### 运行（三方联调）
1. UE 前台开 airbase 关卡；ProjectAirSim 载 `scene_airbase_px4.jsonc`。
2. 启 PX4 SITL：`export PX4_SIM_HOST_ADDR=172.21.192.1 && cd ~/PX4-Autopilot && make px4_sitl none_iris`。
3. `python src/airsim/servo_closed_loop_px4.py --template pictures/尾翼.jpg --stats-episode ... --face-ned -64.2 -18.5 --start-altitude 40`。

### 待 SITL 实测校准（第一次跑重点看）
- **NED 原点对齐**：MAVSDK 的 NED 相对 PX4 EKF home；出生点已设 000，需确认 `--face-ned`/`--stop-alt` 坐标对得上。
- **offboard 设定值频率**：闭环周期含 DINO 单帧推理，须 <0.5s（GPU 上没问题；CPU 慢会触发 offboard failsafe）。
- 目标是否在 PX4 世界可见（同一 UE 关卡，出生点朝向能否让目标进画面）。
- 真机（后续）：机载 GPU(Jetson)、向下测距（判接触高度）、相机 -50°/FOV 标定、接触安全。

---

## §2026-07-24　PX4 SITL 首飞冒烟测试（三大风险全部清掉 ✅）

### 设置
- 三方联调：UE(airbase 关卡，前台) + PX4 SITL(`make px4_sitl none_iris`, `PX4_SIM_HOST_ADDR=172.21.192.1`)
  + `servo_closed_loop_px4.py`（MAVSDK offboard 直控）。
- 保守跑法：只验平台层，不冒险接触。
  `--template pictures/机头.jpg --face-ned -64.2 -18.5 --start-altitude 40 --stop-alt 30`
- 本机 GPU：RTX 4060 Ti（DINO 单帧推理够快，不会拖垮 offboard 频率）。

### 结果：全链路一次跑通
时序：ProjectAirSim 载入 `SceneAirbasePX4` → DINO 指纹就绪 → MAVSDK 连上 PX4 → 位置就绪
→ arm → 爬升 40m → `--face-ned` 转向对准 → 视觉伺服闭环 36 步 → 40m 降到 29.9m → `--stop-alt` 正常触发停止。

**07-23 记的三大风险逐条清掉：**

| 风险点 | 实测结果 |
|---|---|
| NED 原点对齐 | ✅ 出生点 000 生效；爬升高度、`--face-ned` 转向后 `cx` 从 +0.34 收敛到 ~0.00，坐标系对得上 |
| offboard 设定值频率 | ✅ 连续 36 步 dt=0.3 全程无 failsafe，PX4 全程听命令 |
| 目标可见性 | ✅ 目标进画面并被指纹锁住，横向对准可靠（与 07-22 simpleflight 表现一致） |

- **控制律在 PX4 上行为与 simpleflight 一致**：`cy` 在 −0.15~−0.43 间波动，`down` 随之在 0.54~1.12 调节
  → **cy 视线角闭环在 PX4 机体系下发(`set_velocity_body`)上正常工作**，机体系移植没引入偏差。
- `mass≈0.000~0.005`、`peak≈0.08~0.26` 偏低，但 30–40m 高度属**预期**（07-22 机头模板也是到 14m 才 mass=0.20）。
- PX4 侧仅 `Preflight Fail: system power unavailable`（SITL 无电源模块，无害），无 offboard failsafe。

### 代码修正（本次发现的坑）
- `servo_closed_loop_px4.py` 的 `--stats-episode` 默认值是 `airbase_tgt1_100m.h5`（**7-21 录、相机还是 -35°**），
  而当前相机是 **-50°**，对应 `airbase_tgt1_50m.h5`（7-22 录）。相机角度不同→画面分布不同→DINO 特征统计不同，
  用错会让指纹信号整体偏掉。**默认值已改为 50m**。（docstring 例子本来就是对的，只是 argparse 默认没跟上。）

### 产物
- `outputs/runs/servo/px4_机头_0724_112632/`：`signals.csv` + `run.h5` + `viz_dump/`

### 下一步
1. **第 2 步完整伺服接触**（未跑）：`--stop-alt 7` + 机头模板，放开降到接触高度。
2. 07-22 遗留的"低空自动减速"（治红星水平冲过头）仍未实现；因场景中各模型尺寸不一，
   靠调 `mass` 阈值不通用，故优先级排在平台验证之后。
