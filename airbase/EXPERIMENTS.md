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

### 第 2 步：完整接触（`--stop-alt 7`）— 成功 ✅
- `40m → 6.8m`，85 步，`--stop-alt 7` 触发停止，`已贴近 ✅`。
- 横向对准全程锁死（`cx` 贴 0，最大 ±0.18）。
- **视线角闭环在近距离发力**：降到 ~25m 后 `cy` 转正（+0.13→+0.49，目标沉入画面下半），
  `down` 随之加大到 1.74 沿视线追下去——正是"沿视线接触"，无旧版"落在目标前面"的毛病。
- 结论：PX4 平台上纯视觉伺服可从 40m 一路斜下接触到 ~7m，控制律与 simpleflight 完全一致。

### 运维坑：PX4/UE 联调的重启规律（血泪，务必遵守）
反复重启把三方联调搞挂调了很久，最后测清楚了每次重跑的**最省事配方**：

> **UE 里 End PIE → 重新 Play（几秒，不用关编辑器）** + **起一个全新 PX4** + **跑一次伺服**。

对照实测：

| 操作 | 结果 |
|---|---|
| 同一个正在跑的关卡直接跑第二次（不重置） | ❌ 连不上（4560 不重建） |
| **停关卡 + 重新 Play**，UE 编辑器不关 | ✅ 可以（最省事） |
| 关掉整个 UE 重开 | ✅ 可以但没必要，慢 |

- **根因**：① PX4 SITL 的模拟器链路（TCP 4560）**一次性**——伺服脚本退出会 `client.disconnect()`
  断开 ProjectAirSim，PX4 的 4560 随之断且不再接受新连接，故 PX4 每次必须重起；
  ② ProjectAirSim 在 UE 里的 px4-api 桥，客户端断开后**不自动复位**，需 End PIE + 重新 Play 才干净。
- **失败症状**：伺服卡在"连接 PX4"、`ss` 看 TCP 4560 为空、`udpin 14540` 收不到心跳。
  只要看到这三样，别瞎查——直接按上面配方重来一次。
- 切忌 UE 关卡还开着时反复杀/重启 PX4，会把 UE 的桥搞进坏状态，越弄越乱。

### 大坑：MAVSDK 连不上 PX4 的真正根因 = 14580/14540 端口撞车（时好时坏的元凶）
症状：伺服卡在 `连接 PX4: udpin://0.0.0.0:14540`，但 **TCP 4560 是 ESTAB、PX4 完全正常**
（在 `pxh>` 敲 `commander takeoff` 无人机能起飞，证明 PX4↔仿真走 4560 好好的）。问题纯在 14540 这条 MAVSDK 链路。

- **根因**：PX4 的 offboard 链路是 **instance #1**（`mavlink status` 里 `mode: Onboard, UDP 14580→14540`）。
  而 config `robot_quadrotor_px4_airbase.jsonc` 的 `control-port-local/remote: 14580` 让 **ProjectAirSim 也去连 14580**。
  两个抢同一条链路，谁先连谁当 partner：
  - MAVSDK 先抢到 → partner=WSL 本地 → PX4 把 14540 数据发给 MAVSDK → **成功**（早期几次的运气）
  - ProjectAirSim(Windows 172.21.192.1) 先抢到 → PX4 把 14540 数据**全发去 Windows** → WSL 的 MAVSDK 一个包收不到 → **卡死**
- **铁证**：`mavlink status` 的 instance #1 显示 `partner IP: 172.21.192.1`、`Received Messages: sysid:135`
  （sysid 135 = ProjectAirSim，非 QGC——实测没开地面站也这样）。裸 python 监听 14540 八秒零包。
- 这是**抢占竞态**，不是死锁，所以"时好时坏"。脚本里先载场景(ProjectAirSim 连 14580)再连 MAVSDK，本该总是 ProjectAirSim 赢。
- 三条链路各司其职（别混）：`4560`=ProjectAirSim⇄PX4 传感器/电机(仿真必需)；`14580`=ProjectAirSim 多余的控制链(方案B用不到但它照连)；`14540`=PX4→MAVSDK offboard(被上面挤掉)。

**解法（不改文件、不重编译）：给 MAVSDK 单开一条 ProjectAirSim 碰不到的链路（14549）：**
1. UE Play → 起 PX4 → 跑伺服（`--mav-url udpout://127.0.0.1:14549`）→ 伺服停在"连接 PX4"干等（不会崩）。
2. 等 PX4 打印 `Simulator connected`、`pxh>` 活了，在 pxh 敲：`mavlink start -u 14549 -r 4000000`。
3. PX4 立刻在 14549 开链路，MAVSDK(udpout 主动连，像 QGC)接上 → arm → 爬升。
- 只用 **14549 一个端口**，两处对齐（pxh `-u 14549` ↔ 伺服 `udpout://127.0.0.1:14549`），ProjectAirSim 只认 14580、碰不到 14549。
- 想省掉每次手敲 `mavlink start`，可把那行写进 PX4 启动脚本 `px4-rc.mavlink`（会改 PX4 文件，暂不做）。

### 下一步
- 07-22 遗留的"低空自动减速"（治红星水平冲过头）仍未实现；因场景中各模型尺寸不一，
  靠调 `mass` 阈值不通用，故优先级排在平台验证之后。当前机头模板 + `--stop-alt` 已能稳定接触。
