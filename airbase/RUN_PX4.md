# airbase PX4 联调 · 启动手册

> 每次跑 `servo_closed_loop_px4.py` 照这一页做即可，不用回忆。
> 实验结论记在 [`EXPERIMENTS.md`](EXPERIMENTS.md)，本文件只管**怎么开机**。
>
> **原则：不改 PX4 仓库里的任何文件。** 所有配置都在本仓库内（`sim_config/` + 脚本参数），
> 换设备时 `git clone` 就跟着走；PX4 那边永远保持原版，只需 `make px4_sitl none_iris`。

---

## 三条链路（先看懂这个，出问题不用瞎猜）

| 端口 | 谁跟谁 | 干什么 | 缺了会怎样 |
|---|---|---|---|
| **TCP 4560** | ProjectAirSim(Windows) ⇄ PX4(WSL) | 传感器上行 / 电机下行，**仿真必需** | PX4 卡在没有 `Simulator connected`，飞机不动 |
| **UDP 14540** | PX4 → MAVSDK(WSL) | offboard 遥测 + 我们下发速度设定值 | 伺服脚本卡在「连接 PX4」 |
| **UDP 14588** | ProjectAirSim → 空气 | ProjectAirSim 多余的控制链，方案B 用不到 | 无影响（**故意让它打空**，见下） |

**为什么是 14588 而不是默认的 14580**：PX4 的 offboard 链路（`px4-rc.mavlink` 里
`mavlink start -x -u 14580 -m onboard -o 14540`）不带 `-t`，默认把 14540 发到 `127.0.0.1`，
**直到有人往 14580 发包**才把目标切成那个来源 IP。ProjectAirSim 原本就在往 14580 发，
一发就把 14540 的数据流整条拽去 Windows，WSL 里的 MAVSDK 一个包收不到 —— 这就是以前
「时好时坏、要手敲 `mavlink start -u 14549`」的元凶。把 ProjectAirSim 的控制链改到
PX4 从不监听的 14588，PX4 就一直保持 localhost 模式，MAVSDK 用默认 14540 直接连上。
配置在 `sim_config/robot_quadrotor_px4_airbase.jsonc` 的 `control-port-remote/local`。

---

## 第 0 步：确认宿主机 IP（**换设备/重装 WSL 后必做**）

Windows 宿主的 IP = WSL 的默认网关，现取现用，别背：

```bash
ip route show default | awk '{print $3}'
```

当前设备是 `172.21.192.1`。这个值出现在 **3 个地方**，换设备时要一起改：

| 位置 | 字段 |
|---|---|
| `sim_config/robot_quadrotor_px4_airbase.jsonc` | `"local-host-ip"` |
| `src/airsim/servo_closed_loop_px4.py` | `--address` 的默认值 |
| 启动 PX4 的终端 | 环境变量 `PX4_SIM_HOST_ADDR` |

---

## 每次跑：四步

### 1. UE：前台开 airbase 关卡，点 **Play**

- UE 编辑器必须**在前台**（后台会掉到 3Hz）。
- 关卡里 ProjectAirSim 载入的是 `sim_config/scene_airbase_px4.jsonc`（`id: SceneAirbasePX4`）。

### 2. 终端 A：起 PX4 SITL

```bash
export PX4_SIM_HOST_ADDR=$(ip route show default | awk '{print $3}')
cd ~/PX4-Autopilot && make px4_sitl none_iris
```

✅ **成功标志**：打印 `Simulator connected`，随后出现活的 `pxh>` 提示符。
（`Preflight Fail: system power unavailable` 是 SITL 没有电源模块，无害，忽略。）

### 3. 终端 B：跑视觉伺服

```bash
conda activate airsim
cd /home/hui/my
python airbase/src/airsim/servo_closed_loop_px4.py \
    --template airbase/pictures/机头.jpg \
    --face-ned -64.2 -18.5 \
    --start-altitude 40 --stop-alt 7
```

✅ **正常时序**：`Project AirSim 已连接` → `SceneAirbasePX4` 载入 → DINO 指纹就绪 →
`PX4 已连接` → `PX4 位置就绪` → `arm` → `爬升到 40m` → `对准目标, yaw→...` →
`=== 进入视觉伺服闭环 ===` → 每步打 `alt/center/mass/peak` → `已贴近 ✅`。

**用默认值就好，不用记的参数**：`--mav-url`(=`udpin://0.0.0.0:14540`)、
`--stats-episode`(=`airbase_tgt1_50m.h5`，对应 -50° 相机，**别换成 100m 那条**，那是 -35° 录的)。

常用变体：
- 只想冒烟测试、不真接触 → `--stop-alt 30`
- 换目标部位 → `--template airbase/pictures/{机头,尾翼,整机,红星,驾驶舱}.jpg`

产物落在 `airbase/outputs/runs/servo/px4_<模板>_<时间戳>/`（`signals.csv` + `run.h5` + `viz_dump/`）。

### 4. 要再跑一次？**按这个配方重来**（血泪，别省步骤）

> **UE 里 End PIE → 重新 Play**（几秒，不用关编辑器） + **Ctrl-C 掉旧 PX4、起一个全新的** + 再跑伺服。

原因：PX4 的 4560 模拟器链路是**一次性**的 —— 伺服脚本退出时会 `client.disconnect()`，
4560 随之断开且不再接受新连接；而 ProjectAirSim 在 UE 里的 px4-api 桥客户端断开后不自动复位。

| 操作 | 行不行 |
|---|---|
| 同一个正在跑的关卡直接跑第二次 | ❌ 连不上（4560 不重建） |
| **End PIE + 重新 Play**，UE 编辑器不关 | ✅ 最省事 |
| 关掉整个 UE 重开 | ✅ 但慢，没必要 |

⚠️ 切忌在 UE 关卡还开着时反复杀/重启 PX4，会把 UE 那侧的桥搞进坏状态，越弄越乱。

---

## 卡住了？对症查表

| 症状 | 多半是 | 怎么办 |
|---|---|---|
| 伺服卡在 `连接 PX4: udpin://0.0.0.0:14540` | 14540 数据流被拽去 Windows 了 | 在 `pxh>` 敲 `mavlink status`，看 instance #1 的 `partner IP`：若是宿主 IP(172.21.192.1) 而非 127.0.0.1，说明 14588 那两行没生效 —— 确认 UE 载的是改过的 `robot_quadrotor_px4_airbase.jsonc`，并重走第 4 步配方 |
| PX4 一直没有 `Simulator connected`；`ss -tlnp \| grep 4560` 为空 | 4560 没建立 | 走第 4 步配方（End PIE + 重 Play + 全新 PX4） |
| 伺服连上了但一直不 arm | PX4 没定位好 | 等 `PX4 位置就绪`；EKF home 要靠出生点 000，检查场景是 `scene_airbase_px4.jsonc` |
| 闭环中途 PX4 抢回控制 | offboard failsafe（设定值间隔 >0.5s） | 检查 DINO 是不是跑在 CPU 上（脚本首行会打 `device=`，应为 `cuda`） |
| 帧率掉到 ~3Hz | UE 跑到后台了 | 把 UE 窗口拉回前台 |

**兜底方案（只在上面那条 14588 的修法失效时临时用，用完请查原因，别固化）**：
跑伺服时加 `--mav-url udpout://127.0.0.1:14549`，脚本会停在「连接 PX4」干等；
等 `pxh>` 活了，在里面敲 `mavlink start -u 14549 -r 4000000`，MAVSDK 立刻接上。
这行**只在 pxh 里临时敲，不写进 PX4 的任何启动脚本** —— 写进去换设备就丢，而且掩盖真正的端口冲突。

---

## 换设备迁移 checklist

1. 装 PX4：clone 官方 `PX4-Autopilot`，`make px4_sitl none_iris` 能编过即可，**不改它任何文件**。
2. 改 3 处宿主 IP（见第 0 步表格）。
3. conda 环境 `airsim` 里要有 `mavsdk`：`pip install mavsdk`（纯新增，不影响 torch/projectairsim）。
4. DINO 权重是软链 `weights/dinov2_vits14_pretrain.pth → ../../weights/`，确认软链没断。
5. Windows 侧 UE + ProjectAirSim 插件按原样装，关卡里指向本仓库的 `airbase/sim_config/`。
