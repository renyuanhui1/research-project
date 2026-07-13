# 任务说明：UAV 世界模型数据采集与训练流水线（交给 Claude Code）

## 背景与目标

硕士课题：基于冻结 DINOv2 特征的潜空间世界模型（DINO-WM 范式）做无人机视觉语言导航。
当前阶段目标：在 ProjectAirSim 仿真里采集 (图像, 动作) 序列数据，训练一个
action-conditioned latent predictor（即世界模型），它学习的映射是：

    (z_t = DINO(图像_t),  a_t = 动作)  →  z_{t+1} = DINO(图像_{t+1})

注意：训练时不存 latent，只存原始图像；latent 在训练时用冻结 DINO 现场抽。
采集不需要人工标注，不需要指令，不需要 PX4（用 ProjectAirSim 自带 Simple Flight）。

---

## 已验证的环境事实（重要，基于实际运行确认，勿改）

- 仿真：ProjectAirSim，Python 客户端，已确认能连接、加载场景、收到相机帧。
- 场景名：`/Sim/SceneDroneSensors`；scene 配置：`scene_drone_sensors.jsonc`
- robot 配置：`robot_quadrotor_fastphysics_sensors.jsonc`
- 无人机名：`Drone1`
- 控制器：Simple Flight（非 PX4）
- 飞行 API（异步，asyncio）：
  `drone.enable_api_control()` → `drone.arm()` → `await drone.takeoff_async()`
  → `await drone.move_by_velocity_async(v_north, v_east, v_down, duration)`
  → `await drone.land_async()` → `drone.disarm()`
- 单位 SI（米/弧度），坐标 NED。
- 相机走 Pub/Sub（端口 8989），控制走 Req/Rep（端口 8990）。
- 可用相机：`Chase`、`FrontCamera`（FrontCamera 另有 depth）。
- 已成功订阅：
  `/Sim/SceneDroneSensors/robots/Drone1/sensors/FrontCamera/scene_camera`
  订阅方式：`client.subscribe(topic, callback)`，回调签名 `callback(topic, image_msg)`
- **相机图像消息 image_msg 的字段（已确认）**：
  `time_stamp, height, width, encoding, big_endian, step, data,
   pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, annotations`
  —— 即每帧自带：时间戳、位姿(pos_xyz + 四元数 rot_wxyz)、原始像素 data。

### 运行前需确认（第一个脚本要先解决）
- 相机 capture-settings 里必须 `capture-enabled: true`（否则脚本收不到像素，
  streaming-enabled 只是网页预览，不送像素到脚本）。当前已能收到帧，说明已满足。
- 需打印并确认 `encoding` 的确切值（如 "rgb8"/"bgra8" 等）+ `len(data)`，
  用来确定 reshape 的通道数与顺序。这是脚本 1 的首要任务。

---

## 需要的脚本（按依赖顺序）

### 脚本 1：`decode_check.py` —— 解码验证（最先做，最小步）
**目的**：确认能把 image_msg["data"] 正确解成图像并存成 png。
**要做的**：
1. 连接、加载场景、初始化 Drone1、订阅 FrontCamera/scene_camera。
2. 回调里对第一帧：打印 `encoding, height, width, step, len(data)`。
3. 根据 encoding 把 data（bytes）reshape 成 numpy 数组 (H, W, C)。
   - 通道数 C 用 len(data)/(height*width) 反推校验，别硬编码。
   - 注意 encoding 决定通道顺序（RGB vs BGR vs BGRA），存图前转成标准 RGB。
4. 用 cv2 或 PIL 存成 `frame_000.png`，肉眼确认画面正确（不是白图/花屏）。
5. 起飞飞几秒确保有帧，存 3~5 张不同时刻的图。
**通过标准**：存出的 png 是清晰正确的仿真画面。

### 脚本 2：`collect_episode.py` —— 单条 episode 采集
**目的**：飞一条预设轨迹，同步记录 (图像, 动作, 位姿, 时间戳)，存成 HDF5。
**要做的**：
1. 复用脚本 1 验证过的解码逻辑（建议抽成 `decode_image(image_msg)` 函数复用）。
2. 回调把最新帧（解码后的图像 + 自带的 pos/rot/time_stamp）存入一个共享缓存。
3. 主循环：按固定 dt（先用 0.1s，即 10Hz）执行预设速度指令序列；
   每个时间步：
     - 下发 move_by_velocity_async(vx, vy, vz, duration=dt)
     - 记录这一步的动作 a_t = [vx, vy, vz, yaw_rate]（yaw_rate 见下方说明）
     - 从缓存取当前帧图像 + 位姿 + 时间戳
     - 把 (image, action, pose, sim_time) 追加进 episode 列表
4. 预设轨迹要"任务相关"：包含前进、上升(v_down 负)、左右横移、偏航的组合，
   不要只飞直线（原因：世界模型要覆盖 MPPI 推理时会用到的机动）。
   每条基础轨迹叠加随机扰动（动作加噪 + 起点位姿随机化），便于后续批量生成多样数据。
5. 存 HDF5：一个文件 = 一条 episode，datasets 至少含：
     - rgb:    (T, H, W, 3) uint8
     - action: (T, 4) float32   # vx, vy, vz, yaw_rate
     - pose:   (T, 7) float32   # pos_xyz + quat_wxyz（调试/replay 用，不喂模型）
     - time:   (T,)  int64      # time_stamp（纳秒）
**关键约束（务必正确，否则训练数据是错的）**：
   动作与图像必须时间对齐：a_t 必须是"导致 图像_t → 图像_{t+1} 的那个动作"。
   即在 t 时刻记录的图像，配的是 t 时刻"即将执行"的动作。整条序列保持这个约定一致即可。
**关于 yaw_rate**：确认 move_by_velocity_async 这版怎么传偏航（是单独 yaw_mode/yaw_rate
   参数，还是别的形式）。先查 API 签名再写，别假设。如这版不支持 yaw_rate，
   先用 vx/vy/vz 三维动作，action 维度相应改成 3，后续再加偏航。

### 脚本 3：`replay_check.py` —— 采集质量验证
**目的**：把采好的 HDF5 放出来，肉眼确认图像序列流畅、和动作/位姿对得上、无白图错位。
**要做的**：
1. 读 HDF5，按时间顺序把 rgb 帧逐张显示（或拼成视频），叠加显示当前 action、pose。
2. 检查：帧数与 action 数一致、时间戳单调递增、图像不空、轨迹与 pose 变化一致。
**通过标准**：能确认这条 episode 数据干净、对齐。

### 脚本 4：`batch_collect.py` —— 批量采集
**目的**：把脚本 2 放进循环，生成多条 episode（先目标 20~50 条跑通，再扩到几百条）。
**要做的**：
1. 定义一组基础轨迹模板 + 随机化参数（起点、速度幅值、扰动强度）。
2. 循环调用 episode 采集，每条存独立 HDF5（命名含编号），记录元信息（轨迹类型等）。
3. 注意 ProjectAirSim 一次只允许一个客户端连接，循环里管理好连接/场景重置。

### 脚本 5：`extract_dino_features.py` —— 抽 DINO 特征（训练前处理）
**目的**：用冻结的 DINOv2 把所有 episode 的 rgb 抽成 latent，供训练用。
**要做的**：
1. 加载冻结 DINOv2（先用 ViT-S/14 或 B/14，输入按 DINO 要求 resize/normalize）。
2. 对每条 episode 的每帧抽 patch feature 或 CLS+patch（按 DINO-WM 论文用 patch tokens）。
3. 存成与原 episode 对应的 latent 文件（z: (T, num_patches, dim) 或按设计展平）。
   注意：DINO 冻结、不训练，所以这步是一次性预处理，可缓存复用。

### 脚本 6：`train_predictor.py` —— 训练世界模型（核心）
**目的**：训练 action-conditioned latent predictor。
**要做的**：
1. Dataset：从 latent + action 构造 (z_t, a_t, z_{t+1}) 样本（或多步 (z_t..z_{t+k})）。
2. 模型：一个以 (z_t, a_t) 预测 z_{t+1} 的网络（参考 DINO-WM：ViT/transformer 类
   预测器在 patch latent 上做动作条件预测）。先求结构最简、能跑通、loss 能降。
3. 损失：预测 latent 与真实 latent 的 L2（或按 DINO-WM 设计）。
4. 先小数据过拟合验证（确认能学到东西），再上全量。
**通过标准**：训练 loss 稳定下降，能在留出序列上做出合理的多步 latent rollout。

---

## 给 Claude Code 的执行建议
- 严格按 1→6 顺序，每个脚本跑通验证后再写下一个，不要跳步。
- 脚本 1、2、3 是当前重点（数据闭环）；4、5、6 数据闭环通过后再推进。
- 凡涉及 ProjectAirSim API 具体签名（subscribe、move_by_velocity_async 的 yaw 参数、
  连接/场景重置、解码 data 的字段语义），先查本地 projectairsim 包的实际签名/示例，
  不要凭印象假设；不确定就先写一个最小验证打印出来确认。
- 所有"魔法数"（dt、分辨率、轨迹参数、DINO 型号）集中成配置，便于调。

## 暂不需要做的（避免分散精力）
- 不接 PX4（那是后期真机部署）。
- 不换无人机模型 / 不改物理结构（X vs + 构型不影响采数据）。
- 不引入 ROS2 / Gazebo / CBF（与当前阶段无关）。
- 不调 WorldVLN（它只是参考对象，方法已吸收，不在主线）。
