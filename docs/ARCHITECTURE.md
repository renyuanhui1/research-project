# 架构文档

> 维护本文件以描述应用内部架构与工作原理（见 `CLAUDE.md` 第 5 条）。

## 3a. MPC 演示（`src/mpc/`，独立模块）

与主采集/世界模型链路解耦的教学演示，验证 MPPI（Model Predictive Path Integral）规划器：

- `mppi_demo.py`：ROS2 Humble 节点。差速小车（unicycle，状态 `[px,py,theta]`、控制 `[v,omega]`）
  从起点导航到目标、绕开圆形障碍。每步采样 K 条控制序列→向量化 rollout→按代价指数加权得最优控制，
  执行第一步后滚动优化。发布 `visualization_msgs/MarkerArray`（障碍/目标/小车/采样轨迹束/最优轨迹）
  和 `nav_msgs/Path`（已走路径），fixed frame = `world`。
- `mppi_demo.rviz`：预配好的显示项。
- `mppi_demo.launch.py`：用 `ExecuteProcess` 同时起节点和 rviz2，无需 colcon 构建。
- 运行：`ros2 launch src/mpc/mppi_demo.launch.py`（依赖 numpy，可在 `airsim` conda 环境里跑）。

`aerial_pursuit_demo.py`：3D 空中跟踪版本。无人机（二阶积分，状态 `[p,v]` 6 维、加速度控制）
用 MPPI 持续悬停到一个沿 8 字曲线移动的地面目标正上方。期望悬停点 = 对目标位置+速度做**匀速外推
预测**后 + 悬停高度偏移，体现"预测目标未来 + 滚动重规划"。发布 `aerial/markers`（地面/目标/悬停点/
无人机/高度线/采样束/最优轨迹）和两条 `Path`（无人机、目标轨迹）。配套 `aerial_pursuit.rviz`、
`aerial_pursuit.launch.py`，运行 `ros2 launch src/mpc/aerial_pursuit.launch.py`。

### 3a′. 为什么还需要世界模型（demo 的边界与创新点定位）

上面的 MPPI demo 效果很好，但**它被白送了三样真实无人机问题里没有的东西**，想清楚这三样即可定位世界模型的价值与创新点：

| | MPPI demo | 真实无人机问题 |
| --- | --- | --- |
| 状态从哪来 | 直接拿到精确坐标 `[x,y,z]`、目标 `[x,y]` | 只有 FrontCamera 像素，无上帝视角坐标（森林树冠下 GPS 拒止） |
| 模型从哪来 | 二阶积分/unicycle 解析公式 | 场景演化（前进→树逼近/遮挡/视差）写不出方程，须从数据学 |
| 代价从哪来 | 到已知坐标的欧氏距离，又光滑又单调 | 只有 DINO 潜距，又平又非单调（见 §5 根因） |

**世界模型的优势 = 在拿不到上述三样时仍能工作**：①从像素规划、不需度量状态；②动力学学出来而非写出来（脚本 6 的 action-conditioned predictor，离线 held-out 已验证有效/听命令/泛化）；③目标可用"一张图/视觉概念"指定；④一个模型多任务复用、样本效率高于 model-free RL。

**但边界要诚实**：若真实任务只是"接近一个可被检测器还原成坐标的目标"，那么 检测器 + 经典 MPC（就是这个 demo）可能就够了，不必上世界模型。世界模型真正不可替代的场景，是目标无法还原成干净坐标、动力学无法写成方程时（如森林里朝视觉指定目标穿行）。

**创新点定位（与 §5 一致）**：世界模型（预测器）本身不是瓶颈，瓶颈是目标函数。demo 之所以丝滑，正因白送了一个完美单调的 cost；闭环之所以卡，正是缺这个。故真正的创新点不在"再训一个更好的预测器"，而在 §5 结尾的方向——**在没有度量状态、没有解析模型的像素世界里，造出一个像 demo 里那样单调好用的 cost**（target-conditioned 目标：模板→DINO patch 指纹→"目标居中+变大"，又尖又单调、且泛化）。一句话：**demo 免费拥有的好模型 + 好 cost，正是本研究要挣来的东西；预测器已验证，单调目标函数是当前该 all-in 的创新点。**

## 3b. 世界模型数据流水线（DINO-WM，进行中）

目标见 `docs/claude_code_task_pipeline.md`：采集 (图像, 动作) 序列训练
action-conditioned latent predictor。脚本按 1→6 顺序、逐个跑通验证。

- **脚本 1 `src/airsim/decode_check.py`（已跑通验证）**：Pub/Sub 订阅
  `FrontCamera/scene_camera`，回调缓存最新帧，`decode_image()` 把 `image_msg["data"]`
  解成标准 RGB 存 png。`decode_image()` 设计为可复用（脚本 2 直接复用）。
- 已确认相机帧事实：`encoding='BGR'`、1920×1080×3、`len(data)=6220800`；
  Pub/Sub 帧 data 是 bytes（reqrep get_images 则是 list）；存 cv2 需 RGB→BGR 再写。
- 已确认 API：`move_by_velocity_async(..., yaw=, yaw_is_rate=True)` 支持 yaw_rate，
  故动作可取 4 维 `[vx, vy, vz, yaw_rate]`；`takeoff_async`/`move_*` 均返回需 await 的 task。
- **脚本 2 `src/airsim/collect_episode.py`（已跑通验证）**：复用 `decode_image`，
  飞预设任务相关轨迹（前进/爬升/横移/偏航分段组合 + 动作高斯噪声 + 起点随机化），
  存 HDF5：rgb(T,224,224,3)u8 / action(T,4)f32 / pose(T,7)f32 / time(T,)i64。
  - 时间对齐约定：每步先取帧再下发 a_t，即 a_t 驱动 image_t→image_{t+1}。
  - **节拍=事件驱动均匀 10Hz**：相机 `FrontCamera` capture-interval 改为 0.1（10Hz），
    循环每步 `wait_new_frame()` 等下一张新帧即记录+派发动作 → dt 均匀 ≈100ms
    （实测 min/中位=100ms、std≈15ms；旧"await 完成"版本是 7.5Hz 且 dt 双峰抖动）。
  - 派发动作只发送、不等执行完成（`await` 仅提交指令拿 task 句柄）；指令时长 dt×1.5
    使整步内速度持续有效、无滑行空档。打印 `赶不上相机` 步数判断是否 CPU/RPC 受限。
  - 每条 ~22MB(gzip)。依赖：`h5py`（已装入 conda airsim 环境）。
- **脚本 3 `src/airsim/replay_check.py`（已跑通验证）**：读 HDF5 做数值检查
  （长度一致 / 时间单调 / 无空图 / 位移合理），并拼成带 action/pose/time 标注的 mp4
  供肉眼复核对齐。视频 fps 默认按真实时间戳算（视频时长=真实飞行时长）。离线脚本，不连仿真。
- **脚本 4 `src/airsim/batch_collect.py`（已跑通验证）**：连接一次、循环内**重载场景**
  做干净重置（无人机回 spawn，无跨 episode 漂移），复用脚本 2 的 `run_episode/save_hdf5`。
  - 4 个轨迹模板（mixed/forward_climb/yaw_scan/lateral）轮询保证覆盖均衡，每条随机缩放
    速度幅值（0.8~1.2）、seed 驱动动作噪声与起点随机化；每条存 `episode_XXXX.h5` + 写 `manifest.json`。
  - 每条流程：reset_cache→subscribe→arm/takeoff→run_episode→save→land→unsubscribe。
  - **开跑前相机热身**（`warmup_camera`）：空拉帧直到帧间隔稳定到 ~100ms 才开始 ep0，
    解决相机冷启动（前几条 dt 偏大、抖动大，实测 ep0 332ms→131ms）。`--no-warmup` 可关。
  - 估时：~固定开销17s + steps×dt 飞行；200步/条 ≈ 40s/条（50条≈33min）。
  - 至此**数据闭环（脚本 1→4）打通并验证**。
- **最终训练数据集**：`outputs/datasets/episodes_dataset/`，**53 条**（连续编号 + `manifest.json`）。
  由多批采集合并整理而成（`src/airsim/consolidate.py`：剔除 <5Hz 退化条、裁掉撞门冻结尾、
  连续重编号）。全部 ≥8Hz、dt 一致、模板均衡（mixed16/forward_climb14/yaw_scan13/lateral10），
  长度 131~200 帧（8 条裁过尾）。
- **脚本 5 `src/airsim/extract_dino_features.py`（已写，已跑）**：冻结 DINOv2 抽 latent，
  纯 torch+h5py 离线预处理，不连仿真。读 `episode_*.h5` 的 rgb(224×224)→ ImageNet 归一化
  → `model.forward_features` 取 patch tokens，存 `episode_XXXX_dino.h5` 的 `z (T,256,dim)`
  （默认 fp16；`--include-cls` 另存 CLS）。已存在则跳过、可续跑。
  - 待办：装 `torch torchvision`；首次 `torch.hub.load("facebookresearch/dinov2", ...)` 联网下权重。
  - 默认 `dinov2_vits14`(dim384)；224/14=16→256 patch。
- **脚本 6 `src/airsim/train_predictor.py`（已写，已跑）**：训练 action-conditioned latent
  predictor（世界模型核心）。读 latent(`z`)+action 构造 `(z_t, a_t, z_{t+1})`，
  最简 **patch-latent Transformer**（动作过 MLP 广播加到每个 patch + 位置编码 + 残差预测），
  损失 **MSE**。支持单步(`--horizon 1`)/多步 rollout、`--overfit` 小数据验证、留出集 rollout 评估。
  - 时间对齐：样本 = `(z[t], action[t], z[t+1])`，即 a_t 驱动 z_t→z_{t+1}。
  - 跑在统一的 `airsim` 环境；当前权重存于 `weights/predictor_h5.pt`。
  - **训练要点**：latent 必须标准化（否则卡 identity，见 §5）；多步训练 `--horizon`
    对规划很关键（h5 长 rollout 比 h1/h3 更稳）。看效果用"超过 identity 基线多少%"+
    `verify_predictor.py`（验证集单步 vs identity），别只看绝对 loss（会被尺度骗）。
- **脚本 7 `src/airsim/plan_mppi.py` / `plan_closed_loop.py`（进行中）**：MPPI 目标图像规划。
  - 阶段 A `plan_mppi.py`（离线潜空间，已验证）：起点 z0、目标 z_goal=z[H]，采样动作序列
    用世界模型 rollout、按到目标距离 softmax 加权更新（CEM 式 sigma 退火）。温度 0.1 是关键
    （1.0 太软不收敛）。结果 d_mppi 明显小于 do-nothing，规划逻辑通过。
  - 阶段 B `plan_closed_loop.py`（本地一体化）：每步 FrontCamera→DINO→z_t→MPPI→执行→重规划。
    本地 4060 Ti：torch 2.6+cu124 装入 `airsim` 环境，DINOv2 仓库在 `/home/hui/my/dinov2`、
    权重在 `/home/hui/my/weights/`；fp16+warm-start、显存 346MB。
    机制层已修：`--init-vx`(前进初始化，否则从零速搜索得零动作)、`--iters`(多迭代+sigma退火)、
    `--smooth-beta`(AR(1)时间相关噪声，治小鸡啄米)、`--cost-metric poolcos`(池化余弦目标)、
    `--record-goal`(自飞录"保证可达"的目标图)、`--dry-run`/`--diag` 自测。
    **仍不收敛，根因是目标函数(DINO潜距)，见 §5。下一步：target-conditioned 目标(模板+DINO指纹)。**
  - `check_cost_monotonic.py`（**已跑，2026-07-03，结论性结果**）：离线单调性对比。同一条真实
    飞近目标的 episode（0050，23.75m→0，目标=绿门）上逐帧算三种 cost，以 pose 真实距离为参照
    （坐标只当离线评估尺子，cost 本身不碰坐标），输出曲线 png + csv + Spearman ρ；
    偏航段（|yaw_rate|≥0.3）标灰并额外算剔除后的 ρ_facing（米制距离不懂朝向，偏航段 cost 上涨
    是正确行为，ρ_all 会被污染）。**结果**：patchmse ρ≈-0.11（正对着飞也 -0.10 —— 旧 cost
    非单调的定量实锤，与偏航无关）；poolcos ρ_facing=+0.68；**target 指纹调优后 ρ_facing=+0.82**
    （关键：mass_thresh 0.35→0.2 把躺平的面积项救活成单调主力，权重改为
    conf 0.5 / center 0.4 / size 1.2 / sharpness 10）。分量结论：mass=单调主力，conf≈常数开关，
    center=转向信号（偏航段上涨是特性）。**验证阶梯**：①真实帧单调 ✓ → ②预测 latent 下动作
    排序 ✓（2026-07-03，`check_target_action_ranking.py` + REAL_future 真实帧对照行）→ ③闭环
    （`plan_closed_loop.py` 传同组权重，起点建议 ≤12m）。
  - **②级结果**：frame100/170 处 dataset_true 预测 delta 与真实帧 delta 几乎一致
    （-0.055 vs -0.058、-0.235 vs -0.233）→ 世界模型对指纹特征的预测是准的，模型再次排除嫌疑。
    frame170（~3m）排序教科书级：yaw_right(居中)>dataset_true>forward>...>zero，接近动作大幅
    领先不动。zero 在远/中距离有轻微漂移增益（-0.08/-0.11），近距离消失。
  - **模板对决（全程素材）**：`record_approach.py` 录 `outputs/recordings/approach/approach_full.h5`
    （出生点→绿环门前 0.2m，41.6m 直线，14m 高度；因实际 8.3Hz vs 名义 10Hz 超程 ~2m，
    末 ~30 帧贴脸退化）。三模板全程 ρ：环带 `tmpl_ring_rim.png` **0.847** > 整环 real.jpg 0.781
    > 旧远景糊模板 0.626（远段专用，近段崩）。环带 cost 碗底在门前 4.8m（-0.26），41m 起全程
    有梯度，<5m 回升，贴脸跳回 +0.25。**goal-thresh 重标定为 -0.18（≈门前 5~6m）**；换模板/权重
    必须重标定（量表会平移，旧默认 0.5 曾致起点即误报到达）。poolcos/patchmse 在该素材 ρ=0.85
    是假象：目标图=退化末帧、无梯度靠尾部跳崖。闭环建议 ~11m 高度（WM 训练集 6~11m）。

## 4. 关键配置

- `sim_config/my_scene.jsonc` + `sim_config/my_robot.jsonc`：当前自定义机体与场景（4 旋翼，fast-physics，
  含 Chase / FrontCamera 相机、IMU、Lidar、GPS 等传感器）。
- 相机 `capture-settings` 的 `image-type`：0=RGB 场景，2=深度透视。
- **采集减负**：`robot_quadrotor_fastphysics_sensors.jsonc` 里已关掉吃 GPU/CPU 的传感器
  （只关不删、带注释）：Chase 相机 `enabled:false`、FrontCamera 的 depth `capture-enabled:false`、
  lidar1 `enabled:false`。世界模型只用 `FrontCamera/scene_camera`；IMU/GPS/气压计/磁力计保留
  （Simple Flight 状态估计可能要用）。关后飞行正常，飞行帧率 7.6→8.0Hz。Chase 关后 UE 视角
  自动切 FrontCamera 第一人称。

## 5. 已知坑

- **闭环规划不收敛 = 目标函数问题（非动作 OOD；2026-06-27 的 OOD 判断已于 07-01 系统实测推翻）**：
  ①**模型没问题**：多步开环（真实平滑动作）每个 horizon 都赢 identity（H15 模型 0.47 vs 1.07）；
  动作敏感、误差与动作幅度无关（相关 -0.05）；最近邻可视化前 ~10 步忠实跟踪。数据也不窄
  （mixed/yaw_scan/lateral/forward_climb 四模板，vx 到 2.4、横移 ±1.9、偏航 ±1）。
  ②**planner 机制已修**（见 §3b 脚本7）：前进初始化、多迭代+退火、动作平滑（治小鸡啄米）；
  采样数非瓶颈（64≈512，靠迭代次数而非样本数）。
  ③**真正根因 = 目标函数**：DINO 全局 patch-MSE 潜距又平又非单调——前进 rollout 代价 ≈ 原地
  （前进动作几乎无贡献）；真实帧 frame_k→goal 距离非单调（frame5 比 frame0 还远）→ MPPI 被带去
  横移/冲过头，best 降（幻觉）而真实 dist 涨。换池化余弦（标准化空间）较单调、能缓解但仍不收敛
  （即便"保证可达"的自飞目标，也横移漂 + 冲过头，dist 0.25→0.9）。
  **方向：放弃整图匹配，改 target-conditioned 目标**——给目标模板 → DINO patch 指纹 → 定位
  "目标居中 + 变大"，又尖又单调，且泛化（换目标只换模板图、不用颜色规则），正好对接后续
  "无人机靠近对地目标"任务。原型已验证 DINO 指纹能定位目标，待做：紧模板 + 对比/单调目标 + 接 MPPI + 仿真。
  另：状态 OOD（须爬升到 ~6m）已在 plan_closed_loop 修；颠簸→`--smooth-beta`/`--replan-stride`；
  往下掉→`--vz-max` 限制下降。
- **训练前必须标准化 latent**：DINO patch token 各维方差大且不均（实测每元素方差≈5.7），
  直接 MSE 会让模型退化成 identity（train_mse 卡在基线、调 lr/epoch 都没用）。
  按训练集每维均值/方差标准化后解锁（注意 latent 存 fp16，求统计要先升 float32 否则求和溢出）。
- **采集帧率取决于 UE 是否前台**：UE 在前台且不跑其他占 GPU 的软件时，空载 ~10Hz、飞行 ~8Hz；
  UE 一旦切后台 / 用户操作其他软件，相机渲染被压到 ~3Hz（333ms），热身 60s 都热不起来。
  曾误判为"场景反复重载累积退化"，实为前台/GPU 抢占；"Use Less CPU when in Background"
  没勾也照样掉，是渲染前台行为而非该开关。**铁律：采集全程把 UE 放前台、别同机跑占 GPU 程序。**
  窗口大小无影响（相机渲染到固定 1080p 离屏目标）。3Hz 数据 dt 仍稳但帧间运动 ~2.5 倍，
  与 8Hz 混训会让 (z_t,a_t)→z_{t+1} 不自洽，须剔除。
- **UE 采集纯白**：地图过亮时，采集路径（SceneCaptureComponent2D，无自动曝光历史）会过曝爆白，
  而视口因 Eye Adaptation 看着正常。修法：在地图 Post Process Volume 锁 `Min EV100 = Max EV100`
  为视口收敛的 EV100 值（实测某白天图 ≈13.6）。
- 轨迹脚本对 COLMAP→NED 的坐标系对齐方式（首末对齐 / 仅起点 / 统一 scale）有多个版本，
  见 `src/trajectory/` 各文件头部注释。

## 6. 约定

- COLMAP 等外部输入放 `data/`；生成的数据集、缓存、评估和运行记录按 `outputs/README.md` 分类；模型放 `weights/`。
- **单一 conda 环境**：项目统一使用 `airsim`，同时承担仿真采集、DINO 特征提取、世界模型训练与闭环规划。
  - 数据流：仿真采集 → `episode_*.h5` → DINO 抽特征 → 世界模型训练 → 闭环规划。
