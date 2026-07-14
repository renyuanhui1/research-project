# 架构文档

> 维护本文件以描述应用内部架构与工作原理（见 `CLAUDE.md` 第 5 条）。

## 0. 当前状态速览（更新于 2026-07-13）

**研究主线**：基于冻结 DINOv2 的潜空间世界模型 + **目标条件代价函数**，在纯视觉（目标坐标未知）下逼近/接触目标。远期目标：空中接触大型对地目标的特定部位。

**已确立的结论**：
- **世界模型（预测器）本身可信**，不是瓶颈：离线 held-out 预测准、动作排序对（见 §3 脚本 6、§4）。
- **真正的瓶颈是目标函数**：整图 DINO 潜距（patchmse/poolcos）又平又非单调 —— 14 连败闭环的根因（§4.1、§8）。
- **目标函数演进路线**：整图潜距 ✗ → **target 指纹**（ρ_facing≈0.82~0.85，单调，§4.2）→ **价值函数 `V(z,z_goal)`**（当前主攻，§4.3）。
- **今日关键结果（§4.3）**：价值函数 **v2 在世界模型 rollout 下，远距离(K=25)仍把动作方向指对(余弦 0.71、符号 8/8)，而 patchmse 崩到 0.34** —— "学习式代价成立"（层级 A）落地，方向选对。
- **今日闭环现状（§6）**：value(v2) 闭环在服务器上跑，**卡在 V 的训练半径 KMAX=25**（远目标处 V 饱和成平面、无梯度，drone 原地打转）→ 下一步 **子目标链（≤25 步 waypoint）**。

**计算拓扑（今日起改为分布式，§5）**：仿真在 Windows 宿主机、编码+世界模型+MPPI 在服务器、跨 LAN 连接；本机 8G 显卡撑不住大预算 MPPI。

**两个平行分支（能用但非研究贡献）**：
- **纯视觉伺服 `--visual-servo`**：指纹 center 直接比例控制、恒定前进。近距离能靠近、换目标免标注，但**是写死的约束控制律、无指令泛化**，只作 baseline / 底层执行器。
- **MPC 教学 demo（`src/mpc/`）**：坐标已知的经典 MPPI，用来反衬世界模型的价值（§2）。

---

## 1. MPC 演示（`src/mpc/`，独立模块）

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

`plan_viz_node.py`：**闭环规划过程的 rviz 可视化节点**（服务于主线，非 demo）。读取
`plan_closed_loop.py --viz-dump` 落盘的 `step_*.npz`，tail 式发布 `/plan/markers`（候选束红→绿染色、
最优绿线、无人机、目标环、文字）、`/plan/traj`（已飞路径）、`/plan/view`（机载画面 + 指纹热力图）。
坐标 NED→ENU。`ros2 launch src/mpc/plan_viz.launch.py dump_dir:=...` 一步起节点+rviz。可实时跟播或回放。

## 2. 为什么还需要世界模型（demo 的边界与创新点定位）

上面的 MPPI demo 效果很好，但**它被白送了三样真实无人机问题里没有的东西**，想清楚这三样即可定位世界模型的价值与创新点：

| | MPPI demo | 真实无人机问题 |
| --- | --- | --- |
| 状态从哪来 | 直接拿到精确坐标 `[x,y,z]`、目标 `[x,y]` | 只有 FrontCamera 像素，无上帝视角坐标（森林树冠下 GPS 拒止） |
| 模型从哪来 | 二阶积分/unicycle 解析公式 | 场景演化（前进→树逼近/遮挡/视差）写不出方程，须从数据学 |
| 代价从哪来 | 到已知坐标的欧氏距离，又光滑又单调 | 只有 DINO 潜距，又平又非单调（见 §4、§8） |

**世界模型的优势 = 在拿不到上述三样时仍能工作**：①从像素规划、不需度量状态；②动力学学出来而非写出来（脚本 6 的 action-conditioned predictor，离线 held-out 已验证有效/听命令/泛化）；③目标可用"一张图/视觉概念"指定；④一个模型多任务复用、样本效率高于 model-free RL。

**但边界要诚实**：若真实任务只是"接近一个可被检测器还原成坐标的目标"，那么 检测器 + 经典 MPC（就是这个 demo）可能就够了，不必上世界模型。世界模型真正不可替代的场景，是目标无法还原成干净坐标、动力学无法写成方程时（如森林里朝视觉指定目标穿行）。

**创新点定位**：世界模型（预测器）本身不是瓶颈，瓶颈是目标函数。demo 之所以丝滑，正因白送了一个完美单调的 cost；闭环之所以卡，正是缺这个。故真正的创新点不在"再训一个更好的预测器"，而在——**在没有度量状态、没有解析模型的像素世界里，造出一个像 demo 里那样单调好用、且可泛化的 cost**。一句话：**demo 免费拥有的好模型 + 好 cost，正是本研究要挣来的东西；预测器已验证，目标条件的单调 cost 是当前 all-in 的创新点。** 目标函数的具体演进见 §4。

## 3. 世界模型数据流水线（DINO-WM，脚本 1→7）

目标见 `docs/claude_code_task_pipeline.md`：采集 (图像, 动作) 序列训练
action-conditioned latent predictor。脚本按 1→7 顺序、逐个跑通验证。

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
  - 每条 ~22MB(gzip)。依赖：`h5py`。
  - 连接地址 `DEFAULT_ADDRESS = "172.21.192.1"`（WSL 看 Windows 宿主机；`--address` 可覆盖，见 §5）。
- **脚本 3 `src/airsim/replay_check.py`（已跑通验证）**：读 HDF5 做数值检查
  （长度一致 / 时间单调 / 无空图 / 位移合理），并拼成带 action/pose/time 标注的 mp4
  供肉眼复核对齐。离线脚本，不连仿真。
- **脚本 4 `src/airsim/batch_collect.py`（已跑通验证）**：连接一次、循环内**重载场景**
  做干净重置（无人机回 spawn，无跨 episode 漂移），复用脚本 2 的 `run_episode/save_hdf5`。
  - 4 个轨迹模板（mixed/forward_climb/yaw_scan/lateral）轮询保证覆盖均衡，每条随机缩放
    速度幅值（0.8~1.2）、seed 驱动动作噪声与起点随机化；每条存 `episode_XXXX.h5` + 写 `manifest.json`。
  - **开跑前相机热身**（`warmup_camera`）：空拉帧直到帧间隔稳定到 ~100ms 才开始 ep0。`--no-warmup` 可关。
  - 至此**数据闭环（脚本 1→4）打通并验证**。
- **最终训练数据集**：`outputs/datasets/episodes_dataset/`，**53 条**（连续编号 + `manifest.json`）。
  由多批采集经 `src/airsim/consolidate.py` 合并整理（剔除 <5Hz 退化条、裁掉撞门冻结尾、连续重编号）。
  全部 ≥8Hz、dt 一致、模板均衡（mixed16/forward_climb14/yaw_scan13/lateral10），长度 131~200 帧。
- **脚本 5 `src/airsim/extract_dino_features.py`（已跑）**：冻结 DINOv2 抽 latent，
  纯 torch+h5py 离线预处理。读 `episode_*.h5` 的 rgb(224×224)→ ImageNet 归一化
  → `model.forward_features` 取 patch tokens，存 `z (T,256,dim)`（默认 fp16）。
  默认 `dinov2_vits14`(dim384)；224/14=16→256 patch。
- **脚本 6 `src/airsim/train_predictor.py`（已跑，世界模型核心）**：训练 action-conditioned latent
  predictor。读 latent(`z`)+action 构造 `(z_t, a_t, z_{t+1})`，最简 **patch-latent Transformer**
  （动作过 MLP 广播加到每个 patch + 位置编码 + 残差预测），损失 **MSE**。支持单步/多步 rollout、留出集评估。
  - 时间对齐：样本 = `(z[t], action[t], z[t+1])`。当前权重 `weights/predictor_h5.pt`。
  - **训练要点**：latent 必须标准化（否则卡 identity，见 §8）；多步训练 `--horizon` 对规划关键。
    看效果用"超过 identity 基线多少%" + `verify_predictor.py`，别只看绝对 loss。
- **脚本 7 `plan_mppi.py` / `plan_closed_loop.py`（进行中）**：MPPI 目标图像/目标条件规划。
  - `plan_mppi.py`（离线潜空间，已验证）：起点 z0、目标 z_goal，采样动作用世界模型 rollout、
    按到目标距离 softmax 加权更新（CEM 式 sigma 退火）。温度 0.1 是关键（1.0 太软）。
  - `plan_closed_loop.py`（一体化闭环）：每步 FrontCamera→DINO→z_t→MPPI→执行→重规划。
    - `Planner`：`encode`（DINO+标准化）、`target_components`/`target_cost`（指纹）、`goal_cost`
      （按 `--cost-metric` 选 value / target / poolcos / patchmse）、`plan`（MPPI/CEM）。
    - `--cost-metric` **默认 `target`**（指纹）；`value` 为 opt-in（价值函数，§4.3）。
    - 机制层已修：`--init-vx`（前进初始化）、`--iters`（多迭代+退火）、`--smooth-beta`（AR(1) 噪声治啄米）、
      `--replan-stride`（一次规划连执行几步）、`--record-goal`/`--dry-run`/`--diag`/`--sanity` 自测。
    - 默认预算 `--samples 32 --iters 2 --horizon 5`；服务器上可拉大（见 §5、§6）。
    - **纯视觉伺服分支 `--visual-servo`**（§0）：不走世界模型/MPPI，`target_components` 的 center
      直接比例控制 `vy=kp·cx, vz=kp·cy`、恒定前进、yaw=0。是 baseline / 底层执行器。
      伺服循环已抽成 `servo_stage()`，纯伺服与两段式共用。
    - **两段式 `--handoff`**：target 指纹 MPPI 远程领路，`dist≤goal-thresh`（当前模板标定 -0.18，
      ≈门前 4.8m）后自动交接 `servo_stage()` 收尾（同一指纹，无需第二张目标图）；
      伺服段预算 `--servo-max-steps`（默认 80），viz 步号接续 MPPI 段，rviz 全程可看。
      给一张紧裁剪目标模板图（`--target-template`）即可跑通"截获→远程→接触"全链。

**目标函数（cost）是本流水线的关键与瓶颈，单列 §4。** 相关离线验证工具：
`check_cost_monotonic.py`、`check_target_action_ranking.py`、`check_target_cost.py`、
`check_lateral_reliability.py`、`train_value.py`、`tools/eval_mppi_dir.py`、`tools/eval_wm.py`。

## 4. 目标函数：从整图潜距到指纹到价值函数（研究主线）

世界模型可信、planner 机制已修，**闭环成败取决于 cost 是否随"接近目标"单调下降**。三代演进：

### 4.1 整图 DINO 潜距（patchmse / poolcos）—— 失败
- `check_cost_monotonic.py`（2026-07-03）在真实飞近目标的 ep0050（23.75m→0，目标=绿门）上逐帧算 cost，
  以 pose 真实距离为参照（坐标只当离线尺子），算 Spearman ρ；偏航段（|yaw_rate|≥0.3）标灰、另算 ρ_facing。
- **结果**：patchmse ρ≈**-0.11**（正对着飞也 -0.10）——旧 cost 非单调的定量实锤；poolcos ρ_facing=+0.68。
- 后果：前进 rollout 代价≈原地，MPPI 被带去横移/冲过头，best 降（幻觉）而真实 dist 涨 → 14 连败根因。

### 4.2 target 指纹（`target_components`）—— 单调，作 `--cost-metric target`
- 给目标模板 → DINO patch 指纹 → 定位"目标居中(center) + 变大(mass) + 检测强(peak/conf)"。
  又尖又单调、且泛化（换目标只换模板图、不用颜色规则）。
- 调优后 **ρ_facing≈0.82**（mass_thresh 0.35→0.2 把面积项救活成单调主力）。分量：mass=单调主力、
  conf≈常数开关、center=转向信号（偏航段上涨是特性）。
- **模板对决**（`record_approach.py` 录 41.6m 全程素材）：环带 `tmpl_ring_rim.png` ρ=**0.847** > 整环 0.781。
  cost 碗底在门前 4.8m；**goal-thresh 重标定为 -0.18（≈门前 5~6m）**。换模板/权重必须重标定量表。
- **验证阶梯**：①真实帧单调 ✓ → ②预测 latent 下动作排序 ✓（`check_target_action_ranking.py`：
  frame170 排序 yaw_right>forward>...>zero；且 dataset_true 预测 delta 与真实帧几乎一致 → 世界模型预测准）
  → ③闭环。指纹目标能让闭环从出生点起飞、飞 ~11m 高（近段可靠）。
- **局限**：指纹是"接近某个特定目标"的固定行为，表达力有限；给不了任意指令的泛化（同伺服的病）。

### 4.3 价值函数 `V(z, z_goal)`（`train_value.py`）—— 当前主攻，指令泛化正道
- **思路**：hindsight —— 同轨迹 frame t→t+k 就是 k 步；训 `V(pool(z_t), pool(z_goal)) ≈ k`。
  按定义单调、梯度强；**给任意目标图就能打分 → 天然跨目标/指令泛化**（补上指纹/伺服缺的泛化）。
- **实现**：对标准化 patch latent 做 **4×4 粗空间池化**（保留左右/上下/远近布局，能区分前进 vs 横移）
  → 6144 维 → MLP 输入 `cat[z, g, g−z]`。回归到步距 + **时间一致性 TD 损失** `V(z_t,g)≈1+V(z_{t+1},g)`
  + 跨集负样本(=KMAX)。产物 `weights/value_fn.pt`；**KMAX=25**、pool_grid=4。闭环用 `--cost-metric value`。
- **今日离线验证（`tools/eval_mppi_dir.py`，在服务器 ep0052 上，比"规划平均动作 vs 真实平均动作"的余弦/主符号）**：

  | 目标半径 | patchmse | value v1 | **value v2** |
  | --- | --- | --- | --- |
  | K=8（近） | 0.77 (7/8) | 0.58 (7/8) | **0.91 (8/8)** |
  | K=25（远） | 0.34 (7/8) ⬇崩 | 0.63 (8/8) | **0.71 (8/8)** |

  **判读（层级 A 成立）**：patchmse 随距离拉远崩（0.77→0.34），**value 远处仍稳（v2 0.71、符号 8/8）**
  → 学习式代价成立，走"把 V 做扎实"而非重训 latent。**v2 全面强于 v1，且近处已反超 patchmse**。
- **v1 vs v2**：结构完全相同（d=6144、pool_grid=4、kmax=25），**权重完全不同 = 独立重训**（非微调，
  相对差 >1）。**统一用 v2**（`weights/value_fn_v2.pt`）。价值函数只跟它配对训练的 predictor 一起才对
  （取其 z_mean/z_std）；服务器上配对的是 `.../outputs/local_pair/predictor_h5.pt`（与另一个 predictor md5 相同）。
- **注意"乐观差"**：`eval_mppi_dir` 用 `256采样×6迭代×5重跑`重平滑、且比整段平均动作；真实闭环是
  `32×2、单次、只走第一步`。离线过关是**必要非充分**，须在闭环真实预算下复验（这也是把闭环搬服务器用大预算的动因，§5）。
- **待做**：① 单调性 vs 真实距离（复用 check_cost_monotonic 的 pose 尺子）；② 换训练外目标图的泛化验证；
  ③ 闭环（现状见 §6）。

## 5. 分布式计算与可视化拓扑（2026-07-13 起）

本机 4060 Ti 8G 撑不住 value-MPPI 大预算（256×6）+ UE 渲染，故把编码+世界模型+MPPI 移到服务器：

| 角色 | 机器 | 说明 |
| --- | --- | --- |
| 仿真（UE + ProjectAirSim 服务端） | **Windows 宿主机 192.168.31.178** | 渲染 + 发相机帧 + 收控制。须保持前台（否则掉 3Hz，§8） |
| 编码 + 世界模型 + MPPI | **服务器 192.168.31.237** | conda 环境 `ryh-dinov2`（非本地 `airsim`）；已装 projectairsim 客户端 |
| 可视化 rviz | **本地** | ROS2 只在本地；服务器没装 |

- 连接：`plan_closed_loop.py --address 192.168.31.178`（`ProjectAirSimClient` 本就是网络化 C/S）。
  相机帧 ~6MB/帧、~10Hz ≈ 500Mbps，须同一千兆 LAN（已满足）。留意 `赶不上相机` 提示。
- 服务器路径（非默认，运行时显式传）：predictor/value 已放入 `research-project/weights/`；
  DINO 权重 `.../dinov2/weights/`、DINO 仓库 `.../dinov2`（含 hubconf.py）、数据集 `.../dinov2/my/datasets/episodes_dataset`。
- **可视化数据通道**（ROS2 只在本地）：服务器 `--viz-dump <目录>` 落 `step_*.npz` →
  本地 `sshfs` 挂载该目录 → 本地 `ros2 launch src/mpc/plan_viz.launch.py dump_dir:=<挂载点>` 实时跟播。
  或跑完 `scp` 回本地回放。dump 带 RGB，跑久了清理服务器 `--viz-dump` 目录。
- 服务器装 projectairsim：`ProjectAirSim/client/python/projectairsim`（6.9M，gitignore 不上库）scp 过去，
  `pip install --no-build-isolation ./projectairsim`（清华镜像；若代理死了用 `env -u *_proxy` 去掉代理）。

## 6. 闭环现状与诊断（2026-07-13）

首次在服务器上跑 value(v2) 闭环（连宿主机仿真），**未收敛，但根因清晰且与离线一致**：

- **现象**：目标 = ep0050 末帧（离出生点 23.7m、200 帧）。`dist`(=V) 起步 ~23.8、全程平在 23~24；
  drone 原地不动（累计位移 0.1m），planner 甚至命令满速倒退（vx=-1.2）。换近目标 `--goal-frame 20`（仅 2m 远）后**原地左右偏航打转**。
- **根因：V 超出 KMAX=25 就饱和**。ep0050 前期 ~0.1m/帧，**V 的 25 帧半径 ≈ 仅 ~3m 前进**；
  末帧远超半径 → 起点处 V 顶到天花板、平、无梯度 → planner 在平面上乱走。**离线在 K≤25 内成立、闭环一上来就站在半径外**，不矛盾。
- **打转（近目标）**：疑似 yaw 通道对 V 反应大但方向无意义（整幅画面变），而前进梯度弱 → planner 专挑拧 yaw。
  诊断手段：rviz 看候选束（§5 可视化）/ 临时 `--yaw-max 0.0` 隔离 yaw 通道。
- **下一步方向**：
  1. **子目标链**：把远目标拆成 ≤25 步（~3m）的 waypoint，逐段推进（对接 V 的有效半径）。
  2. 闭环真实预算复验价值函数（消"乐观差"，§4.3）。
  3. 完成 §4.3 待做的单调性 / 泛化两项离线验证，再坐实层级 A。

## 7. 关键配置

- `sim_config/scene_drone_sensors.jsonc`（场景 `SceneDroneSensors`，drone `Drone1`，Simple Flight）+
  `robot_quadrotor_fastphysics_sensors.jsonc`（机体）。当前 origin `xyz "0.9 3.6 -7.5" / rpy "0 0 0"`（采集/原始出生点）。
- 相机 `capture-settings` 的 `image-type`：0=RGB 场景，2=深度透视。
- **采集减负**：`robot_quadrotor_fastphysics_sensors.jsonc` 里已关掉吃 GPU/CPU 的传感器（只关不删、带注释）：
  Chase 相机 `enabled:false`、FrontCamera 的 depth `capture-enabled:false`、lidar1 `enabled:false`。
  世界模型只用 `FrontCamera/scene_camera`；IMU/GPS/气压计/磁力计保留。关后飞行帧率 7.6→8.0Hz。
  FrontCamera 装在机头前 0.5m（避开拍到两个前臂），FOV=90°。

## 8. 已知坑

- **闭环规划不收敛 = 目标函数问题（非动作 OOD；2026-06-27 的 OOD 判断已于 07-01 系统实测推翻）**：
  ①模型没问题（多步开环每个 horizon 都赢 identity；动作敏感、误差与动作幅度无关）；
  ②planner 机制已修（前进初始化、多迭代+退火、动作平滑；采样数非瓶颈，靠迭代非样本）；
  ③真正根因 = 目标函数（整图 DINO 潜距又平又非单调，§4.1）。**方向已定：目标条件代价 —— 指纹(§4.2) + 价值函数(§4.3)。**
- **价值函数只在 KMAX=25 半径内有梯度，超出即饱和成平**（§6）。远目标闭环必须走子目标链，别直接喂远末帧。
- **训练前必须标准化 latent**：DINO patch token 各维方差大且不均（每元素方差≈5.7），直接 MSE 会退化成 identity。
  按训练集每维均值/方差标准化后解锁（latent 存 fp16，求统计先升 float32 否则求和溢出）。
- **采集/闭环帧率取决于 UE 是否前台**：UE 前台且不跑其他占 GPU 软件时空载 ~10Hz、飞行 ~8Hz；
  一旦切后台/操作其他软件，相机渲染压到 ~3Hz。**铁律：全程把 UE 放前台、别同机跑占 GPU 程序。**
  分布式后此规则针对 Windows 宿主机的 UE（§5）。3Hz 数据与 8Hz 混训不自洽，须剔除。
- **UE 采集纯白**：地图过亮时采集路径（SceneCaptureComponent2D，无自动曝光历史）过曝爆白，视口因 Eye Adaptation 正常。
  修法：地图 Post Process Volume 锁 `Min EV100 = Max EV100`（实测某白天图 ≈13.6）。
- **相机 FOV 光学假象**：FOV=90°、机头前 0.5m，环状目标的边缘在 `d < R/tan(FOV/2)` 时出画
  → FPV 看着"已穿过环"而观察视角还没到，是光学效应非装配 bug；也表现为 mass 在接触前就饱和。
- 轨迹脚本对 COLMAP→NED 的坐标系对齐有多个版本，见 `src/trajectory/` 各文件头部注释（COLMAP 属旧路线）。

## 9. 约定

-生成的数据集、缓存、评估、运行记录按 `outputs/README.md` 分类；模型放 `weights/`。
- 目录布局与迁移见 `docs/FILE_LAYOUT.md`；源码模块划分见 `src/README.md`。
- **conda 环境**：本地 `airsim`（仿真采集 / DINO 抽特征 / 训练 / 本地闭环 / ROS2 可视化）；
  **服务器 `ryh-dinov2`**（编码 + 世界模型 + MPPI，分布式闭环，§5）。
- 数据流：仿真采集 → `episode_*.h5` → DINO 抽特征 → 世界模型训练 → 目标函数 → MPPI/伺服闭环。
