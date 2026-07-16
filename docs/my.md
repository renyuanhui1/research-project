# 研究路线与建议（2026-07-13 讨论整理）

> 本文是与 Claude 讨论的整理稿：研究定位、已完成工作的证据盘点、剩余工作量、飞控与实机路线。
> 架构细节见 `ARCHITECTURE.md`，实验数据见 `EXPERIMENTS.md`。

## 一、论文的核心故事（已成型，有数据支撑）

**一句话主张**：在无度量状态、目标坐标未知的纯视觉世界里，冻结 DINO 世界模型 + 学习式目标条件价值函数，让"给一张目标图就能规划"成为可能。

支撑链条每一环都有实测数字：

| 环节 | 证据 | 角色 |
| --- | --- | --- |
| 整图潜距不行 | ρ≈-0.11；K=25 余弦崩到 0.34 | 反例 = 论文动机章 |
| 世界模型可信 | held-out 赢 identity；动作排序对 | 方法的地基 |
| 指纹 cost 单调 | ρ_facing 0.85 | 中间贡献 |
| **价值函数远距离保方向** | **v2: K=25 余弦 0.71、符号 8/8（patchmse 同处 0.34）** | **核心贡献第一块硬证据（07-13）** |

"坏基线 vs 好方法"的对比很干净，方向不用再怀疑，往下做深即可。

## 二、KMAX 饱和不是 bug，是下一章

闭环实测：V 超出训练半径 KMAX=25（≈3m 前进）即饱和成平面、无梯度 → 远目标下 drone 原地打转。
这个"坏消息"恰好引出**分层规划**：

```
远目标 → 子目标链(每段 ≤ V 半径 ~3m) → 每段 V+MPPI 推进 → 近段伺服/指纹收尾
```

子目标来源的三个梯度（工作量任选）：
1. **最简**：沿目标方向每 K 步把"当前预测的 z"设为下一锚（纯机制，先跑通）；
2. **中等**：重训 V 把 KMAX 拉大（50/100）+ 分段一致性，直接扩半径；
3. **最有味道**：世界模型 rollout 在想象里**搜索**可达中间 z 当 waypoint——"在想象中打锚点"，真正的世界模型式规划，论文最亮点。

这一块 + 伺服（baseline / 底层执行器）= 已拍板的"高层想象选机动、低层执行"分层架构。

## 三、工作量盘点（诚实版：已完成 vs 待做）

### 已完成、有落盘证据的

| 工作 | 支撑产物 |
| --- | --- |
| 数据流水线 | 53 条 / 10367 帧数据集 + manifest + 全套采集/清洗代码 |
| 世界模型 | `predictor_h5.pt`；verify/eval 脚本可随时复跑出"赢 identity" |
| cost 三代对比 | `outputs/evaluations/cost_monotonic/` 12 组 csv+png |
| 闭环记录 | `runs/mppi/` 13 次（14 连败=动机章证据）+ `runs/servo/` 6 次 |
| 价值函数方向验证 | 07-13 K=8/25 对比（EXPERIMENTS.md；eval 脚本已加 `--out` 落盘） |

### 待做（真正的剩余工作量，勿当成已完成）

| 模块 | 内容 | 估时 |
| --- | --- | --- |
| 价值函数做扎实 | 单调性 vs 真实距离、换目标图泛化、真实预算复验乐观差 | ~2-3 周 |
| **分层闭环（核心）** | 子目标链 + V-MPPI + 伺服收尾，闭环成功率统计 | ~1-2 月 |
| 对比实验 | vs 纯伺服、vs 指纹 MPPI、vs 无世界模型贪心 | 分层后顺手 |
| 多场景 | 换 1~2 个场景复制流水线、扩数据、合并重训 | 方法定型后 |
| 飞控+实机 | 见下节 | 后期 |

总量对硕士毕业富余；分层闭环做好可冲一篇会议文章。

### 关于两个担忧的判断

- **数据量（53 条/1 万帧）少吗？** 对"单场景概念验证"够用且已被证明（predictor held-out 有效；DINO 冻结、只训小模型，样本效率高——DINO-WM 范式的卖点）。对最终主张不够，须扩。
- **换场景**：不是负担，就是剩余工作量本身——新数据 + 泛化实验 + 论文"多场景验证"章，一石三鸟。采集全自动（50 条≈33 分钟），边际成本在 UE 地图。**节奏：先当前场景把子目标链闭环打通（方法定型），再换场景复制（扩到 150~200 条），最后合并重训。** 方法未定型前不急扩数据，防返工。

### 实验留痕纪律（07-13 起执行）

评估结果必须落盘（csv），不能只活在终端/聊天里。流程：**服务器跑 → `--out` 落盘 → scp 回本地 `outputs/` 归档 → `EXPERIMENTS.md` 记一行**。固定 seed 保证可复现。

## 四、飞控接入：架构天生好接，按三级走

系统对飞控的全部要求：**接收 10Hz 机体速度指令 `[vx,vy,vz,yaw_rate]`**。上层（DINO/WM/MPPI/V）不动，换飞控只换动作下发层。

1. **PX4 SITL（仿真里换飞控）**：ProjectAirSim 原生支持，`sim_config/scene_px4_sitl_wsl2.jsonc`、`src/px4/` 现成。Simple Flight → PX4 SITL、速度指令走 Offboard。在仿真里提前吃掉真飞控的时延/内环特性，实机前最有价值的一步。
2. **实机架构**：机上只跑相机 + MAVSDK/MAVROS 速度透传，DINO+WM+MPPI 放地面站（=现在服务器的角色），WiFi 图传下行、指令上行。**当前的分布式闭环（仿真宿主机 + 计算服务器）本质就是实机架构的彩排。**
3. **简单复现**：室内/园区，一张目标图，飞 5~10m 用 V+伺服接近，足够写"真机部署验证"章。降级路径：V 域差大就退伺服（center 控制对域差最鲁棒），论文照样成立。

## 五、执行顺序（未来 6-8 周）

1. **本周**：V 的单调性 + 换目标泛化离线验证、闭环真实预算复验——层级 A 钉死（脚本 Claude 写，服务器跑）。
2. **紧接着**：最简子目标链进闭环，rviz 看行为，拿到第一次"value 闭环真的往目标走"。
3. **然后**：分层完整版（WM 选机动 + 伺服收尾）+ 成功率统计 → 论文核心章。
4. **并行慢烧**：PX4 SITL 通道在仿真打通。
5. **最后**：实机简单复现。

## 六、整体流程速记（当前形态）

```
【数据】宿主机 UE(前台!) → FrontCamera 10Hz → episode_*.h5(53条) → DINO latent
【模型】predictor(世界模型,已验证) + value_fn v2(目标条件代价,方向已验证)
【规划】plan_closed_loop: 服务器算(编码+WM+MPPI) --address 连宿主机仿真
        cost 演进: patchmse✗ → target指纹 → value(主攻); 伺服=baseline/执行器
【可视化】服务器 --viz-dump → 本地 sshfs 挂载 → plan_viz_node + rviz
【代码】本地改 → GitHub(renyuanhui1/research-project) → 服务器 pull(服务器不改源码)
【产物】权重 weights/, 数据 outputs/datasets/, 评估 outputs/evaluations/, 台账 docs/EXPERIMENTS.md
【卡点】V 半径 KMAX=25 → 子目标链; 近目标打转 → rviz 诊断/关 yaw 隔离
```

## 七、启动命令速查（两段式闭环 --handoff + rviz）

> 前提：① 宿主机 192.168.31.178 上 UE 仿真已跑起来且**在前台**；② 服务器已 `git pull`（或应用 bundle）。
> 服务器权重路径与本地不同（都被 gitignore，不随 git 同步，必须显式覆盖）：
> - DINO 主干在 dinov2 仓库内：`/home/pc/works/2025ryh/dinov2/weights/dinov2_vits14_pretrain.pth` → 必须 `--dino-weights` 指到此**文件**。
> - 世界模型 `predictor_h5.pt` 在 research-project 的 `weights/` 里 → 走默认，无需 `--ckpt`。
> - dinov2 仓库在 research-project 外的上一级 → `--repo-dir /home/pc/works/2025ryh/dinov2`（指到**目录**，别和 --dino-weights 混）。

### A. 服务器：跑两段式闭环（指纹 MPPI 远程领路 → dist≤goal-thresh 交接视觉伺服收尾）

```bash
# conda activate ryh-dinov2; cd /home/pc/works/2025ryh/research-project
python src/airsim/plan_closed_loop.py \
  --address 192.168.31.178 \
  --repo-dir /home/pc/works/2025ryh/dinov2 \
  --dino-weights /home/pc/works/2025ryh/dinov2/weights/dinov2_vits14_pretrain.pth \
  --cost-metric target \
  --target-template outputs/references/templates/tmpl_ring_rim.png \
  --goal-thresh -0.18 \
  --handoff \
  --servo-stop-mass 0.5 \
  --max-steps 140 \
  --samples 256 --iters 6 \
  --viz-dump outputs/runs/mppi/handoff03
# 说明：tmpl_ring_rim 已标定(ρ=0.847, goal-thresh -0.18)，换目标图必须先离线重标定阈值。
#      需原地转圈搜目标时加 --acquire；嫌 MPPI 慢降 --samples 64 --iters 3。
#      --max-steps 是 MPPI 远程段上限；交接后伺服段另有 --servo-max-steps(默认80)。
#      起点：sim_config/scene_drone_sensors.jsonc 出生点 x=22.3(距目标~20m)；采集需改回 x=0.9。
```

### B. 本地：sshfs 挂载 + rviz 可视化（rviz 与节点分开跑，便于反复重播）

```bash
# 1) 挂载服务器 dump 目录（重启后需重挂；非空/已挂先 fusermount -u ~/mnt/server_runs 再挂）
mkdir -p ~/mnt/server_runs
sshfs pc@192.168.31.237:/home/pc/works/2025ryh/research-project/outputs/runs/mppi ~/mnt/server_runs

# 2) 终端A：只开 rviz（全程不动）
rviz2 -d src/mpc/plan_viz.rviz

# 3) 终端B：单独跑可视化节点（重播=Ctrl+C 后重跑，rviz 不用重启；换 run 改 --dump-dir）
#    必须用系统 python /usr/bin/python3（rclpy C 扩展是 3.10 编的，conda python 会 import 失败）
/usr/bin/python3 src/mpc/plan_viz_node.py --dump-dir ~/mnt/server_runs/handoff03 --rate 5
# --rate = 回放速度(步/秒)：每 1/rate 秒播一步；调小慢放看细节，调大快进
```
