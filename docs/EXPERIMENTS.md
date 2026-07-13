# 实验台账

> 每次关键实验跑完记一行/一节：日期、命令、关键数字、结论。论文实验章节的素材库。
> 落盘产物按 `outputs/README.md` 分类；本文件只记结论和指针，不贴大段原始输出。

## 记录格式

```
### <日期> <实验名>
- 命令: <可复现的命令(含关键参数)>
- 环境: <本地 airsim / 服务器 ryh-dinov2>, <权重版本>
- 产物: <落盘路径 或 "仅终端(已抄录下方)">
- 结果: <关键数字>
- 结论: <一句话>
```

---

## 2026-07 之前（补记,详见 docs/ARCHITECTURE.md §4 与 outputs/evaluations/）

### 2026-07-03 cost 单调性三代对比（check_cost_monotonic.py）
- 产物: `outputs/evaluations/cost_monotonic/`（12 组 csv+png）
- 结果: patchmse ρ≈-0.11（正对飞也 -0.10）; poolcos ρ_facing=+0.68; target 指纹调优后 ρ_facing=+0.82
- 结论: 整图潜距非单调实锤（14 连败根因）; 指纹 cost 单调可用

### 2026-07-03 动作排序验证（check_target_action_ranking.py）
- 产物: 仅终端（关键数已录 ARCHITECTURE.md §4.2）
- 结果: frame170(~3m) 排序 yaw_right>dataset_true>forward>...>zero; dataset_true 预测 delta≈真实帧 delta（-0.235 vs -0.233）
- 结论: 世界模型对指纹特征预测准; target cost + WM 能把合理动作排前

### 2026-07-0x 模板对决（record_approach.py + check_cost_monotonic.py）
- 产物: `outputs/recordings/approach/approach_full.h5`; `outputs/evaluations/cost_monotonic/cost_mono_approach_*.csv/png`
- 结果: 环带 tmpl_ring_rim ρ=0.847 > 整环 0.781 > 旧糊模板 0.626; 碗底门前 4.8m; goal-thresh 重标定 -0.18
- 结论: 换模板/权重必须重标定量表

### 闭环历史记录
- 产物: `outputs/runs/mppi/run01~run14`（旧 cost 下 14 连败的 viz dump）; `outputs/runs/servo/run01~06`（视觉伺服成功接近）
- 结论: 旧目标函数闭环不收敛; 伺服近段可靠但无泛化

---

## 2026-07-13 价值函数方向验证（本页核心）

### eval_mppi_dir: value vs patchmse, K=8/25, v1 与 v2（服务器 ryh-dinov2, ep0052）
- 命令: `python src/airsim/tools/eval_mppi_dir.py --K {8,25} --value-fn-path <v1|v2> ...`
  （256采样×6迭代×5重跑×8起点; 重平滑=乐观上界）
- 产物: v2 两组已重跑落盘并复现（0.907/0.713,与首跑一致,seed=0 可复现 ✓）:
  `outputs/evaluations/mppi_dir/eval_ep0052_K8_v2.csv`、`eval_ep0052_K25_v2.csv`（服务器跑,已 scp 回本地归档）。
  v1 两组仅终端抄录（v1 已淘汰,需要时可随时重跑）。
- 结果（余弦均值 / 主分量符号对）:

| 目标半径 | patchmse | value v1 | value v2 |
| --- | --- | --- | --- |
| K=8（近） | 0.77 (7/8) | 0.58 (7/8) | **0.91 (8/8)** |
| K=25（远） | **0.34 (7/8) 崩** | 0.63 (8/8) | **0.71 (8/8)** |

- 权重版本: v1=`value_fn.pt`(07-02 15:47, 与 local_pair predictor 配对), v2=`value_fn_v2.pt`(07-02 16:23, 独立重训, 结构同 v1: d=6144/grid=4/kmax=25; 两 predictor md5 相同故通用)
- 结论: **patchmse 远处崩、value 远处稳 → 学习式代价成立（层级 A）; v2 全面强于 v1 且近处反超 patchmse; 统一用 v2**

### value(v2) 闭环首跑（服务器算, 连宿主机仿真 192.168.31.178）
- 命令: `plan_closed_loop.py --address 192.168.31.178 --cost-metric value --samples 256 --iters 6 --replan-stride 3`
- 产物: `outputs/runs/mppi/server_diag`（服务器侧）
- 结果: 目标=ep0050 末帧(23.7m 远): dist(V) 全程平在 23~24, drone 原地不动甚至倒退; `--goal-frame 20`(2m): 原地左右偏航打转
- 结论: **V 超 KMAX=25 半径(前期≈3m)即饱和无梯度 → 远目标必须子目标链**; 近目标打转疑 yaw 通道, 待 rviz 诊断 + `--yaw-max 0` 隔离

### eval_value_fn: V(v2) 单调性 + 梯度强度曲线（服务器, ep0050, 2026-07-13 晚）
- 命令: `python src/airsim/tools/eval_value_fn.py --value-fn-path weights/value_fn_v2.pt ...`
- 产物: `outputs/evaluations/value_fn/value_episode_0050_value_fn_v2.{csv,png}`
- 结果 B（梯度强度 ΔV/步, 理想=1）: K≤12 全程 ≈1.0（方向 74~81%）; K=16~20 ≈0.9; **K=25 腰斩 0.51; K≥30 归零(方向≈50%)**
  → **有效半径 ≈20 步(≈2.5m), 子目标间距取 12~16 步**。与 eval_mppi_dir 的 vx 衰减(K8:0.89→K25:0.29)互证。
- 结果 A（单调性, goal=末帧, **看图后修正过的解读**）: 24m→5m V 稳定贴 25 平台（**正确饱和, 不假报近**）;
  13~15m 鼓包=偏航段目标出视野, V 升=方向正确; **<5m 干净陡峭单调滑到 0 = 工作区**。
  ρ=-0.03 是统计口径假象（70% 帧在平台上, 平段把全程秩相关拉没; 只看 <5m 段应近 1）——指标设计问题, 非 V 的问题。
- 结论: **V = "5m 内的精密计步器"**: 远处老实说"很远"（到达判据不会假触发）, 近场梯度陡且单调到 0。
  半径外无方向（A 平台=B 归零）→ 远程领路靠指纹(41m 已验证), 近程/到达判据归 V。
  重训 KMAX=100 目的=**纯扩工作区**（5m→10~15m?）, 不是修 bug。

### value_fn_k100 重训 + 曲线对比（2026-07-13 晚, 服务器）
- 命令: `train_value.py --kmax 100 --k-per-frame 8 --out weights/value_fn_k100.pt`（回归 huber 39.7→3.6; 留出集单调相关 0.78/0.94/0.90）
- 产物: `outputs/evaluations/value_fn/value_episode_0050_value_fn_k100.{csv,png}`（v2 对照同目录）
- 结果 A: **ρ=+0.868**; 平台缩到 24→15m, **~13m 起稳定下坡到 0（工作区 5m→13m, 与 100步×0.13m 自洽）**
- 结果 B: **K=2~50 ΔV 全程 0.8~1.07 无拐点**; 代价=噪声×2.5(std 2.3~4.9)、单步方向正确率降 ~10pp(57~71%)
- 结论: **半径买大成功, 12m 任务一把尺子可罩住**; 近场精度 v2 更优 → 候选架构"k100 远程领路 + v2/伺服近场收尾"。
  待 eval_mppi_dir(K=25/50, k100) 验证噪声是否伤 MPPI 排序, 过了就闭环重飞 12m 场景。

### eval_mppi_dir: k100 在 MPPI 层面的表现（2026-07-13 晚, 服务器, ep0052）
- 产物: `outputs/evaluations/mppi_dir/eval_ep0052_K25_k100.csv`（K=50 待跑）
- 结果 K=25 三方对比（余弦均值 / 主分量符号对）:

| K=25 | patchmse | value v2 | value k100 |
| --- | --- | --- | --- |
| 余弦/符号 | 0.34 (7/8)（三次复现一致） | **0.71 (8/8)** | 0.49 (6/8, t40 倒退/t60 原地) |

- 结论: **k100 的单步噪声(×2.5)真实传导到 MPPI 排序**——在 v2 舒适区(K≤25)内 k100 打不过 v2。
  分工进一步坐实: **≤25 步用 v2, k100 的价值只在 25 步以外**。
- **K=50 判决（未过关）**: patchmse -0.05(4/8, 彻底躺平) vs k100 **+0.34(4/8)**——比 patchmse 强但半数起点排错,
  规划动作全 |vx|≤0.17(MPPI 没敢离开零初始化)。产物 `eval_ep0052_K50_k100.csv`。
  根因=信噪比: horizon 5 步的真实 V 差≈5, 而 k100 在 K=50 单次评估噪声 std≈4.4 → SNR≈1, softmax 被噪声主导。
  （B 曲线均值贴 1 是几百帧平均的结果, MPPI 单帧享受不到。）
- **结论: "k100+MPPI 远程领路"暂不成立**; 主线不受阻——远程归指纹(41m 已验证), ≤3m 归 v2(0.71~0.91), 伺服收尾。
  k100 救法待办(低优先): ①ensemble/强一致性重训治噪声; ②`--samples 512 --avg-runs 10` 复跑 K=50 分离"噪声 vs 偏差"。

### 待做（层级 A 收尾）
- [ ] 换训练外目标图的泛化验证（脚本待写）
- [ ] 闭环真实预算(32×2×1)下复跑 eval_mppi_dir, 量化"乐观差"
- [ ] 子目标链闭环（间距 12~16 步, 依据上表）
