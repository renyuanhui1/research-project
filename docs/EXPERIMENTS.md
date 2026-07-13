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
- 产物: 当时仅终端（本表即抄录）; **脚本已加 `--out` 落盘,今后自动存 `outputs/evaluations/mppi_dir/`**
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

### 待做（层级 A 收尾）
- [ ] V 单调性 vs 真实距离（pose 尺子, 脚本待写）
- [ ] 换训练外目标图的泛化验证（脚本待写）
- [ ] 闭环真实预算(32×2×1)下复跑 eval_mppi_dir, 量化"乐观差"
- [ ] 子目标链闭环
