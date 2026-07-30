# 文件布局与迁移记录（老场景）

> ⚠️ **本文档描述的是「老场景」的目录布局，截至 2026-07-13，不含 airbase。**
> airbase 的目录说明见 `airbase/README.md`；脚本清单见 `airbase/src/airsim/README.md`。

## 2026-07-10 输出目录整理

本次整理只移动文件，没有删除文件。旧路径到新路径的映射如下：

| 旧路径 | 新路径 |
| --- | --- |
| `outputs/episodes_dataset/` | `outputs/datasets/episodes_dataset/` |
| `outputs/approach/` | `outputs/recordings/approach/` |
| `outputs/pooled_latents_g4.npz` | `outputs/cache/pooled_latents_g4.npz` |
| `outputs/tmpl_*.png` | `outputs/references/templates/` |
| `outputs/goal_*` | `outputs/references/goals/` |
| `outputs/1.jpg` 等手工图片 | `outputs/references/manual/` |
| `outputs/cost_*.csv/png` | `outputs/evaluations/cost_monotonic/` |
| `outputs/*_resp.png`、`vis_pred_ep0000.png` | `outputs/evaluations/responses/` |
| `outputs/viz_runN/` | `outputs/runs/mppi/runNN/` |
| `outputs/viz_servoN/` | `outputs/runs/servo/runNN/` |
| `outputs/view_red*/` | `outputs/runs/views/redNN/` |
| `outputs/viz_test/` | `outputs/runs/tests/viz_test/` |

同时统一了源码中的默认输出路径。模型权重继续保存在 `weights/`，没有移动。

如需人工回退，只需按表格反向移动；但回退后还需要同步恢复源码中的路径配置。

