# 机场场景

`airbase/` 包含机场卫星图与三维目标模型的场景配置、脚本、模板、模型及运行数据。
工程结构与数据流见 [架构说明](../docs/ARCHITECTURE.md)。

## 目录

| 路径 | 用途 |
|---|---|
| `src/airsim/wm/` | 采集、特征提取、训练及模型评价 |
| `src/airsim/loop/` | 运行、诊断、评价及回放导出 |
| `src/airsim/probe/` | 场景、传感器和地图检查 |
| `src/airsim/px4/` | PX4 联调 |
| `sim_config/` | 场景与机器人配置 |
| `mission_config/` | 三机任务配置 |
| `pictures/` | 机场模板图 |
| `weights/` | 机场模型与训练曲线 |
| `outputs/` | 数据集、特征、评价、地图及运行记录 |
| `dinov2` | 指向根目录编码器源码的符号链接 |

## 场景配置

| 配置 | 用途 | 机器人配置 |
|---|---|---|
| `scene_airbase.jsonc` | 单机 Simple Flight | `robot_quadrotor_fastphysics_sensors.jsonc` |
| `scene_airbase_three_drones.jsonc` | 三机 Simple Flight | 同一机体配置，每架机拥有独立传感器 topic |
| `scene_airbase_px4.jsonc` | PX4 SITL | `robot_quadrotor_px4_airbase.jsonc` |

机器人配置定义相机、传感器与飞控连接。场景检查脚本位于 `src/airsim/probe/`；
PX4 的启动、连接和重置步骤见 [RUN_PX4.md](RUN_PX4.md)。

## 运行路径

```bash
conda activate airsim
cd ~/research-project/airbase
python src/airsim/wm/decode_check.py --help
```

脚本按 `sim_config/` 定位场景根目录，默认输入输出均相对 `airbase/`。
各入口用途和输入要求见 [脚本索引](src/airsim/README.md)。在线运行时需保持 UE 前台 Play。

模板诊断需显式指定 `--episode` 和 `--target-template`。
`loop/plan_closed_loop.py` 按模式显式接收 `--target-template` 或 `--goal-episode`；
`wm/record_approach.py` 显式接收 `--goal-ned` 和 `--goal-alt`，避免使用其他场景的隐含输入。

## 输出与可视化

| 路径 | 内容 |
|---|---|
| `outputs/datasets/` | 采集的 episode 数据和 manifest |
| `outputs/features/` | 编码器特征 |
| `outputs/recordings/approach/` | 接近过程录制 |
| `outputs/evaluations/` | 离线诊断和运行评价 |
| `outputs/maps/` | 离线体素地图 |
| `outputs/runs/` | 单机、PX4 和三机运行记录 |

共享可视化入口位于仓库根目录的 [`src/mpc/`](../src/mpc/)，用法见
[可视化说明](../src/README.md)。运行记录可导出为 `viz_dump/step_*.npz`；
三机记录按 `Drone1/`、`Drone2/`、`Drone3/` 分目录保存。

## 本地模型依赖

`dinov2 -> ../dinov2` 共享编码器源码，
`weights/dinov2_vits14_pretrain.pth -> ../../weights/dinov2_vits14_pretrain.pth` 共享冻结权重。
迁移时应检查符号链接，并单独准备 `weights/` 中的机场模型。
实验结论见 [EXPERIMENTS.md](EXPERIMENTS.md)，研究资料见 [RELATED_WORK.md](RELATED_WORK.md)。
