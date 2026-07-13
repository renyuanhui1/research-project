# 源码说明

本目录保存项目自研代码。第三方仿真器在 `ProjectAirSim/`，DINOv2 上游源码在 `dinov2/`，二者不属于本目录。

项目统一使用 conda 环境 `airsim`。

## 本机与服务器路径

脚本不绑定 `/home/hui` 或 `/home/pc`。默认路径由脚本自身位置自动推导项目根目录，因此把整个项目放到服务器任意目录后仍可直接运行。带 `--base`、`--output`、`--output-dir` 等参数的脚本也可以在命令行显式覆盖路径。

推送到服务器时应保持项目内部相对结构，例如 `src/`、`weights/`、`outputs/` 和 `dinov2/` 之间的层级不变。

## 主流程

```text
ProjectAirSim FrontCamera
  → 采集 RGB / action / pose episode
  → DINOv2 patch latent
  → action-conditioned 世界模型
  → 目标代价 / 价值函数
  → MPPI 或视觉伺服闭环控制
```

主要产物位置：

- 正式数据集：`outputs/datasets/episodes_dataset/`
- DINO latent：`outputs/features/latents/`
- 模型权重：`weights/`
- 评估结果：`outputs/evaluations/`
- 闭环运行记录：`outputs/runs/`

## 模块

| 路径 | 职责 |
| --- | --- |
| `airsim/` | 数据采集、世界模型训练、目标函数验证和闭环规划 |
| `mpc/` | MPPI 教学演示以及规划过程的 ROS2/RViz 可视化 |
| `px4/` | PX4 飞控下的起飞、偏航和双相机采集 |
| `ros2/` | EGO-Planner 与 ProjectAirSim 的 ROS2 桥接 |
| `trajectory/` | 按 COLMAP 轨迹驱动无人机 |
| `colmap/` | COLMAP 轨迹转换工具 |

## `airsim/` 主链路

### 1. 数据采集

- `decode_check.py`：验证 FrontCamera 消息解码。
- `collect_episode.py`：采集一条 `(rgb, action, pose, time)` episode。
- `replay_check.py`：检查 HDF5 并生成回放视频。
- `batch_collect.py`：批量采集多种轨迹模板。
- `consolidate.py`：清洗、裁剪并合并为正式数据集。
- `record_approach.py`：录制直线接近目标的标定素材。

### 2. 世界模型

- `extract_dino_features.py`：用冻结 DINOv2 提取 patch latent。
- `train_predictor.py`：训练动作条件 latent predictor。
- `verify_predictor.py`：与 identity 基线对比，验证预测器。
- `tools/eval_wm.py`：多步 rollout 离线评估。

### 3. 目标函数与规划

- `plan_mppi.py`：离线 MPPI 验证。
- `plan_closed_loop.py`：仿真闭环 MPPI、target cost 和视觉伺服入口。
- `train_value.py`：训练视觉时间距离价值函数。
- `check_target_cost.py`：检查模板指纹目标函数。
- `check_cost_monotonic.py`：评估目标代价随真实距离的单调性。
- `check_target_action_ranking.py`：检查世界模型是否为合理动作排序。
- `check_lateral_reliability.py`：诊断横向控制可靠性。
- `tools/eval_mppi_dir.py`：离线比较 patch MSE 与价值函数规划方向。

### 4. 辅助入口

- `test_connection.py`：最小连接测试。
- `direct_takeoff_yaw_capture.py`：起飞、偏航并采图。
- `keyboard_control.py`：键盘控制。
- `motion_planner.py`：ProjectAirSim A* 示例。
- `feasibility_check.py`：本地 GPU 推理可行性检查。

## 常用运行顺序

```bash
conda activate airsim

python src/airsim/test_connection.py
python src/airsim/decode_check.py
python src/airsim/collect_episode.py
python src/airsim/replay_check.py outputs/datasets/episodes/episode_000.h5
python src/airsim/extract_dino_features.py
python src/airsim/train_predictor.py
python src/airsim/verify_predictor.py
python src/airsim/plan_closed_loop.py --dry-run
```

运行参数以各脚本的 `--help` 为准。

## 放置约定

- 新的 ProjectAirSim 数据采集脚本放入 `airsim/` 或 `px4/`。
- 离线评估工具放入 `airsim/tools/`。
- ROS2 节点、launch 和 RViz 配置按用途放入 `ros2/` 或 `mpc/`。
- 源码目录不保存 HDF5、模型权重、图片、视频或运行日志。
- 新脚本不要把过程文件直接写到 `outputs/` 根目录，应使用 `outputs/README.md` 中的分类目录。
