# 机场脚本索引

在 `airbase/` 下运行各入口；脚本按 `sim_config/` 自动定位机场根目录。

## 采集与模型工具：`wm/`

| 脚本 | 用途 |
|---|---|
| `decode_check.py` | 相机连接与消息解码检查 |
| `collect_episode.py` | 采集机场 episode；`--num` 指定批量条数 |
| `record_approach.py` | 显式给定目标点的接近录制 |
| `grab_nadir_frames.py` | 俯视素材录制 |
| `replay_check.py` | HDF5 数值检查与视频回放 |
| `consolidate.py` | 合并和整理采集目录 |
| `interleave_dataset.py` | 按目标与距离带整理 episode 顺序 |
| `extract_dino_features.py` | 冻结编码器特征提取 |
| `train_predictor.py` | 动作条件模型训练 |
| `verify_predictor.py` | 单步预测检查 |
| `eval_wm.py` | 多步预测与基线对照 |
| `train_value.py` | 价值函数训练 |
| `eval_value_fn.py` | 价值函数离线评价 |
| `train_part_dist.py` | 部位距离模型离线训练 |
| `eval_part_dist.py` | 部位距离模型离线评价 |

## 运行与诊断：`loop/`

| 脚本 | 用途 |
|---|---|
| `servo_closed_loop.py` | 单机运行入口 |
| `multi_servo.py` | 三机运行入口 |
| `plan_closed_loop.py` | 模型规划运行入口 |
| `plan_mppi.py` | 离线规划检查 |
| `eval_run.py` | 保存记录的离线评价 |
| `eval_target_localization.py` | 模板定位的离线评价 |
| `check_target_signal.py` | 模板响应诊断 |
| `check_correspondence_signal.py` | 特征对应信号诊断 |
| `check_action_snr.py` | 动作信号与预测误差检查 |
| `check_target_cost.py` | 模板目标函数逐帧检查 |
| `check_cost_monotonic.py` | 目标函数单调性对照 |
| `check_target_action_ranking.py` | 候选动作排序诊断 |
| `check_lateral_reliability.py` | 横向和垂直预测诊断 |
| `eval_mppi_dir.py` | 离线规划方向评价 |
| `depth_map.py` | 深度帧到点云转换 |
| `dump_servo_viz.py` | HDF5 记录转 RViz 回放数据 |

## 场景检查：`probe/`

| 脚本 | 用途 |
|---|---|
| `probe_spawn.py` | 出生位姿与地面检查 |
| `probe_target_object.py` | 场景对象信息检查 |
| `probe_segmentation.py` | 分割相机和 ID 映射检查 |
| `probe_depth_seg_flight.py` | 深度与分割录制检查 |
| `probe_image_msg.py` | 原始图像消息检查 |
| `probe_voxel_map.py` | 离线体素地图生成与检查 |

## PX4：`px4/`

- `px4_yaw_probe.py`：航向联调检查。
- `servo_closed_loop_px4.py`：PX4 运行入口，启动要求见 [RUN_PX4.md](../../RUN_PX4.md)。

## 输入与输出

- 模板诊断脚本显式传入 `--episode`、`--target-template`；横向诊断显式传入 `--target-template`。
- 规划入口按模式传入 `--target-template` 或 `--goal-episode`。
- 接近录制显式传入 `--goal-ned` 和 `--goal-alt`；价值函数评价显式传入 `--ep`。
- 数据目录为 `airbase/outputs/datasets/`，模型为 `airbase/weights/`，模板为 `airbase/pictures/`。
- 特征提取应为每个数据集单独指定输出目录，避免同名 episode 的特征混用。
- 模型计算需要相应依赖与权重；在线入口需要已启动的机场仿真。

各脚本参数以 `--help` 为准。可视化说明见 [`src/README.md`](../../../src/README.md)。
