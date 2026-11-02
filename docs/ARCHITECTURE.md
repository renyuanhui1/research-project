# 机场场景架构

工程由机场场景模块、离线模型工具与 ROS2/RViz 可视化组成。

## 模块边界

| 路径 | 职责 |
|---|---|
| `airbase/src/airsim/wm/` | 相机解码、数据采集、特征提取、模型训练与评价 |
| `airbase/src/airsim/loop/` | 闭环运行、信号诊断、运行评价及回放数据导出 |
| `airbase/src/airsim/probe/` | 传感器、场景对象与体素地图检查 |
| `airbase/src/airsim/px4/` | 机场 PX4 联调入口 |
| `airbase/sim_config/` | 场景和机器人配置 |
| `airbase/mission_config/` | 三机任务参数 |
| `src/mpc/` | 读取机场记录并发布 ROS2 轨迹、图像和地图 |

单机、三机和 PX4 是同一机场场景的运行配置。各配置引用的机器人文件位于同一 `sim_config/` 目录。

## 路径与依赖

机场脚本向上查找包含 `sim_config/` 的目录，将 `airbase/` 作为场景根目录。
各流程目录加入导入路径，跨组复用相机解码、模型加载和运行评价等模块。

- `airbase/dinov2 -> ../dinov2` 提供本地编码器源码。
- `airbase/weights/dinov2_vits14_pretrain.pth -> ../../weights/dinov2_vits14_pretrain.pth` 提供冻结权重。
- `ProjectAirSim/` 提供仿真客户端源码和机器人配置引用的 schema。
- `environment.yml` 描述仿真客户端环境；模型计算环境还需准备对应训练依赖。

## 数据流

```text
机场配置 + ProjectAirSim
  -> 相机检查 / episode 采集
  -> airbase/outputs/datasets/
  -> 特征提取 -> airbase/outputs/features/
  -> 模型训练 -> airbase/weights/

机场运行入口
  -> airbase/outputs/runs/
  -> 运行评价 -> airbase/outputs/evaluations/
  -> viz_dump/ -> src/mpc/ -> ROS2 / RViz

体素检查
  -> airbase/outputs/maps/
  -> src/mpc/voxel_map_viz* -> ROS2 / RViz
```

采集 HDF5 保存 RGB、动作、位姿、时间以及可用的深度和分割数据。
模型输入与输出的具体格式由各脚本及检查点描述；使用数据时应保留其坐标系和时间对齐约定。

## 文件使用约定

- 机场输入模板放在 `airbase/pictures/`，生成物放在 `airbase/outputs/`。
- 模板诊断脚本需显式指定 `--episode` 和 `--target-template`。
- 规划入口按模式显式指定 `--target-template` 或 `--goal-episode`。
- 接近录制需显式指定 `--goal-ned` 和 `--goal-alt`；价值函数评价需显式指定 `--ep`。
- 可视化从机场输出目录读取记录；共享源码的位置不作为场景根目录使用。
- 回放导出脚本可按当前场景根目录解析运行记录中的场景内绝对路径。
- 本地模型和运行数据不纳入 Git，需随部署单独准备。
