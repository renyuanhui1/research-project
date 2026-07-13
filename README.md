# my — 无人机世界模型项目

围绕 **ProjectAirSim 仿真** 的无人机数据采集、轨迹驱动与世界模型（图生视频 / COLMAP）链路。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `ProjectAirSim/` | 上游仿真引擎（外部依赖，**已 gitignore**，需单独获取/构建） |
| `sim_config/` | 仿真配置：场景 `scene_*.jsonc`、机器人 `*_robot.jsonc`、任务 `mission.plan` 等 |
| `src/airsim/` | simple-flight 控制器下的采集/控制脚本（起飞、偏航、采图、键盘控制） |
| `src/px4/` | PX4 飞控下的控制与双相机采集脚本 |
| `src/ros2/` | EGO-Planner ↔ AirSim 控制桥（ROS2 launch / bridge node / rviz） |
| `src/trajectory/` | 按 COLMAP 轨迹驱动无人机飞行（playback） |
| `src/colmap/` | COLMAP 轨迹处理工具（`colmap_airsim.py`） |
| `data/` | 可选的 COLMAP 轨迹 CSV；需要轨迹回放时创建 |
| `outputs/` | 数据集、缓存、评估结果和闭环运行记录，详见 `outputs/README.md` |
| `weights/` | DINOv2、世界模型和价值函数权重 |
| `docs/` | 架构与说明文档 |

## 环境

仿真脚本运行在 conda 环境 `airsim`（Python 3.10，含 `projectairsim`、`opencv`、`pynng` 等）。

```bash
# 用导出的环境文件重建
conda env create -f environment.yml
conda activate airsim
```

## 快速开始

```bash
conda activate airsim
# 1) 确认能连上仿真服务器
python src/airsim/test_connection.py
# 2) 起飞 + 偏航 + 采集 Chase/Front 图像
python src/airsim/direct_takeoff_yaw_capture.py
# 3) 采集一条 episode
python src/airsim/collect_episode.py
```

## 注意事项

- **采集纯白问题**：某些 UE 地图过亮，采集路径无自动曝光会爆白。在该地图的 Post Process Volume 里锁 `Min EV100 = Max EV100`（取视口收敛值）。详见各场景配置与团队记忆。
- 轨迹脚本里的 `CSV_PATH` 指向 `data/` 下的 csv；新增轨迹数据请放 `data/`。

代码入口见 [src/README.md](src/README.md)，输出文件分类见 [outputs/README.md](outputs/README.md)，更多内部架构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
