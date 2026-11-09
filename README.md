# 无人机视觉抵近研究

使用ProjectAirSim/UE机场场景，以冻结DINO模板匹配与深度测距驱动无人机抵近，并在ROS2/RViz中同步显示占用栅格、位置、轨迹和匹配画面。

| 目录 | 内容 |
|---|---|
| `src/` | 采集、视觉感知、抵近控制、调试、评价、可视化与PX4代码 |
| `configs/` | 场景档案、机器人传感器、任务参数和RViz配置 |
| `data/` | 机场采集、目标模板、地图与目标几何元数据 |
| `debug/` | 相机帧率、图像解码和匹配诊断结果 |
| `weights/` | 预训练视觉编码器权重 |
| `runs/` | 每次运行的配置、图像、录像、位姿和日志 |
| `reports/` | 实验索引、分析与验证报告 |
| `logs/` | 未归属具体试验的客户端日志 |
| `third_party/` | ProjectAirSim客户端及DINOv2源码 |
| `docs/` | 架构、运行、场景切换和可视化说明 |
| `tests/` | 离线回归检查 |

## 环境

先将ProjectAirSim与DINOv2源码分别准备在`third_party/ProjectAirSim/`、`third_party/dinov2/`，将编码器权重放入`weights/`。本地数据、权重和引用项目独立保存，不随代码仓库自动分发。

```bash
cd /home/hui/research-project
conda env create -f environment.yml
conda activate airsim
python -m pip install -e third_party/ProjectAirSim/client/python/projectairsim
python src/capture/prepare_scene.py --profile airport --validate-only
```

ROS2/RViz使用系统Python及ROS环境；仿真、视觉计算使用`airsim`环境。

## 运行

[运行说明](docs/RUNNING.md)给出模板准备、实时抵近和日志保存命令；[可视化说明](docs/VISUALIZATION.md)给出地图与回放命令。

当前场景档案为[configs/scenes/airport.json](configs/scenes/airport.json)，任务参数为[configs/missions/airport_approach.json](configs/missions/airport_approach.json)。默认任务为100米水平起点、相机下俯30°、水平速度上限3米/秒、4米深度停车。模板准备使用已知几何布置采集和起点；正式闭环速度根据图像与深度计算，地图用于显示。

## 文档

- [架构与数据流](docs/ARCHITECTURE.md)
- [场景配置与切换](docs/SCENES.md)
- [代码入口索引](src/README.md)
- [实验结果与记录索引](reports/EXPERIMENTS.md)
- [PX4接口](docs/PX4.md)
- [相关研究](docs/RELATED_WORK.md)
