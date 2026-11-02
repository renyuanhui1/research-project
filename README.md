# research-project — 机场无人机场景

`airbase/` 保存机场场景的代码、配置、模型和实验产物；`src/mpc/` 提供机场运行记录的 ROS2/RViz 回放。

## 目录

| 路径 | 用途 |
|---|---|
| `airbase/src/airsim/` | 相机检查、采集、训练、评价和闭环入口 |
| `airbase/sim_config/` | 机场单机、三机和 PX4 配置 |
| `airbase/mission_config/` | 三机任务配置 |
| `airbase/pictures/` | 机场目标模板图 |
| `airbase/weights/` | 机场模型权重及训练曲线 |
| `airbase/outputs/` | 机场数据集、特征、评价、地图和运行记录 |
| `src/mpc/` | 单机、三机和体素地图可视化 |
| `weights/dinov2_vits14_pretrain.pth` | 机场通过符号链接使用的冻结编码器权重 |
| `dinov2/` | 本地 DINOv2 源码依赖 |
| `ProjectAirSim/` | 仿真客户端与配置 schema 等上游依赖 |
| `docs/ARCHITECTURE.md` | 模块边界、路径关系和数据流 |

## 使用

```bash
conda env create -f environment.yml
conda activate airsim
cd airbase
python src/airsim/wm/decode_check.py --help
```

在线脚本需要 ProjectAirSim/UE 场景，UE 应保持前台 Play。运行时通过脚本参数设置服务地址。
训练与离线模型评价需要具备 PyTorch、DINOv2 和相应权重的环境。

## 文档

- [机场配置与入口](airbase/README.md)
- [脚本索引](airbase/src/airsim/README.md)
- [架构说明](docs/ARCHITECTURE.md)
- [可视化用法](src/README.md)
- [PX4 联调](airbase/RUN_PX4.md)
- [机场实验记录](airbase/EXPERIMENTS.md)
- [相关研究](airbase/RELATED_WORK.md)

## 本地资产

数据和运行结果统一位于 `airbase/outputs/`，模型位于 `airbase/weights/`。
冻结 DINOv2 权重通过 `airbase/weights/dinov2_vits14_pretrain.pth` 链接到根目录 `weights/`；
`airbase/dinov2` 链接到根目录 `dinov2/`。迁移仓库时需同时准备这些本地依赖。
