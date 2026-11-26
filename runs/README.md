# 运行记录

本目录按实验阶段和任务类型归档离线训练、在线控制、多机协同及可视化回放数据，完整索引与结论见 [实验报告](../reports/EXPERIMENTS.md)，启动命令见 [使用手册](../docs/USAGE.md)。

| 目录 | 用途 |
|---|---|
| `approach/` | 机场单机部位抵近、两级粗→细控制及主被动对比试次（含 `run.h5`、录像与 `viz_dump/`） |
| `evaluation/` | 机场离线消融、模板核验、P1 全场景复验、P2.6 分级门控与 P3 校准结果 |
| `training/` | 机场局部匹配器、样本 HDF5（`patches.h5`）、模板条件化与核验器训练权重 |
| `transfer/` | DINOv2/DINOv3 及模板模型跨骨干、跨目标迁移评估产物 |
| `opening-validation/` | 机库洞口固定模板离线 DINO/深度几何匹配诊断 |
| `opening-alignment/` | 机库洞口上方定高水平对准控制记录与 `rviz/` 回放资产 |
| `opening-entry/` | 机库垂直/斜俯视视觉抵近穿洞入室控制记录与 `rviz/` 特征展示 |
| `indoor/` | 机库通道碰撞采样与坐标引导入室记录 |
| `coordination/` | 多机两阶段协同调度、SQLite 状态库、5 Hz 真轨迹与分屏录像 |
| `scenes/` | 沿海渔港 UE 关卡构建、表面修复、光照排查、性能优化与校验快照 |
| `harbor/` | 沿海渔港三类巡检任务（部位复查、破口入室、沿线巡检）实飞结果与双视角录像 |
| `inspection/` | 机场 19 个目标资产在线核对与观察取图 |
| `catalog/`、`capture/` | 目标库构建与多目标采集登记索引 |
| `validation/`、`visualization/` | 离线/服务端环境就绪审计，以及 19 目标 RViz 联合审阅启动配置 |
| `waypoint/` | 航点飞行输出目录 |

> 准备与控制入口均要求使用空输出目录，严禁覆盖既有试次。
