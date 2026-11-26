# 数据输入

本目录保存仿真采集素材、目标模板、几何地图与对象元数据。各次运行的快照与校验哈希同时保存在对应 `runs/` 目录。

| 目录 | 内容 |
|---|---|
| `captures/scene-inspection/` | 机场范围、相机与障碍物检查素材 |
| `captures/aircraft-overhead/` | 两架机场飞机俯视原图、飞行记录与录像 |
| `captures/hangar_2019/` | 指定机库的俯视原图、深度、运行配置、位姿及采集日志 |
| `captures/aircraft_a_tail/`、`captures/aircraft_b/` | 飞机 A/B 训练与独立视角斜视 RGB-D 采集及部位掩码 |
| `captures/airport_targets/` | 机场 19 目标小样本（`pilot`）及 1,920 组多目标采集（`multitarget`） |
| `templates/airport_7890/`、`templates/airport_7888/` | 机场飞机可复用的俯视、斜俯视模板和特征统计帧 |
| `templates/airport_targets_20261116/` | 机场 19 目标俯视模板库、整组/部位掩码及 `catalog.json` 划分协议 |
| `templates/coastal_harbor_pilot/` | 沿海渔港任务 1 单机模板规划参考图与特征统计 |
| `maps/airport_scene/` | 机场全场碰撞采样、目标三角面、占用栅格与验证报告 |
| `maps/airport_targets_20261116/` | 机场 19 目标及配套 Actor 的世界 ENU 三角网格与变换快照 |
| `labels/airport_targets/` | 多目标同画面上下文实例监督标签 |
| `metadata/targets/` | 由机场网格推导的目标包围盒，仅供离线评价 |
| `metadata/coastal_harbor/` | 沿海渔港 280 个巡查对象表（`targets.json`）及鱼市参考机位（Git 跟踪） |
