# 数据输入

| 目录 | 内容 |
|---|---|
| `captures/scene-inspection/` | 机场范围、相机与障碍物检查素材 |
| `captures/aircraft-overhead/` | 两架飞机俯视原图、飞行记录与录像 |
| `captures/hangar_2019/` | 指定机库的俯视原图、深度、运行配置、位姿及采集日志 |
| `templates/airport_7890/` | 可复用的俯视、斜俯视模板和特征统计帧 |
| `maps/airport_scene/` | 全场碰撞采样、目标三角面、占用栅格与验证报告 |
| `metadata/targets/` | 由机场网格推导的目标包围盒，仅供离线评价 |

采集入口可将episode与特征分别写入`data/datasets/`和`data/features/`。每次运行使用的输入快照保存在对应`runs/`目录，便于按原始配置复查。
