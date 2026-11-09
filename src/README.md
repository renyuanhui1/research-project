# 代码入口

从项目根目录执行，完整参数见各入口`--help`。

| 目录 | 入口与用途 |
|---|---|
| `capture/` | `prepare_scene.py`准备模板和起点；`collect_episode.py`采集episode；`record_approach.py`坐标接近录制；`grab_nadir_frames.py`俯视采集；其余入口整理与检查HDF5 |
| `perception/` | `fingerprint.py`模板匹配；`extract_dino_features.py`冻结DINO特征提取；`depth_map.py`深度转点云 |
| `control/` | `run_approach.py`机场主流程；`servo_closed_loop.py`公共控制律与通用入口；`multi_servo.py`显式任务配置的多机入口 |
| `debug/` | 图像解码、出生点、目标、深度分割、模板响应与碰撞体素探测 |
| `debug/maps/` | UE导出、对象范围、全场碰撞采样、网格转换和栅格生成 |
| `evaluation/` | `eval_run.py`离线轨迹与包围盒评价；`eval_target_localization.py`模板定位对照 |
| `visualization/` | `map_view.py`地图发布；`plan_viz_node.py`轨迹与匹配图；`matching.launch.py`单机启动；`multi_viz.launch.py`多机回放 |
| `px4/` | MAVSDK/PX4接口与航向探测 |
| `common/` | 路径、配置、日志、解码和分割ID工具 |

仿真与视觉入口使用`airsim`环境；ROS2入口使用ROS环境的系统Python。主流程见[运行说明](../docs/RUNNING.md)，配置见[场景说明](../docs/SCENES.md)。

模板响应诊断显式指定`--episode`、`--target-ned N E`及`--target-alt`，坐标仅用于离线距离评价。深度分割飞行检查显式指定`--views ALT:YAW ...`。
