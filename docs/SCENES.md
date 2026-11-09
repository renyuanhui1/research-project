# 场景配置

`configs/scenes/airport.json`描述当前机场的连接地址、目标Actor、地图、模板、权重和准备几何。`configs/simulation/`定义ProjectAirSim的无人机、相机、传感器与时钟；`configs/missions/airport_approach.json`定义本次任务的距离、速度、俯角、部位和停车深度。

| 配置项 | 用法 |
|---|---|
| `simulator` | 服务地址、场景JSONC、机器人、无人机名称和前摄ID |
| `assets` | 模板、统计帧、准备输入、DINO源码与权重、评价元数据 |
| `map` | 地图数据根目录、目标Actor和显示名称 |
| `target`、`parts` | 当前目标、机体轴向、部位几何筛选与分割ID；颜色需实测后填写 |
| `preparation` | 巡航点、俯视起点与模板采集距离；使用NED米 |
| `controller` | ROI切换距离、近距限速与保护阈值 |
| `mission` | 默认任务JSON路径 |
| `editor` | UE几何导出的临时目录名称、WSL读取位置 |

路径相对项目根目录填写。可用`--profile airport`、`--profile configs/scenes/my_scene.json`或环境变量`UAV_SCENE_PROFILE`选择配置。单次准备可用`--mission`指定同场景任务，再用`--speed`、`--start-distance`等参数覆盖。

## 更换UE场景

1. 在UE中打开目标关卡，确认ProjectAirSim插件、地面和目标碰撞配置，进入Play。
2. 建立对应场景JSON，填写服务地址、Actor、准备点和独立资产目录；复制无人机JSONC时同步更改机器人引用。
3. 采集并核对目标位置、比例、机体轴向与相机视角。准备该场景的模板和统计图像，重新导出网格、碰撞栅格及评价元数据；不要沿用其他场景的目标坐标和ROI。
4. 运行配置检查，再按[运行说明](RUNNING.md)执行准备与匹配预检。

```bash
python src/capture/prepare_scene.py --profile airport --validate-only
python src/debug/probe_target_object.py --profile airport --name StaticMeshActor_7890
```

ProjectAirSim场景JSONC配置当前UE关卡中的机器人与传感器，不会自动替用户切换UE关卡。准备入口会重载机器人配置并执行坐标飞行，控制入口读取本次`runtime-scene.json`和`scene-profile.json`快照。

## 地图采集入口

```bash
python src/debug/maps/read_scene_bounds.py --profile airport
python src/debug/maps/generate_global_map.py --profile airport
python src/debug/maps/prepare_meshes.py --profile airport
python src/debug/maps/build_grid_maps.py --profile airport
```

前两条连接UE读取对象与碰撞。`prepare_meshes.py`读取`editor.wsl_export_dir`中的OBJ及变换，`build_grid_maps.py`离线生成栅格。运行前需在UE Python中执行`src/debug/maps/export_aircraft.py`导出目标，保持脚本所在项目结构和`configs/`可访问；`capture_airport.py`可选导出带坐标俯视图，图像不参与栅格生成。

配置中的Actor、准备几何和网格必须与实际关卡一致。地图数据说明见[data/maps/airport_scene/README.md](../data/maps/airport_scene/README.md)。
