# 使用手册

本文包含机场抵近运行、场景切换、栅格地图与RViz显示，以及PX4接口的使用方法。项目结构与数据流见[架构说明](ARCHITECTURE.md)，实验结果见[实验索引](../reports/EXPERIMENTS.md)。

- [实时抵近](#实时抵近)
- [场景配置与切换](#场景配置与切换)
- [RViz地图与匹配显示](#rviz地图与匹配显示)
- [匹配热力图的显示与解读](#匹配热力图的显示与解读)
- [附录：PX4接口](#附录px4接口)

仿真和视觉计算使用`airsim`环境，ROS2/RViz使用系统Python及ROS环境。除PX4源码构建命令外，命令从项目根目录`/home/hui/research-project`执行。

## 实时抵近

在UE中打开机场关卡并保持Play及窗口展开。用`airsim`环境运行仿真与视觉脚本，先检查[场景档案](../configs/scenes/airport.json)的地址、目标与地图。

### 准备与运行

为每次试验选用空目录，下列命令需从项目根目录执行：

```bash
conda activate airsim
cd /home/hui/research-project
session_dir="$PWD/runs/approach/my-session"
mkdir -p "$session_dir"
python src/capture/prepare_scene.py --profile airport --out-dir "$session_dir" > "$session_dir/prepare.log" 2>&1
```

准备脚本会重载无人机传感器、起飞，按已知几何前往35、8、60、100米等位置采集模板及验证帧，最后布置100米起点。默认相机下俯30°、水平速度上限3米/秒、深度停车4米。坐标采集不计为视觉抵近成功。

在准备成功后启动[RViz同步显示](USAGE.md#rviz地图与匹配显示)，再执行：

```bash
python src/control/run_approach.py --profile airport --out-dir "$session_dir" > "$session_dir/control.log" 2>&1
```

控制脚本先校验远近景模板落点，全部通过后连接仿真执行视觉与深度闭环。失败记录、碰撞保护和其他提前停车均以`report.json`为准。`--out-dir`必须与准备阶段一致；脚本拒绝覆盖已有控制结果。

### 参数与离线检查

```bash
python src/capture/prepare_scene.py --profile airport --validate-only
python src/capture/prepare_scene.py --help
python src/control/run_approach.py --help
```

准备阶段可用`--start-distance 100 --speed 3 --camera-pitch 30 --stop-distance 4`覆盖任务参数。`--nadir`选择俯视；`--oblique`选择斜俯视。`--target-part wing`选择已配置的机翼区域；机翼匹配仍须通过本次预检。

已有准备目录可离线检查匹配：

```bash
python src/control/run_approach.py --out-dir runs/approach/close4-1793954736186 --validate-only
```

`--validate-only`不连接UE、不下发飞行指令，也不改写已有报告。机翼外侧记录的预检会按实际匹配结果报失败。

### 运行文件

| 文件 | 内容 |
|---|---|
| `runtime-scene.json`、`scene-profile.json`、`controller-setup.json` | 传感器、场景与控制参数快照 |
| `template.json`、模板PNG、`stats-episode.h5`、`validation-frames.npz` | 本次模板、统计与匹配预检输入 |
| `matching-preflight.json` | 远近景匹配落点校验 |
| `run.h5`、`poses.json`、`signals.csv` | 匹配、位姿、速度指令与时间戳 |
| `front.mp4`、`chase.mp4`、`*-frames.csv` | 双路录像与源帧索引 |
| `viz_dump/` | 供RViz读取的逐帧匹配和位姿 |
| `report.json` | 停车原因、实际源帧率与是否满足深度阈值 |
| `prepare.log`、`control.log`、`projectairsim_client.log`、`rviz.log` | 阶段输出与连接日志 |

地图用于空间对照；参考点用于准备与评价，闭环不从目标坐标生成速度。4米指相机深度阈值，并不自动等于到指定部位表面的4米。

## 场景配置与切换

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

### 更换UE场景

1. 在UE中打开目标关卡，确认ProjectAirSim插件、地面和目标碰撞配置，进入Play。
2. 建立对应场景JSON，填写服务地址、Actor、准备点和独立资产目录；复制无人机JSONC时同步更改机器人引用。
3. 采集并核对目标位置、比例、机体轴向与相机视角。准备该场景的模板和统计图像，重新导出网格、碰撞栅格及评价元数据；不要沿用其他场景的目标坐标和ROI。
4. 运行配置检查，再按[运行说明](USAGE.md#实时抵近)执行准备与匹配预检。

```bash
python src/capture/prepare_scene.py --profile airport --validate-only
python src/debug/probe_target_object.py --profile airport --name StaticMeshActor_7890
```

ProjectAirSim场景JSONC配置当前UE关卡中的机器人与传感器，不会自动替用户切换UE关卡。准备入口会重载机器人配置并执行坐标飞行，控制入口读取本次`runtime-scene.json`和`scene-profile.json`快照。

### 地图采集入口

```bash
python src/debug/maps/read_scene_bounds.py --profile airport
python src/debug/maps/generate_global_map.py --profile airport
python src/debug/maps/prepare_meshes.py --profile airport
python src/debug/maps/build_grid_maps.py --profile airport
```

前两条连接UE读取对象与碰撞。`prepare_meshes.py`读取`editor.wsl_export_dir`中的OBJ及变换，`build_grid_maps.py`离线生成栅格。运行前需在UE Python中执行`src/debug/maps/export_aircraft.py`导出目标，保持脚本所在项目结构和`configs/`可访问；`capture_airport.py`可选导出带坐标俯视图，图像不参与栅格生成。

配置中的Actor、准备几何和网格必须与实际关卡一致。地图数据说明见[data/maps/airport_scene/README.md](../data/maps/airport_scene/README.md)。

## RViz地图与匹配显示

ROS2节点使用系统Python。地图与保存记录的回放无需UE运行。

### 显示机场栅格

```bash
cd /home/hui/research-project
source /opt/ros/humble/setup.bash
ros2 launch src/visualization/matching.launch.py profile:=airport
```

默认显示二维占用格、背景碰撞体素及两架飞机表面体素。Fixed Frame为`world`，采用ENU米。白色为空闲、黑色为占用、灰色为未知；飞机颜色表示高度。全场分辨率5米，飞机本体及5米边距范围分辨率0.25米。

### 回放与同步

```bash
ros2 launch src/visualization/matching.launch.py profile:=airport replay:=true dump_dir:="$PWD/runs/approach/close4-1793954736186/viz_dump" rate:=2.85
```

将`dump_dir`改为正在运行的试验目录即可同步读取新帧。`rate`是显示每秒读取记录的数量；已存在地图节点时追加`publish_map:=false`。相同目录重新启动节点从头回放。ROS终端输出可重定向到本次试验的`rviz.log`。

| 话题 | 内容 |
|---|---|
| `/near_nose/grid/global` | 全场OccupancyGrid |
| `/near_nose/grid/StaticMeshActor_7890`、`/near_nose/grid/StaticMeshActor_7888` | 飞机精细二维OccupancyGrid |
| `/near_nose/map` | 栅格色块、背景和飞机体素MarkerArray |
| `/near_nose/plan/traj` | 无人机历史轨迹 |
| `/near_nose/plan/markers` | 无人机与运行标记 |
| `/near_nose/plan/view` | 匹配画面 |

地图节点同时发布标准OccupancyGrid和批量等值行段色块。色块直接表达每格状态，避免索引贴图着色器问题。地图话题使用TRANSIENT_LOCAL，后启动的订阅者也能读取。

### 匹配热力图的显示与解读

启动参数`heatmap_mode`默认`control`，可改为`similarity`。对保存的原始余弦相似度`s_i`：

| 模式 | 红色强度 | 含义 |
|---|---|---|
| `similarity` | `clip((s_i-0.2)/0.4, 0, 1)` | 原始相似度绝对大小 |
| `control` | `sqrt(softmax(s_i / 0.005))` | 控制中心使用的相对权重；最高响应小于0.2时不叠红色 |

平方根只影响显示强度，不参与控制。绿色十字是实际控制使用的中心，经系数0.3指数滤波，绘制在叠色之后。红区缩小不说明部位定位更准确；两种热力图均不是部位分割掩膜，softmax权重也不是校准后的识别置信度。

原始响应位于`run.h5`的`similarity`和`viz_dump/step_*.npz`的`sim`。回放直接读取记录，不重跑DINO。核对其他试验时需使用对应控制温度。

## 附录：PX4接口

`src/px4/`通过MAVSDK连接PX4，使用ProjectAirSim提供相机。配置为`configs/simulation/scene_airport_px4.jsonc`与`robot_quadrotor_px4.jsonc`。

| 端口 | 用途 |
|---|---|
| TCP 4560 | ProjectAirSim与PX4仿真传感器、电机链路 |
| UDP 14540 | MAVSDK接收offboard遥测 |
| UDP 14588 | ProjectAirSim配置中的控制链端口 |

在UE选择机场关卡并进入Play；在PX4终端设置宿主地址后启动SITL：

```bash
export PX4_SIM_HOST_ADDR=172.21.192.1
cd ~/PX4-Autopilot
make px4_sitl none_iris
```

宿主地址还需与机器人JSONC的`local-host-ip`和场景档案的`simulator.address`一致。连接示例参数可通过入口查看：

```bash
python src/px4/px4_yaw_probe.py --help
python src/px4/servo_closed_loop_px4.py --help
```

航向检查显式指定`--yaw`，视觉入口显式指定与PX4相机视角一致的`--template`和`--stats-episode`。通过`--mav-url`选择MAVSDK端点。当前机场的SimpleFlight记录见[实验索引](../reports/EXPERIMENTS.md)；PX4运行需单独完成本场景的相机和飞控验证。

一次运行的控制记录保存在`runs/approach/`。需要重建仿真连接时，先停止客户端，结束UE Play，再重新启动PX4与Play。
