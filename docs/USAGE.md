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

### 同一F-14起点抵近不同部位

在任务配置`configs/missions/airport_f14_approach.json`的`common_start`中指定共同位置、航向和实际状态容差。三次试验分别重载场景并回到此起点，共用35度下俯相机、整机模板和统计数据。先确认UE机场场景、目标变换及当前无人机状态；准备命令会重载无人机并起飞。使用独立的空会话目录保留本轮全部数据。

```bash
session_dir="$PWD/runs/approach/my-common-start"
part=nose  # 分别使用 nose、wing、tail；tail为配置的一侧垂尾
python src/capture/prepare_scene.py --profile airport_f14 --address 172.21.192.1 \
  --common-start --reference-dir "$session_dir/reference" \
  --target-part "$part" --out-dir "$session_dir/$part"
python src/control/run_approach.py --out-dir "$session_dir/$part" --validate-only
python src/control/run_approach.py --out-dir "$session_dir/$part" --max-duration 180
python src/evaluation/eval_part_approach.py --run-dir "$session_dir/$part"
```

首次准备创建`reference/`，后两项复用该目录。准备时所有部位须同时在初始画面内；匹配预检要求同一画面上的三个响应中心各自落在对应区域。预检失败不启动抵近；实际起点位置、朝向、速度或倾斜超限也拒绝运行。`--validate-only`不连接UE。一次控制和终点采图结束后再准备下一部位，不同时连接多个完整传感器客户端。

`start-state.json`和控制报告中的`start_state`分别记录准备及正式起点；`shared_asset_sha256`用于核对共用输入。`matching-reference.png`和`matching-start.png`显示三个部位的预检落点；`viz_dump/step_*.npz`包含`crop_rgb`、`crop_xyxy`、`crop_similarity`和`target_part`，可展示从整机响应到部位响应的计算过程。终点核验见`evaluation/part-surface-validation.json`与`part-validation.png`；使用同一评价命令追加`--offline`可复核保存数据。

从不同视角分别试验部位时，可在离线检查及正式控制命令中显式追加`--preflight-selected-part-only`。它仍输出三个部位的结果，只要求本次所选部位在参考图和本次起始图上通过；失败部位不启动抵近。使用`--mission <本轮任务JSON>`向准备入口指定独立的共同起点，保留原始任务配置和参考来源。控制报告会记录预检范围，不能将所选部位通过写成三个部位全部通过。

为每个起始观察方向准备对应视角的固定模板，三种部位共用该组参考图；参考采集与控制起始帧分别记录。检查预检中的`reference_patch_count`：若为0，该部位在当前模板特征网格上无法建立原型，`failure_reason`为`no_reference_patches`，不会执行抵近。该情况与已经建立原型但起始图匹配错误分别记录。

窄ROI没有覆盖格子中心时，在离线预检和控制两条命令中均追加`--thin-roi-refinement`。该选项自动补一格上下文，按交叠面积生成DINO候选，再以固定原图局部配准精修落点；原始部位框不扩大，预检精修点须落在框内。相关分数低于0.85、竞争峰差低于0.15时停止运动，感知耗时超过0.75秒退出并悬停。默认全部部位预检仍适用，需要单独评价时再追加`--preflight-selected-part-only`。

查看`reference_center_patch_count`与`reference_patch_count`可分别核对中心选格数和实际参与数。`control-signals.json`保存`point_source`、`refinement`、`coarse_dino_center`、未滤波匹配点和深度取样位置；`viz_dump`也保存点来源与配准分数。此模式的尾翼落点来自局部配准，DINO热力图只显示候选响应。深度取样邻域随注册部位尺寸限制，避免窄目标被周围背景深度淹没。终点评价照常运行，结果包含实际匹配点来源。

终点评价默认重新初始化匹配器。核验窄ROI的连续跟踪时，先保留这项结果，再使用`eval_part_approach.py --run-dir <目录> --offline --continue-tracking`，以最后接受的控制帧配准状态初始化，对已独立采集的终点图像重新匹配。输出的`matching_initialization`和`tracking_seed`区分重新截获与延续跟踪；延续通过不能代表近景重新截获通过。

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

### 洞口匹配与入室记录回放

在项目根目录使用ROS2系统Python运行。该入口读取本地记录，不需要UE；`--export-only`只生成回放包。启动洞口离线匹配展示：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 src/visualization/opening_replay.py --mode matching --rviz
```

先以Ctrl+C停止当前回放，再将`--mode`改为`entry`可显示坐标引导入室的真实轨迹和阶段照片。两种模式使用同组`/opening/`话题，不能同时在同一ROS域发布。`--rate`表示每秒播放的保存样本数量，不是相机帧率；匹配模式默认0.5，入室模式默认8。

点云为9组开发RGB-D的投影结果；灰点为采集位姿，蓝球为当前回放位姿，青点及连线为离线投影空隙参考，不代表飞过的轨迹或三维净空。入室模式只回放已有4组照片，持帧期间显示照片时间差。窗口不添加目标名称或分辨率浮动标注。每种模式的`rviz/`目录保存manifest、点云及验证记录。

回放实时水平对准时，使用`alignment`模式并指定控制目录；以下以2倍速度循环播放全部保存观测，窗口同时显示实际轨迹、RGB-D点云和对应匹配画面：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 src/visualization/opening_replay.py --mode alignment \
  --flight-dir runs/opening-alignment/20261112-hangar2019/control \
  --speed 2 --rviz
```

该模式按控制记录中的时间间隔播放，`--speed 1`为原记录速度，循环末暂停2秒；传`--rate`可改为按索引播放。点云从本次飞行每10组RGB-D及末帧生成，位姿和图像保留全部记录；它是保存深度与仿真相机位姿的投影，不是实时SLAM。回放包位于控制目录的`rviz/`，包含manifest、点云、窗口配置及验证结果。停止其他`/opening/`发布进程后再运行。

### 匹配热力图的显示与解读

机库飞行的飞机风格展示可先在`airsim`环境离线生成，再使用ROS系统Python播放：

```bash
python src/visualization/render_opening_features.py \
  --run-dir runs/opening-entry/20261112-hangar2019/control \
  --out-dir runs/opening-entry/20261112-hangar2019/control/rviz/feature-view
source /opt/ros/humble/setup.bash
/usr/bin/python3 src/visualization/opening_replay.py --mode alignment \
  --flight-dir runs/opening-entry/20261112-hangar2019/control \
  --feature-view-dir runs/opening-entry/20261112-hangar2019/control/rviz/feature-view \
  --speed 2 --image-window --rviz
```

生成入口要求空输出目录；已有完整展示目录时直接运行播放命令。机库展示中，红色是DINO空间patch相似度的相对响应，绿色大十字是控制准星，密集绿色小点是RGB-D跟踪角点。小点包含关键帧刷新后新提取的特征，不等于全部已通过匹配的内点。红色强度经峰值归一化增强可见性，不是识别置信度或分割结果；绿色准星也不是红色峰值。界面标注`RECOMPUTED RESPONSE`，表示特征由保存图像离线重建，原轨迹及控制指令保持不变。

以下`heatmap_mode`参数和`CONTROL WEIGHTS`说明适用于飞机抵近回放：

启动参数`heatmap_mode`默认`control`，可改为`similarity`。对保存的原始余弦相似度`s_i`：

| 模式 | 红色强度 | 含义 |
|---|---|---|
| `similarity` | `clip((s_i-0.2)/0.4, 0, 1)` | 原始相似度绝对大小 |
| `control` | `sqrt(softmax(s_i / 0.005))` | 控制中心使用的相对权重；最高响应小于0.2时不叠红色 |

平方根只影响显示强度，不参与控制。绿色十字是实际控制使用的中心，经系数0.3指数滤波，绘制在叠色之后。红区缩小不说明部位定位更准确；两种热力图均不是部位分割掩膜，softmax权重也不是校准后的识别置信度。

原始响应位于`run.h5`的`similarity`和`viz_dump/step_*.npz`的`sim`。回放直接读取记录，不重跑DINO。核对其他试验时需使用对应控制温度。

## 洞口匹配开发验证

固定参考图、模板裁剪框和开发图像清单位于`configs/missions/hangar_opening_validation.json`。离线入口生成模板、参考图统计HDF5、逐图匹配及总览，不连接UE：

```bash
python src/evaluation/eval_opening_matching.py \
  --out-dir runs/opening-validation/<id>
```

补采独立开发序列时，在机场场景进入Play。该入口要求当前没有Drone1，随后加载机器人，从出生点按已知目标坐标采集100、80、65、55、45米及65米横向偏移视图；结束在屋顶上方悬停。它不执行视觉抵近或入室：

```bash
python src/capture/collect_opening_views.py --profile airport \
  --out-dir data/captures/hangar_2019/<id>
```

本机WSL可按网络情况显式传`--address 172.21.192.1`。离线验证追加该批数据时，传`--capture-dir data/captures/hangar_2019/<id>`并使用新的空输出目录。开发图不能作为独立测试集统计成功率。绿色准星为DINO响应中心，红色为softmax相对权重，均不表示可通行区域；当前输出不作为实飞放行依据。

增加`--depth-geometry`可比较深度候选加空间DINO排序，输出`*-geometry.png`及`geometry-overview.png`。灰框为其他候选，黄框为最高分候选，绿色为其当前可见深层射线区域，青色准星为投影空隙建议点。`min_spatial_dino_score`是开发阈值，不是校准置信度；弱匹配不显示建议点。该图不证明完整三维通道可以穿越，查询帧仍需独立标注和后续飞行验证。

## 洞口实时水平对准

在机场UE场景进入Play，先核对已有无人机状态。当前没有Drone1时，可准备坐标引导的65米高起点；入口发现已有Drone1会拒绝重载：

```bash
python src/capture/collect_opening_views.py --profile airport \
  --address 172.21.192.1 --alignment-start-only \
  --out-dir runs/opening-alignment/<id>/preparation
```

用该准备目录的`runtime-scene.json`描述当前无人机。先运行观察模式，记录实时RGB-D、DINO候选及拟发速度，不发送飞行指令：

```bash
python src/control/run_opening_alignment.py --profile airport \
  --address 172.21.192.1 \
  --runtime-scene runs/opening-alignment/<id>/preparation/runtime-scene.json \
  --out-dir runs/opening-alignment/<id>/observation
```

在另一空目录运行带`--execute`的同一入口可执行水平对准：

```bash
python src/control/run_opening_alignment.py --profile airport \
  --address 172.21.192.1 \
  --runtime-scene runs/opening-alignment/<id>/preparation/runtime-scene.json \
  --out-dir runs/opening-alignment/<id>/control --execute
```

入口要求无人机已经在空中并启用API控制，保持初始高度。每次水平速度脉冲后悬停并重新取图；弱匹配、失锁、过期图像、目标跳变及高度异常停止。`--max-duration`限制为1—240秒。默认观察8组独立RGB-D，可用`--observe-frames`调整。

`frames/`保存原始图、深度和匹配诊断，`signals.jsonl`记录候选、图像误差、拟发速度及指令结果，`report.json`记录停止原因、初末状态和保护参数。对准确认只适用于该实验设定，不代表三维净空或穿越成功。原生相机话题只允许一个完整客户端，准备、观察和控制按顺序运行。

## 洞口视觉抵近与入室

使用已准备好的空中无人机及相机配置快照，传`--enter --execute`执行完整流程。入口在水平对准后核验RGB-D采样通道，初始化视觉锚点，再分段下降、穿越洞口并在内部悬停采图：

```bash
python src/control/run_opening_alignment.py --profile airport \
  --address 172.21.192.1 \
  --runtime-scene runs/opening-entry/<id>/preparation/runtime-scene.json \
  --out-dir runs/opening-entry/<id>/control --enter --execute
```

起点准备使用前述`collect_opening_views.py --alignment-start-only`，将输出目录设为`runs/opening-entry/<id>/preparation`。`--enter`模式时限默认900秒，可用`--max-duration`缩短。省略`--execute`可只观察；省略`--enter`只做定高对准。已有Drone1时先核对其状态，不能直接重新运行准备入口。

三维通道检查使用保存和当前深度的自由射线，视觉跟踪、深度或通道条件不满足会停止并悬停。该流程假设场景静止，相机近盲区和已出视野的包络依赖此前观测，不能据此宣称动态避障。`report.json`中的`entered=true`表示完成内部悬停确认；`interior-observation.png`保存内部观察图。`signals.jsonl`分别保存视觉进度、实际位姿、通道检查与指令，便于区分视觉控制和独立安全保护。

完整视觉入室记录沿用控制回放格式，保留实际下降轨迹及原时间间隔；RGB-D交接后显示当时原始图像及状态：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 src/visualization/opening_replay.py --mode alignment \
  --flight-dir runs/opening-entry/20261112-hangar2019/control --speed 4 --rviz
```

## 斜俯视抵近与入室

在UE机场场景进入Play后，准备侧上方起点。镜头向下70度，默认按屋面高度计算水平偏移；准备阶段使用目标包围盒坐标，正式控制依据实时RGB-D：

```bash
python src/capture/collect_opening_views.py --profile airport \
  --address 172.21.192.1 --alignment-start-only --camera-down-angle 70 \
  --start-offset-ne -23 0 \
  --out-dir runs/opening-entry/<id>/preparation
```

若当前已有Drone1，默认拒绝重载。明确需要重新准备仿真实验时追加`--reset-scene`，入口先记录当前状态再重载；`--start-offset-ne N E`可指定相对于目标包围盒中心的水平起点偏移。

先观察8组独立输入，核对DINO匹配、相机实际朝向及拟发控制，不发送飞行指令：

```bash
python src/control/run_opening_alignment.py --profile airport \
  --address 172.21.192.1 --enter --oblique \
  --runtime-scene runs/opening-entry/<id>/preparation/runtime-scene.json \
  --out-dir runs/opening-entry/<id>/observation
```

观察满足条件后执行斜向抵近，前进与下降同时进行，进入后确认悬停并采图：

```bash
python src/control/run_opening_alignment.py --profile airport \
  --address 172.21.192.1 --enter --oblique --execute \
  --runtime-scene runs/opening-entry/<id>/preparation/runtime-scene.json \
  --out-dir runs/opening-entry/<id>/control
```

斜向入室要求近似水平屋面，并核验包含整机扫掠范围的采样通道；相机角度与路径角度分别记录。水平速度上限0.25米/秒，下降随路径斜率协调限速，近屋面上限0.35米/秒。光轴附近距离小于5米、视觉失锁、未知通道或深度异常都会停止。70度在合成视野检查中可完整覆盖指定包络，实际场景仍须通过实时检查；角度更浅时可能因包络出视野被拒绝。

洞口上下文接近图像边缘、最小边界余量降至8%或更少时，当前帧DINO与RGB-D几何一致性仍须通过，再交接给RGB-D跟踪；到估计屋面25米也会交接。`dino_handoff`记录实际触发原因。该视角与起点偏移属于开发参数，观察失败时先核对记录，再决定是否重新布置起点。

`report.json`记录`oblique`、交接洞口位置、路径倾角和实际完成状态。`signals.jsonl`记录视觉路径横向误差、下行进度和路径前视点；`render_opening_features.py`与`opening_replay.py`支持这些记录。未完成内部悬停确认的执行不报告成功。

本轮斜向入室记录的响应与轨迹回放：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 src/visualization/opening_replay.py --mode alignment \
  --flight-dir runs/opening-entry/20261112-oblique-hangar2019/control \
  --feature-view-dir runs/opening-entry/20261112-oblique-hangar2019/control/rviz/feature-view \
  --speed 4 --image-window --rviz
```

## F-14机头、主翼与垂尾

目标配置为`airport_f14`，Actor为`StaticMeshActor_7888`。`nose`选择机头，`wing`选择一侧主翼，`tail`选择同侧垂尾。每个部位分别准备与归档；准备阶段重载机器人、按已知坐标采集模板并布置起点，控制阶段根据DINO与深度抵近。

```bash
python src/capture/prepare_scene.py --profile airport_f14 \
  --address 172.21.192.1 --target-part nose \
  --out-dir runs/approach/<id>/nose
python src/control/run_approach.py --out-dir runs/approach/<id>/nose \
  --max-duration 600
```

机翼、垂尾分别把`nose`换成`wing`、`tail`，准备和控制使用同一个部位目录。控制省略`--profile`时读取该目录已保存的场景和实际传输地址。默认35度下俯、60米水平起点、远段前向速度1.5米/秒、6米深度停车，12米内合速度上限0.35米/秒；停车深度与到所选部位表面的距离分别评价。

准备后可先用`run_approach.py --out-dir <部位目录> --validate-only`离线核对远景、35米视角和8米视角的模板落点。预检失败先核查`matching-preflight.json`、模板ROI与校验图；对应部位的视觉控制须全部通过预检再开始。`report.json`保存实际停止原因和到达确认，`front.mp4`、`chase.mp4`及`viz_dump/`用于录像和DINO匹配展示。

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
