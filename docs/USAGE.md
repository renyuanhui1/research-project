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

入口要求无人机已经在空中并启用API控制，保持初始高度。每次水平速度脉冲后悬停并重新取图；弱匹配、失锁、过期图像、目标跳变及高度异常停止。`--max-duration`设置正的有限执行时限。默认观察8组独立RGB-D，可用`--observe-frames`调整。

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

### 部位冷启动离线对照

使用`airsim`环境，读取已保存的窄ROI垂尾试验，不需要启动UE：

```bash
python src/evaluation/eval_part_reacquisition.py \
  --run-dir runs/approach/20261113-f14-tail-local-refinement/tail \
  --out-dir runs/approach/<研究id>
```

输出目录须独立于原试验目录，且不能含已有的`report.json`或`records.json`。入口对比冷启动、单因素历史和搜索范围/尺度的组合，固定配准分数0.85与峰差0.15；条件扩展方法只在默认冷启动失败时扩大三倍搜索半径。原始图和投影框擦除图分别运行。`report.json`保存环境、参数、距离分组、输入哈希和评价限制，`records.json`保存每帧每种方法的诊断，`summary.csv`保存汇总，`comparison.png`展示落点，`negative-examples.png`展示合成擦除图。

样本为起始图、全部无损全画面记录和独立终点图，距离分布不均。中间帧几何使用异步位姿，仅作近似诊断；终点使用相机元数据。继承状态的方法使用之前已接受的记录，起始图没有历史时不参与。运行需保留原试验与共用参考资产；每帧耗时仅统计配准，不能当作实飞帧率或独立测试成功率。

### 固定方案的新视角采集

在UE机场关卡进入Play后，使用已有垂尾试验作为固定参考；采集会加载机器人并执行坐标飞行：

```bash
python src/capture/collect_part_views.py \
  --source-run runs/approach/20261113-f14-tail-local-refinement/tail \
  --out-dir data/captures/f14_tail/<采集id> --validate-only
```

预检通过后去掉`--validate-only`开始采集。方案来自`configs/missions/f14_tail_reacquisition_validation.json`：45、30、18米水平距离，−12/0/+12度侧向角，−8/+8度航向偏移，每个条件3组RGB-D，共18个条件、54组。坐标只用于采集与事后核验；结束后在巡航位置悬停。输出目录必须为空。入口发现已有机器人会停止，不直接重载。

本入口若在取图前中断，可先检查原机器人状态，再用新的空采集目录和`--resume-from <中断目录>`恢复；仅接受零采样中断、相同配置且原机器人仍在出生点或预定航点附近悬停的情况。已有采样的中断目录应保留，不能套用该恢复方式。

采集完成后运行离线评价，无需保持UE运行：

```bash
python src/evaluation/eval_part_reacquisition.py \
  --run-dir runs/approach/20261113-f14-tail-local-refinement/tail \
  --capture-dir data/captures/f14_tail/<采集id> \
  --out-dir runs/approach/<评价id>
```

评价核对采集前冻结的代码与参考资产，参数不匹配时停止。`report.json`分别汇总帧与观察条件，`records.json`保存逐帧判断，`view-overview.png`显示全部观察条件的首帧，`comparison.png`显示三种方法的落点。新视角来自同一静态飞机场景，不代表跨飞机泛化或在线失锁恢复。

### 训练视角采集

`capture/collect_training_views.py`按`configs/missions/f14_training_views.json`在飞机前向坐标系内随机采样位姿，用`SetPose`逐帧瞬移取图，不执行航线飞行：

```bash
python src/capture/collect_training_views.py \
  --source-run runs/approach/20261113-f14-tail-local-refinement/tail \
  --out-dir data/captures/f14_tail/<采集id> --validate-only
```

预检只核对地图间距与参考点是否在画面内，不连接仿真；通过后去掉`--validate-only`，在UE关卡进入Play后开始采集。无需改动关卡的曝光设置：采集相机的自动曝光在场景加载后约30帧内从纯白收敛到正常亮度，采集器在记录前持续取图，直到连续3帧均值低于220且标准差大于5才进入采样，120秒仍未收敛则报错停止；收敛过程产生的帧不会写入数据集。准备阶段创建共用参考图时使用同一判据等待曝光收敛，避免模板和统计量取自过曝帧。默认5000个位姿，距离按70%取12–60米、30%取60–150米分层，对应相对垂尾约8–105米高度；侧向角±40度（相对飞机前向），航向偏移±20度，相机保持固定−35度。近中段用于部位匹配，远段用于整机粗定位。每帧保存原生RGB、深度、机头/主翼/垂尾/整机掩码、相机元数据和机体位姿，`report.json`每50帧写一次。`--count`覆盖采样数量；输出目录必须为空；`--resume-from`只接受准备阶段的零采样中断。

掩码由`capture/part_masks.py`用目标网格射线投射生成，部位区域与评价标签使用同一规则。部位成员同时受场景档案中的前向、横向、高度边界和以共用尾参考点为圆心的半径约束；机头与主翼的半径放宽到30米，实际只由分区边界决定，垂尾保留2.5米半径。位姿取自图像消息自带的相机位置与四元数，而不是取图后读取的机体运动学；后者可能晚于图像约0.5秒，无人机在此期间会掉落，直接用会造成米级偏差。掩码只依赖位姿、网格和区域配置，不读图像，因此可在不连仿真时预先统计各视角的可见性。

改动部位区域后，用保存的相机位姿离线重渲染掩码，不需要重新仿真：

```bash
python src/capture/rerender_part_masks.py \
  --capture-dir data/captures/f14_tail/<采集id> \
  --run-dir runs/approach/<运行目录>
```

加`--patches-dir <样本.h5或图片目录>`与`--features <features.jsonl>`时按已记录的搜索块边界重写64×64目标；在没有Open3D的机器上用`--targets-only`只从已渲染的掩码重建目标。

### 训练样本与局部匹配器

特征导出用`--patches <文件.h5> --all-parts`把RGB裁剪块及各部位目标写入一个HDF5文件；不加`--all-parts`时只写任务指定部位。`--out-dir`保存`features.jsonl`与`summary.json`。例如：

```bash
python src/training/extract_part_features.py \
  --capture-dir data/captures/f14_tail/20261114-training-views \
  --run-dir runs/approach/20261113-f14-tail-local-refinement/tail \
  --out-dir runs/training/<运行id>/patches-all \
  --patches runs/training/<运行id>/patches-all/patches.h5 --all-parts --skip 30
```

F-14现有4970帧样本保存在`runs/training/f14_tail_acceptance/patches-all/patches.h5`，训练命令为：

```bash
python src/training/train_local_matcher.py \
  --features runs/training/f14_tail_acceptance/patches-all/features.jsonl \
  --patch-dir runs/training/f14_tail_acceptance/patches-all/patches.h5 \
  --capture-dir data/captures/f14_tail/20261114-training-views \
  --parts nose wing tail --out-dir runs/training/<运行id>/local-matcher
```

训练和跨模板评价的`--patch-dir`、标签重建的`--patches-dir`均可传HDF5文件或PNG/NPY目录。每折权重为`fold*.pt`，指标为`metrics.json`，逐帧预测为`oof.jsonl`；接受层输出另含`acceptance.joblib`，每个候选模型的分数直接读取`oof.jsonl`中的`scores`。

### F-35部位参考与训练采集

`configs/scenes/airport.json`已按F-35配置机头、主翼和一侧垂尾区域，共同参考点取机头。准备共同参考后按同一入口采集训练视角：

```bash
python src/capture/prepare_scene.py --profile configs/scenes/airport.json \
  --mission configs/missions/f35_approach.json --common-start \
  --reference-dir runs/approach/<运行目录>/nose/reference \
  --target-part nose --out-dir runs/approach/<运行目录>/nose
python src/capture/collect_training_views.py \
  --source-run runs/approach/<运行目录>/nose \
  --out-dir data/captures/f35/<采集id> \
  --protocol configs/missions/f35_training_views.json
```

参考图以机头为中心时，三个部位ROI覆盖整机裁剪中心，冻结匹配器不产生候选，特征提取只输出裁剪块和三个部位的64×64目标，没有NCC特征；训练局部匹配器时使用`--parts nose wing tail`。

### 跨模板与跨机型评价

`evaluation/eval_cross_template.py`读取带`patch_xyxy`的特征文件和对应的HDF5样本文件或图片目录，按训练相同的（15度侧向扇区×距离档）分组重建5折。`--mode self`每折只评自己的留出帧，用于复现训练指标；`--mode transfer`用全部折模型推理另一份数据，用于跨采集或跨机型评价。两种模式都输出3像素正确率、峰值落在掩码内的比例，以及除以掩码等效半径√(面积/π)的面积归一化距离。迁移模式下每个样本有5个模型的预测，结论以每折均值±标准差为准。

```bash
python src/evaluation/eval_cross_template.py --mode transfer \
  --features runs/training/f35/patches-all/features.jsonl \
  --patch-dir runs/training/f35/patches-all/patches.h5 \
  --capture-dir data/captures/f35/20261115-training-views \
  --weights-dir runs/training/f14_tail_acceptance/local-matcher-allparts-fixed \
  --out-dir runs/evaluation/<运行id>
```

`--limit`用于小样本冒烟检查。结果写入`oof.jsonl`与`metrics.json`，逐帧记录折号、部位、预测点、有符号距离、掩码面积和归一化距离。

### 俯视模板基线

`evaluation/eval_nadir_template.py`从记录的高空俯视图裁出目标模板：按采集报告里的相机位姿投影目标网格顶点，外扩`margin=1.15`取正方形裁片。随后在给定采集的每帧上比较俯视模板与斜视模板的整机响应，报告峰值/质心是否落在整机掩码内及落点距离。标准化统计量取被评采集自身等间隔32帧（`--stats-frames`），两个模板共用。

```bash
python src/evaluation/eval_nadir_template.py \
  --capture-dir data/captures/f35/20261115-training-views \
  --overhead-image data/captures/aircraft-overhead/StaticMeshActor_7890_overhead.png \
  --overhead-report data/captures/aircraft-overhead/capture-report.json \
  --mesh data/maps/airport_scene/meshes/StaticMeshActor_7890.npz \
  --actor StaticMeshActor_7890 \
  --oblique-template data/templates/airport_7890/oblique/aircraft-template.png \
  --out-dir runs/evaluation/<运行id> --skip 30
```

`--skip 30`去掉曝光未收敛的开头帧。每个模板同时取0/90/180/270度四个旋转，`metrics.json`的`single`为原方向结果，`best_of_4`为逐帧按峰分选最高旋转的非真值结果。输出`template-nadir.png/json`、`records.jsonl`（逐帧、逐模板、逐旋转）和按距离档的叠加图`peak-<距离档>-<旋转>.png`。

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

## 远端粗定位与近端部位精对准

UE Airport处于Play后，按目标配置执行斜视准备、匹配预检、控制和独立终点核验。第一项F-14示例：

```bash
python src/capture/prepare_scene.py --profile runs/approach/20261116-coarse-to-fine/01-f14-intact/profile.json --oblique --out-dir runs/approach/20261116-coarse-to-fine/01-f14-intact/control
python src/control/run_approach.py --out-dir runs/approach/20261116-coarse-to-fine/01-f14-intact/control --validate-only
python src/control/run_approach.py --out-dir runs/approach/20261116-coarse-to-fine/01-f14-intact/control --max-duration 600
python src/evaluation/eval_part_approach.py --run-dir runs/approach/20261116-coarse-to-fine/01-f14-intact/control
```

远端以整目标原型定位，连续3帧深度进入45米后切换指定部位ROI；6米停车要求3张独立深度帧确认。水平起点距离、相机俯角和实际高度分别记入任务与准备记录。准备、匹配、控制或终点核验失败时停在当前目标，分析后再执行；停车与独立终点核验均通过才进入下一目标。

局部部位在场景档案的`parts`中按名称配置，任务用`target_part`选择。机鼻使用参考球域；其他部位配置前向、横向和高度边界、`seed_longitudinal_m`、`seed_lateral_m`、`seed_altitude_m`及`radius_m`，准备入口从真实网格中选择参考点并生成ROI。高度字段使用世界高度米。`approach_axis`取`rear`时从后方接近，取`negative_side`时从负横向接近，不设置时从正横向接近。方向选择不改变局部区域边界，非机鼻部位近端预检使用0.02归一化容差。

模板中局部框含有大面积相似背景或完整舱沿时，可在该部位配置`template_roi_fraction`，四个比例相对于投影局部框，例如`[0.1,0.1,0.9,0.65]`选择内部纹理。独立评价仍使用原几何区域。屋顶任务的整目标接近使用`run_approach.py --handoff-on-roi`，确认`handoff_ready`后交给洞口匹配和RGB-D通道入口；这一停止不作为近端任务到达。

屋顶任务使用`--roof-handoff`代替普通部位的深度停车：在实时屋面距离进入45米后悬停，读取`part_handoff`中的测量来源和确认帧数，再运行洞口入口。可在`signals.csv`检查`roof_vertical_range_m`和`phase_range_m`；深度分层缺失或屋面余量低于15米时不会继续接近。

普通部位抵近在首次进入`close_range_m`后保持`close_speed_mps`限速，即使后续取样落到较远背景也不重新加速。查看报告的`close_speed_latched_at_step`与`signals.csv`的`close_speed_latched`；该状态只在当前运行内有效。

给同质纹理部位配置独立近端参考时，设置`part_template_distance_m`和该距离原生图上的归一化`part_template_roi`，并把参考距离及`controller.close_range_m`加入`preparation.calibration_distances_m`。准备输出`part-template.png`、模板元数据和`fine-validation.png`。控制从整目标原型开始，原切换距离使用整目标模板中的部位ROI，进入`close_range_m`后停车并切到独立近端原型。远景、原部位、近端切换图及8米图分别预检；实际精细模板切换见报告`part_template_switch`。部位几何和终点评价条件独立于模板ROI。

粗定位报告确认`handoff_ready`后，用`python src/capture/prepare_opening_handoff.py --coarse-run-dir <粗定位目录> --out-dir <相机交接目录> --camera-down-angle 45`准备洞口相机。洞口入口的`--runtime-scene`指向该目录的`runtime-scene.json`，先不加`--execute`观察独立帧，再加`--execute --enter --oblique`执行。相机交接入口不会加载场景或移动无人机。

需要围绕已选表面点提取局部纹理时，在部位字段设置`template_anchor_half_size`，例如0.0625；它与`template_roi_fraction`共同约束实际匹配框，不修改几何评价区域。准备统计可设置`preparation.statistics_source=far_start`，使用最后6张未标注远端标定帧；默认`all_calibration`使用所有标定帧。元数据同时保存参考相机、完整目标投影框、参考点在模板中的位置和统计来源。

粗定位交接前会减速并实测停车；相机交接要求机体速度低于0.1米/秒、偏航角速度低于0.02弧度/秒，且当前航向与粗定位报告相差不超过5度。检查失败时先处理停车与航向，不直接开始入室。

任务允许从同一目标的任一屋顶破口进入时，可给洞口入口添加`--whole-target-run-dir <已确认粗定位目录>`。该模式用整目标模板及原门槛确认屋面归属，随后按真实RGB-D对准同一屋面内的闭合破口；报告`identity_mode=whole_target_then_any_opening_on_same_roof`。指定洞口身份验证继续使用原俯视参考入口，不添加此参数。

`--alignment-pulse <秒>`配置水平对准的速度脉冲时长，默认0.3秒；速度限幅仍由控制器执行，每次脉冲后实测停车并读取新RGB-D。运行记录保存实际时长，指令等待时限随该时长计算。

近端图像区域中断后，若无人机与相机保持原位，可添加`--resume-from <上一观察或控制目录>`重新获取目标。入口核对引用、配置及机体状态；下降记录只在仍高于已观测屋面至少`--min-roof-range`米时允许重新获取，默认15米。入口重新确认独立帧、屋面及完整通道，不恢复旧视觉位姿或通道缓存；接近或穿过屋面后的中断不能这样恢复。

任一破口任务会在投影对准后检查同屋面候选的完整三维通道，按实际射线选择通过者；查看`entry_path_candidates`与`handoff`中的路径，不以二维半径代替通道通过。`--entry-pulse <秒>`配置入室脉冲，默认0.8秒，报告保存实际值。

入室在已完整核验的通道内连续发送短速度指令，每次使用新RGB-D更新视觉位姿和通道；到达、等待或异常时执行实测停车。`--min-roof-range`配置获取屋面所需的最小观测距离，不改变完整通道、45秒观测缓存和碰撞保护。

屋顶正上方垂直入室可添加`--vertical-entry --enter`，初始斜视相机由控制器逐步跟随破口至垂直，保持高度水平对准后核验竖直下降通道。搭配`--whole-target-run-dir`确认目标屋面，运行快照保存相机实际安装姿态。普通`--oblique --enter`继续沿视觉斜线通道执行。

任一屋顶破口的垂直入室依次执行破损组上方对准、垂直视角下梁间空隙精对准、完整通道核验和下降。查看`roof_group_alignment`及逐帧`alignment_phase`；组中心可以位于梁上，仅作上方航点，下降使用另行精对准并核验的真实空隙。初始不足3帧稳定时会清零计数并悬停，`acquisition_resets`记录次数；确认后的跳变仍停止。

组水平段在获取完成后固定航点，通过近层屋面RGB-D视觉锚点跟踪相对运动；查看`roof_group_visual_anchor`及逐帧`group_visual_anchor`的配准质量。分割中心保存在`group_alignment_geometry_point_rgb_xy`，实际跟踪点保存在`group_alignment_point_cv_m`，两者用途不同。

`group_visual_anchor.inside_matched_group`表示锚点投影位于当前匹配破损组内；该条件与已验证视觉配准成立时，`region_continuity`可记录组框分裂后的连续身份。精对准及实际入室仍使用原真实空隙和通道条件。

垂直入室可直接使用粗定位的35度相机快照，无需另行切换相机：`--runtime-scene <粗定位目录>/runtime-scene.json --whole-target-run-dir <粗定位目录> --enter --vertical-entry`。该模式允许35—90度并跟随破口，普通斜线入口仍为40—80度。

## 目标库、俯视模板与数据划分

从保存的资产核对俯视图和导出网格离线建库，无需UE或无人机：

```bash
python src/capture/build_target_catalog.py --config configs/datasets/airport_targets.json
python src/capture/build_target_catalog.py --config configs/datasets/airport_targets.json --validate-only
```

输入路径、裁片余量、类别映射和训练/验证/测试划分由配置指定。输出`data/templates/airport_targets_20261116/catalog.json`，每个目标目录包含`template.png`、`assembly-mask.png`、`part-mask.png`；总览、构建日志和核验报告位于`runs/catalog/20261117-airport-targets/`。`--validate-only`核对输入/输出哈希及划分，不重写数据。

训练按`catalog.json.splits`中的目标ID选择数据，未来同目标的全部视角须跟随该目标的分区。`appearance`按主体网格分组；`tower_category`完全排除训练及验证中的塔台，测试结果区分未见类别和已见类别的新外观；`hangar_instance`仅作同网格机库的位置/背景对照，不能报告为新外观泛化。不同协议须分别训练，测试搜索图不得用于标定统计或阈值。

`catalog.json.primary_protocol`指定主实验。主体外观划分为训练11/验证3/测试5个目标，塔台类别留出为10/3/6，同网格机库对照为14/1/4。主实验训练包含机鼻、主翼、尾翼和座舱；座舱只有一个资产，仅作训练示例，塔台外观协议没有独立塔台验证样本。

`region.visible_in_nadir`和`visible_pixels`描述网格自遮挡后的部位可见性，`projected_crop_roi`仅是几何投影范围。空部位掩码不能作为可见部位正例；屋面掩码也不能作为可通行开口标签。模板检查没有实测深度，后续训练采集仍需用真实深度排除外部遮挡。

## 多目标斜视采集小样本

先离线查看采样计划，再在UE Play且无无人机Actor时执行：

```bash
python src/capture/collect_target_pilot.py --protocol configs/datasets/airport_pilot.json --validate-only
python src/capture/collect_target_pilot.py --protocol configs/datasets/airport_pilot.json
```

协议配置目标编号、距离、各类方位、相机尺寸和深度容差；小样本限定使用主协议训练目标。采集采用无物理相机，结束后移除；输出目录必须为空，避免覆盖已有样本。现有配置为F-35座舱、机库910、轻损塔台共44组，RGB 1280×720、深度及标签640×360。

数据位于`data/captures/airport_targets/20261117-pilot/`，包括`samples.h5`、`report.json`、`plan.json`、相机配置和`overview.png`。HDF5每组的`rgb_png`可用`common.patches.read_rgb`读取，`depth_m`为float16米制透视距离；掩码字段见[架构说明](ARCHITECTURE.md#多目标斜视采集)。训练时用`valid_mask`排除未知像素，不能将所有非目标像素都直接当成背景。完整原生图像和深度只有一份，不逐帧保存诊断PNG；总览为离线生成的标签叠加图。

## 多目标分片采集与整目标训练

采集在连接UE的本机执行；保持Airport处于Play、窗口可见，场景中没有无人机Actor：

```bash
python src/capture/collect_target_pilot.py --protocol configs/datasets/airport_multitarget.json --validate-only
python src/capture/collect_target_pilot.py --protocol configs/datasets/airport_multitarget.json
python src/capture/collect_target_pilot.py --protocol configs/datasets/airport_multitarget.json --resume
```

前两条分别查看计划和首次采集，第三条用于同协议中断恢复。输出`data/captures/airport_targets/20261117-multitarget/<目标ID>/`，每目标一个HDF5及报告。配置采用训练目标各128组、验证及测试目标各64组，共1920组；RGB 896×504，深度及标签640×360。根报告只列已完成并回读校验的分片。原始RGB保留，`ignored_mask`在编码前擦除后续分区资产；测试图不擦除。

将代码、目标库模板、完成的采集分片和DINO权重准备到237同一项目根下，再执行：

```bash
python src/training/extract_template_features.py --config configs/training/airport_template_matcher.json
python src/training/train_template_matcher.py --config configs/training/airport_template_matcher.json
python src/evaluation/plot_template_comparison.py --config configs/training/airport_template_matcher.json
```

提取器可添加`--target <目标ID>`逐个处理已完成分片，输入哈希一致时复用缓存。已有完整特征时直接运行训练命令；开发训练只要求所用分片可核验。配置指定数据协议、目录和参数，不能复用其他划分训练出的权重作为本协议结果；结果位于`runs/training/20261117-template-identity-context/`。

配置`context_labels`时，先在具备导出网格和Open3D的本机离线生成同图其他目标标签，无需UE Play：

```bash
python src/capture/label_target_context.py --config configs/training/airport_template_matcher.json
```

输出`data/labels/airport_targets/20261117-context/`的`labels.h5`与`report.json`；已有完成标签时直接复用，随后将该目录同步到237相同相对路径，再运行训练。只处理训练帧，原始RGB-D不变；训练入口核对标签、原始采集、特征及目标库的哈希。

查看`comparison.json`比较命中率、接受精度、覆盖率和错误模板误接受；`frozen-validation-selection.json`记录选定阈值，`manifest.json`记录Git提交及数据、配置、代码哈希。目标坐标和几何仅参与标签、负例构造与评价，模型输入只有模板及搜索图特征。经验95%验证精度不保证测试或在线精度，空接受集不能解释为控制可用。

`evaluation_partition=validation`用于开发，只读取训练和验证特征；`paired_variants`配置响应差异损失权重，`confidence_folds`配置按主体网格分组的交叉拟合折数。每轮按指定精度下的有效正例召回选模，置信度头使用折外预测拟合。`selection.json`记录所选方法及开发验收，`minimum_verified_recall`配置最低有效正例召回；开发结果不能代替独立测试。

绘图入口读取已有结果和本地原始分片，生成`comparison.png`、`prediction-examples.png`及`figure-selection.json`。示例固定为各评估目标每距离档第一张正例和第一张错误模板负例，不按预测成败选择；图中圆点表示接受、叉号表示拒绝，角落显示请求模板。

237上的独立代码目录、原始分片和特征缓存位置见[远端产物清单](../runs/training/20261117-template-identity-context/remote-artifacts.json)。最终权重、指标和日志保存在本地结果目录；重新训练须在具备所用特征缓存的项目根下执行。各定位器目录的`model.pt`包含定位器、置信度头、统计量、`thresholds`及配置，离线加载使用`torch.load(..., map_location=...)`；`presence/`和`crossfit/`分别保存两种分数的指标，`confidence-fold-*/`保存交叉拟合依据。控制入口尚未读取这些权重。

## 候选裁剪的模板核验

在237准备已有RGB分片、全图特征、冻结DINO权重及配置中的`coarse_checkpoint`，离线执行：

```bash
python src/training/train_template_verifier.py --config configs/training/airport_template_verifier.json
```

不需要UE Play。`crop_margin`、`crop_minimum_fraction`和`crop_logit_delta`控制图像候选裁剪，`reprojection_threshold`控制DINO图像块对应的仿射核验；`rgb_ratio`及`rgb_reprojection_threshold`控制RGB局部描述子的唯一性和几何核验，`classifier_seeds`指定接受层的随机种子。查询世界坐标和网格不参与核验。

`reuse_verification`可指向完整的DINO核验结果目录，复用已冻结候选及特征；训练入口核对配对清单和特征索引。未配置时重新生成折外候选与DINO裁剪特征。启用RGB核验后只补充局部统计，不复制原生RGB。

结果保存在`runs/training/20261117-template-verification-rgb/`。`proposals.npz`记录框、点和响应统计，`verification-features.npz`记录训练/验证特征与拟合标签，`verifier.pt`保存粗定位及接受层，`validation-records.json`保存各方法逐对预测。`selection.json`记录开发门槛是否达到；外层验证用于选模和定阈值，不能当成独立测试结果。用同一配置运行`plot_template_comparison.py`生成曲线及固定示例。

冻结产物可直接接收RGB数组进行离线推理：

```python
from perception.template_verification import TemplateVerifier
model = TemplateVerifier("runs/training/20261117-template-verification-rgb/verifier.pt", device="cuda")
model.set_reference(template_rgb)
result = model.predict(search_rgb)
```

数组为H×W×3、uint8、RGB顺序；沿用同一模板时只调用一次`set_reference`。`point_xy`在[0,1]内，`box_xyxy`使用原图像素；`accepted`表示是否通过选定阈值，拒绝时的点仍是待核验候选。接口不连接UE或执行控制。

冻结推理按配置的`feature_batch`及`batch`复现编码和粗定位的算术路径；单帧自动填充，不增加任务观测。完整RGB重放见`runtime-replay.json`，导出边界核验见`numeric-export-audit.json`。`accepted-overview.png`展示全部接受配对，包含误接受案例。

服务器代码、共享原始数据与特征缓存位置见[远端产物清单](../runs/training/20261117-template-verification-rgb/remote-artifacts.json)。`manifest.json`分别记录训练提交与导出提交、权重哈希及来源，`result-audit.json`记录原生标签复算和推理一致性核验。

## DINO骨干离线对照

准备配置指定的RGB-D分片、目标库、固定验证配对、本地权重及`third_party/dinov2`、`third_party/dinov3`官方源码后运行：

```bash
python src/evaluation/eval_dino_backbones.py --config configs/evaluation/dino_backbone_comparison.json
python src/evaluation/eval_dino_backbones.py --config configs/evaluation/dino_backbone_comparison.json --stage plot
python src/evaluation/eval_dino_backbones.py --config configs/evaluation/dino_backbone_comparison.json --stage benchmark
```

评价使用训练帧拟合各自的标准化统计，在固定验证图上直接执行整目标及ROI模板匹配，不训练定位网络或接受层。`conditions`指定模型、权重、源码目录及输入尺寸；`feature_batch`指定实际编码批量，`temperature`指定softmax温度。只有任务模板的固定部位ROI参与匹配，查询掩码用于事后判分。

输出位于`runs/evaluation/20261117-dino-backbones/`，包括登记与来源哈希、配对、逐对位置和分数、分组指标、参考原型及图表。各模型分别在验证集校准接受阈值，结果属于开发对照。将评价产物同步到本机后执行`benchmark`，`latency.json`记录本机设备与单帧耗时；`benchmark_frames`、`benchmark_warmup`控制计时样本及预热次数。此入口不连接仿真。

匹配方法消融复用以上结果的训练统计及同一批输入：

```bash
python src/evaluation/eval_dino_matching.py --config configs/evaluation/dino_matching_ablation.json
python src/evaluation/eval_dino_matching.py --config configs/evaluation/dino_matching_ablation.json --plot
python src/evaluation/eval_dino_matching.py --config configs/evaluation/dino_matching_ablation.json --benchmark
```

`normalizations`指定是否在L2前使用通道标准化；`readouts`选择`legacy`、`cell`、`peak`、`soft`、`local`、`mutual`和`affine`。`soft_temperature`控制柔化温度，`local_radius_cells`控制峰值邻域，`reprojection_threshold`及RANSAC参数控制逐块空间核验。部位ROI来自任务模板，匹配不使用查询目标框。

输出位于`runs/evaluation/20261117-dino-matching-ablation/`，`baseline-parity.json`核对基线命中数，`records.json`保留各设置的落点、分数及对应支持，`selection.json`区分命中最优和接受召回最优。所有选择和阈值均来自开发验证数据，不能作为独立测试成绩。

`--benchmark`在本机对基线及各编码器的命中最优设置计时，模板在计时前编码；`latency.json`保存单帧RGB到指定整目标或部位落点的中位与P95耗时。互近邻模式跳过未使用的仿射拟合；该操作不改变互近邻的对应点。

## 模板身份诊断与候选覆盖率

在具备已有RGB-D、模板、同图标签、特征缓存和折外模型的项目根目录运行：

```bash
python src/evaluation/eval_template_selection.py --stage diagnose
python src/evaluation/eval_template_selection.py --stage coverage
python src/evaluation/eval_template_selection.py --stage plot
```

默认配置为`configs/evaluation/template_selection.json`，可用`--config`指定。`diagnose`复算已有开发预测并输出连续分数分组统计，不运行DINO；`coverage`需要CUDA及全图特征，重放折外粗定位器。`candidate_counts`指定K，`nms_iou`指定候选裁剪框抑制阈值，`minimum_crop_coverage`指定目标可见掩码被候选包含的最低比例；峰值命中率另行报告，不以大裁剪框覆盖代替准确定位。

结果位于`runs/evaluation/20261117-template-selection/`，含两个模式的登记、逐对诊断、逐目标候选覆盖、候选清单及图表。统计和覆盖率结果各只写一次，绘图读取已有结果。训练接受分数不属于接受层留出评价，开发验证不属于独立测试。任务部位像素面积不能证明损伤差异在该距离可辨。

237使用`/home/ryh2025/miniconda3/envs/ryh-dinov2/bin/python`。数据与模型通过主项目中的软链接引用已有实验产物，无需复制全量数据；准备检查记录见`runs/validation/20261117-server-ready/report.json`。

## 全目标粗定位监督对照

在237项目根目录使用上述Python环境运行：

```bash
python src/training/train_scene_proposals.py --config configs/training/airport_scene_proposals.json
python src/training/train_scene_proposals.py --config configs/training/airport_scene_proposals.json --plot
```

`evaluation_config`指定数据和候选评价配置，`seeds`指定配对随机种子，`epochs`指定每折轮数；`original`与`all_balanced`分别使用原目标和按实例平衡的全目标标签。训练采用固定末轮，三折来自已有粗定位权重，模型结构及损失函数共用已有实现。

`runs/training/20261117-scene-proposals/`保存登记、监督使用审计、各种子/方法/折的模型与训练曲线、候选和逐目标记录，以及汇总指标和图。`comparison.json`分别报告种子结果、均值/标准差及配对百分点差值；`manifest.json`记录模型哈希。模型仅提供候选，模板身份、接受阈值和连续控制需分别评价。

## 固定候选预算融合

```bash
python src/evaluation/eval_proposal_fusion.py --stage evaluate
python src/evaluation/eval_proposal_fusion.py --stage plot
python src/evaluation/eval_proposal_fusion.py --stage benchmark
```

默认配置为`configs/evaluation/proposal_fusion.json`，`source_config`引用已完成的配对训练，`methods`列出单路及固定融合顺序。候选预算、跨路NMS和标签口径沿用其评价配置；已有缓存每路最多5个候选，结果记录返回不足预算的帧数。缓存评价不运行模型，单路记录以来源文件链接保存。融合顺序不依据原始分数或标签调整。

在本地控制端运行`benchmark`，`benchmark.seed`和`fold`指定模型，`warmup`和`frames`指定预热及测量数量；图像按该折既有记录顺序选择，方法顺序轮转。计时含一次共享DINO编码、一或两个定位头和候选融合；RGB解码及按数据协议擦除在计时前完成。

`runs/evaluation/20261117-proposal-fusion/`保存预先登记、单路复现、各种子候选来源及抑制记录、覆盖指标、图表和`latency.json`。这些是已有训练图的开发评价，不提供独立泛化或在线身份接受结论。
