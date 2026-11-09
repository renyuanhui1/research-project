# 项目结构与运行原理

## 模块

| 路径 | 职责 |
|---|---|
| `src/common/` | 项目路径、场景档案、日志归档、图像解码与分割ID映射 |
| `src/capture/` | 模板与起点准备、传感器采集、HDF5整理 |
| `src/perception/` | 冻结DINO、模板原型、图像全局匹配与深度点云 |
| `src/control/` | 单机抵近控制 |
| `src/debug/` | 传感器、目标对象、匹配与地图诊断 |
| `src/debug/maps/` | UE几何导出、碰撞采样、网格转换及占用栅格生成 |
| `src/evaluation/` | 位姿、包围盒及部位投影的离线评价 |
| `src/visualization/` | ROS2地图发布、匹配图与轨迹回放 |
| `src/px4/` | MAVSDK/PX4接口与航向检查 |

Python入口定位项目根目录，并只将`src/`加入搜索路径；跨模块使用`common`、`perception`、`control`等包导入。资产路径相对项目根目录解析，不依赖启动工作目录。`common/profiles.py`加载`--profile`或`UAV_SCENE_PROFILE`指定的JSON；默认使用`airport`。

## 配置与数据流

```text
configs/scenes/<scene>.json
  ├─ configs/simulation/*.jsonc → ProjectAirSim无人机与传感器
  ├─ configs/missions/*.json → 起点、速度、相机俯角及停车阈值
  ├─ data/templates/ → 模板与特征统计图像
  └─ data/maps/ → 背景栅格、飞机三角面和精细表面体素

src/capture/prepare_scene.py
  → 已知坐标采集模板、几何投影ROI、起点布置
  → runs/approach/<id>/runtime-scene.json + scene-profile.json
  → template.json + validation-frames.npz + controller-setup.json

src/control/run_approach.py
  → 远近景匹配预检
  → RGB → 冻结DINO → 标准化 → 模板相似度/softmax中心
  → 中心滤波 + 深度 → 机体系速度 → ProjectAirSim
  → runs/approach/<id>/原始记录、双路录像、日志和viz_dump

src/visualization/matching.launch.py
  → 地图节点：OccupancyGrid + MarkerArray
  → 回放节点：viz_dump → 位姿、轨迹、匹配画面 → RViz
```

参考点坐标用于采集准备与离线评价，正式抵近的速度不由目标坐标生成。无人机位姿还用于高度保护、轨迹显示与评价。地图提供空间对照，不参与当前速度控制或障碍规避。准备过程需核对UE目标变换与保存地图一致。

## 感知、控制与记录

`Fingerprint`用统计帧计算特征均值、方差，构造整图及ROI原型；不训练编码器。远段使用整机原型，可信深度连续进入45米后切换部位ROI。图像softmax中心经指数滤波后输入控制。

主入口支持斜俯视和俯视。斜俯视100米任务远段水平速度上限3米/秒，近距12米内合速度限制0.35米/秒。停车要求3张独立深度帧满足阈值；保留高度、深度、碰撞、失锁和图像超时保护。翼外侧匹配尚未通过预检，具体结果见[实验索引](../reports/EXPERIMENTS.md)。

前摄与跟随RGB为1920×1080，深度为640×360，采集请求间隔0.0167秒。跟随相机位于机后3米、上方1.5米。深度按`16FC1`米制解码，episode存储保留fp16；无效值记0。不同分辨率测距使用归一化图像坐标。录像按30FPS编码，实际独立源帧率由报告与帧索引给出，含重复帧时不能视为真实30FPS采集。

## 文件与坐标约定

- `data/`保存可复用输入；`debug/`保存诊断输出；`runs/`按试验归档；`reports/`保存汇总分析；`weights/`只保存模型；`third_party/`保存引用源码。
- 每次运行目录保存输入快照、`run.h5`、`poses.json`、`signals.csv`、`report.json`、录像、帧索引与阶段日志。控制入口拒绝覆盖已有结果。
- 飞行接口使用NED米：北、东、下。RViz使用ENU米：东、北、上，转换为`(E,N,-D)`。UE导出位置为厘米，转换时同时处理单位与垂直轴方向。
- 机场运行HDF5含`pose_ned`、`action_body`和`t`；完整姿态保存在同目录`poses.json`。评价器同时识别带`pose`字段的采集格式。
- `data/metadata/targets/*.h5`只含由机场网格推导的有向包围盒属性，用于离线评价。包围盒锚点是几何近似，不能代替真实部位表面距离。

[场景切换](USAGE.md#场景配置与切换)说明哪些配置需要随UE关卡、目标和地图一起准备。

## 文档入口

`docs/ARCHITECTURE.md`说明模块、数据流和坐标约定，`docs/USAGE.md`提供操作命令。`reports/`保存实验记录、验证结果和相关研究调研。
