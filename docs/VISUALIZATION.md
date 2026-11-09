# RViz地图与匹配显示

ROS2节点使用系统Python。地图与保存记录的回放无需UE运行。

## 显示机场栅格

```bash
cd /home/hui/research-project
source /opt/ros/humble/setup.bash
ros2 launch src/visualization/matching.launch.py profile:=airport
```

默认显示二维占用格、背景碰撞体素及两架飞机表面体素。Fixed Frame为`world`，采用ENU米。白色为空闲、黑色为占用、灰色为未知；飞机颜色表示高度。全场分辨率5米，飞机本体及5米边距范围分辨率0.25米。

## 回放与同步

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

## 匹配热力图的显示与解读

启动参数`heatmap_mode`默认`control`，可改为`similarity`。对保存的原始余弦相似度`s_i`：

| 模式 | 红色强度 | 含义 |
|---|---|---|
| `similarity` | `clip((s_i-0.2)/0.4, 0, 1)` | 原始相似度绝对大小 |
| `control` | `sqrt(softmax(s_i / 0.005))` | 控制中心使用的相对权重；最高响应小于0.2时不叠红色 |

平方根只影响显示强度，不参与控制。绿色十字是实际控制使用的中心，经系数0.3指数滤波，绘制在叠色之后。红区缩小不说明部位定位更准确；两种热力图均不是部位分割掩膜，softmax权重也不是校准后的识别置信度。

原始响应位于`run.h5`的`similarity`和`viz_dump/step_*.npz`的`sim`。回放直接读取记录，不重跑DINO。核对其他试验时需使用对应控制温度。

## 多机记录

```bash
ros2 launch src/visualization/multi_viz.launch.py run_dir:=/absolute/run/directory profile:=airport
```

输入目录须含`Drone1`至`Drone3`各自的`viz_dump/`；机场多机任务需另行准备相应任务配置。
