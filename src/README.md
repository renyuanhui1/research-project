# 机场可视化

`src/mpc/` 提供机场运行记录和离线体素地图的 ROS2/RViz 展示，直接读取 `airbase/outputs/`。

| 入口 | 用途 |
|---|---|
| `mpc/plan_viz.launch.py` | 单机轨迹、观测图像与静态地图 |
| `mpc/multi_viz.launch.py` | 三机记录分命名空间回放 |
| `mpc/voxel_map_viz.launch.py` | 离线体素地图与历史轨迹播放 |
| `mpc/plan_viz_node.py` | 读取 `step_*.npz`，发布轨迹、观测和规划信息 |
| `mpc/voxel_map_viz_node.py` | 读取压缩体素地图及可选运行记录 |
| `mpc/*.rviz` | 对应的 RViz 显示配置 |

在仓库根目录、具备 ROS2 和 RViz 的终端中使用：

```bash
ros2 launch src/mpc/plan_viz.launch.py dump_dir:=/path/to/airport/viz_dump
ros2 launch src/mpc/multi_viz.launch.py run_dir:=/path/to/airport/multi_run
ros2 launch src/mpc/voxel_map_viz.launch.py start_rviz:=true
```

单机默认记录目录为 `airbase/outputs/runs/mppi/run01`；地图默认位于
`airbase/outputs/maps/voxel_probe/`。`dump_dir`、`run_dir` 和 `map_file` 可显式覆盖。
`voxel_map_viz.launch.py` 的 `preview:=true` 显示同步观测；`frustum_image:=true` 显示视锥图像。

机场业务脚本见 [`airbase/src/airsim/`](../airbase/src/airsim/README.md)，模块关系见
[架构说明](../docs/ARCHITECTURE.md)。
