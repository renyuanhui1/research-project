# 运行记录

`approach/<id>/`按每次视觉抵近归档，`waypoint/`保存航点飞行录像。目录内保留本次输入、相机配置、匹配响应、位姿、速度、停车原因、录像帧索引和阶段日志。

`indoor/<id>/`保存机库通道碰撞检查与坐标引导入室记录；`opening-validation/<id>/`保存固定模板的离线DINO诊断、参考统计和结果报告，离线匹配完成不代表视觉抵近或穿越成功。

这两类目录的`rviz/`可保存`manifest.json`、`rgbd-map.npz`及验证记录。回放manifest引用原始照片和诊断图；匹配采集位姿与实际入室轨迹分模式显示，入口为`src/visualization/opening_replay.py`。

完整索引见[实验报告](../reports/EXPERIMENTS.md)。准备与控制必须使用同一空输出目录，命令见[运行说明](../docs/USAGE.md#实时抵近)。
