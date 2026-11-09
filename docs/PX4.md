# PX4接口

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
