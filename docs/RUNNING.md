# 实时抵近

在UE中打开机场关卡并保持Play及窗口展开。用`airsim`环境运行仿真与视觉脚本，先检查[场景档案](../configs/scenes/airport.json)的地址、目标与地图。

## 准备与运行

为每次试验选用空目录，下列命令需从项目根目录执行：

```bash
conda activate airsim
cd /home/hui/research-project
session_dir="$PWD/runs/approach/my-session"
mkdir -p "$session_dir"
python src/capture/prepare_scene.py --profile airport --out-dir "$session_dir" > "$session_dir/prepare.log" 2>&1
```

准备脚本会重载无人机传感器、起飞，按已知几何前往35、8、60、100米等位置采集模板及验证帧，最后布置100米起点。默认相机下俯30°、水平速度上限3米/秒、深度停车4米。坐标采集不计为视觉抵近成功。

在准备成功后启动[RViz同步显示](VISUALIZATION.md)，再执行：

```bash
python src/control/run_approach.py --profile airport --out-dir "$session_dir" > "$session_dir/control.log" 2>&1
```

控制脚本先校验远近景模板落点，全部通过后连接仿真执行视觉与深度闭环。失败记录、碰撞保护和其他提前停车均以`report.json`为准。`--out-dir`必须与准备阶段一致；脚本拒绝覆盖已有控制结果。

## 参数与离线检查

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

## 运行文件

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
