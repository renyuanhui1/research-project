# 机场与海港灾损巡查实验记录

本文记录各试验的关键配置、结果和限制。完整逐帧数据、日志、权重与配置快照保留在对应本地实验目录。Git 同步渔港目标表、观察位和关键测量摘要。运行方式见[使用手册](../docs/USAGE.md)，控制与数据流见[架构说明](../docs/ARCHITECTURE.md)，待执行的双场景试验见[研究方案](PLAN.md)。

11 月 16 日的 19 项分阶段验收和 11 月 17 日的 1,920 组 RGB-D 采集均使用 WSL 网关。11 月 18 日的收帧和飞行结果按实际连接地址分别统计，见[当日汇总](#2026-11-18-相机链路与巡检验证)。[分阶段结果](../runs/approach/20261116-coarse-to-fine/summary.json) · [采集报告](../data/captures/airport_targets/20261117-multitarget/report.json)

## 通用口径

- **视觉控制**：冻结DINOv2 ViT-S/14，448输入、32×32特征格、softmax温度0.005，不训练；统计帧只用于计算标准化均值和方差。速度由图像匹配中心与深度生成。
- **坐标准备不计为视觉闭环**：按已知坐标采集模板、布置起点和转场。目标坐标只用于准备和事后评价，无人机位姿另用于高度、范围等独立保护。全部记录都不能证明未知目标坐标下的自主搜索能力。
- **距离分别报告**：停车滤波深度、到参考点、到整机最近表面、到所选部位表面的距离互不替代；深度阈值确认只说明测距满足停车条件。
- **定位指标**：3像素容差与面积归一化距离并列报告。归一化距离为有符号像素距离除以掩码等效半径√(面积/π)，同时给出峰值落在掩码内的比例；3像素容差对大掩码偏宽松。
- **帧率**：录像按30FPS编码并含重复帧，实际独立帧率以`report.json`和`*-frames.csv`为准。
- **热力图**：红色为相似度或softmax相对权重的显示，绿色十字为滤波后的控制中心；两者都不是分割掩膜或校准置信度，见[热力图说明](../docs/USAGE.md#匹配热力图的显示与解读)。
- **成功计数**：部位抵近以`arrival_confirmed=true`、入室以`entered=true`计为完成；预检失败和中断目录保留实际状态，不计为成功。运行目录中各文件的含义见[运行文件](../docs/USAGE.md#运行文件)。
- **起飞接口**：多次准备中`takeoff`（个别为`move`）返回False，每次都由实际爬升与位姿检查确认后继续；原始返回值保留在各`prepare.log`，原因未查明，下文不再逐次重复。
- **收尾状态**：在线试验结束时无人机悬停、客户端断开，未降落或解除电机，另有说明的除外。
- **共同限制**：已有闭环飞行来自静态 UE 机场开发场景。海港记录覆盖场景检查、视口测量与预设机位取图。各试验的重复次数、留出情况及通道验证范围见对应记录。

## 数据位置

| 内容 | 位置 |
|---|---|
| 部位抵近试验 | [runs/approach/](../runs/approach/) |
| 洞口匹配、对准与入室 | [runs/opening-validation/](../runs/opening-validation/)、[runs/opening-alignment/](../runs/opening-alignment/)、[runs/opening-entry/](../runs/opening-entry/)、[runs/indoor/](../runs/indoor/) |
| 离线评价、复验与多目标全场景评估 | [runs/evaluation/](../runs/evaluation/)、[runs/validation/](../runs/validation/) |
| 模型训练与跨骨干迁移实验 | [runs/training/](../runs/training/)、[runs/transfer/](../runs/transfer/) |
| 多机协同调度与端到端仿真 | [runs/coordination/](../runs/coordination/) |
| 沿海渔港场景构建与三类巡检闭环 | [runs/scenes/](../runs/scenes/)、[runs/harbor/](../runs/harbor/) |
| 资产核对、建库、采集索引与 RViz 审阅 | [runs/inspection/](../runs/inspection/)、[runs/catalog/](../runs/catalog/)、[runs/capture/](../runs/capture/)、[runs/visualization/](../runs/visualization/) |
| 采集素材与开发视图 | [data/captures/](../data/captures/) |
| 模板与统计帧 | [data/templates/airport_7890/](../data/templates/airport_7890/)、[data/templates/airport_7888/](../data/templates/airport_7888/)、[data/templates/airport_targets_20261116/](../data/templates/airport_targets_20261116/)、[data/templates/coastal_harbor_pilot/](../data/templates/coastal_harbor_pilot/) |
| 飞机网格、目标库与占用图 | [data/maps/airport_scene/](../data/maps/airport_scene/)、[data/maps/airport_targets_20261116/](../data/maps/airport_targets_20261116/) |
| 相机帧率（含720p、1080p、采集间隔）与网络诊断 | [debug/camera/](../debug/camera/)、[debug/network/](../debug/network/) |
| 航点飞行 | [runs/waypoint/](../runs/waypoint/) |
| 海港目标与观察位 | [海港目标与观察位](../data/metadata/coastal_harbor/) |

包围盒评价读取`data/metadata/targets/`，表面距离使用机场三角面；占用图上的距离不能代替部位的真实距离。`debug/`在`.gitignore`内，不经Git同步。

## 机场单机抵近（StaticMeshActor_7890）

斜俯视起点水平距离100米，相机下俯30°，远段水平速度上限3米/秒，12米内合速度上限0.35米/秒。表中深度为末帧滤波测距，未测的表面距离记为“—”。

| 试验 | 控制周期 | 末帧深度/m | 最近模型表面/m | 结果与记录 |
|---|---:|---:|---:|---|
| 早期俯视机头 | 300 | 11.79 | — | 达到步数上限，未取得3张独立帧停车确认；原始文件已清理 |
| 俯视机头 | 451 | 11.95 | — | 12米深度阈值确认；原始文件已清理 |
| 俯视机头与栅格同步 | 614 | 11.97 | — | 12米深度阈值确认；原始文件已清理 |
| 100米斜俯视机头 | 228 | 11.75 | — | 12米深度阈值确认；原始文件已清理 |
| 机头近距 | 184 | 3.76 | 3.14 | 4米阈值连续独立帧确认；[记录](../runs/approach/close4-1793954736186/README.md) |
| 正前方机翼 | 606 | 8.06 | 3.87 | 碰撞体接触后停滞，由操作中断；[记录](../runs/approach/wing4-1793956203311/README.md) |
| 翼外侧机翼 | — | — | — | 远、近景校验未通过，未执行闭环；[记录](../runs/approach/wing4-outboard-20261106/README.md) |

**2026-11-16清理记录**：按用户要求删除12米停车阈值的四组实验目录：`near-nose_1106`、`near-nose-dual_1106`、`map-session-1793950632689`、`oblique100-1793952243221`，共1,383个文件、1,607,834,612字节。对应的原始数据、模板快照、录像、报告与日志均已删除，仅保留本节历史结果；这些记录不再支持回放或逐帧复核。4米、6米抵近及机库入室数据保留。本次未运行新实验。

- 机头近距：到最终参考点约3.53米，到网格表面约3.14米，见[距离评价](../runs/approach/close4-1793954736186/distance-validation.json)。实际独立帧率约2.82FPS。
- 正前方机翼：到所选机翼区域的表面仍约6.67米，不能用到任意机体表面的3.87米替代；接触信息见[碰撞诊断](../runs/approach/wing4-1793956203311/collision-diagnostics.json)。实际独立帧率约2.84FPS。
- 翼外侧机翼：准星偏出选定区域；校验失败不代表落点一定在整片机翼之外，实际落点见[校验图](../runs/approach/wing4-outboard-20261106/matching-validation.png)。

## 2026-11-10—11：跨机器仿真连接与远程图像流

这一组实验的结论经历过两次修正，按时间顺序列出。

**10日 网络连通**。目的：让178本机WSL与服务器237都能连接178上的UE。Windows新增入站规则，只允许来源237访问TCP 8989/8990/4560、UDP 14540/14588；卡巴斯基设置未修改。初测时237连8989/8990超时，Windows上也没有监听；用临时监听器测路径时与首次Play重叠，UE报`Address in use`，该次仿真启动失败。关闭临时监听器、重新进入Play后，UE（进程35224）在`::`上监听这两个端口，WSL和237均可连接，WSL经`172.21.192.1`也可访问。两端场景档案统一使用`192.168.31.178`。未验证PX4链路；237的默认Python缺少cv2、h5py、numpy、commentjson和projectairsim，未安装。见[TCP日志](../debug/network/20261110-237-to-178/nc.log)、[Windows监听](../debug/network/20261110-237-to-178/windows-listeners.json)、[防火墙规则](../debug/network/20261110-237-to-178/windows-firewall.json)、[放行脚本](../debug/network/20261110-237-to-178/allow-windows-inbound.ps1)、[端口占用日志](../debug/network/20261110-237-to-178/ue-bind-error.log)、[配置检查](../debug/network/20261110-237-to-178/validation.json)。

**10日 远程帧率初测（结论已作废）**。传感器配置不变（前摄1080p RGB与640×360深度、Chase 1080p RGB，采集间隔0.0167秒），无人机未起飞。service往返中位数4.5 ms、p95 7.6 ms；10秒内前摄3.0fps、深度2.8fps、Chase 2.4fps，合计约34.8 MB/s。当时据此认为“网络不是瓶颈”，11日的测量推翻了这一结论，34.8 MB/s属于链路峰值，不代表稳定吞吐。见[测量报告](../debug/camera/fps-test-remote-237/fps-report.json)、[原始输出](../debug/camera/fps-test-remote-237/measurement.log)。

**11日 本机对照**。178的WSL经`172.21.192.1`连接，配置哈希与237一致，未解码、未录像、未飞行，按到达数计帧率，按`GetSimTime`计时钟比。

| 订阅组合 | 本机接收率 | 图像负载MB/s | 仿真时钟/墙钟 |
|---|---:|---:|---:|
| IMU-only | 314.69 Hz | — | 0.953× |
| RGB-only前摄 | 28.57 fps | 177.70 | 0.955× |
| 前摄RGB+Chase RGB | 各24.17 fps | 合计300.75 | 0.952× |
| IMU+前摄RGB | 314.27 Hz / 28.57 fps | 177.73 | 0.955× |

本机订阅图像没有拖慢时钟（约0.95×，并非精确1.00×）。见[说明](../debug/camera/20261111-local-vs-237/README.md)、[结果](../debug/camera/20261111-local-vs-237/results.json)、[逐消息数据](../debug/camera/20261111-local-vs-237/samples.csv)、[复跑脚本](../debug/camera/20261111-local-vs-237/measure_local.py)、[237输入](../debug/camera/20261111-local-vs-237/remote-inputs.json)、[237报告值](../debug/camera/20261111-local-vs-237/remote-reported.json)。

**11日 本机窗口状态**。用Windows API每秒记录窗口状态，各轮的全部采样点都符合设定条件。

| 窗口状态 | RGB-only fps | 双路RGB前摄/Chase fps | 时钟 |
|---|---:|---:|---:|
| 最小化 | 2.80 | 2.86 / 2.86 | 0.951—0.954× |
| 小窗口聚焦 | 28.50 | 23.31 / 23.31 | 0.953× |
| 小窗口可见、不聚焦 | 28.50 | 21.71 / 21.64 | 0.952—0.954× |
| 不聚焦，同时使用其他软件 | 28.84 | 23.98 / 23.98 | 0.963× |

最小化会把本机帧率压到约2.8fps，与历史抵近试验的2.82/2.84FPS一致；可见但不聚焦不会造成同等下降。每种状态只测一轮、按固定顺序执行，其他软件的负载未测。窗口汇总因Windows时间戳有7位小数而首次解析失败，修正后用保留的原始记录补算，测量值未改。见[说明](../debug/camera/20261111-window-state/README.md)与[四轮汇总](../debug/camera/20261111-window-state/comparison.json)，各轮子目录另存逐消息CSV与窗口JSONL。

**11日 远程窗口状态与时钟口径修正**。237按同一脚本测四种窗口状态（由用户在178现场保持，未采集遥测）。远程RGB-only为0.80—1.67fps、4.97—10.36MB/s，与窗口状态没有一致对应关系，比本机同配置低约17倍。16个阶段的`GetSimTime`时钟为0.978—1.002×；而IMU+前摄阶段按消息`time_stamp`计算的比值只有0.118—0.126（最小化组0.578）。所以此前报告的“远程订阅使仿真时钟降到0.11×”测到的是客户端消息积压，不是仿真变慢，该说法已修正。双路RGB在两条流之间的分配不稳定，最小化组前摄15秒内只收到1帧，Chase收到25帧。客户端CPU占用0.02—0.04核。逐阶段表格见Git提交`90d5dae`中的本文件版本。

**11日 纯TCP吞吐**。不经仿真，用1 MiB块发送200 MiB。

| 发送端→接收端 | MB/s |
|---|---:|
| 178 WSL → 237 | 10.11、11.60 |
| 237 → 178 Windows | 15.70（受PowerShell接收循环影响） |
| 178 Windows → 237 | 13.09、16.83 |

178→237的纯TCP吞吐为10.1—16.8MB/s，与远程图像流同一量级，瓶颈是178→237的无线链路。测试开始时178连的是2.4GHz（发送速率172Mbit/s），后来改连5GHz，但切换后的速率以及哪几轮是切换后测的都没有记录，5GHz的提升幅度未量化。WSL发送比Windows直发低约15%。

**决定**：仿真调试固定在178用WSL本地连接，237只用于训练；若以后需要远程观察，按720p或更低分辨率重新评估。远程测量与吞吐的原始目录（`debug/camera/20261111-remote-window-state/`、`debug/network/20261111-throughput/`）在本机不存在，只能在当时的测量机器上查找。

## 2026-11-11：机库俯视采集与坐标引导入室

两项都是坐标引导飞行，不计为视觉抵近。目标机库主体为`StaticMeshActor_2019`，包围盒中心NED约`(-28.28,713.39,-12.27)`米，尺寸约68.70×74.99×24.54米。

**俯视采集**。爬升至100米，沿六段航点转场到包围盒中心上方。首张图严重过曝，弃用。在空中重载相机配置后，无人机未保持高度，下落至约24.28米，得到暗图，该轮无效，是否发生碰撞未确认。随后从出生点重新飞完整航线，在Gamma 1.4与−4/−2/0/+2 EV四档中选用+2 EV。最终位置距包围盒水平中心约0.39米、高度100.97米，机库完整入画，近白像素比例约0.00014%。中心射线穿过屋面开口，中心深度99.13米不能当作到屋面的高度。空中重载失高的原因未查明。见[俯视原图](../data/captures/hangar_2019/20261111-150412-overhead/overhead.png)、[采集记录](../data/captures/hangar_2019/20261111-150412-overhead/capture.json)、[深度](../data/captures/hangar_2019/20261111-150412-overhead/depth.npy)、[运行配置](../data/captures/hangar_2019/20261111-150412-overhead/runtime-scene.json)、[客户端日志](../data/captures/hangar_2019/20261111-150412-overhead/projectairsim_client.log)。

**屋顶开口入室**。用实时`ECC_Pawn`碰撞查询，先以1米分辨率采样，再在候选入口以0.25米细查。在入口`N=-30.527、E=717.385`处，3×3×30米竖直盒的占用数为0。飞行时降至35、28、25、22、18、14米指令高度，28米以下的下降速度为0.4米/秒。首次8米/秒航段停止时距航点约3.25米，超过1.7米阈值，于是改为1米/秒修正、6米/秒转场，未重载或瞬移。最终位置距入口水平约0.004米，高度约14.72米，内部墙面、窗带和钢梁可见，飞行中没有收到新的碰撞事件。通道依据只来自采样范围内的碰撞查询，“没有收到碰撞事件”不能代替连续净空验证。见[报告](../runs/indoor/20261111-160335-hangar2019/report.json)、[内部前向图](../runs/indoor/20261111-160335-hangar2019/interior-14m-forward.png)、[内部俯视图](../runs/indoor/20261111-160335-hangar2019/interior-14m-down.png)、[飞行位姿](../runs/indoor/20261111-160335-hangar2019/flight.json)、[细采样切片](../runs/indoor/20261111-160335-hangar2019/entry-fine-slices.png)、[原始体素](../runs/indoor/20261111-160335-hangar2019/entry-fine-occupancy.npz)、[配置](../runs/indoor/20261111-160335-hangar2019/runtime-scene.json)、[客户端日志](../runs/indoor/20261111-160335-hangar2019/projectairsim_client.log)。

## 2026-11-12：洞口匹配离线开发

固定参考取自11日约101米高的俯视图，裁剪框`[805,350,1230,750]`，标准化统计只由参考图计算。开发图包括新采的9组（100、80、65、55、45米，以及65米处南北东西各偏6米）和11日的35/25/22/14米阶段观测。这些图都是同场景开发数据，不是留出测试集。新采9组RGB-D的时间戳互不相同，RGB与深度的时差小于250毫秒。

**平均原型基线（未达到飞行条件）**。参考图自检时，整帧中心落在洞口内，但单次裁剪细化后中心约`(1202,630)`，偏到洞外屋面；80、65、45米视图同样误匹配到屋面，峰值约0.49—0.59，说明较高的响应不能作为对应正确的证据。见[配置](../configs/missions/hangar_opening_validation.json)、[首轮报告](../runs/opening-validation/20261112-hangar2019/report.json)、[首轮对比](../runs/opening-validation/20261112-hangar2019/matching-overview.png)、[开发数据](../data/captures/hangar_2019/20261112-development-views/report.json)、[14图诊断](../runs/opening-validation/20261112-hangar2019-development/report.json)、[80米误匹配](../runs/opening-validation/20261112-hangar2019-development/center-80m-matching.png)。

**深度候选加空间DINO对应**。先用Otsu把深度分成两层，要求两层中位深度相差至少6米，并去除与画外相连的远层，得到封闭候选；再按逐位置patch余弦均值对候选排序。开发阈值0.4，未校准。9张开发视图都选中了中央洞口，空间分数为0.45—0.75。35米处只剩低分候选（0.179），被拒绝；25/22/14米处深度层分不开，不输出建议点。DINO最高分候选与面积最大候选在9张图中完全一致，所以这批数据不能证明DINO排序比简单面积规则有额外收益。调试中改正了Otsu箱边界问题，以及45米视图候选连到画外地面的问题；对应的5项针对性测试通过。见[几何模块](../src/perception/opening_geometry.py)、[离线入口](../src/evaluation/eval_opening_matching.py)、[测试](../tests/test_opening_geometry.py)、[报告](../runs/opening-validation/20261112-hangar2019-geometry/report.json)、[9图总览](../runs/opening-validation/20261112-hangar2019-geometry/development-geometry-grid.png)、[80米对比](../runs/opening-validation/20261112-hangar2019-geometry/baseline-vs-geometry-80m.png)、[评价日志](../runs/opening-validation/20261112-hangar2019-geometry/evaluation.log)。

**RViz回放**。`matching`模式显示9个离散采集位姿和67,284点背景，不发布轨迹，避免被误认为DINO飞行；`entry`模式回放坐标入室的417条位姿，只用已有的4张照片，并标注持帧时间差。实际订阅检查和退出码检查都通过。首轮入室探针错过了临时发布进程，另有一次ROS关停时的发布异常，修复后复验正常。见[回放脚本](../src/visualization/opening_replay.py)、[RViz配置](../configs/rviz/opening.rviz)、[匹配manifest](../runs/opening-validation/20261112-hangar2019-geometry/rviz/manifest.json)、[匹配验证](../runs/opening-validation/20261112-hangar2019-geometry/rviz/validation.json)、[入室manifest](../runs/indoor/20261111-160335-hangar2019/rviz/manifest.json)、[入室验证](../runs/indoor/20261111-160335-hangar2019/rviz/validation.json)、[验证日志](../runs/indoor/20261111-160335-hangar2019/rviz/validation.log)。

## 2026-11-12：洞口实时定高水平对准

目的：把深度候选和空间DINO对应接入UE实时飞行，只做屋顶上方的水平对准和悬停，不下降。参数：增益0.25/秒，合速度上限0.35米/秒，0.3秒脉冲后悬停取新帧；连续3帧确认获取目标，最终确认需≥5帧、≥2秒、视觉平面误差≤0.5米、水平速度≤0.15米/秒。控制律与保护条件见[架构说明](../docs/ARCHITECTURE.md#文件与坐标约定)。

首次只读探针发现没有Drone1，于是停止，未发指令。之后按坐标准备到约66米高的起点；8帧观察阶段DINO分数约0.63，未发指令；随后执行控制。

结果：186组独立RGB-D、177条速度脉冲，约126.53秒后确认对准。视觉平面误差从7.574米降至0.494米，末段连续6帧、约2.28秒，末帧水平速度约0.0066米/秒；净水平位移约6.855米，高度在66.023—66.028米之间。RGB与深度配对时差均为0，指令前最大帧龄约0.333秒，没有新碰撞事件，也没有触发保护。在线与离线共用的匹配代码提取为`perception/opening_matching.py`后，14图回归结果与之前逐项相同，24项测试通过。

限制：0.494米是近层平面的近似视觉误差，没有独立的定位真值，不是通道中心误差，也不代表三维净空。

见[预检](../runs/opening-alignment/20261112-hangar2019/preflight/report.json)、[坐标准备](../runs/opening-alignment/20261112-hangar2019/preparation/report.json)、[观察](../runs/opening-alignment/20261112-hangar2019/observation/report.json)、[控制报告](../runs/opening-alignment/20261112-hangar2019/control/report.json)、[逐帧记录](../runs/opening-alignment/20261112-hangar2019/control/signals.jsonl)、[原始帧](../runs/opening-alignment/20261112-hangar2019/control/frames/)、[核验](../runs/opening-alignment/20261112-hangar2019/control/validation.json)、[误差与轨迹图](../runs/opening-alignment/20261112-hangar2019/control/alignment-summary.png)、[控制日志](../runs/opening-alignment/20261112-hangar2019/control.log)、[测试日志](../runs/opening-alignment/20261112-hangar2019/regression-tests.log)、[离线回归](../runs/opening-alignment/20261112-hangar2019/regression-validation.json)。回放导出186个位姿和63,597点背景，见[回放核验](../runs/opening-alignment/20261112-hangar2019/control/rviz/validation.json)，启动方法见[回放说明](../docs/USAGE.md#洞口匹配与入室记录回放)。

## 2026-11-12：RGB-D入室交接离线验证

目的：洞缘出画后，不再依赖完整洞口匹配，改由RGB-D相对位姿完成交接，并核验采样通道。使用保存的186帧对准序列，不连接UE；仿真位姿只用于事后评价误差。

- **PnP（失败）**：跟踪了185帧，中位重投影残差≤0.522像素，但最大位置误差约2.206米、末帧约1.221米，米制误差检查未通过。
- **两帧RGB-D三维刚体配准**：初试最大位置误差约0.120米。在50米合成下降中，固定的像素残差保护在22米处被触发（当时位置误差仍是厘米级）；于是增加米制残差检查和视觉关键帧刷新。最终最大误差约0.094米、末帧约0.079米，内点最少858个，用了3个关键帧。
- 通道共24,563个采样点。新增7项测试，总计31项通过。

离线结果本身不构成实时入室。见[PnP失败记录](../runs/opening-entry/20261112-hangar2019/pnp-diagnostic.json)、[PnP日志](../runs/opening-entry/20261112-hangar2019/pnp-diagnostic.log)、[大幅下降初次失败](../runs/opening-entry/20261112-hangar2019/large-descent-failure.log)、[离线验证](../runs/opening-entry/20261112-hangar2019/offline-validation.json)、[离线日志](../runs/opening-entry/20261112-hangar2019/offline-validation.log)、[模块](../src/perception/rgbd_entry.py)、[测试](../tests/test_rgbd_entry.py)。

## 2026-11-12：俯视视觉抵近、穿越洞口与内部悬停

流程：水平对准→RGB-D通道核验→下降→近距交接→穿越→内部悬停采图。参数：入室阶段水平上限0.25米/秒；远段下降上限0.8米/秒，距估计屋面10米内0.35米/秒；0.8秒脉冲后悬停复测；距屋面25米以外保留DINO身份检查；目标点为估计屋面下方8米。通道与保护条件见[架构说明](../docs/ARCHITECTURE.md#文件与坐标约定)。

**中断**：首次观察因没有收到RGB内参而停止，未发指令。把订阅顺序改为先收齐两路内参后，8帧观察完成。

**结果**：控制约263.74秒，292组独立RGB-D，276条指令均返回True。高度从66.02米降到13.37米（总下降约52.645米），完成内部悬停确认，末段连续7帧、约2.15秒。RGB-D估计的下降量为52.631米，与独立高度记录的差值全程最大约0.410米；内点最少761个，共14个关键帧，通道缓存最长约16.04秒。末帧视觉水平误差约0.0063米，这不是独立的定位误差。没有新碰撞事件，也没有保护介入。

**内部采图**：Chase前向图严重过曝，标为无效。随后把FrontCamera临时转为前向补采，再恢复俯视，过程中未发飞行指令；有效内部图的均值约189.92、近白比例约0.059%，墙面、窗带、钢梁和货箱都能辨认。

**限制**：起点由坐标准备，UE高度作为独立保护参与，不能据此声称不依赖外部位姿的部署能力；采样棱柱、初始近盲区和45秒静态缓存的限制仍在。

见[预检](../runs/opening-entry/20261112-hangar2019/preflight.json)、[坐标准备](../runs/opening-entry/20261112-hangar2019/preparation/report.json)、[首次观察停止](../runs/opening-entry/20261112-hangar2019/observation/report.json)、[有效观察](../runs/opening-entry/20261112-hangar2019/observation-ready/report.json)、[控制报告](../runs/opening-entry/20261112-hangar2019/control/report.json)、[逐帧记录](../runs/opening-entry/20261112-hangar2019/control/signals.jsonl)、[核验](../runs/opening-entry/20261112-hangar2019/control/validation.json)、[轨迹与高度图](../runs/opening-entry/20261112-hangar2019/control/entry-summary.png)、[内部图](../runs/opening-entry/20261112-hangar2019/control/interior-observation.png)、[过曝图](../runs/opening-entry/20261112-hangar2019/control/interior-chase-overexposed.png)、[补采记录](../runs/opening-entry/20261112-hangar2019/control/recapture/report.json)、[控制日志](../runs/opening-entry/20261112-hangar2019/control.log)、[测试日志](../runs/opening-entry/20261112-hangar2019/tests.log)。

**回放与特征展示**：回放导出292个位姿和98,870点背景，4倍速播放（[回放核验](../runs/opening-entry/20261112-hangar2019/control/rviz/validation.json)）。展示图从保存的数据离线重算：229帧有候选，其中第0—216帧显示DINO红色响应，第217帧起显示RGB-D跟踪角点；重算的DINO分数与记录值最大相差约1.19×10⁻⁷，位姿差为0。密集的绿色小点是LK跟踪特征，不全是通过检验的内点。见[生成脚本](../src/visualization/render_opening_features.py)、[展示报告](../runs/opening-entry/20261112-hangar2019/control/rviz/feature-view/report.json)、[展示核验](../runs/opening-entry/20261112-hangar2019/control/rviz/feature-view/validation.json)。

## 2026-11-12：斜俯视抵近与入室

目的：从洞口侧上方出发，让前进和下降同时进行，避免正上方起点掩盖水平抵近的难度。参数：相机向下70°，水平入室上限0.25米/秒，横向偏差超过0.35米时暂停下降，准备高度65米，起点水平半径保护60米。斜向通道和交接规则见[架构说明](../docs/ARCHITECTURE.md#文件与坐标约定)。

**离线**：在理想闭环模型中，60°和70°都能进入内部确认状态，加入0.5米横向扰动后各暂停下降3步；这些是合成结果，不是UE飞行结果。60°和65°相机各有11个近处包络点看不到，被未知空间判定拒绝；70°覆盖23,760个采样点，加入侧面横梁时能正确拒绝穿越。因此选定70°，未放宽未知通道判定。

**首轮中断**：起点最初在仓库中心南侧25米，首次观察因候选分差只有0.09973而停止，未发指令。准备阶段再向北平移3米时，临时连接的5秒接收超时短于动作所需时间，客户端断开；改用60秒超时重连并确认悬停，实际向北移动约2.19米。正式控制完成对准和约17.7米下降后，DINO分数降到0.393，在固定的25米交接点之前触发失锁停止，未入室。

**交接修复**：增加一条交接条件：候选上下文距图像边界的最小余量≤8%，且当帧DINO身份、几何一致性、RGB-D位姿和通道检查都通过时，交接给RGB-D。在保存序列上做反事实回放，会在第286帧交接（分数0.434）。8%是开发参数，未经留出验证。39项测试通过。

**完整重试**：375组RGB-D、360条脉冲，用时约331.51秒。第280帧在距估计屋面30.96米处交接（分数0.4333，边界余量7.87%）。视觉交接后的斜向段水平移动19.16米、下降52.45米，净路径倾角69.93°（计划69.91°）。视觉下降量与独立记录最大相差0.237米，内点最少506个，斜向段视觉横向误差最大0.164米（不是独立定位精度）。完成屋顶穿越、内部悬停和前向采图（均值163.13、近白比例0），没有新碰撞事件，也没有保护介入。

**未验证**：更浅的角度、不同起点、退出与损伤评估。

见[首轮中断](../runs/opening-entry/20261112-oblique-hangar2019/control-stopped-dino/report.json)、[中断分析](../runs/opening-entry/20261112-oblique-hangar2019/control-stopped-dino/failure-analysis.json)、[平移超时](../runs/opening-entry/20261112-oblique-hangar2019/reposition/report.json)、[恢复悬停](../runs/opening-entry/20261112-oblique-hangar2019/recovery/report.json)、[交接核验](../runs/opening-entry/20261112-oblique-hangar2019/handoff-validation.json)、[正式报告](../runs/opening-entry/20261112-oblique-hangar2019/control/report.json)、[完整核验](../runs/opening-entry/20261112-oblique-hangar2019/control/validation.json)、[内部图](../runs/opening-entry/20261112-oblique-hangar2019/control/interior-observation.png)、[回放核验](../runs/opening-entry/20261112-oblique-hangar2019/control/rviz/validation.json)、[离线核验](../runs/opening-entry/20261112-oblique-hangar2019/offline-validation.json)、[视野诊断](../runs/opening-entry/20261112-oblique-hangar2019/visibility-diagnostic.json)、[测试日志](../runs/opening-entry/20261112-oblique-hangar2019/tests.log)、[操作说明](../docs/USAGE.md#斜俯视抵近与入室)。

## 飞机A部位抵近（StaticMeshActor_7888）

以下五项共用的设置：WSL经`172.21.192.1`连接，相机下俯35°，远段前向上限1.5米/秒，12米内合速度上限0.35米/秒，6米深度停车并要求3张独立深度帧确认。终点评价在实际相机姿态下采一张新图，从DINO中心发出射线，与飞机三角面求交，判断是否命中所选部位，这项几何只用于事后评价。共同起点的状态准入容差为位置0.35米、航向0.75°、速度0.08米/秒、倾斜1°。表中起点一致性是仿真记录的一致性，不是定位精度。部位选取与两级匹配见[架构说明](../docs/ARCHITECTURE.md#aircraft-a部位抵近)，命令见[使用手册](../docs/USAGE.md#同一aircraft-a起点抵近不同部位)。

### 2026-11-12：按部位分别准备

每个部位单独采集35、8、60米视角的模板与预检帧，从各自的60米起点出发，时限600秒，主翼与垂尾的ROI预检容差为0.02。三个部位都以`range_threshold`完成，没有新碰撞。这三项是分别重载和准备的，不是连续的自主转场。

| 部位 | 控制观测 | 停车滤波深度/m | 到所选部位表面/m | 实际独立帧率/FPS |
|---|---:|---:|---:|---:|
| 机头 | 318 | 5.839 | 5.591 | 12.33 |
| 主翼 | 366 | 5.817 | 4.865（任意表面4.861） | 12.11 |
| 垂尾 | 358 | 5.704 | 5.457 | 14.13 |

RGB与深度最大时差为84—110毫秒，39项测试通过。离线复核命令：`python runs/approach/20261112-aircraft-a-parts/evaluate_part.py nose --offline`（`wing`、`tail`同理）。见[汇总](../runs/approach/20261112-aircraft-a-parts/summary.json)、[三处核验图](../runs/approach/20261112-aircraft-a-parts/summary.png)、[轨迹图](../runs/approach/20261112-aircraft-a-parts/trajectories.png)、[部位选区](../runs/approach/20261112-aircraft-a-parts/part-definitions.json)、[场景核验](../runs/approach/20261112-aircraft-a-parts/preflight.json)、[测试日志](../runs/approach/20261112-aircraft-a-parts/tests.log)；各部位的报告、终点核验与录像在`nose/`、`wing/`、`tail/`子目录中。

### 2026-11-13：侧前方共同起点

目的：在同一起点、同一模板与统计下只改变所选部位，排除“按部位预先移到有利方位”的混杂因素。起点NED`(516.8,144.9,-32.0)`、航向−8.557°，时限180秒。首次准备时采集整机参考图和6帧统计，后两项复用同一份。正式匹配先由整机响应生成裁剪框，再在高清原图裁剪上匹配部位ROI，在线裁剪不读取网格投影或部位坐标。

几何预检显示三个部位都在初始视场内，机头、主翼、垂尾各覆盖约1×1、1.5×1.5、1.2×1.2个特征格。首次离线预检因通用参数校验要求ROI切换距离为正而退出，未连接仿真；共同模式不使用这个参数，修复后通过，原始日志保留。

| 部位 | 控制观测 | 末帧时间/s | 轨迹长度/m | 停车滤波深度/m | 到所选部位表面/m | 实际独立帧率/FPS |
|---|---:|---:|---:|---:|---:|---:|
| 机头 | 185 | 51.05 | 43.43 | 5.782 | 5.767 | 17.53 |
| 近侧主翼 | 179 | 51.57 | 39.32 | 5.720 | 5.099 | 19.52 |
| 一侧垂尾 | 214 | 58.67 | 43.97 | 5.144 | 4.892 | 20.14 |

三次起点最大两两距离为0.003米。三项都以`range_threshold`停止，终点射线命中所选部位，没有新碰撞。用全部578帧重建滤波中心，最大差约3.89×10⁻⁷；43项测试通过。

结论：从共同起点出发，改变部位选择会产生不同的抵近轨迹。参考图与机头预检用的是同一张采集图，不属于独立测试。

见[汇总](../runs/approach/20261113-aircraft-a-common-start/summary.json)、[轨迹图](../runs/approach/20261113-aircraft-a-common-start/summary.png)、[三列DINO视频](../runs/approach/20261113-aircraft-a-common-start/matching-comparison.mp4)、[几何预检](../runs/approach/20261113-aircraft-a-common-start/geometry.json)、[参考ROI](../runs/approach/20261113-aircraft-a-common-start/reference/template-location.png)、[参数退出日志](../runs/approach/20261113-aircraft-a-common-start/nose/preflight-argument-failed.log)、[匹配重建核验](../runs/approach/20261113-aircraft-a-common-start/matching-reconstruction.json)、[测试日志](../runs/approach/20261113-aircraft-a-common-start/tests.log)；各部位的报告与终点核验在子目录中。

### 2026-11-13：正前上方起点

后面三项共用正前上方起点NED`(556.8,178.88,-32.0)`、航向−90°（从机头方向的斜上方看向飞机，不是低空平视），时限180秒。三项的变化依次是：参考图来源、窄ROI的原型方法。

| 试验 | 参考 | 机头 | 近侧主翼 | 一侧垂尾 |
|---|---|---|---|---|
| A 跨视角 | 侧前方参考，逐字节复制 | 完成：151观测，5.783 / 5.419米 | 完成：216观测，5.793 / 4.890米 | 起始落点错误，预检拒绝 |
| B 同视角 | 正前上方新采参考，统计不变 | 完成：132观测，5.773 / 5.321米 | 完成：192观测，5.522 / 3.700米 | 参考原型为0格，预检拒绝 |
| C 窄ROI精修 | 同B | — | — | 完成：193观测，5.739 / 5.342米 |

表中“a / b米”为停车滤波深度和到所选部位表面的距离。完成的各项均以`range_threshold`停止，正式控制中没有新碰撞或其他保护介入。被拒绝的垂尾没有抵近轨迹、终点或录像，但准备阶段的起飞和复位飞行已经执行。

**A 跨视角**。默认的全部位预检中，垂尾落点错误，导致整次被拒绝。为了独立评价每个部位，新增`--preflight-selected-part-only`：三个部位的诊断都保存，但只要求所选部位通过，阈值不变；44项测试通过。在同一张起始图上，机头、主翼、垂尾的峰值分别约为0.649、0.694、0.610，垂尾的响应点落在选区下方，可见单看峰值不能判断匹配是否正确。垂尾投影宽约6.31像素，只占约0.285个特征格。这里的困难与正面薄边成像和跨视角外观变化都相符，两者的影响未分离。实际独立帧率为19.55和18.37FPS。见[汇总](../runs/approach/20261113-aircraft-a-front-start/summary.json)、[轨迹图](../runs/approach/20261113-aircraft-a-front-start/summary.png)、[起始响应](../runs/approach/20261113-aircraft-a-front-start/start-matching.png)、[全部位拒绝日志](../runs/approach/20261113-aircraft-a-front-start/nose/preflight-all-parts-rejected.log)、[垂尾预检](../runs/approach/20261113-aircraft-a-front-start/tail/matching-preflight.json)、[输入来源](../runs/approach/20261113-aircraft-a-front-start/experiment.json)、[测试日志](../runs/approach/20261113-aircraft-a-front-start/tests.log)。

**B 同视角模板**。在起点悬停时采6帧参考，过程中未发飞行指令。整机裁剪`[797,385,1123,726]`；统计文件与A逐字节相同，避免同时改变两个因素。机头、主翼、垂尾的ROI分别选中18、24、0个特征格：垂尾框太窄，没有包含任何格中心，原实现在构造它的原型时会抛异常，连带阻断其他部位。修改为把这种部位记为`no_reference_patches`，单独拒绝，其他部位照常运行；未扩大ROI或降低阈值。45项测试通过，用旧输入回归的结果差为0。主翼到所选表面3.700米、到任意表面3.546米，都小于6米，说明测距阈值不等于机体到所有表面的净距。实际独立帧率只有约2.92FPS，重复帧约90%，原因未定位，所以不能比较模板视角对收敛速度的影响。见[同视角模板](../runs/approach/20261113-aircraft-a-front-matched-template/reference/aircraft-template.png)、[ROI与特征格](../runs/approach/20261113-aircraft-a-front-matched-template/reference-feature-grid.png)、[汇总](../runs/approach/20261113-aircraft-a-front-matched-template/summary.json)、[轨迹图](../runs/approach/20261113-aircraft-a-front-matched-template/summary.png)、[垂尾预检](../runs/approach/20261113-aircraft-a-front-matched-template/tail/matching-preflight.json)、[数值回归](../runs/approach/20261113-aircraft-a-front-matched-template/baseline-regression.json)、[测试日志](../runs/approach/20261113-aircraft-a-front-matched-template/tests.log)。

**C 窄ROI原型与局部配准**。离线比较了五种原型方法，用6张同场景图。其中“补一格”只有在额外加0.02框容差时才全部通过，但这个容差约19像素，远大于约6像素的垂尾宽度，而且没有一张图的粗DINO中心落在原始框内，因此不采用。最终方案是：以DINO候选为中心，在1.5格半径内做原图相关配准。6张图的精修点都落在原始框内；擦除目标的合成图全部被拒绝；8项缩放与旋转变换中7项通过（+5°因峰差0.146被拒）。阈值（相关分数0.85、峰差0.15）是用这批开发图选的，未独立校准。实现为显式选项`--thin-roi-refinement`，只作用于中心选格为空的部位；50项测试通过，不启用时回归差为0。

在线结果：初次客户端话题握手阻塞，被中断，未发指令。正式控制193组观测、62.29秒、轨迹50.01米；配准最低分数0.8621，单次感知最长0.155秒，实际独立帧率约18.06FPS。录像统一起点设在启动后0.2秒，所以第一条记录的时间约为−0.113秒，原时间保留。

**终点核验（冷启动失败）**：重新初始化匹配器时，粗DINO候选落在垂尾下方，配准分数0.796，低于0.85，被拒绝。用第192步的配准状态继续跟踪（`--offline --continue-tracking`）时，分数0.869，射线命中所选垂尾。延续跟踪通过不能代替冷启动重新截获。

见[原型比较](../runs/approach/20261113-aircraft-a-tail-roi-study/pooling-comparison.json)、[配准初验](../runs/approach/20261113-aircraft-a-tail-roi-study/refinement-verification.json)、[精修示意](../runs/approach/20261113-aircraft-a-tail-roi-study/proposal.png)、[汇总](../runs/approach/20261113-aircraft-a-tail-local-refinement/summary.json)、[轨迹图](../runs/approach/20261113-aircraft-a-tail-local-refinement/summary.png)、[控制报告](../runs/approach/20261113-aircraft-a-tail-local-refinement/tail/report.json)、[配准诊断](../runs/approach/20261113-aircraft-a-tail-local-refinement/tail/control-signals.json)、[延续跟踪核验](../runs/approach/20261113-aircraft-a-tail-local-refinement/tail/evaluation/part-surface-validation.json)、[冷启动失败](../runs/approach/20261113-aircraft-a-tail-local-refinement/tail/evaluation/cold-start-validation.json)、[冷启动日志](../runs/approach/20261113-aircraft-a-tail-local-refinement/tail/evaluation-cold-start.log)、[协议检查](../runs/approach/20261113-aircraft-a-tail-local-refinement/protocol-check.json)、[测试日志](../runs/approach/20261113-aircraft-a-tail-local-refinement/tests.log)。

**三项结论**：在正前上方起点，机头和主翼在跨视角和同视角模板下都完成了抵近。垂尾先因跨视角落点错误被拒，再因特征格为空被拒，最后通过局部配准完成了一次连续抵近，但近景冷启动重新截获没有通过。未测试的项：更高的特征分辨率、主动调整视角、低空平视正面、其他飞机。

### 2026-11-13：垂尾冷启动重新截获离线对照

**目的**：分离连续跟踪与冷启动的差异，检验搜索中心、尺度、角度和搜索范围分别对拒绝结果的影响。全程离线，未连接UE或发送飞行指令。

**环境与输入**：Python 3.10.20、PyTorch 2.6.0+cu124、OpenCV 4.13.0、RTX 4060 Ti；Torch两线程、OpenCV一线程。沿用正前上方垂尾试验的冻结DINO、448输入、固定模板与统计，配准分数阈值0.85、峰差阈值0.15。使用本次起始原图、`viz_dump`中全部10张原生全画面无损RGB，以及另采的终点原图，共12张；没有从录像补帧。按保存的滤波深度分组：≥35米1张、12—35米2张、<12米9张。深度分组不是独立距离真值，样本集中于近距且高度相关。

**方法**：每张图清空匹配器状态重新计算DINO候选，固定该候选比较八项搜索设置。位置历史单因素保留冷启动尺度与角度；尺度、角度单因素各只替换对应搜索列表，仍围绕当帧DINO候选搜索。完整历史使用该图之前已接受的配准记录，起始图没有历史，不参与四项历史对照。加密尺度为0.5至12倍的49个对数均匀值；扩大范围为原1.5特征格半径的三倍。每张图另生成投影框擦除图，并在擦除后重新计算DINO候选。几何只用于标注和合成删除，不提供搜索中心或尺度。

| 方法 | 原图配准接受数 | 擦除图误接受数 | 配准中位耗时/ms |
|---|---:|---:|---:|
| 默认冷启动 | 7/12 | 0/12 | 75.3 |
| 仅位置历史 | 11/11 | 0/11 | 75.1 |
| 仅尺度历史 | 6/11 | 0/11 | 44.5 |
| 仅角度历史 | 6/11 | 0/11 | 74.7 |
| 完整历史 | 11/11 | 0/11 | 42.8 |
| 无条件扩大范围 | 10/12 | 0/12 | 179.5 |
| 仅加密尺度 | 7/12 | 0/12 | 337.7 |
| 扩大范围并加密尺度 | 10/12 | 0/12 | 899.3 |
| 默认失败后才扩大范围 | 12/12 | 0/12 | 87.0 |

表中最后一项是在查看前八项结果后提出、随后在同一批数据重新运行的开发方案，不是独立验证。公平比较历史因素时，默认冷启动在相同11张图上为6/11；仅提供位置历史便恢复为11/11。默认冷启动按远/中/近分组为1/1、1/2、5/9；条件扩展为1/1、2/2、9/9。接受结果全部通过对应的部位框或射线诊断，但中间帧诊断存在下述位姿限制。

**原因定位**：五张默认冷启动失败图中，完整历史得到的配准点都在原候选搜索范围之外。独立终点的DINO候选到该配准点的垂直偏移约107.69像素，原垂直搜索半径仅50.625像素。仅提供位置历史时终点分数约0.8700；仅提供尺度或角度历史仍被拒绝。证据支持这批图的主要限制是候选偏移与有限搜索范围，不能据此排除其他数据中的尺度或外观问题。

**扩大范围的副作用**：第188、189步原本通过；无条件扩大范围后，分数仍约0.8504、0.8554，但峰差降到0.1412、0.1494，被原阈值拒绝。因此条件扩展保留原范围已接受的结果，只对失败项扩大一次，不降低阈值。独立终点以0.8700分、0.1932峰差通过，使用真实相机元数据的射线命中所选垂尾；五张失败图都在扩大阶段恢复，其余七张保留原范围结果。条件扩展的最大单张配准耗时347.8毫秒；扩大范围并加密尺度最长984.0毫秒。耗时为逐项单次CPU配准，不含DINO计算，不能当作实飞帧率。

**核验与限制**：52项测试通过；修改前保存的终点冷启动和完整历史输出，与增加离线搜索参数接口后的各字段完全一致；两轮前八项的接受数、几何诊断计数和误接受数相同。飞行入口仍使用默认搜索参数。原图中被拒绝的最优配准点也做了几何诊断，但不计作接受；原终点评价在拒绝后输出的是粗DINO中心，两种射线不可混用。起始图使用保存的投影框，中间10帧由异步机体位姿和固定相机外参生成近似射线，只有另采终点有对应的精确相机元数据。投影框擦除是合成负例，不等于真实遮挡测试。没有独立留出图、不同视角、其他飞机、新飞行或真实失锁重启验证。

**结论与下一步**：形成了可复现的失败归因，以及不依赖历史位置的条件扩展候选方案。下一步应冻结该方案与阈值，补采不同起点和观察方向的原生RGB-D，加入相似部位与真实遮挡负例，先做独立冷启动评价，再决定是否接入在线重新截获。

见[评价入口](../src/evaluation/eval_part_reacquisition.py)、[复现命令](../docs/USAGE.md#部位冷启动离线对照)、[报告与距离分组](../runs/approach/20261113-aircraft-a-tail-reacquisition/report.json)、[逐项数值](../runs/approach/20261113-aircraft-a-tail-reacquisition/records.json)、[汇总表](../runs/approach/20261113-aircraft-a-tail-reacquisition/summary.csv)、[落点对照图](../runs/approach/20261113-aircraft-a-tail-reacquisition/comparison.png)、[合成擦除示例](../runs/approach/20261113-aircraft-a-tail-reacquisition/negative-examples.png)、[首轮筛选](../runs/approach/20261113-aircraft-a-tail-reacquisition/screening.json)、[首轮日志](../runs/approach/20261113-aircraft-a-tail-reacquisition/screening.log)、[完整评价日志](../runs/approach/20261113-aircraft-a-tail-reacquisition/evaluation.log)、[数值与重复核验](../runs/approach/20261113-aircraft-a-tail-reacquisition/validation.json)、[测试日志](../runs/approach/20261113-aircraft-a-tail-reacquisition/tests.log)。

### 2026-11-13：垂尾重新截获的独立视角验证

**目的**：在不参与阈值调整的新起点和新观察方向上评价三种冻结方法，检验上一轮"条件扩展恢复5张失败图"能否在独立视角上复现。

**环境与参数**：同上一轮的UE机场飞机A（`StaticMeshActor_7888`）与WSL `airsim`环境。采集18个条件×3帧共54帧：水平距离45/30/18米、侧向角−12/0/+12°、航向偏移−8/+8°，相机下俯35°、速度2 m/s。评价离线：Python 3.10.20、PyTorch 2.6.0+cu124、OpenCV 4.13.0、RTX 4060 Ti，CUDA。冻结阈值分数0.85、峰差0.15、扩展半径3倍，方法`cold`、`radius_3x`、`cold_then_radius_3x`。采集前把源文件SHA-256、源运行与参考资产写进`protocol.json`，评价入口逐项校验采集完成、协议哈希和冻结源，任一不符即中止。

**方法**：每帧清空匹配器状态，从DINO整机候选冷启动；扩展方法只在冷启动拒绝后按3倍半径重搜。每帧另用投影框填中值生成一张合成负例。坐标仅用于采集和事后射线核验，不提供搜索中心或尺度。

| 方法 | 接受/54 | 接受且命中垂尾 | 合成负例误接受 | 全3帧通过的条件 | 配准中位/最大 ms |
|---|---:|---:|---:|---:|---:|
| cold | 7 | 7 | 0/54 | 1/18 | 69/77 |
| radius_3x | 3 | 3 | 0/54 | 1/18 | 169/194 |
| cold_then_radius_3x | 7 | 7 | 0/54 | 1/18 | 233/286 |

按距离分组（`cold`与组合方法相同）：45米1/18、30米6/18、18米0/18；唯一三帧全通过的条件是`r30_b+00_y+08`。

**拒绝原因**：`cold`为43次`registration_confidence`、4次`no_coarse_candidate`，扩展方法为47/4。43帧被拒但算出分数，分数中位0.782、最高0.918，峰差中位0.082；其中仅2帧分数过线而峰差不足。接受的7帧分数0.851—0.925、峰差0.150—0.188，全部贴着阈值。

**采集核验**：54帧状态均为`completed`；位置最大偏差0.347米、航向最大0.183°、RGB与深度时差0秒；结束时在巡航点悬停、未降落。

**失败与限制**：采集前准备阶段中止四次，均为0样本：一次`Takeoff did not produce altitude`（`landed_state`仍为1，未发飞行指令）；一次深度RPC返回460800个元素而解码预期230400（每像素2字节），相机配置修正后恢复；两次观察位未稳定（同为`r45_b-12_y-08`），放宽容差后通过。这些过程日志随临时目录清理，未单独保留。评价阶段，扩展方案没有增加任何接受（7→7），单独扩大半径反而更少（3），上一轮开发集"扩展恢复5张失败图"未复现；被拒帧的主因是配准置信度（外观与尺度），不是搜索半径；18米全部0接受，近距失效原因未定。仍是同一静态场景、同一飞机；每个条件3帧高度相关，不能当成54个独立样本；投影框擦除是合成负例，不等于真实遮挡；深度取中心邻域值；耗时只含配准、不含DINO，不能当作实飞帧率；未验证在线失锁恢复飞行。

**结论与下一步**：冻结方法在新视角上精度可用（0误接受、7/7命中）但召回低（最好7/54），条件扩展不是有效修复。下一步先定位近距（<20米）下整机候选与模板尺度失配，补真实遮挡与相似部位负例；召回提升前不接入在线重新截获。

见[采集协议](../data/captures/aircraft_a_tail/20261113-independent-views/protocol.json)、[采集报告](../data/captures/aircraft_a_tail/20261113-independent-views/report.json)、[采集日志](../runs/approach/20261113-aircraft-a-tail-independent-validation/capture.log)、[评价入口](../src/evaluation/eval_part_reacquisition.py)、[采集入口](../src/capture/collect_part_views.py)、[采集配置](../configs/missions/aircraft_a_tail_reacquisition_validation.json)、[复现命令](../docs/USAGE.md#固定方案的新视角采集)、[评价报告](../runs/approach/20261113-aircraft-a-tail-independent-validation/report.json)、[逐项数值](../runs/approach/20261113-aircraft-a-tail-independent-validation/records.json)、[汇总表](../runs/approach/20261113-aircraft-a-tail-independent-validation/summary.csv)、[距离分组对照图](../runs/approach/20261113-aircraft-a-tail-independent-validation/comparison.png)、[视角总览](../runs/approach/20261113-aircraft-a-tail-independent-validation/view-overview.png)、[合成擦除示例](../runs/approach/20261113-aircraft-a-tail-independent-validation/negative-examples.png)、[测试日志](../runs/approach/20261113-aircraft-a-tail-independent-validation/tests.log)。

### 2026-11-14：冻结匹配器特征上的学习型接受层

**目的**：冻结匹配器的召回瓶颈在固定接受阈值（分数0.85、峰差0.15），大量配准点其实落在部位内却被拒。用同一批冻结特征训练接受层，检验在保持高精度的前提下能否把召回从约7%提升上去。

**环境与参数**：UE机场飞机A（`StaticMeshActor_7888`）、WSL `airsim`环境采集；特征提取与训练在服务器（RTX PRO 6000 Blackwell 96 GB、PyTorch 2.11.0+cu128、scikit-learn 1.7.2）。采集5000个位姿：水平距离70%取12–60米、30%取60–150米，侧向角±40°、航向偏移±20°、固定−35°相机，每帧保存原生RGB、深度与机头/主翼/垂尾/整机掩码。匹配器与上一轮冻结版本一致（DINOv2 ViT-S/14、448输入、瘦ROI原型、局部NCC细化，阈值分数0.85/峰差0.15）。

**采集核验与预热**：场景加载后相机自动曝光从纯白逐帧收敛，前19帧全白（`std`<5）、第20–29帧均值248→179，第30帧起稳定在164–185。提取时跳过前30帧，其余4970帧全部可用；`report.json`状态`completed`，位置最大偏差与RGB-深度时差均满足采集协议。掩码由目标网格按相机元数据射线投射，抽查近距（19米）与远距（134米）帧均与机体对齐。

**方法**：每帧运行冻结匹配器，记录10个在线可得特征：整机DINO峰值、部位DINO峰值、局部NCC分数、峰差、尺度、角度、是否在搜索半径内、细化点到粗候选的距离、裁剪框宽高。标签为细化点落在垂尾掩码3像素以内（有符号距离），4970帧中正例1919（38.6%）。按（15°侧向扇区×距离分层）分组做5折交叉验证，避免同一视角单元的帧同时进入训练和测试；比较逻辑回归与梯度提升树，阈值取OOF上精度≥95%时的最大召回。

| 方法 | AUROC | 精度≥95%时召回 | 精度≥90%时召回 | 精度≥85%时召回 |
|---|---:|---:|---:|---:|
| 冻结规则 | — | 7.1% | — | — |
| 逻辑回归 | 0.883 | 52.9% | 62.8% | 69.8% |
| 梯度提升树 | 0.872 | 57.7% | 64.2% | 69.2% |

梯度提升树工作点：阈值0.752，精度0.950，召回0.577；冻结规则在同一批4970帧接受137帧、精度1.000、召回0.071。

**分组表现**（工作点阈值下）：12–30米精度0.97/召回0.55，30–60米0.97/0.68，60–100米0.84/0.44，100–150米0.91/0.36；侧向角−40°~−20°与20°~40°接受数分别只有13和111（各约1200帧），精度0.85/0.90。特征重要性（置换AUROC下降）：分数0.238、细化点到粗候选距离0.048、整机峰值0.021、部位峰值0.012，其余≤0.012；`inside_radius`为0。提升主要来自把NCC分数与候选偏移联合校准，而不是引入标签侧信息。

**失败与限制**：仅一个场景、一架飞机、一个部位（垂尾）和一个模板，不能外推到其他部位或其他飞机；标签来自网格掩码，远距垂尾掩码可小到40像素，3像素容差偏严；工作点阈值在同一批OOF分数上选取，报告精度略偏乐观；所有帧来自同一次连续采集，没有独立飞行或新视角复核；接受层尚未接入在线控制器，也未与路径2的局部匹配器联合评价。

**结论与下一步**：学习型接受层把95%精度下的召回从7.1%提到57.7%，说明固定阈值损失了大量正确配准，路径1成立。下一步实现路径2（可学习的局部匹配器），在同一批帧上比较端到端局部定位精度；接受层暂不接入在线链路。

见[特征提取入口](../src/training/extract_part_features.py)、[训练入口](../src/training/train_acceptance.py)、[采集报告](../data/captures/aircraft_a_tail/20261114-training-views/report.json)、[特征与标签](../runs/training/aircraft_a_tail_acceptance/patches-all/features.jsonl)。 本节对照实验的权重和逐帧数据已于2026-11-16清理，汇总结果保留在本节。

### 2026-11-14：学习型局部匹配器（路径2）

**目的**：冻结匹配器的局部细化用固定10尺度×3角度NCC搜索，近距和极端方位下大量配准点偏离部位。用同一批搜索块训练热图回归网络，替换这段局部搜索，并与路径1的接受层用同一标签和分组比较。

**环境与参数**：同上一节的数据与冻结DINO整机粗定位。每帧以整机粗框外扩25%取正方形搜索块（夹到图像内），缩放到256×256；目标为垂尾网格掩码在64×64上的面积下采样并高斯平滑，归一化为空间分布。网络为4级卷积编码器（32/64/128/128）加一级上采样融合，输出64×64单通道；损失为空间softmax的负对数似然；Adam 1e-3、30轮、批32，训练时水平翻转和亮度/对比度抖动。按与路径1相同的（15°侧向扇区×距离分层）5折分组验证。4970帧中4961帧的搜索块含至少20个垂尾像素，9个远距帧未参与。

**方法**：每帧只用冻结DINO粗候选生成搜索块，网络峰值映射回原图后计算到垂尾掩码的有符号距离，≤3像素记为正确。再用网络峰值的热图形状统计量（最大概率、熵、峰邻域3×3质量、最大logit、logit间隔、峰到质心距离）与路径1的NCC特征合并，训练接受层，比较95%和90%精度下能接受多少帧。

| 局部定位 | 3像素内正确 | 中位有符号距离 |
|---|---:|---:|
| 冻结NCC细化 | 38.6%（1919/4970） | +7.6像素 |
| 热图网络 | 88.2%（4374/4961） | −2.0像素 |

网络按距离：12–30米97.4%、30–60米87.1%、60–100米83.3%、100–150米80.5%；按侧向角：−40°~−20° 93.6%、−20°~0° 82.2%、0°~20° 81.9%、20°~40° 94.8%。

**接受层与并列比较**：热图统计量本身区分度很弱（正确/错误的均值：最大概率0.087/0.101，熵3.43/3.24，峰邻域质量0.53/0.60，logit间隔15.65/15.49），单独用作置信度时95%精度下召回接近0。把统计量与NCC特征合并后，逻辑回归AUROC 0.719，95%精度下接受2103帧（覆盖率42.4%，其中1998帧正确）；90%精度下接受4632帧（覆盖率93.4%，4169帧正确）。路径1在同一批帧上95%精度覆盖率为23.4%（1107帧正确）。

按距离的95%精度覆盖率：12–30米路径2 83%（精度0.98）对路径1 19%；30–60米35%对31%；60–100米23%对20%；100–150米16%对12%。提升集中在近距。

**失败与限制**：仍是一个场景、一架飞机、一个部位和一个模板；标签来自网格掩码，远距掩码小，3像素容差偏严；搜索块由冻结粗候选决定，粗定位失败时网络无法补救；训练与接受层共用同一分组划分，接受层的特征来自交叉验证预测，属于常规堆叠，存在轻微乐观偏差；工作点阈值在同一批OOF上选取；未接入在线控制器，也未做独立飞行或新视角复核。

**结论与下一步**：学习型局部匹配器把3像素内定位从38.6%提到88.2%；在95%精度约束下，可信且正确的帧覆盖率从23.4%提到42.4%，90%精度下可达93.4%。路径2优于路径1，且合并两者的接受层是最佳配置。下一步把该局部匹配器与接受层接入离线重新截获评价，在独立视角上复核。

见[裁剪块与特征导出](../src/training/extract_part_features.py)、[局部匹配器训练](../src/training/train_local_matcher.py)、[接受层训练](../src/training/train_acceptance.py)、[裁剪块数据](../runs/training/aircraft_a_tail_acceptance/patches-all/features.jsonl)。 本节对照实验的权重和逐帧数据已于2026-11-16清理，汇总结果保留在本节。

### 2026-11-14：局部匹配器的数据量学习曲线

**目的**：确定换模板或换部位时最少需要采多少帧，以及定位精度和可信工作点分别在哪个数据量饱和。

**环境与参数**：沿用上一节的4961个裁剪块与5折（侧向扇区×距离分层）划分。每折训练集按视角单元配额分层抽到300/1000/2000帧，或保留全部（每折约3970帧）；网络结构与30轮训练不变。每档用同一套OOF预测训练组合接受层（NCC特征+热图统计量），阈值取OOF上精度≥95%和≥90%时的最大召回。

| 每折训练帧数 | 定位3px内正确 | 接受层AUROC | 95%精度覆盖率 | 90%精度覆盖率 |
|---:|---:|---:|---:|---:|
| 300 | 72.6% | 0.647 | <1% | 3.7% |
| 1000 | 82.7% | 0.645 | 7.7% | 24.2% |
| 2000 | 84.2% | 0.606 | <1% | 0.4% |
| 约3970（全量） | 87.6% | 0.706 | 47.7% | 84.3% |

**结果解读**：定位精度随数据量单调上升，1000帧达到全量的94%、2000帧96%，之后收益递减。可信覆盖率不随数据量单调：2000帧低于1000帧，接受层AUROC在0.61—0.65之间波动，全量才到0.706。检查各档排序前5%的精度：300帧0.89、1000帧0.95、2000帧0.88、全量0.96，说明小数据下错误预测无法被稳定排到后面，95%工作点因此无法达到。

**失败与限制**：2000帧的回落约2σ，未做多种子重复，不能排除随机波动；每档只训练一次；全量档与上一节独立运行的定位率相差0.5个百分点（87.6%对88.2%），来自多进程数据加载的随机性；仍未换模板或部位验证，本曲线只说明同一模板下的数据量需求。

**结论与下一步**：换模板/换部位时，定位网络可用1000–2000帧起步；若需要95%精度的可信输出，当前方法仍要接近全量数据，瓶颈在接受层的区分度而不是定位精度。下一步优先改进置信度信号（联合训练置信度头或引入时序一致性），再重测小数据工作点。

见[学习曲线训练入口](../src/training/train_local_matcher.py)、[接受层入口](../src/training/train_acceptance.py)。 本节对照实验的权重和逐帧数据已于2026-11-16清理，汇总结果保留在本节。

### 2026-11-14：部位标签修正与三分支共享模型

**目的**：核实多部位共享模型的标签正确性。此前把同一网络的三个通道当作机头、主翼、垂尾训练，机头通道报出91.1%，但叠加图显示机头掩码与垂尾掩码几乎重合、主翼掩码全为零。目标是查清原因、修正区域定义并用同一批帧重训。

**问题与修正**：`capture/part_masks.py`的部位成员判定中，半径一律以共用尾参考点为圆心。原配置只给机头`radius_m 2.2`、主翼横向从4.5米起，于是机头半径圈住的其实是垂尾附近，主翼的分区则完全不与网格相交。修正后机头为前向10.0—13.6米、横向±2.5米、高度0—3米；主翼为横向2.5—8.5米、前向−4.5—1.5米、高度0.5—3.8米；两者半径提到30米，使半径约束失效、只由分区边界决定成员；垂尾定义不变。机头91.1%的结果作废（无效标签的旧运行已删除，结论不采用）。

**环境与参数**：同一批5000帧（4970帧用于训练）与同样的DINO粗候选搜索块。用保存的相机位姿和目标网格离线重渲染全部掩码（`src/capture/rerender_part_masks.py`，不连仿真），再按已记录的搜索块边界重建64×64目标；机头在4029帧的搜索块内有目标像素，主翼和垂尾全部4970帧有。网络、损失、优化器和5折（15°侧向扇区×距离分层）分组与上一节相同，服务器RTX PRO 6000，约13分钟。

**结果**：每折30轮，合并5折OOF。

| 部位 | 3px内正确 | 中位有符号距离 | 参与评价帧 | 95%精度下召回 |
|---|---:|---:|---:|---:|
| 机头 | 74.3%（2950/3968） | +1.0像素 | 3968 | 0.1% |
| 主翼 | 98.7%（4906/4969） | −18.8像素 | 4969 | 100% |
| 垂尾 | 86.5%（4290/4961） | −2.0像素 | 4961 | 0% |

按距离的3px内正确率：

| 部位 | 12—30米 | 30—60米 | 60—100米 | 100—150米 |
|---|---:|---:|---:|---:|
| 机头 | 78.4% | 74.1% | 75.3% | 72.3% |
| 主翼 | 100% | 99.7% | 98.2% | 94.4% |
| 垂尾 | 97.0% | 84.8% | 80.7% | 79.2% |

**解读**：垂尾标签未变，86.5%与此前87.6%和88.2%在运行间噪声内，说明离线重渲染、目标重建和训练流程一致。共享模型能同时学三个部位，垂尾精度与单部位模型相同。机头74.3%且各距离档都在72—78%，误差集中在3—6像素（p90为5.4像素），不是远距分辨率问题，更像机头锥尖端的系统性偏移。主翼98.7%不能与另两个部位并列：主翼掩码面积大（近距单帧可达数万像素），3像素容差几乎必然落在掩码内，中位距离−18.8像素只说明预测落在主翼内部深处。

**失败与限制**：采集视角以垂尾为中心，近距时机头出画，机头只在3968/4970帧参与评价。网络内部置信度（峰值概率）对机头和垂尾仍无区分度，95%精度下召回接近0，主翼的100%是98.7%基率造成的假象，三者都不能当接受判据；这与上一节“热图统计量单独不可用”的结论一致。三个部位的掩码面积相差一个数量级，3像素容差不是等价的指标，跨部位比较需要面积归一化。

**结论与下一步**：部位标签错误已修正，此前的多部位机头结果无效；共享模型可行，垂尾与单部位持平，机头明显更难。下一步先定义面积归一化的部位定位指标，再沿用组合接受层（NCC特征+热图统计量）重做可信工作点。

见[掩码重渲染入口](../src/capture/rerender_part_masks.py)、[部位配置](../configs/scenes/airport_aircraft_a.json)、[三分支指标](../runs/training/aircraft_a_tail_acceptance/local-matcher-allparts-fixed/metrics.json)、[逐帧OOF](../runs/training/aircraft_a_tail_acceptance/local-matcher-allparts-fixed/oof.jsonl)、[训练日志](../runs/training/aircraft_a_tail_acceptance/local-matcher-allparts-fixed/train.log)。

## 飞机B部位数据与三分支训练（StaticMeshActor_7890）

### 2026-11-15：飞机B采集与共享匹配器

**目的**：在第二架飞机上复现部位定义、准备、采集和三分支共享训练链路，判断前一轮方法是否依赖飞机A的具体外形，并为跨模板泛化提供第二份数据。

**环境与参数**：UE机场飞机B（`StaticMeshActor_7890`），采集在WSL `airsim`环境，训练在服务器（RTX PRO 6000、PyTorch 2.11.0+cu128）。部位区域按网格分析写入`configs/scenes/airport.json`：机头前向10.0—13.9米、横向±2.5米、高度1.0—4.6米、半径3.5米；主翼横向3.5—7.4米、前向−6.2—2.0米、高度2.2—3.1米、半径30米；垂尾横向2.0—3.6米、前向−8.4—−3.4米、高度4.0—6.3米、半径30米。共同参考点取机头（`prepare_scene.py --common-start`），任务为`aircraft_b_approach.json`（斜俯视35度、起点距参考点60米）。采集方案`aircraft_b_training_views.json`：5000个位姿，70%取12—60米、30%取60—150米，侧向角±40度、航向±20度、固定−35度相机、掩码步长2。

**参考过曝与修正**：首次准备的参考图平均亮度242.4，来自场景加载后自动曝光尚未收敛。`prepare_scene.py`加入与采集器相同的等待判据（连续3帧均值<220且标准差>5，上限120秒）后重做准备，参考图均值169.7、标准差36.0；首次目录未用于后续步骤。

**采集核验**：5000帧、18 GB。预热34帧后收敛，帧亮度119.9—220.1，无全白或过暗帧被跳过；机头、主翼、垂尾掩码非空的帧分别为4991、4678、4454。`report.json`状态`completed`，参考图的三个部位ROI叠加检查通过。

**方法**：网络、损失、优化器和分组与飞机A三分支相同。整机DINO粗候选外扩25%取搜索块，缩放到256×256；目标为各部位网格掩码在64×64上的面积下采样、高斯平滑并归一化；4级卷积热图网络，空间softmax负对数似然，Adam 1e-3、30轮、批32，水平翻转与亮度抖动；按（15度侧向扇区×距离分层）5折分组；峰值映射回原图后到部位掩码的有符号距离≤3像素记为正确。

**冻结匹配器特征缺失**：特征提取摘要显示5000帧全部`no_candidate`。本次参考图以机头为中心，三个部位ROI都覆盖整机裁剪中心，`thin_roi_refinement`没有产生局部参考点，冻结匹配器未给出候选，也没有NCC细化特征。因此这批数据只能训练局部匹配器（路径2），路径1的接受层没有可用的特征。

**结果**：14973个部位样本中13748个掩码面积不少于20像素，合并5折OOF。

| 部位 | 3px内正确 | 中位有符号距离 | 距离p90 | 参与帧 | 95%精度下召回 |
|---|---:|---:|---:|---:|---:|
| 机头 | 95.7%（4618/4824） | −12.8像素 | −2.8像素 | 4824 | 100% |
| 主翼 | 95.4%（4359/4570） | −14.0像素 | −3.2像素 | 4570 | 100% |
| 垂尾 | 79.2%（3450/4354） | −1.0像素 | +6.1像素 | 4354 | 0.4% |

按距离的3px内正确率：

| 部位 | 12—30米 | 30—60米 | 60—100米 | 100—150米 |
|---|---:|---:|---:|---:|
| 机头 | 98.7% | 96.3% | 94.1% | 89.8% |
| 主翼 | 97.3% | 96.0% | 93.1% | 91.6% |
| 垂尾 | 85.6% | 75.6% | 81.0% | 76.4% |

**与飞机A对比**：飞机A同口径为机头74.3%、主翼98.7%、垂尾86.5%。机头差距不能归因于机型：飞机B的参考点就是机头，搜索块以机头为中心，且机头掩码是3.9米长的锥体，中位−12.8像素说明预测落在机头内部；飞机A机头在3968帧参与评价且远离参考点。主翼两机都在95%以上，p90仍在掩码内3像素以上，属于大掩码下3像素容差的宽容度问题。垂尾是唯一可直接比较的小掩码部位：飞机B 79.2%低于飞机A 86.5%，p90为+6.1像素，是真实的几像素级偏移；30—60米档75.6%为最低点，与飞机A同距离档84.8%的下降趋势一致。95%精度下机头、主翼召回100%仍是基率高于95%的假象，垂尾0.4%说明网络内部置信度不可用作接受判据，与飞机A结论一致。

**失败与限制**：只有一架飞机B、一个场景、一个模板，且只有路径2指标，接受层无法与飞机A对齐评价。3像素容差在大掩码上不等价，跨部位、跨机型比较仍需面积归一化指标（上一节已提出，尚未实现）。采集为一次连续瞬移采样，没有独立飞行或新视角复核；除参考ROI外没有逐帧人工检查掩码。两机的差异同时混有外形、参考点位置和采集中心的区别，不能单独归因于机型。

**结论与下一步**：部位定义、准备、采集和训练链路在第二架飞机上直接复用，不需要改训练代码；垂尾是最能暴露误差的部位，飞机B上为79.2%。下一步做跨模板评价：用飞机A训练的模型直接推理飞机B搜索块，再与各自训练的模型比较，检验共享模型能否跨机型使用。

见[飞机B部位配置](../configs/scenes/airport.json)、[准备任务](../configs/missions/aircraft_b_approach.json)、[采集方案](../configs/missions/aircraft_b_training_views.json)、[准备入口](../src/capture/prepare_scene.py)、[采集报告](../data/captures/aircraft_b/20261115-training-views/report.json)、[参考目录](../runs/approach/20261115-aircraft-b-training-v2/nose/)、[特征提取摘要](../runs/training/aircraft_b/patches-all/features.jsonl)、[训练指标](../runs/training/aircraft_b/local-matcher/metrics.json)、[逐帧OOF](../runs/training/aircraft_b/local-matcher/oof.jsonl)、[训练日志](../runs/training/aircraft_b/local-matcher/train.log)。

## 2026-11-16：跨机型权重迁移与面积归一化指标

**目的**：检验飞机A三分支局部匹配器不做微调、直接推理飞机B数据时的定位能力，并把3像素指标与面积归一化指标并列，确定阶段3模板条件化模型要超过的基线。

**环境与输入**：本地WSL，RTX 4060 Ti 8 GB，Python来自conda `airsim`（PyTorch 2.6.0+cu124、OpenCV 4.13.0）；全程离线，未连接UE或发送飞行指令。飞机A使用`patches-all`（4970帧）与三分支5折权重，飞机B使用`patches-all`（4991帧）与三分支5折权重。评价入口[src/evaluation/eval_cross_template.py](../src/evaluation/eval_cross_template.py)按训练相同的（15度侧向扇区×距离档）分组重建5折。

**方法**：`--mode self`只用各折留出帧，复现训练时的OOF；`--mode transfer`用飞机A全部5折模型推理飞机B全部帧，报5个模型均值±标准差。标签与训练一致，为峰值到全分辨率部位掩码的有符号距离，过滤搜索块内该部位像素少于20的样本。指标包括3像素正确率、峰值落在掩码内的比例、面积归一化距离（除以掩码等效半径）不超过1个半径的比例。

| 评估 | 机头3px | 主翼3px | 垂尾3px | 机头掩码内 | 主翼掩码内 | 垂尾掩码内 |
|---|---:|---:|---:|---:|---:|---:|
| 飞机A自评 | 74.4% | 98.7% | 86.5% | 35.9% | 97.2% | 65.0% |
| 飞机A→飞机B迁移 | 46.9%±9.3 | 52.2%±12.5 | 9.4%±4.2 | 33.4% | 46.7% | 3.6% |
| 飞机B自评 | 95.7% | 95.4% | 79.2% | 94.2% | 93.9% | 59.0% |

面积归一化距离不超过1个等效半径的比例：飞机A自评为90.8%、99.8%、95.9%（机头、主翼、垂尾），迁移为80.5%、80.8%、28.4%，飞机B自评为97.8%、98.0%、92.2%。

**复现**：两轮自评的pooled结果与[飞机A三分支指标](../runs/training/aircraft_a_tail_acceptance/local-matcher-allparts-fixed/metrics.json)和[飞机B三分支指标](../runs/training/aircraft_b/local-matcher/metrics.json)逐项一致（飞机A为0.7437/0.9873/0.8647，飞机B为0.9573/0.9538/0.7924），确认权重与patches管线正确。

**迁移的分组表现**：飞机A→飞机B的主翼随距离快速下降，12—30米93.3%、30—60米42.9%、60—100米26.1%、100—150米22.7%，近距高值由大掩码基率造成；垂尾各档为8.9%、8.3%、12.4%、12.3%，没有可用档位；机头各档为46.3%、46.8%、44.6%、50.7%。5折波动较大，主翼35.6%—73.7%，机头37.5%—62.0%。

**结论**：固定通道的局部匹配器不能跨机型使用。垂尾只剩9.4%，是小掩码部位最干净的失败证据；机头、主翼的约一半不能当作可用，主翼近距结果被掩码面积抬高。迁移时同时改变飞机外观、部位边界定义和搜索块中心（飞机B以机头为参考中心），掉分不能单独归因于外观。面积归一化结果表明3像素正确里很大一部分只是落在掩码边界附近：飞机A机头3像素正确74.4%，但落在掩码内只有35.9%。

**限制**：每类只有一架飞机、一个场景、固定−35度相机；飞机B数据来自一次连续瞬移采集，没有独立飞行或新视角复核；迁移结果只覆盖定位，没有接受层，因为飞机B的冻结匹配器5000帧全部`no_candidate`。迁移模式的pooled统计每个样本有5个折模型预测，不能当作独立样本，结论以每折均值±标准差为准。

见[评价入口](../src/evaluation/eval_cross_template.py)、[飞机A自评指标](../runs/evaluation/20261116-aircraft-a-self-allparts/metrics.json)、[飞机A自评逐帧](../runs/evaluation/20261116-aircraft-a-self-allparts/oof.jsonl)、[迁移指标](../runs/evaluation/20261116-aircraft-a-to-aircraft-b-transfer/metrics.json)、[迁移逐帧](../runs/evaluation/20261116-aircraft-a-to-aircraft-b-transfer/oof.jsonl)、[飞机B自评指标](../runs/evaluation/20261116-aircraft-b-self-allparts/metrics.json)、[飞机A自评日志](../runs/evaluation/20261116-aircraft-a-self-allparts/run.log)、[迁移日志](../runs/evaluation/20261116-aircraft-a-to-aircraft-b-transfer/run.log)、[飞机B自评日志](../runs/evaluation/20261116-aircraft-b-self-allparts/run.log)。

## 2026-11-16：俯视模板与斜视模板的离线跨视角基线

**目的**：新主线要求用高空俯视图裁出的目标模板驱动抵近，因此先回答两个问题：整机俯视模板能否作为粗定位模板；它与传统斜视模板在相同指标下差多少。全程离线，不连接仿真。

**环境与输入**：本地WSL，RTX 4060 Ti 8 GB，conda `airsim`（PyTorch 2.6.0+cu124、OpenCV 4.13.0）。飞机B取`data/captures/aircraft_b/20261115-training-views`（4970帧，跳过前30帧曝光未收敛帧），飞机A取`data/captures/aircraft_a_tail/20261114-training-views`（同样4970帧）。俯视模板由`src/evaluation/eval_nadir_template.py`按记录的相机位姿投影目标网格顶点、外扩1.15倍裁出（飞机B 319×319，飞机A 322×322）；斜视模板分别用`data/templates/airport_7890/oblique/aircraft-template.png`与`data/templates/airport_7888/tail/aircraft-template.png`。冻结DINOv2 ViT-S/14，输入448，32×32 patch。

**方法**：模板编码后对全部patch取归一化均值得到整机原型。标准化统计量取被评采集自身等间隔32帧的DINO token均值与标准差，两个模板共用，以隔离视角因素；统计不使用任何标签。每帧计算响应图，峰值为最大响应patch的中心，质心为温度0.005的softmax加权中心；命中定义为落点位于整机掩码内。整机原型对朝向敏感，因此每个模板再取0/90/180/270度四个旋转，逐帧按峰分选最高者，记为`best_of_4`；该选择不使用真值。

| 评估 | 模板 | 原方向峰命中 | 4方向峰命中 | 原方向质心命中 | 4方向质心命中 |
|---|---|---:|---:|---:|---:|
| 飞机B | 俯视 | 59.3% | 57.7% | 78.7% | 77.2% |
| 飞机B | 斜视 | 45.5% | 46.3% | 62.0% | 62.2% |
| 飞机A | 俯视 | 26.1% | 48.9% | 55.9% | 65.7% |
| 飞机A | 斜视 | 56.7% | 54.3% | 69.4% | 69.6% |

4方向峰命中按距离档（12–30 / 30–60 / 60–100 / 100–150米）：飞机B俯视85.9 / 58.3 / 39.2 / 27.5%，飞机B斜视64.3 / 50.8 / 30.5 / 19.7%，飞机A俯视69.3 / 52.7 / 31.1 / 21.2%，飞机A斜视69.0 / 62.2 / 36.8 / 23.9%。所选旋转：飞机B俯视r0占84.8%，飞机A俯视r0仅1.0%、r180占47.0%、r270占38.6%。

**复现**：200帧冒烟结果与逐帧诊断一致（飞机A俯视best_of_4峰命中0.48、质心0.665），全量两套各4970帧。指标见[飞机B结果](../runs/evaluation/20261116-nadir-template-aircraft-b/metrics.json)与[飞机A结果](../runs/evaluation/20261116-nadir-template-aircraft-a/metrics.json)，逐帧记录为同目录`records.jsonl`，日志为`run.log`。

**结论**：俯视模板可作为粗定位模板，不弱于斜视模板。飞机B上俯视在峰命中、质心命中两项均高约11–17个百分点；飞机A上经4方向搜索后俯视峰命中低约5个百分点、质心低约4个百分点，而斜视模板来自同一架飞机、同一接近方式，俯视模板没有这种视角便利。整机原型对朝向很敏感：飞机A俯视不做旋转搜索时峰命中只有26.1%，搜索后升至48.9%，说明单朝向评价会误判模板可用性。峰值分数不能跨模板比较：飞机B斜视峰分更高（0.507对0.423）但命中率更低。

**限制**：每类只有一架飞机、一个场景、一个模板，固定35度俯仰相机；指标只判断整机掩码命中，没有部位级定位和接受层。标准化统计来自被评采集本身，属离线可用、在线需改由抵近起始帧估计的做法。4方向搜索只覆盖90度倍数，其他朝向未验证。飞机B采集首帧亮度220（曝光未收敛），已用跳过前30帧排除；稳定帧均值156，与飞机A的163基本一致。

**下一步**：模板条件化学习模型的最低基线是近距（12–30米）峰命中60–86%，并需在60–150米显著优于本基线的20–39%；同时验证单朝向模板不依赖人工选向。

见[评价入口](../src/evaluation/eval_nadir_template.py)、[飞机B俯视模板](../runs/evaluation/20261116-nadir-template-aircraft-b/template-nadir.png)、[飞机A俯视模板](../runs/evaluation/20261116-nadir-template-aircraft-a/template-nadir.png)、[飞机B日志](../runs/evaluation/20261116-nadir-template-aircraft-b/run.log)、[飞机A日志](../runs/evaluation/20261116-nadir-template-aircraft-a/run.log)。


## 2026-11-16：飞机A训练样本合并与重复输出清理

**目的**：精简`runs/training/aircraft_a_tail_acceptance`的样本散文件和重复输出，保持样本、模型及实验结果一致。

**环境与方法**：本地WSL，conda `airsim`，Python 3.10.20、h5py 3.16.0、NumPy 2.2.6、OpenCV 4.13.0、PyTorch 2.6.0+cu124；预测检查使用CUDA，训练更新检查使用CPU。将4970张256×256 RGB图像和14910份64×64 float16标签合并到一个HDF5文件，RGB保存为无损PNG编码，标签用gzip压缩。逐项核对全部图像与标签的形状、类型和数值；核对两份特征表的共有字段、CSV内容及6份分数数组与对应JSONL的一致性，再删除原散文件和重复输出。

**结果**：文件数19933→46，文件系统分配空间约779→646 MiB（减少约133 MiB）。删除19880个样本散文件及10个重复输出；保留较完整的`patches-all/features.jsonl`和摘要，以及各组权重、指标、逐帧预测和日志。保留的43个文件SHA256全部一致。按特征行序等间隔选32帧，五折三部位预测在两种存储下完全相同；固定种子7、取2帧进行一次Adam更新，损失均为8.20649242401123，更新后的全部网络状态逐值一致。真实采集单帧特征导出成功；训练输入（含增强和单部位标签）、标签重建两项回归测试通过，五个入口的`--help`检查通过。

**结论与限制**：合并只改变存储和输出方式，没有重新训练或全量重跑评价，不据此报告精度提升。初次直接压缩RGB数组产生更大的HDF5，已改用无损PNG编码并重新全量校验；过程文件已清理。现有PNG/NPY目录仍可作为训练、评价和标签重建输入。后续特征导出只保存一份JSONL，接受层分数直接保存在逐帧JSONL中。

见[样本数据](../runs/training/aircraft_a_tail_acceptance/patches-all/patches.h5)、[特征表](../runs/training/aircraft_a_tail_acceptance/patches-all/features.jsonl)、[使用命令](../docs/USAGE.md#训练样本与局部匹配器)。

样本合并校验摘要：4970张RGB与14910份标签逐值一致，特征共有字段、CSV和6份分数数组与JSONL一致；32帧×5折×3部位预测一致，种子7的两帧Adam更新后网络状态一致；真实单帧导出、2项回归测试和5个入口帮助检查通过。合并数据SHA256为`cd1368bf92edde853f80a522f4f4cfb871e897ee35586d69232fd9ea0a4f0a25`。独立存储校验文件的要点归入本节，不另保留过程文件。

## 2026-11-16：飞机A训练目录只保留样本与三部位模型

**目的与范围**：按确认清单精简本地`runs/training/aircraft_a_tail_acceptance`，保留当前训练样本与三部位模型；不修改训练代码，不运行训练或评价。

**方法**：删除8个`lc-*`学习曲线目录、`acceptance-local`、`acceptance-tol3`、`local-matcher-v2`，以及已归纳到上一节的两个`storage-validation`文件。三部位训练日志移入模型目录，名称为`train.log`。实验汇总表保留，移除已删除数据的链接，方案中的现有权重清单同步移除单垂尾条目。

**结果**：删除11个实验目录、35个文件；文件数46→11，顶层只剩两个目录。文件分配空间约646.0→609.7 MiB，释放约36.3 MiB。保留的11个文件在清理前后SHA256一致，包括4970帧样本、特征和摘要、三部位5折权重、指标、逐帧预测及训练日志。

**结论与限制**：本地已删除对照实验的权重和逐帧数据，历史指标仍保存在本文对应小节；本次没有重训或复算指标。

见[样本](../runs/training/aircraft_a_tail_acceptance/patches-all/patches.h5)、[三部位结果与权重](../runs/training/aircraft_a_tail_acceptance/local-matcher-allparts-fixed/)、[训练日志](../runs/training/aircraft_a_tail_acceptance/local-matcher-allparts-fixed/train.log)。

## 2026-11-16：飞机B训练样本合并与目录精简

**目的与环境**：精简本地`runs/training/aircraft_b`，沿用飞机A的HDF5存储接口；WSL、conda `airsim`，NumPy 2.2.6、h5py 3.16.0、PyTorch 2.6.0+cu124。未修改训练代码，未重训或全量重跑评价。

**方法**：把4991张PNG和14973份部位标签写入一个HDF5文件，图像保留PNG原始字节，float16标签使用gzip无损压缩。全量核对PNG字节以及图像、标签的类型、形状和数值；读取特征筛选后的样本顺序也保持一致。按特征行序等间隔选32帧，用飞机B五折三部位模型在CUDA上比较两种存储的预测结果。全部通过后删除19964个样本散文件，将特征表移入样本目录、训练日志移入模型目录。

**结果**：文件数19973→10，顶层只剩`patches-all`与`local-matcher`。文件分配空间约729.3→602.1 MiB，减少约127.2 MiB。全部样本逐值一致，32帧×5折×3部位预测完全一致；原有9个特征、权重、指标、逐帧预测及日志文件的SHA256保持不变。特征表完整保留5000行，其中9行原本没有裁剪框，不生成额外样本。数据文件SHA256：`e913a86e8b13be7235eabeb1f0604bf676c9b59aeab026ddb74fe4460d71c020`。

**结论与限制**：仅改变存储与目录布局，未删模型或实验结果，未重新计算总体精度。校验要点直接记录在本节，不保留单独的过程脚本和校验输出文件。

见[合并样本](../runs/training/aircraft_b/patches-all/patches.h5)、[完整特征表](../runs/training/aircraft_b/patches-all/features.jsonl)、[模型与指标](../runs/training/aircraft_b/local-matcher/)、[训练日志](../runs/training/aircraft_b/local-matcher/train.log)、[评价命令](../docs/USAGE.md#跨模板与跨机型评价)。

## 2026-11-16：19个目标的在线核对与观察采图

**目的与方法**：按[目标清单](../docs/抵近目标清单.md)核对UE Airport资产。本地WSL经ProjectAirSim连接，使用1280×960、90度水平视场的无物理观察相机，每个目标采集俯视和斜视各一张，取图前等待曝光稳定。

**结果**：7934个对象中找到全部19个目标，最大原点偏差约0.6毫米；38张图曝光检查通过。机库含主体、门和钢梁等多个Actor，塔台另有雷达；原点近邻不能代替完整组成核对。首次连接因GameMode未启动AirSim服务失败，调整后成功。本次仅为坐标辅助资产检查，图中破口不构成可穿越证明。

见[对象与包围盒](../runs/inspection/20261116-target-assets/report.json)、[采图记录](../runs/inspection/20261116-target-assets/capture.json)、[原始图像](../runs/inspection/20261116-target-assets/images/)、[连接日志](../runs/inspection/20261116-target-assets/projectairsim_client.log)。

## 2026-11-16：试跑清理与终点评价修复

**试跑清理**：此前多数试跑从首帧匹配整主体，未执行远端整目标→近端指定部位，曾有8项主体停车核验通过，但不计入分阶段实验。机库910、209高度保护停止，2019超时，3955中断，其余7项未执行。删除该批约24 GB原始数据，保留[结果摘要](../runs/validation/20261116-endpoint-projection/discarded-trials-summary.json)。

**评价修复**：近景投影框因相机后方点和越界坐标导致绘图中断；改为仅绘制前方有限投影并裁剪图像坐标，射线及表面距离计算不变。离线复算中飞机B仍为碰撞失败，飞机A已通过样本各项数值差为0。另查飞机B物理接触点距可见网格约3.37米，碰撞体与可见几何不一致的原因未确认。

见[离线验证](../runs/validation/20261116-endpoint-projection/annotation-validation.json)、[复算日志](../runs/validation/20261116-endpoint-projection/annotation-replay.log)、[碰撞诊断](../runs/validation/20261116-endpoint-projection/collision-analysis.json)。

## 2026-11-16—17：远端粗定位、近端部位精对准与屋顶入室

178 WSL 经网关 `172.21.192.1` 连接 UE Airport。冻结 DINOv2 用整目标远端定位，进入 45 米后切到部位 ROI。飞机和塔台按 6 米滤波深度停车，机库交接给独立的洞口对准与入室控制。19 个目标经逐目标准备、调整及重跑后通过独立终点验收。

| 类别 | 通过 | 终点 |
|---|---:|---|
| 飞机 | 8/8 | 指定部位命中，末帧滤波深度 5.67–5.92 米 |
| 机库 | 9/9 | 入室并拍照，位于观测屋面下 7.92–8.11 米 |
| 塔台 | 2/2 | 柱体命中，末帧滤波深度 5.31/5.88 米 |

飞机曾出现预检失败和错部位。机库曾因交接未停稳、身份误关联或通道受阻中止。重损塔台有一次因机体距主体仅 1.75 米被排除，换方向完整重飞后通过。该批结果适用于同场景逐目标验收，重复成功率与留出泛化待测。见[逐目标结果](../runs/approach/20261116-coarse-to-fine/summary.json)、[独立审计](../runs/approach/20261116-coarse-to-fine/completed-result-audit.json)、[控制报告示例](../runs/approach/20261116-coarse-to-fine/05-aircraft-b-intact/control/report.json)和[测试日志](../runs/approach/20261116-coarse-to-fine/final-checks.log)。

## 2026-11-17：失败与重复试次数据清理

按用户要求清理 41 个失败、重跑和诊断轮次的大体积数据，共 18.91 GiB。保留关键帧、配置、报告和日志，这些轮次已无法完整回放。最终验收的 28 个控制目录校验一致。见[清理清单](../runs/approach/20261116-coarse-to-fine/failed-data-cleanup.json)。

## 2026-11-17：俯视模板库与目标级数据划分

离线建立 19 个目标、14 组主体网格的俯视模板库，生成整目标和部位掩码。主外观划分为 11/3/5，另有塔台类别留出及同网格机库实例协议。正俯视图遮挡两座塔台柱体，柱体精对准仍需侧视。见[目标库](../data/templates/airport_targets_20261116/catalog.json)、[划分配置](../configs/datasets/airport_targets.json)和[构建日志](../runs/catalog/20261117-airport-targets/build.log)。

## 2026-11-17：多目标斜视RGB-D小样本采集

UE Airport 用无物理观察相机在 35° 俯角、12/35/75/125 米采集飞机、机库和塔台。44 组 RGB-D 时间戳一致，HDF5 回读通过。深度一致性用于遮挡过滤，塔台 8/16 个视图看到指定部位。见[采集报告](../data/captures/airport_targets/20261117-pilot/report.json)、[样本](../data/captures/airport_targets/20261117-pilot/samples.h5)和[采集日志](../runs/capture/20261117-airport-pilot/capture.log)。

## 2026-11-17：多目标扩量与模板条件化整目标模型

178 WSL 经网关采集 19 个目标的 1,920 组斜视 RGB-D，训练/验证/测试为 1408/192/320。相机下俯 35°，覆盖 12–150 米。帧对时差为 0，数据哈希、位姿和标签审计通过。237 用冻结 DINOv2 和交叉注意力训练整目标定位及接受头。

模板条件化模型命中 306/320，无模板对照 310/320，四朝向冻结 DINO 指纹 180/320。错误模板仍命中 269/320，测试无有效接受。模板身份判别仍需独立验证。见[采集报告](../data/captures/airport_targets/20261117-multitarget/report.json)、[数据审计](../runs/capture/20261117-multitarget/data-audit.json)、[完整指标](../runs/training/20261117-template-conditioned/comparison.json)和[训练日志](../runs/training/20261117-template-conditioned/training.log)。

## 2026-11-17：模板身份判别与候选接受层开发

237 离线比较同图模板监督、交叉拟合和候选后核验。单候选 DINO＋RGB 对应核验在 383 个开发验证配对中接受 40 个，38 个正确，召回 20%。两次错模板接受均为 飞机B 不同损伤外观。验证集同时用于开发与校准，独立外观泛化、整图候选选择及闭环待验证。见[开发对照](../runs/training/20261117-template-verification-rgb/comparison.json)、[同图标签](../data/labels/airport_targets/20261117-context/report.json)和[训练日志](../runs/training/20261117-template-verification-rgb/training.log)。

## 2026-11-17：DINOv2与DINOv3冻结模板匹配对照

在 237 复用同一验证集比较冻结骨干及单/四朝向均值原型。DINOv2/448 单朝向整目标命中 53.68%，DINOv3/512 四朝向 49.47%。两组 DINOv3 的总体部位命中下降，可靠接受仍未达标。该结果适用于均值原型匹配。见[对照结果](../runs/evaluation/20261117-dino-backbones/comparison.json)和[评价日志](../runs/evaluation/20261117-dino-backbones/evaluation.log)。

## 2026-11-17：特征归一化、落点与逐块对应消融

在固定开发配对上比较标准化、落点和逐块对应。保留通道标准化时，DINOv2/448 用局部中心的部位命中为 25.00%，DINOv3/512 用互近邻的整目标命中为 60.53%。落点收益尚未转化为可靠身份接受。见[消融结果](../runs/evaluation/20261117-dino-matching-ablation/comparison.json)、[身份诊断](../runs/evaluation/20261117-dino-matching-ablation/identity-diagnostic.json)和[日志](../runs/evaluation/20261117-dino-matching-ablation/evaluation.log)。

## 2026-11-17：模板分数诊断与折外候选覆盖率

在已保存的训练折外和开发验证结果上检查分数重叠与候选缺失。637 张训练图的 K=5 峰值覆盖为原目标 89.80%、同图其他目标 22.97%，后者的裁剪覆盖 51.04%。候选覆盖仍限制了同图指定目标选择。见[分数诊断](../runs/evaluation/20261117-template-selection/diagnosis.json)、[覆盖率](../runs/evaluation/20261117-template-selection/coverage.json)和[覆盖日志](../runs/evaluation/20261117-template-selection/coverage.log)。

## 2026-11-17：全目标粗定位实例平衡监督对照

237 固定数据、三折和训练预算，用三个种子比较只监督原目标与按实例平衡监督。K=5 原目标覆盖由 91.31% 降到 75.67%，同图其他目标由 22.34% 升到 38.23%。收益主要来自拟合组。同种子复跑的权重有数值差异，统计按种子分开保留。见[指标](../runs/training/20261117-scene-proposals/comparison.json)、[复现检查](../runs/training/20261117-scene-proposals/reproducibility-audit.json)和[训练日志](../runs/training/20261117-scene-proposals/training.log)。

## 2026-11-17：固定候选预算的双路融合与共享编码耗时

用已保存候选比较同为 K=5、框 IoU 去重阈值 0.5 的双路融合。原模型优先交替保留 90.06% 原目标和 36.15% 同图其他目标峰值覆盖，选作开发候选协议。本机共享一次 DINO 编码时，候选生成耗时中位/P95 为 24.99/25.98 毫秒，范围到候选框生成。见[融合指标](../runs/evaluation/20261117-proposal-fusion/comparison.json)、[本机计时](../runs/evaluation/20261117-proposal-fusion/latency.json)和[日志](../runs/evaluation/20261117-proposal-fusion/evaluate.log)。

## 2026-11-17：RGB-D俯视重建、免训练匹配与距离口径审计

本机离线比较 1,600 张训练和验证图的斜视/俯视 SIFT、DINO。验证中俯视 SIFT 首候选命中 60/190，原斜视 SIFT 为 25/190，四组验证接受均为 0。任务范围内的既有粗定位首候选命中 186/190。历史 152 个同图留出请求全部在 432–1,070 米，超出主任务 12–150 米，近距同图外观泛化仍待独立评价。首次运行中断后从检查点续完。见[指标](../runs/evaluation/20261117-birdseye-matching/comparison.json)、[距离审计](../runs/evaluation/20261117-birdseye-matching/protocol-audit.json)和[执行日志](../runs/evaluation/20261117-birdseye-matching/execution.log)。

## 2026-11-17：米制匹配、多候选冻结测试与影子运行

本机与 237 离线完成 236 帧米制对齐对照，验证首候选命中 60/190，接受 11/22 正确，未达到替换候选生成器的门槛。固定原模型优先融合 K=5，按折外候选训练 44 维线性接受层，冻结阈值 0.913073。开发验证正确接受 6/190。历史五目标复评正确接受 15/18，召回 15/320，核验重排使首候选命中从 283 降到 259/320。历史测试集已有使用，结论限于复评。见[米制判定](../runs/evaluation/20261117-opus-completion/metric-decision.json)、[冻结清单](../runs/evaluation/20261117-opus-completion/frozen.json)、[复评指标](../runs/evaluation/20261117-opus-completion/test.json)和[复算审计](../runs/evaluation/20261117-opus-completion/audit.json)。

影子飞行在 178 WSL 经宿主地址 `192.168.31.178` 执行，目标 03，100 米水平起点，6 米停车门槛。122 组输入接受 25 组，其中 13 组核验正确、12 组未知。原控制器触发深度停车后，独立终点距参考点约 100 米，稳定到达未通过，偏移根因未定位。影子输出未参与运动。见[影子核验](../runs/approach/20261117-opus-shadow/shadow-evaluation.json)、[控制报告](../runs/approach/20261117-opus-shadow/report.json)和[控制日志](../runs/approach/20261117-opus-shadow/control.log)。

## 2026-11-18 相机链路与巡检验证

实验在 178 的 WSL `airsim` 环境连接 Windows UE。上午部分试次使用宿主局域网地址，随后在 WSL 网关上复验。各条件以单轮为主，帧率均按墙钟时间内收到的独立源帧计算。

| 验证项 | 当日结果 | 证据 |
|---|---|---|
| 单相机链路 | 同一静止前视 1080p 场景，采集间隔 0.0167 秒，WSL 网关 27.71 FPS，局域网 16.05 FPS。网关前视 720p 为 30.12 FPS，局域网单 Chase 720p 为 27.67 FPS。 | [同场景地址对照](../debug/camera/20261118-front-single-resolution/address-recheck/results.json) · [前视 720p](../debug/camera/20261118-front-single-resolution/wsl-720p/results.json) · [Chase 720p](../debug/camera/20261118-chase-single-720p/results.json) |
| 双路录像与同步 | 局域网双路 720p 首轮前视／Chase 为 13.06／13.34 FPS，窗口未最小化且不在前台的复测为 9.26／10.37 FPS。单轮窗口对照不足以解释飞行卡顿。网关双路 1080p 按 5 Hz 请求，35 秒实收三路各 161 帧，全部前视帧能在 50 毫秒内配到深度。 | [双路录制](../debug/camera/20261118-camera-720p/results.json) · [窗口复测](../debug/camera/20261118-camera-720p/windowed-unfocused/results.json) · [网关收帧与配对](../runs/approach/20261118-wsl-gateway-retest/transport/results.json) |
| 到达与停车 | 原控制器在局域网和网关均有通过记录。冻结新模型在局域网两次失锁，终点距部位表面 22.45、13.84 米；网关复测同样失锁，距表面 18.47 米，停稳通过，6 米到达失败。更换链路改善了 Chase 录像连续性，未消除失锁。 | 保留链路实测 [20261118-wsl-gateway-retest/transport/](../runs/approach/20261118-wsl-gateway-retest/transport/)；早期联调目录 `20261118-{aircraft-template-e2e,new-model-e2e,new-model-e2e-repeat}` 及 `attempt-01` 已于 2026-11-26 清理 |
| 同步、身份与交接 | 早期同步试次因 RGB-D 配对超时中断。影子试验中原控制器两轮到达，错模板回放仍有 15 个误确认帧。后续在线影子试次连续 32 个结果身份正确。三轮实际近端接管均命中指定部位并停稳，均先触发高度保护，未达到 6 米。调整后另一次达到停车条件，末帧滤波距离 5.918 米、保持最大漂移 0.087 米，仅验证一次。 | 本地审计 `runs/evaluation/20261118-online-identity-shadow/audit.json`、`20261118-active-part-handoff/` 各轮接管审计及 `20261118-adaptive-height-handoff/flight/adaptive-audit.json`（后两项也位于 `runs/evaluation/`） |

[静止收帧脚本](../runs/approach/20261118-wsl-gateway-retest/probe_transport.py)读取相机、深度和时钟，在指定的新输出目录保存结果。可用归档的[到达时间戳](../runs/approach/20261118-wsl-gateway-retest/transport/sensor-arrivals.csv)、[时钟](../runs/approach/20261118-wsl-gateway-retest/transport/clock.csv)和[运行日志](../runs/approach/20261118-wsl-gateway-retest/transport/projectairsim_client.log)复核本轮采集。

当天尚未验证新模型独立完成远端定位、近端接管及到达。影子结果只评价识别，近端单次通过也不能给出重复成功率。目录 `20261118-airport-handoff-revalidation` 的全场景批次于 11 月 19 日完成，按完成日另计。

## 2026-11-19 Airport 全场景交接复验

通过 178 WSL 网关 `172.21.192.1` 串行运行 19 个目标的斜视巡检。各目标沿用它的历史进场和距离判据。飞机模型和塔台使用原有部位 ROI 交接，机库在确认屋面交接后切换俯视，尝试定高外部水平对准。7889 以基准方向及偏转 30 度各做三轮，同一模板、归一化样本和 ROI。原始匹配图与录制由试验入口保存在各轮目录。

19 项基准复验中，7 项完整通过、4 项止于远端预检、1 项近端命中并停稳但先触发高度保护。7 座机库确认屋面交接，切成俯视后 7/7 的屋面参考点出画，均未发出近端运动指令，退出原因为 5 项匹配失败、2 项高度保护。第 4 项恢复已登记的起点帧归一化后通过。7889 两个方位各通过 1/3，6 轮均停稳。19 项历史证据的报告、终点核验与图像哈希一致。UE Play 已关闭，Windows 无监听，WSL 的 8989 和 8990 均不可达。逐目标日志和失败原因见[结果汇总](../runs/evaluation/20261118-airport-handoff-revalidation/summary.json)、[目标表](../runs/evaluation/20261118-airport-handoff-revalidation/targets.csv)和[关停验证](../runs/evaluation/20261118-airport-handoff-revalidation/shutdown.json)（各目标原始子目录已于 2026-11-26 清理，保留顶层汇总文件）。


## 2026-11-19 新模型远端定位与分阶段接管验证

目的为验证冻结新模型直接提供远段控制点，并在连续确认后交给近端部位 ROI。178 WSL 经网关 `172.21.192.1` 连接 UE，首轮选择完整型目标 `05-aircraft-b-intact` 的机头任务，水平起点 100 米，相机下俯 35 度，45 米切换、6 米连续三张深度确认停车。RGB 双路 1920×1080，深度 640×360，采集间隔 0.0167 秒。采用原冻结权重 `6fe119c4f32056021f0877de4b3554a494235674d2f4de7490a50cddbf0bb629`、阈值 0.9130731477441585 和连续三次确认规则。两阶段接入代码及全部输入另行登记哈希。

首次控制启动止于旧整目标模板预检，近端参考已通过。随后按实际定位来源调整主动分阶段预检，远段由新模型运行时确认放行。此启动未发送抵近指令，日志保存在 `startup-preflight/`。两阶段接入及既有共同起点路径的 9 项回归检查通过。

正式在线共记录 15 次新模型推理。事后 RGB-D 与目标网格核验显示，候选点 15 次均命中请求实例，无未知。模型分数为 0.2803–0.8813，全部低于冻结阈值，连续确认计数始终为零。输入年龄最大 1.686 秒，未触发 2 秒时效限制。远端运动指令为零，未进入近端，最终以失锁保持退出，持续停稳通过，最大漂移 0.0547 米。6 米到达和独立机头终点验收未通过。事后仅按分数改用 0.84 计数，15 次中有 4 次达标，最长连续 2 次；该诊断未执行同框身份排名，不能作为在线规则验收。

结论为本轮候选位置正确，接受分数在当前远距画面中未形成运动放行。新模型驱动的远端运动、近端接管和抵近到达仍待验证。UE Play 在终点取证完成后停止，Windows 已无监听，WSL 8989/8990 均不可达。

同日离线阈值诊断使用原开发验证集的 383 个请求、此次 15 条在线记录和既有同型身份回放。仅按分数筛选时，0.84 在开发集正确接受 15/19，0.82 为 16/20。原生图重新推理与在线分数完全一致；此次序列在 0.84 加同框比较下最长连续 2 次，0.82 下最长 3 次，仅形成一次连续确认。对完整型、折翼侧向、折翼偏转及库外存档回放，0.82 加同框比较未出现错误或未知的三帧确认，但有 3 次错误单帧接受。这些序列已用于开发，0.82 仅列为新一轮验证的候选值。数据见[阈值回放](../runs/evaluation/20261119-new-model-staged-active/threshold-sweep.json)与[同框逐帧判分](../runs/evaluation/20261119-new-model-staged-active/threshold-identity-replay.json)。

见[协议与输入登记](../runs/evaluation/20261119-new-model-staged-active/protocol.json)、[汇总](../runs/evaluation/20261119-new-model-staged-active/summary.json)、[模型及控制逐帧记录](../runs/evaluation/20261119-new-model-staged-active/flight/active-records.jsonl)、[独立逐帧判分](../runs/evaluation/20261119-new-model-staged-active/flight/active-evaluation-records.json)、[控制日志](../runs/evaluation/20261119-new-model-staged-active/control.log)、[原生 RGB-D](../runs/evaluation/20261119-new-model-staged-active/flight/active-inputs.h5)和[关停核验](../runs/evaluation/20261119-new-model-staged-active/shutdown.json)。

同日用 WSL 本地 Python 对 0.82 做单帧历史测试集复评。使用 320 张图的 640 个请求，目标可见与目标缺席各 320 个，图像样本名与选阈值用的开发验证集无交集。按已保存的冻结模型分数分别应用 0.913073 和 0.82，先逐条复现原接受标记。正确接受由 15 增至 40，错误接受由 3 增至 6，目标缺席时的误接受由 2 增至 5。可见正例召回由 4.69% 升至 12.50%。该历史测试集此前已用于模型复评，本轮仅验证单帧分数，三连确认与在线闭环仍待验证。[逐请求输入](../runs/evaluation/20261117-opus-completion/test-records.json) · [复算结果与输入哈希](../runs/evaluation/20261119-new-model-staged-active/threshold-test-reevaluation.json)

## 2026-11-19 塔台固定相机在线阈值验证

WSL 经网关 `172.21.192.1` 连接 UE。非物理观察相机分别朝向 19 号和 18 号塔台，水平距离各 100 米、下俯 35°，RGB 1920×1080、深度 640×360、采集间隔 0.0167 秒。每个视角实时读取 12 张新帧并查询 19 号塔台。以冻结模型分数 ≥0.82、同框严格胜出、连续三次且相邻结果间隔 ≤2 秒判定确认，分别在两段开始时清零计数。

朝向 19 号时目标深度掩膜每帧 3502 像素，所选点命中 0/12，最高分 0.459。朝向 18 号时 19 号仍有 677 像素，所选点命中 0/12，最高分 0.475。因此两段均不能提供目标缺席样本。全程 0 次单帧接受、0 次连续确认。结论限于这两个固定视角的在线识别，未执行飞行或近端交接。观察相机已移除，UE Play 已关闭，双端口核验通过。[试验配置](../runs/evaluation/20261119-tower-live-threshold/protocol.json) · [逐帧记录](../runs/evaluation/20261119-tower-live-threshold/observations.jsonl) · [统计](../runs/evaluation/20261119-tower-live-threshold/summary.json) · [运行日志](../runs/evaluation/20261119-tower-live-threshold/live.log) · [关停记录](../runs/evaluation/20261119-tower-live-threshold/shutdown.json)

## 2026-11-19 塔台新视角在线复验

在已开启的 UE Airport 中用 178 WSL 经网关采集四段各 8 帧的实时 RGB-D。非物理观察相机固定下俯 35°，RGB 为 1920×1080，深度为 640×360，请求模板始终为 19 号塔台。冻结权重、模板和推理代码哈希通过核对。原阈值为 0.913073，候选规则要求分数至少 0.82、同一候选裁剪上高于 18 号模板、连续三帧且帧间结果间隔不超过 2 秒。用深度和场景网格独立核验请求目标可见性及候选点。

| 观察视角 | 19 号可见帧 | 候选命中 | 19 号最高分 |
|---|---:|---:|---:|
| 19 号，55 米、偏转 45° | 8/8 | 4/8，另有 1 帧候选点深度未知 | 0.390 |
| 19 号，95 米、偏转 90° | 8/8 | 8/8 | 0.536 |
| 18 号，65 米、偏转 90° | 0/8 | 0/8 | 0.485 |
| 13 号机库，75 米、基准方向 | 0/8 | 0/8 | 0.315 |

两项缺席对照逐帧确认 19 号塔台完全出画且观察对象可见。32 张源帧时间戳各异，RGB 与深度最大时差 0，段内相邻结果最大间隔 1.946 秒。两种规则均为 0 次单帧接受和 0 次连续确认。95 米视角持续命中，请求实例在该段仍因分数低于门槛而未被确认。16 帧缺席对照无误接受。四段为固定视角，未发飞行指令，远端运动、近端接管与停车待验证。观察相机已移除，复查两个仿真端口均可达。

[试验配置](../runs/evaluation/20261119-tower-live-independent/protocol.json) · [原生 RGB-D](../runs/evaluation/20261119-tower-live-independent/frames.h5) · [逐帧日志](../runs/evaluation/20261119-tower-live-independent/observations.jsonl) · [回读审计](../runs/evaluation/20261119-tower-live-independent/audit.json) · [运行日志](../runs/evaluation/20261119-tower-live-independent/live.log)

## 2026-11-19 塔台双轮闭环对照

在已开启的 UE Airport 中，经 178 WSL 网关对 19 号塔台外部柱体执行两轮飞行。两轮分别重新起飞、采参考图并布置 150 米水平起点，运行时场景和档案哈希一致，实际控制起点相差 0.022 米。相机下俯 35°，远近切换门槛 45 米，连续三帧滤波深度不高于 6 米时停车。结束后保持 15 秒、被动观察 10 秒，再独立采集终点位姿和图像。准备阶段的坐标移动与闭环速度指令分别记录。

| 控制器 | 远段与近段 | 停车与终点评价 |
|---|---|---|
| 冻结模型主动控制 | 15/15 次候选点经深度及网格核验命中请求实例，分数 0.261–0.458，均低于原冻结阈值 0.913073。最大输入年龄 1.235 秒，RGB-D 配对时差均为 0。无连续确认、无非零控制指令、无水平位移，未切入近段。 | 失锁退出，停稳通过，最大漂移 0.091 米。未达到停车条件，独立终点未通过，距指定部位表面 178.55 米。 |
| 原控制器 | 水平移动 147.16 米，在前一帧滤波深度 42.93 米时切换柱体 ROI，按任务配置进一步切换近端模板。闭环运动段约 141.42 秒。 | 连续三帧满足深度停车，最后滤波深度 3.979 米，保持与观察最大漂移 0.090 米。独立终点图射线命中指定部位，机体距部位表面 2.612 米，无碰撞记录，整轮通过。 |

主动轮在该起点的失败发生在远端接受阶段。此次闭环使用原冻结阈值；候选阈值 0.82 的同框规则在前一项固定相机实验中单独测过。原控制器结果为本起点下一次独立通过，主动模型的近端接管和停车仍未验证。试验结束后仿真端口仍可达。

[协议](../runs/approach/20261119-tower-closed-loop/protocol.json) · [双轮审计汇总](../runs/approach/20261119-tower-closed-loop/summary.json) · [主动逐帧判分](../runs/approach/20261119-tower-closed-loop/active/active-audit-records.json) · [主动 RGB-D](../runs/approach/20261119-tower-closed-loop/active/active-inputs.h5) · [主动控制日志](../runs/approach/20261119-tower-closed-loop/active/control.log) · [原控制日志](../runs/approach/20261119-tower-closed-loop/baseline/control.log) · [原控制独立终点](../runs/approach/20261119-tower-closed-loop/baseline/evaluation/part-surface-validation.json) · [原控制追踪录像](../runs/approach/20261119-tower-closed-loop/baseline/chase.mp4)

## 2026-11-19 冻结模型主动闭环全目标覆盖

在同一 UE Airport 中用 178 WSL 网关补测剩余 17 个目标。使用原冻结权重和 0.913073 接受门槛，连续三次接受才放行。沿用原任务配置，名义水平起点 100–245 米，相机下俯 35°。先真实布置起点，再启用冻结模型产生远端控制点。飞机、塔台依原任务检查部位接管、深度停车和独立终点图，机库检查外部屋面交接与停稳后的对准。使用原生 RGB-D、目标网格和逐帧指令事后复核模型落点及运动。

| 补测类别 | 实时推理覆盖 | 至少一次单帧接受 | 远端运动 | 通过近端与终点 |
|---|---:|---:|---:|---:|
| 飞机 | 7/7 | 3/7 | 0/7 | 0/7 |
| 机库 | 9/9 | 5/9 | 2/9 | 0/9 |
| 塔台 | 1/1 | 0/1 | 0/1 | 0/1 |

13 号机库水平推进 55.28 米，发出 138 条非零运动指令，其中 136 条关联候选点命中请求实例、2 条因 RGB-D 时差超过 50 毫秒归为未知。15 号机库发出 1 条非零指令，关联候选点命中，水平位移约 0.13 米。两项均以失锁结束，未到屋面交接。02 号飞机出现一次未命中请求实例的单帧接受，未形成连续确认。其余 15 项补测未放行远端运动。17 项均停稳，无新增碰撞记录。

加上此前 05 号飞机与 19 号塔台的分阶段主动试次，19/19 个目标均进入在线控制并产生模型推理，8/19 有单帧接受，2/19 有远端运动，0/19 进入近段或通过完整任务。这批结果显示远端可出现局部推进，但确认无法持续到交接。结束时仿真仍可连接。

[补测协议](../runs/approach/20261119-active-remaining-targets/protocol.json) · [全目标覆盖](../runs/approach/20261119-active-remaining-targets/coverage.json) · [17 项逐目标结果](../runs/approach/20261119-active-remaining-targets/targets.csv) · [批次日志](../runs/approach/20261119-active-remaining-targets/batch.log) · [13 号控制日志](../runs/approach/20261119-active-remaining-targets/13-hangar-4321/control.log) · [13 号逐帧审计](../runs/approach/20261119-active-remaining-targets/13-hangar-4321/active-audit-records.json)

## 2026-11-19 多视角实例确认离线初测

目的为检验固定观测预算下，按历史区分度选择角度能否改善模板实例确认。使用本机 Python 与已保存的冻结核验分数，复用 UE 场景的机库和塔台外部 RGB-D 记录。每组固定同一个实际可见模板、同一个不同主体网格的同类缺席模板、同一采集对象及距离档，两种请求共用搜索帧。历史负例定义为该模板主体网格的全部目录实例均在扩张视锥外。历史训练、验证、测试分别生成 57、11、16 个互不重复用帧的观测组，按搜索图来源的主体网格组划分。验证组来自 1 个主体网格组，测试组来自 2 个主体网格组，测试图此前已经用于开发复评。

三种策略共用预先确定的首帧，最多使用 3 张图。近角度按与首帧的角度差选图，固定多视角按首帧方位 ±90° 选最近可用图，学习策略使用训练组中正确命中与缺席请求的分数差，按首帧分数档和相对方位差预测备选角度的区分度。选择发生在读取备选图得分之前。三种策略用所选图的平均分判断接受，用分数最高的已选图的候选点判分。另对三张图均须超过原冻结阈值的规则做对照。历史观测组中的独立位姿样本不作为固定机位时间序列。

以 3 图预算和验证错误确认占所有请求不超过 10% 选择阈值，测试各有 16 个可见和 16 个缺席请求。错误接受包括缺席接受和可见请求落点错误。

| 选图策略 | 正确接受／16 | 错误接受／32 | 缺席误接受／16 |
|---|---:|---:|---:|
| 近角度 | 4 | 3 | 2 |
| 固定多视角 | 3 | 4 | 3 |
| 学习选角 | 4 | 4 | 3 |

原阈值 0.913073 下，三种选图的三图全过规则均无确认。即使事后任选某个单帧，16 个测试组中也只有 1 组存在正确且过原阈值的图，该组首帧已满足。32 个测试请求中，学习策略有 14 个选择了与固定多视角不同的视角集合。另回放 4 段真实固定机位各 8 帧数据，原三连规则确认 0 次。将同一历史图组的 81 个可能首帧逐一轮换后，在同一验证错误上限下，近角度、固定多视角、学习选角分别得到 30／23／23 次正确接受、13／17／11 次错误接受。轮换是重复回放，81 个首帧不能当作相互独立的试次。

结论为本轮未验证学习选角能稳定提升正确确认。原核验分数偏低限制了仅改选图的提升空间。由于缺少标注过的参考图不可区分实例对，暂时无法验证视角可恢复与参考信息不足的区分。实验只使用存档数据，未连接仿真。

[协议与输入哈希](../runs/evaluation/20261119-view-selection-pilot/protocol.json) · [观测组](../runs/evaluation/20261119-view-selection-pilot/panels.json) · [逐请求决策](../runs/evaluation/20261119-view-selection-pilot/decisions.json) · [结果汇总](../runs/evaluation/20261119-view-selection-pilot/summary.json) · [首帧敏感性与复算核验](../runs/evaluation/20261119-view-selection-pilot/start-sensitivity.json) · [执行日志](../runs/evaluation/20261119-view-selection-pilot/evaluate.log)

## 2026-11-19 冻结候选的同裁剪模板分差诊断

使用本机 Python 复算多视角初测的存档分数，定位候选覆盖、排序和低分各自造成的拒绝。冻结候选位置和原核验分数，在每个裁剪上对已登记的两个同类模板比较 logit 分差。每组将两个模板分别视作请求，使用相同候选与固定多视角日程。对照为原分数、保留原排序仅以 sigmoid 分差接受、按分差重排并接受。三个策略均在 11 个历史验证图组上按相同错误确认上限校准，再用于 16 个历史测试图组。无新推理和仿真连接，两个参照模板均已知。

81 张测试正例图中，5 张无正确候选，22 张有正确候选但首选错误，53 张已正确首选但未过原阈值，1 张正确首选并过阈值。即使在每张图的正确候选中事后选最高分，也仅 1 张能过原阈值 0.913073。将 logit 分差直接用于重排，首选命中从 54/81 降到 34/81。

验证错误确认占全部请求的上限设为 10%，三图预算结果如下。

| 策略 | 正确接受／16 | 错误接受／32 | 目标缺席误接受／16 |
|---|---:|---:|---:|
| 原分数 | 3 | 4 | 3 |
| 保持原排序，同裁剪分差接受 | 4 | 2 | 0 |
| 同裁剪分差重排 | 2 | 1 | 0 |

保留原排序时，分差对这批缺席请求有诊断价值，但可见请求的错误落点接受由 1 增至 2。分差直接重排不适合当前候选集。本轮对照基于已有的成对模板和重复使用过的图像，验证目标来源单一，仍需在独立、同框相似实例条件下验证相对证据，才能判断是否带来可重复收益。

[协议](../runs/evaluation/20261119-instance-score-diagnosis/protocol.json) · [逐候选诊断](../runs/evaluation/20261119-instance-score-diagnosis/candidate-diagnosis.json) · [逐请求记录](../runs/evaluation/20261119-instance-score-diagnosis/requests.json) · [结果汇总](../runs/evaluation/20261119-instance-score-diagnosis/summary.json) · [运行日志](../runs/evaluation/20261119-instance-score-diagnosis/evaluate.log)

## 2026-11-19 同框交换模板与已知裁剪对照

在本机 WSL 的 `airsim` 环境和 RTX 4060 Ti 上，对历史无擦除外部 RGB-D 做离线复评。用深度与场景网格共同确认同类的两个实例每个至少有 64 个可见深度像素，且相机到两者的水平距离均不超过 150 米。每个配对包含采集原目标。模型推理前固定 196 张图、200 个图像与实例对组合。测试来源 11 张均为 16、17 号机库这一对，验证来源 36 张对应 37 组，训练来源 149 张对应 152 组。没有在本轮图像上拟合阈值，并且保留分区来源。旧冻结清单的飞行控制及影子脚本哈希不符，首轮启动在推理前退出。用推理配置、权重哈希和阈值均相同的 11 月 19 日清单完成评分，输入哈希检查通过。

对每张图只生成一次候选，交换模板分别查询 A、B，需两次落点都命中各自实例才计为换模板成功。原分数与同裁剪分差使用此前确定的单图阈值。另在验证和测试图上直接提供深度掩码的两个外接裁剪，用同一模型对两张模板计算二乘二分数。联合分配仅作事后诊断，以两个模板均在场为前提，比较正确排列与交换排列的得分，或从自动候选中为两张模板分配不同候选以最大化总分。

| 历史测试来源 11 张 | 两次落点均正确 | 两次均正确且接受 | 两次选同一候选 |
|---|---:|---:|---:|
| 自动候选及原分数，原冻结阈值 | 2 | 0 | 8 |
| 自动候选及原分数，先前校准阈值 | 2 | 0 | 8 |
| 保留原排序，同裁剪分差接受 | 2 | 0 | 8 |
| 同裁剪分差重排 | 2 | 0 | 0 |
| 以深度核验的两个实例裁剪为候选 | 1 | 0 | 10 |

自动候选同时覆盖两个实例的有 6/11，其中用每个实例的第一处正确候选做事后比较，4/6 的两个裁剪分别更支持正确模板。验证来源 37 组中，自动候选覆盖两者 28 组、原分数双命中 5 组、分差重排双命中 14 组。训练、验证、测试合计覆盖两者 157/200，原分数换模板双命中 28/200，原冻结阈值下双确认 0/200。测试图中按原分数的先前校准阈值，单次查询有 6 次正确接受、1 次接受了另一实例。保留原排序的分差接受仅有 1 次正确接受、无接受另一实例。直接给真实裁剪并强制联合分配时，测试来源配对正确 7/11，验证来源 36/37。联合分配用于自动候选时，分别为 3/11 和 12/37，两组均无双确认。

结论为本组历史测试图同时暴露了候选覆盖不足和实例核验不足。保留原排序的分差在此前缺席对照中减少误接受，这里没有带来双确认。已知裁剪上的联合分配表明某些图像仍有相对辨识信号。测试只覆盖一对机库且为历史数据，结果限于诊断，无法说明对未见实例对的泛化。

[几何核验清单](../runs/evaluation/20261119-same-frame-template-swap/index.json) · [冻结评分协议](../runs/evaluation/20261119-same-frame-template-swap/protocol.json) · [逐图评分](../runs/evaluation/20261119-same-frame-template-swap/observations.jsonl) · [分区汇总](../runs/evaluation/20261119-same-frame-template-swap/summary.json) · [真实裁剪得分](../runs/evaluation/20261119-same-frame-template-swap/known-crops.json) · [复核与联合分配](../runs/evaluation/20261119-same-frame-template-swap/audit.json) · [推理日志](../runs/evaluation/20261119-same-frame-template-swap/evaluate.log)

## 2026-11-19 机场全目标网格地图与回放审阅

在 ROS 2 环境中，为地图节点扩展 `--target-catalog` 支持，将全场 19 个目标（8 架飞机、9 座机库、2 座塔楼）由局部体素展示升级为三维三角网格展示，并在 RViz 中统一审阅全目标外观及历史抵近轨迹。使用目标目录引用的 ENU 三角网格，自动计算各目标包围盒并剔除重叠的 3,995 个 5 米背景碰撞体素。飞机网格按 0.25 米网格聚类合并顶点以减轻渲染负载，机库与塔楼使用存档三角网格。回放节点依次加载历史通过轮次轨迹、参考图及逐帧相机位姿。

发布 19 个目标共 35 个 Actor，总计 207,292 个显示三角面，背景保留 34,229 个体素。RViz 顺利完成 19 个目标的连续加载回放，消除了此前完好飞机表面体素的块状显示失真。该节点仅用于离线显示与空间对照，不参与飞行控制。

[回放脚本](../runs/visualization/20261119-airport-target-review/review.py) · [启动脚本](../runs/visualization/20261119-airport-target-review/launch.sh) · [RViz 配置](../runs/visualization/20261119-airport-target-review/airport-review.rviz) · [回放日志](../runs/visualization/20261119-airport-target-review/replay.log) · [RViz 日志](../runs/visualization/20261119-airport-target-review/rviz.log)

## 2026-11-19 分级门控离线开发、推理降延与多候选/同网格对照（P0/P1排查/P2/P4）

全程离线（未连接 UE，实测 178 上 `172.21.192.1:8989/8990` 关闭）。本机 178（WSL `airsim`，RTX 4060 Ti 8 GB）完成数据登记、P1 复验日志排查、控制端耗时基准、保序校准、P4.3 同网格对照与最终策略回放；服务器 237（RTX PRO 6000 Blackwell 96 GB、24 核 CPU）完成 P4.1 候选预算评估及 1,101 帧 16 核全模板评分缓存。

**P0 数据边界登记与 P1 复验根因定位**。登记 89 个 11.18—11.19 开发集文件（含 SHA256 与 11/3/5 分区）及 4 个仅作回归的历史测试集文件（0 缺失），并预登记 P5 新方位（±30°、150 m、152 轮）独立测试协议。排查 11.19 全场景复验 7/19 失败原因：7 座机库失败系批次脚本 `run_batch.py` 在距机库中心 45–62 m 的交接点强行 `--camera-down-angle 90` 一步切垂直且漏传 `--oblique --vertical-entry`（默认 `min_altitude_m=45 m`）；06 号飞机机翼参考点高 1.52 m、35° 下 6 m 理论高度 4.96 m 低于其 5.0 m 高度保护；04 号脚本读了未同步 `far_start` 的旧快照；03/10/11 为现场重准备导致的冻结 DINO 远端预检压线抖动；18 号塔台重复 3 轮 6 m 停稳但终点射线漂到塔体上方。

**P2.1 离线诊断与 P2.5 控制端推理降延**。855 帧 11.18—11.19 在线序列已知命中率 **94.83%（734/774）**：80–120 m 命中 86.7%、自运动补偿重投影落框 96.4%、射线角误差中位 0.02°，但分数中位仅 0.6955、单帧接受 3.3%（4/120）；进入 45–80 m 后同一模型命中 100%、分数中位 0.9730、接受率 98.1%（51/52）。178 逐环节计时定位 411.1 ms 主瓶颈为单线程 SIFT（123.8 ms，30%）和 5 候选 × 4 朝向 RANSAC（151.0 ms，37%）。在 `MultiCandidateVerifier.prepare_query` 实现无填充 GPU 前向及线程池重叠 SIFT（保留 `TemplateVerifier.encode` 兼容），178 实测中位耗时 **414.5 ms → 286.3 ms**（p95 333.6 ms），与在线实录分数最大差 `3×10⁻⁷`。

**P2.2—P2.4 + P4.2 分级门控与同类模板库离线回放**。按查询图实际观测主体距离（非缺席目标世界坐标距离）在 `appearance.train`（3,966 条折外请求）上拟合距离分档 PAVA 保序校准器，未参与拟合的 `appearance.validation`（383 条）正确正例均值 LLR `+0.426`、缺席负例均值 `-0.530`。在 237 缓存的 1,101 帧（11.19 全部 19 目标 + 6 条 100 m→6–14 m 飞机长航程，含验证集留出外观 06 号）及 32 帧塔台固定相机观测上评估：45–80 m 中距离同类模板库 Top-1 区分率 **94.7%**（飞机B 四种损伤变体间 **109/114 = 95.6%**，正例 logit 裕量中位 `+1.47`，错同类 P90 `-1.09`），80–120 m 仅 64.1%。

| 门控策略 | 11.19 远段放行（19目标） | 长航程正例确认与近段放行（6条） | 同类错目标长航程误确认（42条） | 跨类错目标误动/误确认（26条） | 固定相机缺席误动/误确认（16帧） |
|---|---:|---:|---:|---:|---:|
| 旧冻结 0.913 三连阈值 | 3/19（在线2/19） | 5/6（留出06号未过） | 4/42（2/42进近段） | 0/26 / 0/26 | 0 / 0 |
| 分级门控 + 单模板 SPRT | 17/19 | 6/6 | 20/42（20/42进近段） | 0/26 / 0/26 | 0 / 0 |
| 分级门控 + SPRT + P4.2 同类相对裕量（≥0 / ≥0.3） | 17/19 | 6/6（含留出06号，裕量+0.542） | 0/42（0进近段） | 0/26 / 0/26 | 0 / 0 |

另查单模板首轮回放中 02 号飞机（10 帧）与 17 号机库（4 帧）的 `point_hit=False` 全部为 `ctr_in_box=True`（15/15 框覆正确目标，仅 64×64 单峰值落在翼尖外 1 格或重损机库屋顶破洞内）。

**P4.1 候选预算与 P4.3 同网格上下文**。200 对同框实例上，K=5 精确复现 157/200（验证 29/37、测试 6/11）；K=8 升至 **170/200**（验证 **32/37**、测试 6/11），K=10/12 在验证与测试集饱和。4 组同网格机库（9 座 × 10 帧 30–90 m RGB-D 俯视重建，随机基线 44.4%）单帧区分：`tight_dino` 51/90（56.7%）、`wide_dino` 57/90（63.3%）、`tight_sift` **71/90（78.9%）**（`12-16` 与 `13-17` 各 90%，`09-14` 75%，`10-11-15` 67%）、`wide_sift` 65/90（72.2%）。

**结论与限制**：分级门控配合中距离同类模板库相对裕量在开发序列上同时解决了远段锁死（3/19→17/19）和同机型不同损伤误确认（4/42→0/42），全量 173 项离线单元测试通过。限制：6 条长航程仅覆盖飞机（缺机库、塔台 150→45 m 连续飞行序列，待 P1.3 影子采集补齐）；两座塔台在 150 m 处分数仅 0.26–0.46，35° 侧视柱体与俯视屋顶模板跨视角鸿沟仍未解决；三孪生同网格机库 `10-11-15` 单帧区分仅 67%，高置信识别仍需多帧或上层位置先验；全部回放基于开发集，尚未联机接入控制或进行 P5 独立新方位实飞验证。

[P0 登记](../runs/evaluation/20261119-dev-registry/registry.json) · [P2.1 序列诊断](../runs/evaluation/20261119-p21-diagnostics/sequence-diagnostics.json) · [P2.1 耗时拆分](../runs/evaluation/20261119-p21-diagnostics/latency-breakdown.json) · [P2.3 校准器](../runs/evaluation/20261119-p2-gating-offline/calibrator-train.json) · [P2.4/P4.2 全量回放](../runs/evaluation/20261119-p2-gating-offline/multitemplate-replay-summary.json) · [P4.1 预算对照](../runs/evaluation/20261119-p41-k-budget/summary.json) · [P4.3 同网格对照](../runs/evaluation/20261119-p43-same-mesh-context/summary.json) · [分级门控模块](../src/perception/graded_gating.py) · [离线回放入口](../src/evaluation/eval_graded_gating.py) · [单元测试](../tests/test_graded_gating.py)

## 2026-11-20 P1.2—P1.3 原控制器修正复验与全场景影子飞行采集（第 1 轮 19×1）

在 178 本机 WSL（`airsim`，RTX 4060 Ti 8 GB）经 `172.21.192.1:8989/8990` 联机 UE，使用复用 11.16 核验模板的批跑脚本执行全 19 目标第 1 轮复验，并全程挂载 `--shadow-verifier`（权重与阈值不变，仅更新 3 个已改源码哈希）同步录制连续飞行 RGB-D 与新模型影子预测供 P3 使用。修正项：04 号机恢复 `far_start` 统计、06 号机 `close_altitude_guard_m` 调为 `4.0 m`、9 座机库改为复用粗接近结束的 35° `runtime-scene.json` 并调用 `run_opening_alignment.py --execute --enter --vertical-entry --alignment-pulse 0.8 --entry-pulse 0.8 --min-roof-range 15.0 --max-duration 1200` 及 `eval_opening_entry.py`。

**执行过程异常与说明**：先在 `06-aircraft-b-folded-wing` 单测联调（`runs/evaluation/20261119-p1-smoke/06-aircraft-b-folded-wing/`），停稳于 `5.26 m`、影子 123 帧零报错，但终态高度 `3.89 m` 处冻结 DINO 射线偏离机翼判 `part_not_verified`。正式批跑早期因平台 29 分钟后台命令超时中断两次（保留已完成的 `01–08`），且首版脚本漏传 `--alignment-pulse 0.8` 导致 `09` 号机库在垂直下降至 `11.04 m` 时触发 `900 s` 超时；补齐脉冲参数并改用 `IsDaemon=true` 常驻运行后从 `09` 重跑至 `19` 顺利完成。挂影子推理使实测 `input_fps` 较 11.18 无影子基线低约 1.5–3 FPS（飞机 10.7–17.0 FPS、机库 10.6–11.0 FPS），仍高于 5 Hz 控制周期且所有影子帧 `skipped_without_fresh_pair=0`。

**第 1 轮开发集单轮结果（14/19 通过，73.7%，Wilson 95% CI 51.2%–88.2%）**：该数字基于在这 19 个目标上排查后修正的配置，且机库验收口径已由 11.19 的屋顶外部对准恢复为完整入室，仅作修正配置后的开发集单轮基线参考，不能直接视为相对 11.19 `7/19` 的泛化提升幅度。
- **飞机（7/8 通过）**：`01, 02, 04, 05, 06, 07, 08` 全部 `range_threshold` 稳停（终距 `5.69–5.90 m`）且独立部位核验 `passed`；`03-aircraft-a-nose-damage` 远端预检失败（`inside=true`，`peak=0.167 < 0.20`；结合 11.18 `0.186`，两次均低于 `0.20`，属可复现的远端预检失败，非随机抖动）。
- **机库（5/9 完整入室通过，7/9 粗交接通过）**：`10, 12, 15, 16, 17` 全部 `entered_and_stable_visual_hover` 且独立入室核验 `passed`；`09, 14` 完成粗交接与影子录制，但在近端斜视对准第 377/320 步触发 `opening_or_aperture_jump` 安全悬停；`11, 13` 远端预检失败（峰值稳定落在画面下缘 `cy=0.612 / 0.930`，超出目标框下界 `inside=false`）。
- **塔台（2/2 通过）**：`18-tower-7889`（终距 `5.67 m`）、`19-tower-7892`（终距 `5.76 m`）均完整通过。
- **在线并发负载下的影子推理实测耗时（未达 ≤0.3 s 目标）**：与 178 单跑基准（286.3 ms）不同，真实飞行并发负载（UE + 5 Hz 控制 DINO + 双路 1080p 录像 + `torch.set_num_threads(2)` / `cv2.setNumThreads(1)`）下，16 目标影子推理中位耗时为飞机 **0.50–0.72 s**（p95 `0.70–0.82 s`，实际采样间隔 `0.81–1.01 s`）、机库/塔台 **0.66–0.83 s**（p95 `0.77–1.04 s`，采样间隔 `1.01–1.21 s`）。旧单帧冻结阈值 `0.913` 在机库/塔台连续飞行上几乎零确认（16 目标中机库仅 `15` 号确认 3 帧，塔台 `18/19` 均为 0 帧）。
- **P3 连续飞行影子采集与选择偏差**：16/19 目标完成录制，合计 **1,743 帧**零报错 RGB-D 与预测（飞机 794 帧、机库 680 帧、塔台 269 帧）；按 `appearance` 划分为训练目标 10/11（1,052 帧，缺 11）、验证目标 3/3（365 帧）、测试目标 3/5（326 帧，缺 3/13，封存至 P5）。注意该数据存在明显选择偏差：旧控制器远端最难的 `03/11/13` 未起飞而缺帧，且每目标仅 1 条轨迹、帧间高度自相关，有效独立样本远少于帧数。
- **未完成项与关停状态**：本次完成了第 1 轮 `19×1`；收尾用 `SendKeys('{ESC}')` 自动关停返回 `PLAY_PORTS_STILL_LISTENING`（`closed=false`），由人工停止 Play；已将 Win32 `PostMessage(VK_ESCAPE)` 关停逻辑整合进 `run_p1_batch.py` 的 `finally` 块，但该实现尚未在运行中的 UE Play 上实测验证（仅在已关闭状态下返回 `ALREADY_STOPPED`）。

[批跑脚本](../runs/evaluation/20261119-p1-revalidation/run_p1_batch.py) · [逐目标索引](../runs/evaluation/20261119-p1-revalidation/index.json) · [批跑日志](../runs/evaluation/20261119-p1-revalidation/batch.log) · [影子清单](../runs/evaluation/20261119-p1-shadow/frozen.json) · [关停记录](../runs/evaluation/20261119-p1-revalidation/shutdown.json) · [06 号单测目录](../runs/evaluation/20261119-p1-smoke/06-aircraft-b-folded-wing/report.json)

## 2026-11-20 推理二次降延、P3 连续飞行校准与 P2.6 分级门控离线集成

全程离线在 178 本机 WSL（`airsim`，RTX 4060 Ti 8 GB）完成（UE 未启动；服务器 237 当时不可达 `No route to host`）。P1.3 实飞采得的 3 个 `appearance.test` 目标（`08, 17, 19`，共 326 帧）及缺失目标 `03, 13` 严格封存并登记 SHA256，未参与任何评分、校准或分析。

**GPU 互近邻特征归约与单候选同类评分**：针对 P1.3 联机实测 0.50–0.83 s 瓶颈（将 20–32 个 `1024×1024` 矩阵拷回 CPU 并在 1–2 线程上执行 `np.partition`/`argmax`），在 `src/perception/template_verification.py` 新增 `_correspondence_features_torch`，将 `clamp`、`topk(2)`、`max` 与互近邻筛选全部留在 CUDA 上完成，仅回传互近邻坐标对供 `cv2.estimateAffine2D`；同时在 `src/perception/multicandidate_verification.py` 新增 `score_candidate` 供 P4.2 同类裕量只评唯一已选候选。在 `12-hangar-3955` 10 帧实飞图上校验，与原在线实录分数最大差 `5.76e-08`，与 CPU 路径最大差 `5.96e-08`。178 实测延迟如下（并发负载模拟含 5 Hz 控制 DINO + 双路编解码 + `torch.set_num_threads(2)`）：

| 条件 | 优化前中位 | 优化后中位 | 优化后 p95 | 加速比 |
|---|---:|---:|---:|---:|
| K=5 单跑 | 473.4 ms | **146.7 ms** | 165.6 ms | 3.2× |
| K=5 并发模拟 | 580.4 ms | **157.8 ms** | 199.5 ms | 3.7× |
| K=8 单跑 | 823.2 ms | **191.6 ms** | 220.9 ms | 4.3× |
| K=8 并发模拟 | — | **214.1 ms** | 231.4 ms | — |

**P3 连续飞行分布漂移与仅训练集保序校准**：在 178 上对全部 13 个非测试目标（1,417 帧：10 个训练目标 1,052 帧 + 3 个验证目标 365 帧）× 19 模板完成评分缓存。训练目标连续飞行相对静态瞬移折外（`teleport_oof`）存在显著类别分化：
- **飞机**：连续飞行得分高于静态瞬移（`45–80 m` 中位 `0.753→0.940`，命中 `96.3%`；`80–120 m` `0.589→0.859`，命中 `98.0%`；`0–45 m` `0.797→0.897`，命中 `100%`）。
- **机库**：`120+ m` 中位 `0.814`（命中 `94.7%`）→ `80–120 m` `0.640`（`85.4%`）→ `45–80 m` 降至 **`0.459`**（`64.4%`，因建筑过大溢出裁剪框）。因此机库必须在远距至 `160 m` 起累积确认 LLR，且 P4.2 同类裕量须排除 `appearance_group` 相同的同 mesh 副本。
- **塔台（`18-tower-7889`）**：全距离中位 `0.308–0.448`，`acc_0913=0%`，`120 m` 内命中 `75%–89%`（`120+ m` 为 `26.1%`）。

仅在 `appearance.train`（3,966 条静态折外 + 1,052 帧连续飞行）上拟合混合保序校准器 `calibrator-blended-train.json`，并在 365 帧未参与拟合的 `appearance.validation`（`06, 12, 16`）与 1,052 帧 `appearance.train` 上回放部署配置 `p26_blended_h160`（机库确认上限 `160 m` + 异 mesh 同类 peer）：

| 分区 / 策略 | `pos_perm` | `pos_conf` | `pos_near` | `hit_on_perm` | `diff_mesh_ws_conf` | `same_mesh_ws_conf` | `cx_move` | `cx_conf` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `val` / `legacy_0913` | 1/3 | 1/3 | 1/3 | 100.0% (1/1) | 0/21 | 0/2 | 0/31 | 0/31 |
| `val` / `blended_cal_p42_m03`（机库 ≤80 m） | 3/3 | 1/3 | 1/3 | 99.43% (175/176) | 0/21 | 0/2 | 0/31 | 0/31 |
| **`val` / `p26_blended_h160`** | **3/3** | **3/3** | **2/3** | **98.91% (182/184)** | **0/21** | **0/2** | **0/31** | **0/31** |
| `train` / `legacy_0913` | 6/10 | 6/10 | 5/10 | 100.0% (198/198) | 2/62 | 0/6 | 0/112 | 0/112 |
| **`train` / `p26_blended_h160`** | **10/10** | **9/10** | **5/10** | **98.87% (526/532)** | **0/62** | **2/6** | **2/112** | **0/112** |

注：`val` 上 `16-hangar-7920` 已确认（`mid_llr=0.751`），`pos_near` 为 `2/3` 系其粗接近影子轨迹在屋顶交接前未跨越 `<=45 m` 候选深度；`train` 上 5 架飞机与 4 座机库全部确认，仅 `18-tower-7889` 因跨视角鸿沟未确认（`final_llr=-1.08`）。

**P2.6 在线分级门控集成与离线烟测**：在 `src/perception/active_verification.py` 与 `src/control/run_approach.py` 接入可选的 `graded_gating` 模式（默认关闭，兼容原三连确认），按配置加载校准器与异 mesh 同类 peer 库，执行阶段限速、确认后方准近端/屋顶交接、身份拒绝停机。全量单元测试 **176/176 通过**（含新增 3 项分级门控用例）。生成冻结清单 `runs/evaluation/20261120-p26-graded-active/frozen.json` 并在 178 上加载真实清单烟测实飞帧：`06-aircraft-b-folded-wing`（7 peer）中位 `152.8 ms` 且进入 `MID_CONFIRMED`；`12-hangar-3955`（7 异 mesh peer，已排除 `16`）中位 `132.9 ms`。

**结论与未验证项**：GPU 归约将 178 模拟并发推理压至 `157.8 ms`，`p26_blended_h160` 在验证集实现 `3/3` 正例确认与 `0/21` 异 mesh 同类误确认、`0/31` 跨类误确认。限制与待实测项：（1）`train` 上 `same_mesh_ws_confirmed=2/6` 表明仅靠裁剪外观无法区分同 mesh 机库副本；（2）`18-tower-7889` 仍未跨过确认门槛；（3）P3 仍存在 `03/11/13` 缺失造成的选择偏差与单轨自相关；（4）真实 UE 并发耗时、P2.6 `19×1` 联机闭环与 Win32 `PostMessage(VK_ESCAPE)` 自动关停均待下次开 UE 实测。

[P3 评估脚本](../src/evaluation/eval_p3_flight_calibration.py) · [封存清单](../runs/evaluation/20261120-p3-flight-calibration/sealed-test-manifest.json) · [分布漂移](../runs/evaluation/20261120-p3-flight-calibration/distribution-shift.json) · [混合校准器](../runs/evaluation/20261120-p3-flight-calibration/calibrator-blended-train.json) · [回放汇总](../runs/evaluation/20261120-p3-flight-calibration/val-replay-summary.json) · [P2.6 冻结清单](../runs/evaluation/20261120-p26-graded-active/frozen.json) · [主动核验模块](../src/perception/active_verification.py) · [单测](../tests/test_active_verification.py)

## 2026-11-20 30 Hz 与 60 Hz 相机采集对照

验证相机改为约 30 Hz 后的实际收帧率、30 FPS 录像重复帧及 RGB 与深度的同步情况。178 上的 UE 机场通过 WSL 网关 `172.21.192.1` 连接，使用 `airsim` 环境和 P2.6 的 01 号场景快照。前视和追踪 RGB 均为 1920×1080，前视深度为 640×360。两台相机同时设为 `0.0167` 或 `0.0333333` 秒，按 60、30、60、30 Hz 顺序分别重载，每轮订阅三路图像，预热 3 秒后用原录制器录制 10 秒。无人机保持在地面，未运行控制或模型推理。

| 配置 | 前视 RGB / 帧每秒 | 前视深度 / 帧每秒 | 追踪 RGB / 帧每秒 | 前视录像重复帧 | 追踪录像重复帧 |
|---|---:|---:|---:|---:|---:|
| 约 60 Hz | 23.5–23.7 | 23.6–23.7 | 23.3–23.8 | 20.7%–21.3% | 21.0%–23.0% |
| 约 30 Hz | 22.6–23.5 | 22.5–23.5 | 22.2–23.5 | 21.7%–24.7% | 21.7%–26.3% |

8 个 MP4 均验证为 30 FPS、300 帧。四轮 RGB 与最近深度帧的源时间戳差 p95 均为 0 ms。30 Hz 第一轮有一张 RGB 缺少同时间戳的深度帧，最近深度相差 45.1 ms，其余三轮采样窗口内逐帧匹配。30 Hz 配置可用于静态录制，实测独立画面更新约 22–24 帧/秒，与 60 Hz 基线相近但未观察到流畅度提升。飞行并发负载下的表现待测。

测试后恢复 `DefaultScene`，新连接确认默认时钟有效，试验机器人服务和图像流均已卸载。话题列表仍有五条旧相机元数据，首次恢复确认因此报失败，二次检查结果已补记到报告。

[测量报告](../runs/evaluation/20261120-camera-rate-ab/121740/report.json) · [客户端日志](../runs/evaluation/20261120-camera-rate-ab/121740/projectairsim_client.log) · [四轮录像、配置和逐帧时间戳](../runs/evaluation/20261120-camera-rate-ab/121740/)

## 2026-11-20 P2.6 联机闭环采集 19×1

本轮评估冻结 P2.6 分级门控在 19 个机场场景中的闭环结果。上午采集 01–09，下午在 178 的 UE 机场继续采集 10–19。使用 WSL `airsim` 环境、网关 `172.21.192.1`、同一冻结清单与原批处理入口。前视和追踪相机均按 `0.0167` 秒采集，双路 1920×1080 MP4 按 30 FPS 写入，场景特定参数和参考来自原档案。每项保留双路录像及时间戳、控制报告和门控观测。

| 场景 | 试次 | 通过 | 结果 |
|---|---|---:|---|
| 飞机 | 01–08 | 6/8 | 03、08 失锁 |
| 机库 | 09–17 | 0/9 | 均曾确认身份，随后粗定位失锁，未进入洞口阶段 |
| 塔台 | 18–19 | 0/2 | 均未确认身份，因失锁停止 |

正式试次 19 项齐全，总计 6 项通过。续采的 20 个 MP4 均核验了 30 FPS、帧数和首帧解码，实际收帧率 11.9–16.3 帧/秒。18 号第一次准备后，近景匹配点在预检容差之外约 0.04 像素，未启动正式录像。按原参数重新准备一次后通过预检并采完录像，结果仍为失锁，首轮记录另存。09–12 号失锁前连续 15 步处于门控 `HOLD`，身份仍为已确认。同目标 10 和 12 在 P1 以更低收帧率通过，本轮直接观测到的停止条件是连续失锁，现有结果不能将失败归因于采集帧率。

采集完成后，Windows 锁屏阻止操作 UE 的 Stop 按钮。已通过 ProjectAirSim 恢复空的 `DefaultScene`，确认试验图像流卸载。用户接手结束 UE Play，端口关闭尚未核验。

[批次索引](../runs/evaluation/20261120-p26-graded-active/index.json) · [批次日志](../runs/evaluation/20261120-p26-graded-active/batch.log) · [18 号首次预检](../runs/evaluation/20261120-p26-graded-active/18-tower-7889/repeat-01-incomplete-1795151552/matching-preflight.json) · [18 号正式记录](../runs/evaluation/20261120-p26-graded-active/18-tower-7889/repeat-01/report.json) · [停机记录](../runs/evaluation/20261120-p26-graded-active/shutdown.json) · [P1 对照索引](../runs/evaluation/20261119-p1-revalidation/index.json)

## 2026-11-21 多机两阶段协同调度中心与 Qwen3.5-9B 离线回放验证

**目的**：在不启动 UE 仿真的前提下，验证“中心决策、分机执行”的两阶段多机协同调度骨架（乱序到达、重复回传幂等、上传与返航状态分离、中性编号防真值泄露），并接入 237 服务器上的 `qwen3.5:9b` 实测三类机场目标（飞机、机库、塔台）的“首轮初评 → 复查问题交接 → 次轮补拍复评”闭环。

**环境与参数**：调度端运行在 178 的 WSL（Python 3 + SQLite WAL，无新增第三方依赖），通过 SSH 密钥 `/home/hui/.ssh/id_ed25519_501` 调用 237 服务器（RTX PRO 6000 96 GB）上的 Ollama `qwen3.5:9b`（`think=true`、`temperature=0.1`、强制 JSON Schema）。回放清单 `configs/missions/coordination_replay_3targets.json` 复用已有配对图像，给模型统一使用中性编号 `OBJ-A` / `OBJ-B` / `OBJ-C` 及中性归档文件名，开启 `--stress-test` 逆序交错投递并注入 6 次重复上传。

**方法**：先执行离线单测 `tests/test_coordination.py` 验证幂等去重、乱序处理、标签隔离与单目标 VLM 故障隔离；再执行 `src/coordination/run_replay.py --vlm ollama --model qwen3.5:9b --stress-test`，检查三目标的动态派发、两轮证据引用、耗时与事后对照真值。

**结果**：
- 离线单测 `tests/test_coordination.py` 3/3 通过（0.29 s）。
- 真实 `qwen3.5:9b` 三目标回放 3/3 完成、失败 0、拦截重复上传 6 次、按需创建复查任务 2 个：

| 中性编号 | 事后真值（未送模型） | 首轮视图与耗时 | 是否派发复查 / 建议动作 | 次轮视图与耗时 | 复评判定 | 人工核验 |
|---|---|---|---|---|---|---|
| `OBJ-A` | `02-aircraft-a-wing-curl`（飞机A 主翼上卷） | 正俯视模板 / 19.6 s | 否 / `none` | 未派发 | 无需复查 | **首轮正俯视漏检**：垂直俯视下上卷右翼透视缩短，模型误判外观对称完好 |
| `OBJ-B` | `09-hangar-910`（机库屋面单区破口） | 正俯视模板 / 11.6 s | 是 / `roof_entry` | 入室内部图 / 9.2 s | `confirmed` | 准确识别屋顶破洞并提出查内部梁柱；次轮区分了屋顶断裂与直立支撑柱 |
| `OBJ-C` | `18-tower-7889`（塔台主柱剥落、屋檐破损） | 正俯视模板 / 14.4 s | 是 / `side_inspection` | 近距侧视柱体图 / 14.1 s | `confirmed` | 首轮指出屋顶蓝色异常及侧面盲区；次轮准确识别混凝土主柱纵向裂缝与底部坍塌缺口 |

**结论与未验证项**：
1. 集中式两阶段调度中心已跑通：乱序到达与重复回传均幂等处理，且机库与塔台两类目标展示了清晰的“初评提问 → 针对性补拍 → 依据次轮新证据确认”的协同闭环。
2. 同日补充验证默认 `always_confirm` 派发策略（无论初评是否发现损伤均派发第二批近距复核，离线单测增至 4/4 通过）：重跑 `20261121-qwen35-9b-always-confirm` 后，3/3 完成、拦截重复上传 7 次、创建 3 个复查任务；其中首轮正俯视漏检的 `OBJ-A`（8.5 s，判无损）被第二批近距图（22.7 s）成功改判为 `confirmed`（三目标检出率由 2/3 升至 3/3）。人工核对发现一个 9B 模型细粒度局限：`OBJ-A` 次轮正确发现了上卷翼面上的放射状裂纹，但把竖直上卷的右主翼误称为“垂直尾翼根部”。
3. **尚未验证项**：（1）在真正完好目标（如 `01-aircraft-a-intact`）上验证 `always_confirm` 能否稳定输出 `ruled_out`（确认完好、无误报）；（2）相同补拍预算下“VLM 定向复查 vs 固定预设补拍”的对照评估；（3）ProjectAirSim 中两批共 6 架仿真无人机的在线采集、回传与返航联调。

[首轮仅标疑派发报告](../runs/coordination/20261121-qwen35-9b-3targets/report.md) · [全量近距复核报告](../runs/coordination/20261121-qwen35-9b-always-confirm/report.md) · [全量复核 JSON](../runs/coordination/20261121-qwen35-9b-always-confirm/report.json) · [全量复核 SQLite](../runs/coordination/20261121-qwen35-9b-always-confirm/survey.db) · [回放清单](../configs/missions/coordination_replay_3targets.json) · [单测](../tests/test_coordination.py)

## 2026-11-21 ProjectAirSim 两批 6 机 720p 帧率探测与端到端协同实飞

**目的**：在 UE Airport 实时仿真中加载两批共 6 架 720p + 30 Hz 精简配置无人机（`Scout-1..3`、`Inspect-1..3`，保留 `Chase` 追踪相机用于可视化），先测量 1/3/6 机实时收帧率，再跑通“第一批 3 机并行高空采图返航 → 237 `qwen3.5:9b` 初评与复查交接 → 第二批 3 机并行抵近/破口入室实拍返航 → `qwen3.5:9b` 最终复评”的完整端到端闭环。

**环境与参数**：178 上的 UE Airport（Play 状态），WSL 通过 `172.21.192.1` 以 `airsim` 环境运行 `src/coordination/run_sim_e2e.py`；场景 `configs/simulation/scene_airport_multi6.jsonc`（`robot_quadrotor_multi_720p.jsonc`，`FrontCamera` 1280×720 RGB + 640×360 深度、`Chase` 1280×720 RGB、30 Hz、`IMU1`+`GPS`，关闭分割及多余辅助传感器）；任务 `configs/missions/coordination_sim_6uav.json`（三目标巡航层 75/90/105 m、巡航 12 m/s、首轮采正俯视+斜俯视、次轮抵近/侧视/机库破口入室）；VLM 调用 237 Ollama `qwen3.5:9b`（`think=true`，`dispatch_policy=always_confirm`）。

**结果 1：6 机场景实时收帧率探测（`20261121-sim6-fps-probe`）**
- 6 机场景加载耗时 `3.25 s`，全部流 100% 独立时间戳、零丢帧：

| 订阅规模 | 并发流数 | 单路实测独立帧率 | 帧间隔 p50 / p95 |
|---|---:|---:|---:|
| 1 机（`Scout-1`） | 3 路（Front RGB/Depth + Chase） | **16.58 FPS** | 67–68 ms / 73.5 ms |
| 3 机（单批 `Scout-1..3`） | 9 路 | **11.59 FPS** | 85.5–86.5 ms / 92.4–94.6 ms |
| 6 机全订阅 | 18 路 | **5.66 FPS** | 162.6–163.3 ms / 169.6–174.5 ms |

**结果 2：完整 6 机端到端实飞与两轮评估（`20261121-sim6-e2e-qwen35-9b`）**
- **飞行与安全**：第一批 3 机并行飞采图并返航降落总墙钟 `233.6 s`（`Scout-1/2/3` 分别 `233.6 / 219.9 / 227.2 s`）；第二批 3 机并行抵近/入室实拍并返航降落总墙钟 `251.4 s`（`Inspect-1/2/3` 分别 `215.6 / 241.3 / 251.4 s`，其中 `Inspect-2` 无回退直降穿入机库破洞内拍完再安全飞出）。全部 6 架机全程碰撞计数均为起飞前地面接触 1 次 + 降落接地 1 次，**空中及障碍物碰撞 0 次**。共采集 9 组 `FrontCamera` + `Chase` 1280×720 原图（零过曝、`white_fraction=0.0`）。
- **VLM 两轮评估**（第一轮 3 目标总耗时 `50.4 s`，第二轮总耗时 `50.5 s`）：

| 中性编号 | 事后真值 | 首轮初评（实拍正+斜俯视） | 次轮实拍与复评判定 | 人工核验 |
|---|---|---|---|---|
| `OBJ-A` | `02-aircraft-a-wing-curl` | 18.7 s，`needs_recheck=false` → 触发无损近距复核 | 14.7 s，**`confirmed`** | 实拍近景图清晰对准上卷右翼，成功救回漏检（但模型将卷翼阴影/裂纹误描述为“烧灼痕迹”） |
| `OBJ-B` | `09-hangar-910` | 18.5 s，`needs_recheck=true`（但盯住了相邻完好机库） | 24.7 s，**`ruled_out`（误排除）** | **实拍构图定位根因**：首轮 90 m 未裁剪俯视图画面中心落在了隔壁完好机库上（破洞机库在画面左半区），且次轮室内图桁架完好，模型只回答了“立柱/桁架是否完好”而忽略了画幅顶部的屋顶破洞 |
| `OBJ-C` | `18-tower-7889` | 13.1 s，`needs_recheck=true` | 10.9 s，**`confirmed`** | 实拍近景清晰对准主柱破损，模型准确指出上部垂直裂缝与下部墙体断裂缺口 |

**结论与未验证项**：
1. ProjectAirSim 两批 6 机并发飞行 + 720p 双相机（含 `Chase` 可视化）+ 中心 SQLite 调度 + 237 `qwen3.5:9b` 两轮评估的完整端到端链路已首次实机跑通，全程零空中碰撞、6 机全部安全返航。
2. 端到端实拍暴露出两个值得立刻改进的工程点：（a）高空首轮正俯视未裁剪原图在密集建筑群中容易引入相邻完好建筑干扰（`OBJ-B` 机库偏左），需要调整 `OBJ-B` 首轮正俯视机位使其居中（或做目标框裁剪）；（b）`qwen3.5:9b` 在 `OBJ-B` 室内近景图上只盯了立柱完好而忽略画幅上沿的大破洞，可在提示词中明确“只要两轮任意视角存在屋顶破洞或主体破损即判 `confirmed`”，且后续可对比 237 上现成的 `qwen3.5:27b` 提升细粒度识别准确率。
3. **尚未验证项**：（1）机库首轮居中构图修复 + `qwen3.5:27b` 对照复测；（2）连续 `Chase` 30 FPS 录像录制下的多机帧率表现；（3）真正无损目标（如 `01-aircraft-a-intact`）端到端防误报验证。

[帧率探测数据](../runs/coordination/20261121-sim6-fps-probe/fps_probe.json) · [端到端评估报告](../runs/coordination/20261121-sim6-e2e-qwen35-9b/report.md) · [端到端结构化报告](../runs/coordination/20261121-sim6-e2e-qwen35-9b/report.json) · [仿真指标与耗时](../runs/coordination/20261121-sim6-e2e-qwen35-9b/sim_metrics.json) · [6机场景配置](../configs/simulation/scene_airport_multi6.jsonc) · [6机任务配置](../configs/missions/coordination_sim_6uav.json) · [端到端入口](../src/coordination/run_sim_e2e.py)

## 2026-11-21 6 机两阶段端到端实飞与 3×2 分屏实时录像复测

**目的**：在实飞中验证三项改进：（1）拍照前增加悬停减速调平等待，解决 `OBJ-B` 机库首轮正俯视构图偏左问题；（2）复评提示词明确“任意轮次可见主体破洞即判 `confirmed`”；（3）集成 `src/coordination/video_recorder.py`，在每批仅订阅当前在飞 3 机 `Chase + FrontCamera`（6 路 720p RGB 流）的条件下，实时录制 `1920×776` 3×2 分屏视频。

**环境与参数**：同上 UE Airport + WSL `172.21.192.1` + 237 Ollama `qwen3.5:9b`（`always_confirm`），巡航速度 `16 m/s`，开启 `--record-video`。

**结果**：
- **飞行与录像帧率**：第一批 3 机并飞 `208.3 s`（6 路流各自实收 `2230–2231` 帧、100% 独立时间戳，实测 **10.71 FPS/路**）；第二批 3 机并飞 `221.7 s`（6 路流各自实收 `2328` 帧、100% 独立时间戳，实测 **10.50 FPS/路**）；全程空中与障碍物碰撞 **0 次**（6 机均为初始+降落各 1 次地面接触）。
- **评估结果（3/3 全部 `confirmed`）**：悬停调平后 `OBJ-B` 首轮正俯视将机库屋顶破洞严格居中，`qwen3.5:9b` 首轮直接派发 `roof_entry`，第二批 `Inspect-2` 入室拍图后最终判定 **`confirmed`**；`OBJ-A`（无损复核救回）与 `OBJ-C`（塔台主柱侧视）亦均为 **`confirmed`**，三目标受损检出率达到 **3/3（100%）**。
- **录像产物**：实时 30 FPS 3×2 总览 `mission_6uav_overview.mp4`（`8928` 帧，`297.6 s`，`476.0 MB`）、4 倍速精华版 `mission_6uav_highlight_4x.mp4`（`2401` 帧，`80.0 s`，`197.5 MB`）及网页播放版 `mission_6uav_highlight_4x.webm`。

（本节早期英文版中间调试产物 `20261121-sim6-e2e-video` 已于 2026-11-21 清理，汇总结果保留在本节；完整纯中文轨迹版见下节 `20261121-sim6-e2e-cn-traj-full`。）

## 2026-11-21 纯中文四区实时轨迹录像与 5 Hz 真轨迹导出实测

**目的**：在 UE Airport 实飞中验证新版 `1920×720` 纯中文四区录像（去内部编号 + 右侧真实机场占用栅格二维轨迹态势 + 底部高度剖面 + 6 机色标遥测表）以及 5 Hz 真实轨迹导出 `trajectories.json`（供 ROS2/RViz2 三维回放）。

**环境与参数**：UE Airport + WSL `172.21.192.1` + 237 Ollama `qwen3.5:9b`（本轮启动时仍为 `think=true`，运行后已按用户要求将默认配置改为 `think=false`），巡航 `16 m/s`，开启 `--record-video`。

**结果**：
- **飞行与碰撞**：第一批 `Scout-1/2/3` 并行实飞 `201.0 s` 全部安全返航；第一轮 VLM 评估中 `OBJ-A` 偶发 `think=true` 思考链超过 `180 s` 超时（单目标故障隔离生效，未中断任务，`OBJ-A` 留在 `round1_vlm_failed`）；`OBJ-B`（`roof_entry`）与 `OBJ-C`（`closeup_approach`）正常派发第二批 `Inspect-2/3`（并行实飞 `217.7 s`）并均在终评判为 **`confirmed`**。全部在飞无人机空中与障碍物碰撞 **0 次**。
- **轨迹与视频**：成功导出真实 5 Hz `trajectories.json`（含 NED 与 ENU 坐标），生成 `mission_6uav_overview.mp4`、`mission_6uav_highlight_4x.mp4` 及网页可播 `mission_6uav_highlight_4x.webm`（22.75 MB，652 帧 @ 15 FPS）。
- **结论与后续**：纯中文四区界面、实时二维机场轨迹态势、高度剖面与真实轨迹导出均已实机跑通；默认 `think=false` 已生效于 `vlm.py`/`run_replay.py`/`run_sim_e2e.py`。下一步将第二批 3 架复查机接入真实视觉伺服闭环（DINO 匹配 + 深度测距速度控制律 + 机库破口视觉对准入室）。

（本节不完整超时运行产物 `20261121-sim6-e2e-cn-traj` 已于 2026-11-21 清理，汇总结果保留在本节；完整 6 机版本见下节 `20261121-sim6-e2e-cn-traj-full`。）

## 2026-11-21 `think=false` 完整 6 机实飞、纯中文四区轨迹录像与 5 Hz 真轨迹导出

**目的**：在默认 `think=false` 下完整复跑 6 机两阶段端到端协同流程，确保全部 6 架无人机（`Scout-1/2/3` + `Inspect-1/2/3`）无超时、全量起飞并完成闭环，产出完整 6 机真实轨迹 `trajectories.json`（供 ROS2/RViz2 三维回放）及纯中文四区轨迹录像。

**环境与参数**：UE Airport + WSL `172.21.192.1` + 237 Ollama `qwen3.5:9b`（`--no-think`，`--dispatch-policy always_confirm`），巡航 `16 m/s`，开启 `--record-video`。

**结果**：
- **飞行与碰撞**：6 架无人机全部完整起飞、采图、安全返航降落；第一批 `Scout-1/2/3` 飞行 `199.97 / 183.27 / 194.27 s`，第二批 `Inspect-1/2/3` 飞行 `185.01 / 206.70 / 220.88 s`（`Inspect-2` 经屋顶破洞无回退入室拍完安全飞出）。全部 6 架机全程碰撞均为起飞前+降落各 1 次地面接触，**空中及障碍物碰撞 0 次**。
- **VLM 耗时与判定（`think=false`）**：第一轮 3 目标评估耗时 `6.43 / 2.82 / 2.43 s`，全部派发第二批复查；第二轮终评总墙钟仅 **`9.31 s`**（单目标 `3.42 / 3.22 / 2.51 s`），三目标 **`3/3 confirmed`**（0 超时、0 漏检）。
- **轨迹与录像**：导出完整 6 机 5 Hz 真轨迹 `trajectories.json`（`Scout-1/2/3` 各 `938` 点、最高高度 `78.5 / 95.0 / 110.2 m`；`Inspect-1/2/3` 各 `1044` 点、最高高度 `77.9 / 94.1 / 109.2 m`）；录像 12 路流各自实收 `2582–2593` 帧（100% 独立时间戳、零黑帧/过曝），输出实时版 `mission_6uav_overview.mp4`（4000 帧）、4 倍速精华版 `mission_6uav_highlight_4x.mp4`（1169 帧，38.97 s）及网页播放版 `mission_6uav_highlight_4x.webm`（23.53 MB，585 帧）。
- **尚未验证项**：本轮第二批抵近仍为航点控制；下一阶段按 1A 方案（3 架并行到点、串行独占 GPU 跑 DINOv2 + RGB-D 视觉伺服闭环与机库破口对准入室）替换第二批末端控制。

[评估报告](../runs/coordination/20261121-sim6-e2e-cn-traj-full/report.md) · [结构化报告 JSON](../runs/coordination/20261121-sim6-e2e-cn-traj-full/report.json) · [完整6机真轨迹 JSON](../runs/coordination/20261121-sim6-e2e-cn-traj-full/trajectories.json) · [仿真指标](../runs/coordination/20261121-sim6-e2e-cn-traj-full/sim_metrics.json) · [高清4倍速 H.264 MP4](../runs/coordination/20261121-sim6-e2e-cn-traj-full/videos/mission_6uav_highlight_4x.mp4) · [高清实时版 MP4](../runs/coordination/20261121-sim6-e2e-cn-traj-full/videos/mission_6uav_overview.mp4)

## 2026-11-21 1A 第二批真实视觉伺服闭环接入与实飞调参（v1 / v2）

**目的**：在第二批 3 架复查机（`Inspect-1/2/3`）上以 **1A 方案（并行巡航至伺服起点 + 串行独占 GPU 视觉伺服闭环）** 替换末端航点控制，伺服全程不读取目标真值坐标，同时在前视画面实时叠加 DINOv2 热图 / RGB-D 屋顶破口掩膜、准星、测距与机体系速度指令，并通过静态 `ffmpeg` 自动封装原生 `1920×720` H.264 MP4。

**环境与参数**：UE Airport + WSL `172.21.192.1` + 本地 GPU（`cuda` 加载 `weights/dinov2_vits14_pretrain.pth`，`32×32` 网格）+ 237 Ollama `qwen3.5:9b`（`--no-think`，`--dispatch-policy always_confirm`，`--visual-servo`，`--record-video`）。离线回归单测 `test_project.py` + `test_coordination.py` 共 **14/14 通过**。

**结果与根因分析**：
- **飞行安全与录像**：`v1` 与 `v2` 两轮实飞 6 架无人机全程空中与障碍物碰撞均为 **0 次**，两轮均自动产出原生 `1920×720 @ 30 FPS` H.264 录像（`h264_encoded=true`）与 5 Hz 真实轨迹 `trajectories.json`。
- **`OBJ-A`（飞机 `dino_oblique`）**：两轮均一次闭环收敛（`v1`：`201` 步、`17.75 s` 抵近至 `12.87 m`；`v2`：`183` 步、`16.82 s`，约 `10.9–11.3 Hz` 实时闭环），触发 3 帧 `range_confirmed` 停机并在真实伺服终点直拍近景；`v1` 终评 `confirmed`，`v2` 因 `stop_dist=13.5 m` 稍远在 `think=false` 下偶发判为 `ruled_out`（实拍图主翼上卷清晰可见，缩短至 `11.0 m` 可进一步放大损伤特征）。
- **`OBJ-B`（机库 `opening_nadir`）**：`v1` 未做姿态去倾时，四旋翼水平加速倾斜导致表观洞口振荡（`4.8 → 1.58 → 7.3 m`）；`v2` 加入基于实时相机四元数的姿态去倾投影与定高保持后彻底消除振荡，10 秒内即收敛至 **`0.024 m` 水平误差**（高度稳锁 `38.43 m`），但因传入 `down` 向量时 `opening_geometry.py` 走几何中心 `plane_aperture` 使锚点靠近横梁边缘（净空 `1.06 m < 1.6 m` 阈值）而未进入下降阶段，超时后由安全回退完成入室拍图并终评 `confirmed`。
- **`OBJ-C`（塔台 `dino_oblique`）**：`v1` 抵近至 `12.08 m` 后因模板含塔顶蓝楼滑到画边采到天空深度；`v2` 改用主柱中段模板（离线峰值 `0.934`）后 12 秒即抵近至 **`14.95 m`**，但主柱较窄时镜头微偏扫到 240 m 外的机场另一座同款塔台，深度由 `15.65 m` 骤跳至 `249 m` 触发切目标，超时后走安全回退并终评 `confirmed`。
- **结论与待做修复**：1A 串行 GPU 闭环架构、热图/准星/速度实时叠加、H.264 封包及零碰撞安全护栏全部跑通；仅剩两处 2 行级小调（机库 `opening_candidates` 选最大净空圆心 + 塔台近距深度骤跳防切目标保护 + 飞机 `stop_dist` 调至 `11.0 m`）即可实现三目标 3/3 零回退闭环收敛。

（本节中途调参产物 `20261121-sim6-e2e-visual-servo` 与 `v2` 已于 2026-11-21 清理，根因分析保留在本节；最终 3/3 零回退收敛产物见下节 `20261121-sim6-e2e-visual-servo-v3`。）

## 2026-11-21 1A 视觉伺服闭环完整验收（v3：对齐 6 m 停机标准、3/3 零回退收敛）

**目的**：验证三项定点修复后的完整 6 机端到端闭环：（1）飞机与塔台停机距离对齐项目标准 `stop_dist_m = 5.8 m`（`< 6 m`）、近距限速门槛 `close_range_m = 12.0 m`、深度护栏 `2.5 m`；（2）在 `SharedDinoMatcher.query` 中加入基于同帧深度图的前景深度门控 `foreground_patch_mask`，屏蔽 `> d + 15 m` 远景 patch 防止 240 m 外孪生塔台劫持；（3）机库正俯视 `opening_nadir` 保持姿态去倾与定高控制，改用 `opening_candidates(..., down=None)` 选取最大净空圆心实现真实穿洞入室。

**环境与参数**：UE Airport + WSL `172.21.192.1` + 本地 GPU（`cuda` 加载 `weights/dinov2_vits14_pretrain.pth`）+ 237 Ollama `qwen3.5:9b`（`--no-think`，`--dispatch-policy always_confirm`，`--visual-servo`，`--record-video`）。回归单测 `test_project.py` + `test_coordination.py` 共 **14/14 通过**（新增前景深度门控单测）。

**结果（3/3 闭环收敛、0 回退、0 空中碰撞、3/3 确认受损）**：

| 复查机 / 中性目标 | 控制模式（伺服不读真值坐标） | 闭环步数 / 耗时 | 终态实测指标 | 是否回退 | VLM 终评 |
|---|---|---:|---|---|---|
| `Inspect-1` / `OBJ-A`（飞机） | `dino_oblique`（首轮斜视图原型 + 前景深度门控） | `239` 步 / `21.33 s` | `range_confirmed`，`guard_depth = 5.64 m`，`peak = 0.792`，3/3 确认 | **否** | **`confirmed`** |
| `Inspect-2` / `OBJ-B`（机库） | `opening_nadir`（最大净空圆心 + 姿态去倾 + 通道核验） | `226` 步 / `17.91 s` | `interior_reached`，水平误差 `0.241 m`，净空 `2.76 m`，垂直穿洞至 `16.03 m` 并原位垂直退出 | **否** | **`confirmed`** |
| `Inspect-3` / `OBJ-C`（塔台） | `dino_oblique`（主柱原型 + 前景深度门控防孪生塔台） | `253` 步 / `25.67 s` | `range_confirmed`，`guard_depth = 5.71 m`，`peak = 0.520`，3/3 确认 | **否** | **`confirmed`** |

- **安全、耗时与产物**：6 架机全程碰撞均为起飞前+降落各 1 次地面接触，**空中及障碍物碰撞 0 次**；VLM 首轮评估总墙钟 `12.39 s`，终评总墙钟 **`9.82 s`**；6 机 5 Hz 真轨迹完整导出（`Scout-1/2/3` 各 `935` 点，`Inspect-1/2/3` 各 `1234` 点）；两版 `1920×720 @ 30 FPS` 视频均已自动封装为原生 H.264 MP4（实时版 `4250` 帧 / `141.67 s`，4 倍速精华版 `1232` 帧 / `41.07 s`）。
- **尚未验证项**：真正无损目标（如 `01-aircraft-a-intact`）在端到端视觉伺服闭环下的 `ruled_out` 零误报验证。

[v3 评估报告](../runs/coordination/20261121-sim6-e2e-visual-servo-v3/report.md) · [v3 结构化报告 JSON](../runs/coordination/20261121-sim6-e2e-visual-servo-v3/report.json) · [v3 完整 6 机轨迹 JSON](../runs/coordination/20261121-sim6-e2e-visual-servo-v3/trajectories.json) · [v3 闭环逐帧日志目录](../runs/coordination/20261121-sim6-e2e-visual-servo-v3/servo_logs) · [v3 仿真指标](../runs/coordination/20261121-sim6-e2e-visual-servo-v3/sim_metrics.json) · [v3 高清 4x H.264 MP4](../runs/coordination/20261121-sim6-e2e-visual-servo-v3/videos/mission_6uav_highlight_4x.mp4) · [v3 高清原速 H.264 MP4](../runs/coordination/20261121-sim6-e2e-visual-servo-v3/videos/mission_6uav_overview.mp4)

## 2026-11-21 混合无损阴性对照实飞验证（2 无损飞机 + 1 受损机库，真实视觉伺服闭环）

**目的**：在 `always_confirm` 全量复核策略与真实 1A 视觉伺服闭环下，将机场目标库中全部 2 个完好目标（`01-aircraft-a-intact`、`05-aircraft-b-intact`）与 1 个受损目标（`09-hangar-910`）混排盲测，验证系统能否同时实现“无损目标稳定判 `ruled_out`（零误报）”与“受损目标稳定判 `confirmed`（不漏检）”，并检验 DINOv2 视觉伺服在新目标上的泛化收敛能力。

**环境与参数**：UE Airport + WSL `172.21.192.1` + 任务配置 `configs/missions/coordination_sim_6uav_intact_check.json`（2D 态势图按目标范围自动扩展）+ 本地 GPU `DINOv2 ViT-S/14` + 237 Ollama `qwen3.5:9b`（`--no-think`，`--dispatch-policy always_confirm`，`--visual-servo`，`--record-video`）。回归单测 `14/14` 通过。

**结果（3/3 零回退闭环收敛、0 空中碰撞、3/3 精准区分真值）**：

| 复查机 / 中性目标 | 事后真值 | 闭环模式与耗时 | 终态实测指标 | VLM 终评 | 核对 |
|---|---|---|---|---|---|
| `Inspect-1` / `OBJ-A`（飞机1） | `01-aircraft-a-intact`（无损） | `dino_oblique`，`391` 步 / `31.64 s` | `range_confirmed`，终距 `5.77 m`，`cx = -0.001`，`peak = 0.700` | **`ruled_out`** | 正确（零误报） |
| `Inspect-2` / `OBJ-B`（机库） | `09-hangar-910`（受损） | `opening_nadir`，`242` 步 / `19.21 s` | `interior_reached`，垂直穿洞入室至 `16.08 m` 并安全原位退出 | **`confirmed`** | 正确（命中损伤） |
| `Inspect-3` / `OBJ-C`（飞机2） | `05-aircraft-b-intact`（无损） | `dino_oblique`，`373` 步 / `31.78 s` | `range_confirmed`，终距 `5.70 m`，`cx = +0.014`，`peak = 0.482` | **`ruled_out`** | 正确（零误报） |

- **安全与效率**：6 机空中及障碍物碰撞 **0 次**；VLM 首轮总耗时 `11.79 s`，终评总耗时 **`8.79 s`**；原生 `1920×720 @ 30 FPS` H.264 录像自动封包完成（实时版 `4838` 帧 / `161.27 s`，4 倍速版 `1379` 帧 / `45.97 s`）。
- **综合 `v3` + 本轮两场端到端实飞汇总（6 组目标、6 次真实视觉伺服）**：视觉伺服闭环收敛率 **6/6（100%，零回退、零空中碰撞）**；损伤识别真正率（受损→`confirmed`）**4/4（100%）**；无损排除真负率（无损→`ruled_out`）**2/2（100%，零误报）**。

[阴性对照评估报告](../runs/coordination/20261121-sim6-e2e-intact-control/report.md) · [结构化报告 JSON](../runs/coordination/20261121-sim6-e2e-intact-control/report.json) · [完整 6 机轨迹 JSON](../runs/coordination/20261121-sim6-e2e-intact-control/trajectories.json) · [闭环逐帧日志目录](../runs/coordination/20261121-sim6-e2e-intact-control/servo_logs) · [仿真指标](../runs/coordination/20261121-sim6-e2e-intact-control/sim_metrics.json) · [高清 4x H.264 MP4](../runs/coordination/20261121-sim6-e2e-intact-control/videos/mission_6uav_highlight_4x.mp4) · [高清原速 H.264 MP4](../runs/coordination/20261121-sim6-e2e-intact-control/videos/mission_6uav_overview.mp4) · [对照任务配置](../configs/missions/coordination_sim_6uav_intact_check.json)

## 2026-11-21 结构化语义评估 Schema 升级与 5 目标实飞真图验证

**目的**：将两轮 VLM 评估从简略文本升级为结构化语义输出（目标种类、疑似/确认受损部位、可见损伤语义描述、损伤分级、结构化二次任务单 `secondary_task` 与后续处置建议），并在 `OllamaVLMClient` 中补齐 SSH 握手重置（`rc=255`）自动重试；使用 `v3` 与 `intact-control` 实飞采集的 5 组真实视觉伺服图像（3 受损 + 2 无损）做 237 `qwen3.5:9b`（`think=false`）验证。

**环境与参数**：离线复用真实视觉伺服采集图像，调用 237 Ollama `qwen3.5:9b`（`think=false`，`temperature=0.1`，`always_confirm`）；回归单测 `test_project.py` + `test_coordination.py` 共 **14/14 通过**。

**结果（5/5 判定准确，单次推理 `2.79–6.64 s`）**：
- `OBJ-A`（`02-aircraft-a-wing-curl`）：首轮 `双尾翼固定翼飞机 / none → intact_verification`（6.64 s）；次轮近景改判 **`confirmed / severe`**（3.37 s），成功救回高空漏检。
- `OBJ-B`（`09-hangar-910`）：首轮 `机场航站楼建筑 / severe / 屋顶 → roof_entry（高优先级）`（4.43 s）；次轮入室终评 **`confirmed / severe`**（5.84 s）。
- `OBJ-C`（`18-tower-7889`）：首轮 `机场控制塔台 / none → intact_verification`（3.54 s）；次轮侧视近景改判 **`confirmed / severe / 塔身主体混凝土结构`**（3.58 s）。
- `OBJ-D`（`01-aircraft-a-intact`）与 `OBJ-E`（`05-aircraft-b-intact`）：均稳定输出 **`ruled_out / intact / 无受损部位`**（终评耗时 `2.79 / 3.25 s`，零误报）。
- **人工核验局限（诚实记录）**：`qwen3.5:9b`（`think=false`）在 5/5 总体定性与二次任务调度上表现稳定，但在细粒度部位描述上存在轻微泛化/过推断（如把上卷右翼误称为“左侧垂直尾翼”、对机库室内完好立柱过度推断为“弯曲倾斜”、对塔台未入画的塔顶屋顶外推破损）；后续对比 `--model qwen3.5:27b` 可进一步校准细粒度语义精度。

[语义化评估报告](../runs/coordination/20261121-semantic-vlm-check/report.md) · [语义化报告 JSON](../runs/coordination/20261121-semantic-vlm-check/report.json) · [SQLite 状态库](../runs/coordination/20261121-semantic-vlm-check/survey.db)

## 2026-11-23 沿海渔港灾后关卡构建与单机取图验收

**目的。** 在独立民用场景中提供建筑、水上船、搁浅船及道路车辆的巡查样本，并验证同一研究平台可以读到彩色图和深度图。

**环境与参数。** Windows `project_520`、UE 5.2.1、ProjectAirSim、RTX 4060 Ti 8 GB；WSL 通过 Windows 网关 172.21.192.1 运行 `airsim` Python 环境。研究区域为 2 km × 2 km 的灾后静态关卡。采用单机 FrontCamera、90° 水平视场、30° 下俯，RGB 1920×1080、深度 640×360。[场景与素材](../docs/沿海渔港关卡.md) · [场景配置](../configs/simulation/scene_coastal_harbor.jsonc)

**方法与结果。** 将地图保存为 UE 原生关卡，再用独立编辑器进程重开。本轮读取到 120 栋建筑、40 艘船（32 艘在水上、8 艘搁浅）、120 辆车；32 个重点对象均能按编号定位。5 处垂直碰撞射线检出了陆地 0.00 m、海面 −2.65 m、岸壁平台 +0.17 m、浮桥 −1.72 m 和街道 +0.09 m。按模型凸包支撑点调整了 8 艘搁浅船和 2 辆侧翻货车的接地高度，输出调整量。保存了 10 个 UE 视角。随后启动 Windows 独立游戏进程，在搁浅港作船 `Vessel_035` 的已知观察点取回三组相互独立的 RGB-D，每组双路时间戳一致。末帧有效正深度占比 100%，全图中位深度 24.172 m；该数值包含岸滩背景。[重开清点和碰撞检查](../runs/scenes/20261123-coastal-harbor/scene-check.json) · [接地几何记录](../runs/scenes/20261123-coastal-harbor/ground-contact.json) · [原生画面](../runs/scenes/20261123-coastal-harbor/views/) · [相机记录](../runs/scenes/20261123-coastal-harbor/simulation/check.json) · [无人机前摄](../runs/scenes/20261123-coastal-harbor/simulation/front-rgb.jpg) · [原始深度](../runs/scenes/20261123-coastal-harbor/simulation/front-depth.npz)

**修正与结论。** 首次导入时出现材质引脚名称、引擎纹理查找和 FBX 自动后缀不匹配，修正后完成导入。第一轮近景存在过重阴影、窗户不透明和部分船体离地，随后调整了天空补光、玻璃材质和接地点。首次加载时 UE 仍在编译着色器，部分画面出现灰色棋盘格。最终编辑器复核显式等待资源编译完成；单机取图入口在已知观察点检查红色船底漆显色，再验收三组同步图像。D3D12 离屏编辑器报告过已处理的交换链警告，完成截图与保存。当前结果支持加载场景、识别对象编号和相机采集；多机性能、自动巡查成功率、真实灾损等级和实景泛化仍未测量。[UE 与模拟器日志](../runs/scenes/20261123-coastal-harbor/logs/) · [关卡文件哈希](../runs/scenes/20261123-coastal-harbor/map-sha256.txt)

## 2026-11-24 沿海渔港建筑群与固定机位复拍

**目的与环境。** 检查住宅街、沿街店铺和仓储院落在航拍和抵近观察时的可辨识程度。使用 Windows 上的 UE 5.2.1、Blender 3.6、ProjectAirSim 和 WSL 的 airsim 环境。沿用 2 km × 2 km 任务区、既有对象编号和 32 个复查标签。UE 对照机位使用 72° 水平视场和 1440 × 900 像素。[机位](../runs/scenes/20261124-coastal-streets/street-cameras.json) · [原关卡对象表](../runs/scenes/20261124-coastal-streets/targets-before.json)

**方法与结果。** 沿既有街道布置了 84 栋带老虎窗的双拼、68 栋连排、68 栋四坡顶小屋，以及 22 间环境店铺。两处有编号的鱼市换上条纹雨棚，一端脱落下垂。仓库保留屋顶开口，库外增加装卸道、货箱、托盘和集装箱。重新加载原生地图，核对 120 个建筑、40 艘船和 120 辆车的巡查标签。对地面、水面、码头、栈桥和街道的碰撞探测均在原容差内。另对两间鱼市及仓库中心向下检测，均命中对应建筑。12 个机位分别导出图像，从中选出住宅、鱼市及港区的同机位对照。[街区清单](../runs/scenes/20261124-coastal-streets/street-build.json) · [场景核验](../runs/scenes/20261124-coastal-streets/scene-check.json) · [对照图](../runs/scenes/20261124-coastal-streets/buildings-before-after.jpg) · [最终地图哈希](../runs/scenes/20261124-coastal-streets/map-sha256.txt)

**过程中发现的问题。** 首次写入 UE 网格碰撞设置时遇到属性接口差异，随后改用 UE 5.2 的编辑器属性接口。住宅背面复拍促成了后院边界和附属设施的补建，仓储远景复拍促成了地表纹理更换。混合草地材质第一次连接纹理采样节点失败，修正输入引脚后完成编译和地图保存。无窗口编辑器启动时报告已处理的 DX12 交换链提示，随后完成全部图像导出。首次把无人机从码头直接摆向鱼市时出现仓库碰撞和相机超时，调整为先抬高、再平移和下降的布置顺序。[构建与相机日志](../runs/scenes/20261124-coastal-streets/logs/)

**模拟无人机取图。** 在最终地图上从码头上方布置观察位，前相机取到了 3 组时间戳同步且递增的 RGB 与深度图。RGB 为 1920 × 1080，深度为 640 × 360，深度图中央小区中位数为 19.74 米。固定观察位的布置与取图没有运行自主抵近控制。[相机记录](../runs/scenes/20261124-coastal-streets/simulation/check.json) · [RGB 原图](../runs/scenes/20261124-coastal-streets/simulation/front-rgb.jpg) · [深度数据](../runs/scenes/20261124-coastal-streets/simulation/front-depth.npz)

**结论边界。** 固定机位照片用于检查布局和可见性。此轮没有测视觉模型对损伤的检出率、多机性能或真实灾害中的识别效果。环境房屋不在带编号的巡查真值表中。

## 2026-11-24 渔港曝光与海雾排查

在 Windows UE 5.2 已保存的渔港地图中，离屏 SceneCapture2D 以 68° 水平视场和 1200 × 750 分辨率观察鱼市同一立面。沿同一观察方向将水平距离从 65 米缩短到 14 米，每次稳定 7 秒后采图。分别沿用项目自动曝光和固定曝光（相机最小、最大 EV100 均为 1，仅为诊断值）。远近分别取中心 12 × 12 及 54 × 54 像素，覆盖近似相同的墙面范围。另在同一个高空机位固定曝光，切换海雾。

自动曝光下远、近处墙面输出亮度中位数为 58.8 和 90.2，固定后为 96.7 和 91.9，数值采用 0–255 显示亮度。高空中央区域的亮暗跨度（第 90 与第 10 百分位之差）在有雾时为 47.8，临时关雾时为 55.6。首次采集出现阴影页池溢出提示，正式复拍将离屏阴影页数调为 1536。此对照解释了当前工程能够复现的现象，用户截图的即时曝光及视口参数未知。未保存关卡变动，复核了地图哈希。道路脚本按 205 米等距生成六条平行主路，街区补建沿用相同基线。[原图、脚本与日志](../runs/scenes/20261124-harbor-light-audit/) · [测量数据](../runs/scenes/20261124-harbor-light-audit/measurements.json)

## 2026-11-24 渔港路网重建与固定曝光验证

**目的和环境。** 在原 2 km × 2 km 任务区内重新布置港区、商街、旧镇和外围聚落，核验同一立面抵近时的亮度。以英国纽林的港街与旧聚落关系作形态参照。WSL 负责地块计算，Blender 3.6 导出网格，Windows UE 5.2 保存并重载关卡，ProjectAirSim 在同一关卡采集 RGB-D。关卡固定 EV100 为 1.25，补偿 +1，海雾密度 0.0065。两种取图入口均采用输出伽马 2.2。[场景说明](../docs/沿海渔港关卡.md) · [纽林社区规划](https://www.pznp.co.uk/wp-content/uploads/2023/11/PZNP-Submission-Plan-13-11-23-v2.9.pdf)

**布局。** 24 条道路连成一张路网，共 362 个建筑地块。对 426 个重新摆放的建筑和车辆复核规划中心及重载后的位置。巡查清单中 120 栋建筑和 64 辆车更新了坐标，280 个目标的其余字段全部一致，含 32 个重点巡查目标。32 艘水上船和 8 艘搁浅船保留原位。关卡重载后，24 个路面探针均通过，植被实例为 4,266 株。高空对照取自改造前备份和最终地图，相机位置、68° 水平视场、1440 × 900 分辨率及输出伽马相同，各自保留对应关卡的曝光。[同机位对照](../runs/scenes/20261124-port-town/town-before-after.jpg) · [布局核对](../runs/scenes/20261124-port-town/layout-check.json) · [重载检查](../runs/scenes/20261124-port-town/scene-check.json) · [地块核对](../runs/scenes/20261124-port-town/parcel-check.json) · [地图 SHA-256](../runs/scenes/20261124-port-town/map-sha256.txt)

**鱼市抵近。** UE 相机持续移动约 21 秒，水平距离从 65 米缩短至 14 米，6 个采样点的中心射线均命中 Building_071。保持中心取样的近似物理范围，墙面显示亮度中位数依次为 153.32、152.72、152.64、151.61、151.61、150.75（0–255）。ProjectAirSim 单机在两个端点各取得 3 组同步、递增时间戳，前相机墙面亮度为 152.22、150.61，中心深度为 74.84、16.16 米。这次验证覆盖鱼市这条观察线，模拟器端点使用预设位姿。本次没有执行自主飞行或真实灾害损伤评估。[逐帧测量](../runs/scenes/20261124-port-town/approach/measurements.json) · [65 米 RGB-D](../runs/scenes/20261124-port-town/simulation/65/check.json) · [14 米 RGB-D](../runs/scenes/20261124-port-town/simulation/14/check.json)

**过程中发现的问题。** 初稿中镇外道路过于贴近潮滩，远景农田纹理重复，已调整道路和地表。码头集散棚一度覆盖模拟器出生点，清出净空后重新保存，在出生点及周边四个点向下检测均命中 0.17 米高的码头面。编辑器显示组件会影响部分 Actor 的包围盒，地块中心在无渲染命令行中核对，GPU 重载后再比对 Actor 原点。码头网格导入有切线警告，模拟器日志有虚拟阴影页池告警，个别阴影细节可能缺失。[构建及复验日志](../runs/scenes/20261124-port-town/logs/)

## 2026-11-24 编辑器主视口移动亮块与内陆地块排查

检查用户提供的 12.1 秒、60 fps、1156 × 770 编辑器录像。在 4.20 秒处，亮区同时覆盖道路、住宅及草地。10.49 秒时，近处海面有带直边与阶梯状过渡的亮区。同帧水面亮边内外各取 32 × 16 像素，左上角坐标分别为 (504, 627)、(504, 692)，显示亮度中位数约为 184 和 86（0–255）。这些区域供比较画面亮度，未做辐射定标。[住宅帧](../runs/scenes/20261124-viewport-light-audit/frames/4.20.jpg) · [海面帧](../runs/scenes/20261124-viewport-light-audit/frames/10.49.jpg)

在单独启动的 Windows UE 5.2 编辑器中，通过主透视视口取 1056 × 704 图像。高空、近海和住宅区观察位分别采用 660、145 和 135 米 UE 世界 Z。每次切换后等待至少 5 秒。海面同机位分别关闭 Lumen 漫反射、Lumen 反射、动态阴影、雾和其他候选项，再恢复对照。基线较远和较近采样区的显示亮度约为 92 和 220；关闭 Lumen 漫反射时为 153 和 142，恢复时为 93 和 220。单独关闭反射、阴影或雾仍能看到原来的亮边。亮区对天空光源和 Lumen 辐射缓存也有响应，尚未单独定位引擎内部哪一级采样形成硬边。将渔港全域后处理体积的动态全局光照方法临时设为 None，海面与住宅两处固定机位的亮边均消失，整体色调变浅。没有用该临时设置进行完整航线或无人机 RGB-D 验收。第一次使用 Shot 命令未导出主视口图像，改用 UE AutomationLibrary 的主视口截图。所有诊断只在独立进程生效，保存地图的 SHA-256 复核与建成时一致。[逐项测量](../runs/scenes/20261124-viewport-light-audit/measurements.json) · [主视口原图](../runs/scenes/20261124-viewport-light-audit/viewport/) · [取图脚本与日志](../runs/scenes/20261124-viewport-light-audit/logs/)

对保存的 362 个建筑地块按 UE 世界 Y 统计。在 Y < 250 米处有 259 个，250 至 500 米有 72 个，500 至 750 米有 28 个，750 至 1,000 米只有 3 个。最后一段在任务区内，区内该段巡查清单仅有 3 栋建筑。地块筛选偏向港边与旧镇道路，任务区外另有延伸道路、农田色块、树带和地形，缺少能从空中辨认的聚落层次。图按地块中心绘制，纵轴只展示陆侧和近岸。[地块与道路分布](../runs/scenes/20261124-viewport-light-audit/inland-layout.png) · [原始地块数据](../runs/scenes/20261124-port-town/layout.json)

## 2026-11-24 前港后镇布局与主视口复拍

**目的与方法。** 在 2 km × 2 km 任务区内分散原有 362 个建筑地块，把港口作业与内陆生活区沿道路连接。港边保留 16 个作业建筑地块、两处重点鱼市及码头设施。原有路网在内陆增加 7 条支路，建筑沿山脊、谷地和果园路口选址，住宅地块预留侧院并增大道路后退距离。使用 airsim Python 计算米制地块，Blender 3.6 导出 3 个更新网格，Windows UE 5.2 保存并重载原生地图。施工前的地图、相关 UE 资产和 SourceArt 已备份。[地块与道路](../runs/scenes/20261124-inland-town/layout.json) · [地图校验值](../runs/scenes/20261124-inland-town/map-sha256.txt) · [备份](</mnt/d/Program Files/Epic Games/Project_5.2/project_520/Saved/Backups/CoastalHarbor/BeforeInlandTown>)

**分布与目标验证。** 按建筑地块中心统计，世界 Y < 250 米的地块从 259 个降至 81 个，最北 250 米从 3 个增至 82 个。362 个地块各自最近邻中心距离的中位数从 23.42 米增至 29.76 米。31 条道路连通，重载后 31 个路面探针及 5 个出生点探针通过。426 个重排 Actor 的原点与保存记录一致，植被为 4,115 个实例。巡查清单共 280 项，其中 120 栋建筑和 64 辆车更新了坐标，目标编号、模型、损伤和重点巡查标签均保留。[平面分布](../runs/scenes/20261124-inland-town/building-distribution.png) · [地块统计](../runs/scenes/20261124-inland-town/layout-check.json) · [重载核验](../runs/scenes/20261124-inland-town/scene-check.json) · [本次目标快照](../runs/scenes/20261124-inland-town/targets.json)

**光照对照。** 地图的全域后处理体积将动态全局光照设为 None，天空光强度从 1.45 调至 0.80，固定曝光 EV100 仍为 1.25、补偿 +1。在前次出现亮带的主视口海面机位，按水平视场和分辨率将两个水面采样区映射到相同的视线，较远、较近区域的显示亮度由 93.58、219.78 变为 118.02、107.03。住宅区原问题机位的直边亮块也消失。两次俯视对照保持机位相同，图像保留了各版的原生光照。[海面测量](../runs/scenes/20261124-inland-town/sea-light-check.json) · [改前镇区](../runs/scenes/20261124-inland-town/before/inland.png) · [改后镇区](../runs/scenes/20261124-inland-town/after/inland.png)

**构建记录。** 首次主视口导图使用 Windows 反斜杠导致绝对路径被误拼，修正路径后重拍。更新 FBX 来源记录时有渲染进程占用网格包，待进程退出后重试，3 个网格的来源均指向 Windows 工程内的 SourceArt 并成功保存。UE 提示 TownLots 网格切线退化，离屏视口初始化有 DX12 交换链提示，主视口和离屏采图最终均成功。[构建与采图日志](../runs/scenes/20261124-inland-town/logs/)

**动态与模拟相机复核。** 在主编辑器视口连续移动相机，从内陆山坡经过住宅、码头到海面，再升高观察。检查 13 张沿途取样图，未复现原来的直边亮块。ProjectAirSim 单机前相机在鱼市水平距离 65 米和 14 米处各取得 3 组同步且递增的 RGB-D 时间戳，中心深度分别为 74.84 米、16.16 米。连续移动的是编辑器视口，模拟无人机采用两个固定观察位。[沿途机位与原图](../runs/scenes/20261124-inland-town/traverse/route.json) · [65 米检查](../runs/scenes/20261124-inland-town/simulation/65/check.json) · [14 米检查](../runs/scenes/20261124-inland-town/simulation/14/check.json)

## 2026-11-24 港务办公楼顶层墙脚闪烁排查

用户截图指向 Building_075。用独立的 Windows UE 5.2 主视口复拍蓝门、黄色台阶与空调之间的墙面，移动取景位置并临时关闭动态阴影。深色墙脚的三角斑块在关掉动态阴影后仍然可见。在 Blender 3.6 中读取完整型 harbour_office.fbx，按材质槽提取竖直三角面并在局部 XZ 平面求投影交集。Slot04 与 Slot06 的外表面都在局部 Y ≈ 2.350178 米；在 Z 10.575 至 11.474 米高度带内有 8.273 平方米重合（未缩放的模型面积）。该处发生共面深度冲突。此轮只做诊断，没有重导入模型或保存关卡。[重合面测量](../runs/scenes/20261124-harbour-office-wall/overlap.json) · [主视口复拍](../runs/scenes/20261124-harbour-office-wall/close-normal.png) · [关闭动态阴影](../runs/scenes/20261124-harbour-office-wall/close-off.png) · [取图及日志](../runs/scenes/20261124-harbour-office-wall/)

## 2026-11-24 渔港巡查目标表面闪烁修复

在 Windows `project_520` 的 UE 5.2.1 中重开 CoastalHarbor，逐个读取 120 栋建筑、40 艘船和 120 辆车的可见网格、材质及变换。43 个不同网格构成 39 种模型或部件组合。对每种组合从 25 个方向投影最高细节网格，图像尺寸为 420 × 420。记录前两层近乎平行、深度相差小于 0.15 毫米、材质或 UV 不同且连通面积至少 4 像素的位置。23 种组合出现候选区域，逐组选取代表目标，在 UE 主视口对着疑点小幅横移相机复拍。建筑底面接触、半透明面、材质动画与正常视差也会造成筛查命中或帧差。[目标与部件](../runs/scenes/20261124-target-surface-repair/inventory.json) · [几何筛查](../runs/scenes/20261124-target-surface-repair/exterior-after.json) · [代表目标双机位图](../runs/scenes/20261124-target-surface-repair/paired-views.jpg)

使用 Blender 3.6.23 修复并重导入 8 份网格，涉及 33 个编号目标。16 栋港务办公楼的墙面与包边有重叠。保留可见包边，裁掉藏在后面的部分，再生成风损版。`Building_075` 顶层墙脚的源模型重合面积为 8.273118 平方米，重新从正式 UE 资源导出后，同一区域只剩约 0.000004 平方米的数值残差。正常光照及关闭动态阴影的原机位和横移画面均显示连续墙脚。四间商铺（072、078、084、090）在木门两边保留玻璃，修复前门窗材质相交的连通区有 2,025 至 2,841 像素，修后均低于 4 像素筛查门槛。[办公楼与商铺对照，含自卸车](../runs/scenes/20261124-target-surface-repair/comparison.jpg) · [面积、编号和资源哈希](../runs/scenes/20261124-target-surface-repair/checks.json)

第一次双机位复拍额外确认 `Vehicle_091` 黄色车厢内壁的黑色竖纹。将左右各五条加强筋贴在侧板表面的隐藏端面埋入板内 2 厘米，重新生成完好及风损自卸车，覆盖 11 辆完好车和 2 辆风损车。由正式网格导出的完好车左右侧板上原各有 10 个加强筋端面三角形，修复后两侧均为 0。完好车在同一对机位的中央 128 × 96 像素区中，灰度差超过 24 的像素占比从 6.97% 降至 0。风损车按关卡内侧翻姿态复拍。其他车型、船只和房屋的已拍机位中未确认新的同类大块跳色；未覆盖所有距离、低细节网格、连续飞行轨迹或无人机传感器管线。[自卸车修前两帧](../runs/scenes/20261124-target-surface-repair/truck-before/) · [两款自卸车两侧复拍](../runs/scenes/20261124-target-surface-repair/truck-views/) · [帧差](../runs/scenes/20261124-target-surface-repair/paired-views.json)

首次保存遇到用户编辑器占用，用户关闭工程后完成正式写入。两个导入批次均核对材质槽、碰撞方式和包围盒；独立进程重载后，280 个目标的部件、材质绑定和变换与修前一致。正式地图 SHA-256 与上次保存记录一致。UE 对办公楼提示部分切线退化，对程序化网格提示缺少平滑组，离屏视口启动时有 DX12 交换链提示。各轮复拍均正常导出，复核进程已退出。[导入和渲染日志](../runs/scenes/20261124-target-surface-repair/logs/) · [修前 FBX 与 UE 资源备份](</mnt/d/Program Files/Epic Games/Project_5.2/project_520/Saved/Backups/CoastalHarbor/BeforeTargetSurfaceRepair>)

## 2026-11-24 渔港活动区外围渲染优化

**目的与改动。** 降低从港口飞向内陆时的背景渲染成本，保留 2 km × 2 km 的活动区及近景巡查内容。对有编号的 280 个 Actor 按模型、材质、标签和变换逐项比对，全部一致；4,115 棵树的树种和变换也逐一相同。区内及边界外 120 米缓冲带共 575 棵树保留原设置。其余 3,540 棵单独使用两种远景植被类型，关闭投影，实例按 1,200 米距离上限剔除。远山在沿岸方向的远端与农田后方改用较疏的地形采样，外围源网格由 149,380 降至 89,312 个三角面。农田、核心地面和近岸接缝保持原采样。[对象与资源复核](../runs/scenes/20261124-backdrop-performance/checks.json) · [航线三处同机位画面](../runs/scenes/20261124-backdrop-performance/flight-views.jpg)

**环境和方法。** Windows 11，UE 5.2.1，RTX 4060 Ti 8 GB，D3D12 离屏编辑器主视口。UE 日志记录实际视口为 813 × 655，另存留证图为 1280 × 720。沿同一条从港口到内陆的路径顺序设置 201 个相机位姿。先分别启动新旧版本做独立测量，再在同一进程中按新、旧、旧、新、旧、新的次序重载，旧关卡使用备份的原远山网格。每趟交叉试验均先预热整条航线，使用引擎 CSV Profiler 记录 GPUTime 与 FrameTime，统一取每趟第 8 至 194 条帧记录（各版本 561 帧），避开切换边界。交叉试验用于比较渲染耗时，显存使用独立进程测量，按 PID 采集 Windows 专用 GPU 内存计数器，并限定在每趟 201 个位姿的采样时段。[每趟统计](../runs/scenes/20261124-backdrop-performance/performance.json) · [交叉测试原始日志](../runs/scenes/20261124-backdrop-performance/cross/) · [进程显存采样](../runs/scenes/20261124-backdrop-performance/process-memory.csv)

**结果。** 同进程交叉试验中，GPU 每帧耗时中位数从 35.20 ms 降至 11.35 ms（下降 67.8%）。整帧耗时中位数从 35.63 ms 降至 33.33 ms，第 95 百分位从 51.30 ms 降至 33.91 ms。独立启动的 UE 进程专用显存中位数为 1,649.8 和 1,649.4 MiB，采样峰值为 1,681.8 和 1,649.4 MiB；观察到的稳定收益集中在渲染时间，常态显存相近。高分位耗时改善不能直接换算成多机相机吞吐量。[按航线位置绘制的 GPU 耗时](../runs/scenes/20261124-backdrop-performance/gpu-route.png)

**功能复验。** 从磁盘重载后，280 个巡查编号（含 32 个重点目标）、31 条道路、5 个出生点、地形边缘、海岸和屋面碰撞等既有检查均通过。单独启动 Windows 游戏和 ProjectAirSim 单机，在距离鱼市目标水平 14 米的观察位取得 3 组时间戳同步的 RGB-D；RGB 为 1920 × 1080，深度为 640 × 360，中心中位深度为 16.16 米，取图超时为 0。此测试没有测量多机同时取图性能。[关卡复核](../runs/scenes/20261124-backdrop-performance/optimized/scene.json) · [模拟相机记录](../runs/scenes/20261124-backdrop-performance/simulation/check.json)

**排查记录。** 首次航线试跑只设置了 CSV 文件名，补上启动采集指令后重测；首次使用无渲染命令行核验时地面碰撞探针未命中，改为在已加载的原生编辑器世界中执行相同的碰撞测试，改前和改后均通过。离屏启动日志有 DX12 交换链提示，全部正式采样、截图及模拟取图均完成。修前关卡、外围网格及 FBX 已留存备份。[试验日志](../runs/scenes/20261124-backdrop-performance/logs/) · [Windows 工程修前备份](</mnt/d/Program Files/Epic Games/Project_5.2/project_520/Saved/Backups/CoastalHarbor/BeforeBackdropPerformance>)

## 2026-11-24 将渔港裁成活动区与 120 米缓冲带

**目标与方法。** 保留 2 km × 2 km 巡查区，在东西南北各保留 120 米缓冲，移除多余场景。Windows UE 5.2.1、Blender 3.6.23、RTX 4060 Ti。对正式 FBX 按世界 XY 边界切割外围地形、田块和街道，收短三条越界道路及计划中的中心线。海面从 60 km 收到 2.24 km 宽，保留原水位及世界空间材质。移除两组外围植被，共 3,540 株，保留的 575 株逐一比对了树种和变换。天空球继续负责绘制天空。一些树冠自然伸出边缘数米；从高处向外俯看能看到地表截断。[重载边界清单](../runs/scenes/20261124-activity-crop/after/bounds.json) · [俯视实景](../runs/scenes/20261124-activity-crop/activity-area.png)

**几何与使用验证。** 当前外围地形、街道、田块分别有 4,951、33,152 和 1,441 个三角形。对这三份网格中包围框与活动区相交的三角形逐一配对，材质槽一致，角点偏差不超过 0.14 毫米。首次使用按毫米取整的哈希检查，FBX 往返的浮点舍入使道路哈希不一致，改用一对一匹配和角点最大误差。独立 UE 进程重载正式地图，280 个巡查目标的标签、网格、材质和变换全部与裁切前一致。原 31 条道路与 5 个出生点通过碰撞检查，外边界四侧的 4 个探针均未命中地表或海面。另从原始布局输入完整运行规划脚本，保留地块位置、装饰和树种与已部署计划一致，生成的地表顶点均在边界内。新生成与直接裁切的网格在边缘处有不同的三角划分。[网格测量](../runs/scenes/20261124-activity-crop/geometry-check.json) · [生成复验](../runs/scenes/20261124-activity-crop/rebuild-check.json) · [UE 探针](../runs/scenes/20261124-activity-crop/after/scene.json)

**航线采样。** 在同一台机器、同样的离屏编辑器主视口和港口至内陆的 201 个相机位姿上，新旧版本各独立启动，预热后重复 3 次。每趟去掉首尾后取 187 个位姿，分别合并 561 帧。GPU 帧耗时中位数由 10.98 ms 变为 9.24 ms，整帧耗时中位数均约 33.33 ms。高处向界外观察时两版显示的地形范围不同；本轮未测 UE 进程单独的显存节省量。独立运行 ProjectAirSim，在鱼市水平距离 14 米的机位采到 3 组同步且递增的前向 RGB-D，取图超时为 0。[裁切及性能数据](../runs/scenes/20261124-activity-crop/checks.json) · [无人机取图](../runs/scenes/20261124-activity-crop/simulation/check.json) · [Windows 工程备份](</mnt/d/Program Files/Epic Games/Project_5.2/project_520/Saved/Backups/CoastalHarbor/BeforeActivityCrop>)

## 2026-11-24 港池移动视角闪白排查

在用户 60 fps 录屏的第 497、498 帧，水域采样的灰度中位数由 116 升至 171，两个陆地采样点保持稳定。用 Windows UE 5.2 编辑器在港口上方约 482 米的近似机位小幅转动视角，每组取 26 张普通视口截图。以同一机位主动隐藏海面的画面为参照，比较港池开阔水域的 RGB 平均绝对差，低于 3 计为水面消失。[录像采样参数](../runs/scenes/20261124-water-flash-diagnosis/recording.json) · [原录像第 497 帧](../runs/scenes/20261124-water-flash-diagnosis/video-497.png) · [第 498 帧](../runs/scenes/20261124-water-flash-diagnosis/video-498.png)

默认设置两轮各有 13/26 张水面消失，临时关闭遮挡查询后为 0/26。保持遮挡查询开启，仅将 HarborWater 静态网格组件的 Bounds Scale 从 1 调至 2，水面消失为 0/26。该结果验证了测试机位的遮挡剔除问题，未覆盖完整飞行航线和模拟相机。试验只在内存中改参数，结束后正式地图的 SHA-256 与裁切后的记录相同。[逐帧结果](../runs/scenes/20261124-water-flash-diagnosis/controls.json) · [组件对照日志](../runs/scenes/20261124-water-flash-diagnosis/targeted.log)

## 2026-11-24 渔港水面闪白与地形切口修复

在 Windows UE 5.2.1 正式关卡中，用潮滩等高线生成水面，保留 −2.65 米水位、原海面材质与碰撞，将 HarborWater 的 Bounds Scale 设为 2。使用 Blender 3.6.23 从正式地形提取两侧及内陆端的外缘，生成贴合坡面的岩质侧壁，底边为 −6 米。水面水平投影从 5.018 减至 2.012 平方公里，420 段岸线中点处水边均藏入潮滩约 15 厘米；三个方向的侧壁逐段测得最大水平接缝为零。首次复核受 FBX 往返浮点误差影响，将近似相同的列当成不同的列，导致个别高度探针落空。将横坐标按毫米归并后复核通过。另从地形规划入口重新导出两种网格。[几何复核](../runs/scenes/20261124-water-edge/geometry.json) · [正式文件哈希](../runs/scenes/20261124-water-edge/files.json) · [地图备份](</mnt/d/Program Files/Epic Games/Project_5.2/project_520/Saved/Backups/CoastalHarbor/BeforeWaterEdge/CoastalHarbor.umap>)

用独立 UE 进程从磁盘重载。保持全场遮挡查询开启，在上次复现问题的港口俯视机位小幅转动，26 张普通主视口截图中水面消失次数从 13 降到 0。以主动隐藏水面的同位姿截图为参考，像素判据与上次一致。通过目标签名记录精度核对了全部 280 个目标的模型、变换、标签及材质。31 条道路、5 个出生点、575 棵树、3 处屋顶和界外探针通过原有关卡检查。5 个海上探针均命中原水位，10 个内陆点均无水面，3 个水平探针分别命中三侧边壁。另起游戏进程，在东侧搁浅船机位用 ProjectAirSim 取得三组同步前视 RGB-D，取图超时为零。离屏编辑器启动时记录了 DX12 交换链提示，截图与取图均完成。[移动视口采样](../runs/scenes/20261124-water-edge/frame-check.json) · [边缘探针](../runs/scenes/20261124-water-edge/edge-probes.json) · [场景复核](../runs/scenes/20261124-water-edge/check.json) · [RGB-D 记录](../runs/scenes/20261124-water-edge/simulation/check.json)

在 RTX 4060 Ti 8 GB 上重复港口至内陆的 201 个机位、全线预热和 3 趟采样，按上次相同规则保留 561 帧。与裁切后的上一轮独立进程比较，GPU 帧耗时中位数从 9.2446 变为 9.3065 毫秒，整帧从 33.3263 变为 33.3121 毫秒。本轮观察到的改善是海面稳定和边缘闭合；未测专用显存和多机吞吐。[航线性能及原始 CSV](../runs/scenes/20261124-water-edge/performance/) · [港池画面](../runs/scenes/20261124-water-edge/views/saved-19.png) · [侧面画面](../runs/scenes/20261124-water-edge/views/east-00.png)

## 2026-11-24 住宅沿既有街道成排

Windows UE 5.2.1、Blender 3.6.23。以正式港镇 362 个地块为基准，在各街侧按路口、急弯、非住宅地块及长空档分组。对同排建筑按宽度及 8 米侧院间距定位，临街退让目标为 5 米，受障碍约束处放宽组内间距。290 栋住宅都保留原街道、街侧及建筑模型，中位平移 8.15 米。226 栋形成 73 排，另 64 栋独立占地。按地块沿街投影估算，同排相邻住宅侧间距的第 10、50、90 百分位由 9.96、19.68、41.75 米变为 7.89、8.00、8.09 米。362 个建筑地基逐对检查没有相交，住宅地基与道路及人行道没有重叠。31 条路的中心线、其他建筑和车辆的布局、田间及港区布置、植被计划均与原布局一致。[布置测量](../runs/scenes/20261124-residential-rows/layout-check.json) · [住宅俯视](../runs/scenes/20261124-residential-rows/district.png) · [街面实景](../runs/scenes/20261124-residential-rows/street.png)

首次规划将入户路生成提取后遗漏了鱼摊使用的门前坐标，规划在写入工程前终止，补回坐标后完整生成并复跑，规划与几何文件逐字节一致。先在独立副本复验，用户关闭占用文件的 Windows 编辑器后备份并写回正式工程。独立重载正式地图，所有 290 栋住宅的底面与地基碰撞高度相差小于 0.01 毫米，31 条道路、5 个出生点及 32 个重点目标通过既有检查，三处主视口已实拍。280 个巡查目标的标签、网格及材质一致，210 个非住宅目标的变换也一致。同步了 70 栋编号住宅的坐标。离屏编辑器有 DX12 交换链提示，截图正常完成。本轮未测 GPU 性能或无人机飞行。[正式地图核验](../runs/scenes/20261124-residential-rows/scene-check.json) · [地基探针](../runs/scenes/20261124-residential-rows/foundations.json) · [目标对照](../runs/scenes/20261124-residential-rows/target-check.json) · [修前备份](</mnt/d/Program Files/Epic Games/Project_5.2/project_520/Saved/Backups/CoastalHarbor/BeforeResidentialRows-9k6qVj>)

## 2026-11-24 渔港不规则外围与住宅聚落

Windows UE 5.2.1、Blender 3.6.23、RTX 4060 Ti。以用户选定的草案多边形裁切外围地形、海面、道路与田块，用地形真实切边生成岩质侧壁。陆地与海面水平投影的并集与草案轮廓的对称差小于 0.01 平方米。外轮廓东西 2,605.6 米，南北 2,601.5 米，面积 5.4176 平方公里，完整覆盖任务方形。保存后从磁盘重载，在曲边均匀取 64 个位置，分别向内、外各偏移 3 米进行竖向探测，64 个内点均命中地形或水面，64 个外点均无命中。全景对照使用相同 2,880 米宽的正交机位；展示图抹去地图外黑色区域上的 UE 天空提示，保留原始截图。[几何测量](../runs/scenes/20261124-irregular-districts/measurements.json) · [边缘探针](../runs/scenes/20261124-irregular-districts/edge-check.json) · [原始全景](../runs/scenes/20261124-irregular-districts/views/after-raw.png)

290 栋住宅分配到五个片区，每区 56 至 59 栋，55 排每排 3 至 8 栋；相较上一版，216 栋平移超过 50 米，中位平移 84.3 米。31 条道路在活动区内的中心线与上版逐条相符。重新测量 290 个地基，全部与房屋底面贴合；建筑地块间无重叠，住宅地块未压道路。280 个目标的网格、材质、标签一致，210 个非住宅编号目标的变换一致。已同步 70 个编号住宅的对象表坐标。海面遮挡测试在原俯视机位转动 26 帧，均与隐藏水面参照有显著差异，消失为 0 帧。离屏编辑器启动有 DX12 交换链提示，测量和截图完成。[关卡复核](../runs/scenes/20261124-irregular-districts/scene-check.json) · [地基](../runs/scenes/20261124-irregular-districts/foundations.json) · [海面逐帧结果](../runs/scenes/20261124-irregular-districts/water-flash.json) · [Windows 修前备份](</mnt/d/Program Files/Epic Games/Project_5.2/project_520/Saved/Backups/CoastalHarbor/BeforeIrregularDistricts-FhyFFd>)

单次独立启动编辑器，沿港口至内陆的 201 个机位预热后运行三趟，统一取每趟 187 帧。561 帧的 GPU 耗时中位数为 9.80 毫秒，整帧中位数为 33.32 毫秒。未单测显存、多机或 RGB-D 取图。[性能采样](../runs/scenes/20261124-irregular-districts/performance/)


## 2026-11-24 沿海小镇风损的空中可见性

**目的与环境**　在正式 `project_520/CoastalHarbor` 的住宅街、工作码头和潮滩增强风暴过后的外观证据。Windows UE 5.2、Blender 3.6、2 km × 2 km 活动区。先备份地图和编号表，在独立副本验证写入，再在正式工程复拍。选取模型朝向和暴露区域时使用固定随机种子 11944。9 个相同机位各拍改前、改后 1280 × 720 图像，含住宅街、编号住宅屋面、码头碎片和搁浅拖船。[机位](../runs/scenes/20261124-wind-aftermath/views.json) · [近景机位](../runs/scenes/20261124-wind-aftermath/close-views.json) · [对照图](../runs/scenes/20261124-wind-aftermath/comparison.jpg)

**几何与标注**　环境住宅局部揭顶 34 栋，成片剥落 39 栋。替换三栋已标损住宅的屋面和一艘已标损拖船的桅杆。64 组屋面碎片、断枝、码头残件和漂木通过 UE 碰撞查询落地，重载后实放位置与记录相符。原尺寸编号住宅网格按 0.2 m 间隔垂直采样，约 13.88 m² 的采样范围内，表面比原屋顶低 5–170 cm。拖船网格弯折后最高点下降 1.867 m，此值为源模型尺寸。280 个编号对象的位置、分类和损伤状态逐条相同，仅四项更新了网格名与可见证据描述。32 个重点对象仍包含 19 个损伤样本和 13 个完好对照。[几何实测](../runs/scenes/20261124-wind-aftermath/storm-check.json) · [编号差异](../runs/scenes/20261124-wind-aftermath/catalog-check.json) · [正式地图校验](../runs/scenes/20261124-wind-aftermath/files.json) · [引擎日志](../runs/scenes/20261124-wind-aftermath/logs/live-check_live.log)

**性能与相机**　复用上一版的视口路线、预热、关垂直同步和取样窗口，连续跑三遍，每版统计 561 帧。改前、改后 GPU 帧耗时中位数分别为 9.803 和 9.792 ms，P95 为 19.232 和 19.749 ms，绘制调用数中位数从 888 变为 1117。单条路线的比较未覆盖所有视角及显存峰值。ProjectAirSim 在搁浅拖船机位收到 3 组同时间戳 RGB、深度帧，零超时，保存帧的有效深度比例 100%。此次验证覆盖视觉和静态碰撞，风力和破坏程度采用合成设定。[路线性能数据](../runs/scenes/20261124-wind-aftermath/performance/summary.json) · [无人机 RGB 图](../runs/scenes/20261124-wind-aftermath/simulation/front-rgb.jpg) · [RGB-D 检查](../runs/scenes/20261124-wind-aftermath/simulation/check.json)

**归档复核**　使用 airsim 环境和 Windows 工程保留的规划、对象尺寸与编号表，离线重算风损布局，与已保存的 73 栋受损环境住宅和 64 组散落物的布局逐项一致。核对了前视 RGB-D 分辨率、280 个唯一编号、32 个重点标签及首轮试跑候选。此复核未重新启动 UE 或执行飞行。[风损布局](../runs/scenes/20261124-wind-aftermath/wind-layout.json) · [巡查目标表](../data/metadata/coastal_harbor/targets.json)

## 2026-11-25 沿海渔港三类任务首轮实飞闭环验证

**目的**　按[双场景试验方案](PLAN.md#2026-11-24-双场景灾损复查试验方案)，在正式 `CoastalHarbor` 关卡中完成三类首轮试跑任务的端到端闭环验证与对照评估：（1）指定部位二次抵近观测三条件对照；（2）仓库屋顶破口尺寸量取与入室/受阻核查；（3）连续仓库屋檐沿线巡检与米制损伤起止边界定位。

**环境与参数**　Windows UE 5.2（`CoastalHarbor` Play 模式），WSL `airsim` 环境经 `172.21.192.1:8989` 连接 ProjectAirSim；本地 RTX 4060 Ti 运行冻结 DINOv2 ViT-S/14 实时视觉伺服；237 服务器 Ollama 提供视觉语言模型（`qwen3.5:9b` / `qwen3.5:27b`）。运行时在内存中为 `FrontCamera` 开启深度流推送（不改原 `robot_coastal_harbor_sensors.jsonc`），离线回归测试 `test_harbor_part_recheck.py`、`test_harbor_indoor_entry.py`、`test_harbor_line_inspection.py` 共 **7/7 通过**。目标真值坐标与部位法向仅用于起点准备与事后独立几何评分，不进入在线视觉控制、绕飞圆心估算或 VLM 提示词。

### 任务 1：指定部位二次抵近观测（4 目标 × 2 工况 × 3 条件）

**起点与独立判据**　选用受损渔船 `Vessel_037`、完好渔船 `Vessel_040`、受损鱼市 `Building_071` 和完好商铺 `Building_078`（4 个独立目标），每个目标设置 1 个无遮挡直视起点（`direct`）和 1 个对侧/背面遮挡起点（`occluded`），经 8 起点实拍拼图与几何检查全部核实通过（`direct` 4/4 `vis=True`，`occluded` 4/4 `vis=False`）。独立几何拍清判据事前冻结为：指定部位投影入画（边距 ≥ 4%）+ 正对法向夹角 ≤ 75° + 无前景深度遮挡（差 ≤ 2.5 m）+ 距部位 ≤ 15.0 m 且等效跨度 ≥ 100 px。三种条件共用逐字节相同的首轮远景图，两种飞行条件使用相同起点、时间与取图预算（上限 3 张）。[起点核验图](../runs/harbor/20261125-task1-starts-check/starts_grid.jpg) · [起点检查](../runs/harbor/20261125-task1-starts-check/starts_check.json) · [任务配置](../configs/missions/coastal_harbor_part_recheck.json)

**首轮调试试跑（`qwen3.5:9b --no-think`，`allow_climb=true`，暴露并定位四项工程问题）**　24 组试验 0 碰撞完成，但全部条件指定部位拍清率均为 0/8。逐帧排查确认三项原因：（a）`allow_climb=true` 覆盖了下滑道下降分量，导致无人机沿水平甚至微爬升轨迹从目标头顶掠过；（b）渔船初始中心框偏高将船后黑礁石纳入原型，第 0 步即锁定 37.6 m 外背景石头；（c）商铺 2/2 正确触发 180° 绕飞，但深度只测背面外墙使圆心偏浅 4–5 m；且 `qwen3.5:9b` 在渔船 22 m 远景下无法稳定分辨左右舷。[首轮报告](../runs/harbor/20261125-task1-qwen9b/report.md) · [首轮近景拼图](../runs/harbor/20261125-task1-qwen9b/final_closeups_grid.jpg)

**二轮修正实飞（`qwen3.5:27b --no-think`，下滑道 `allow_climb=false` + 绕飞半厚补偿 + 中心自动放大图）**　16 次视觉伺服飞行 **16/16 全部 `range_confirmed` 收敛、0 空中碰撞**；VLM 在 8 个起点上的主动视角决策达到 **8/8（100%）**（`direct` 4/4 选直接抵近，`occluded` 4/4 全部触发无真值深度估算 180° 绕飞换位）。

| 工况 | 条件 | 试验数(目标) | 指定部位拍清率 | 停距误差(m) | 最佳部位距(m) | 平均飞行耗时(s) | 平均张数 | 碰撞 | 复评正确率(带中心裁图 / 纯全画幅) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 无遮挡(`direct`) | A. 只用首轮远景 | 4 (4) | 0/4 (0.0%) | — | 22.14 | 0.0 | 1.00 | 0 | 1/4 (25.0%) / 0/4 (0.0%) |
| 无遮挡(`direct`) | B. 预设直飞复查 | 4 (4) | **4/4 (100.0%)** | 2.50 | 8.99 | 21.9 | 2.00 | 0 | 2/4 (50.0%) / 1/4 (25.0%) |
| 无遮挡(`direct`) | C. 主动选视角 | 4 (4) | **4/4 (100.0%)** | 1.85 | 8.84 | 26.0 | 2.00 | 0 | 2/4 (50.0%) / 1/4 (25.0%) |
| 有遮挡(`occluded`) | A. 只用首轮远景 | 4 (4) | 0/4 (0.0%) | — | 28.21 | 0.0 | 1.00 | 0 | 0/4 (0.0%) / 0/4 (0.0%) |
| 有遮挡(`occluded`) | B. 预设直飞复查 | 4 (4) | **0/4 (0.0%)** | 1.32 | 14.48 | 22.9 | 2.00 | 0 | 2/4 (50.0%) / 0/4 (0.0%) |
| 有遮挡(`occluded`) | C. 主动选视角 | 4 (4) | **3/4 (75.0%)** | 2.35 | 16.32 | 69.7 | 3.00 | 0 | 2/4 (50.0%) / 2/4 (50.0%) |
| 全部合计(`all`) | A. 只用首轮远景 | 8 (4) | 0/8 (0.0%) | — | 25.18 | 0.0 | 1.00 | 0 | 1/8 (12.5%) / 0/8 (0.0%) |
| 全部合计(`all`) | B. 预设直飞复查 | 8 (4) | 4/8 (50.0%) | 1.91 | 11.74 | 22.4 | 2.00 | 0 | 4/8 (50.0%) / 1/8 (12.5%) |
| 全部合计(`all`) | C. 主动选视角 | 8 (4) | **7/8 (87.5%)** | 2.10 | 12.58 | 47.8 | 2.50 | 0 | 4/8 (50.0%) / 3/8 (37.5%) |

- **失败与局限诚实记录**：（1）`Vessel_037 occluded` 的 `active_viewpoint` 已 180° 绕到左舷（中继图 `vis=True, ang=20.6°`），但绕飞落点距船 30.1 m 偏远，末段 DINO 锁定偏移至沙滩未拍清；（2）`qwen3.5:27b` 终评在零误报（FP=0）的同时，对清晰近景中的局部微损（`Building_071` 右端条纹雨棚下垂、`Vessel_037` 左舷白色船壳局部凹陷）均出现漏报（判为 `ruled_out` 或 `still_uncertain`）；离线测试开启 `--think` 后单次推理达 71–195 s 并超时，且 `Vessel_037` 仍漏报。这表明当前瓶颈已从“相机取证/控制”转移到“开源 VLM 对低对比度微损的细粒度定损召回”。[二轮报告](../runs/harbor/20261125-task1-v2-qwen27b/report.md) · [二轮汇总 JSON](../runs/harbor/20261125-task1-v2-qwen27b/summary.json) · [全画幅离线对照](../runs/harbor/20261125-task1-v2-qwen27b/reassess_27b_fullframe.json) · [二轮近景拼图](../runs/harbor/20261125-task1-v2-qwen27b/final_closeups_grid.jpg)

### 任务 2：海港仓库入室勘测（`Building_098` 与 `Building_095` 破口实测及受阻核查）

**方法与真值**　两栋仓库当前共用 `ShedTornRoof`（31×19×11.4 m），缺失板 `j=4` 理论净宽 5.02 m × 9.50 m = 47.69 m²，但内部保留两根 0.12 m 水平钢檩条，将开口分割为最大 3.39 m 格室（理论最大无障碍内切圆净空半径 `1.695 m`）。机体最大外接半径 `0.472 m`。在 UE Z=34 m 正俯视 RGB-D 下独立量取破口几何，并分别测试 `Building_095`（高余量阈值 `2.0 m`）受阻中止与 `Building_098`（`1.60 m` 阈值）实飞穿洞。[任务配置](../configs/missions/coastal_harbor_indoor_entry.json) · [执行脚本](../src/harbor/indoor_entry.py)

**结果**
- **破口几何测量**：深度图成功检出 12 cm 细钢檩条与木椽子。`Building_098` 实测最大净空半径 **`1.682–1.691 m`**（误差仅 `4–13 mm`），宽 `5.86–5.95 m`（含掀翻板缝隙，误差 `0.84–0.93 m`），缺失面积 **`43.09–43.44 m²`**（相对误差 **`8.9%–9.6%`**）；`Building_095` 实测最大净空半径 **`1.660–1.729 m`**，面积 `42.20–42.95 m²`。
- **受阻中止（`Building_095`，阈值 `2.0 m`）**：进入前即判定 `1.66 m < 2.00 m`，执行 `blocked_abort` 安全中止，未尝试穿洞。
- **原状穿洞实飞核查（`Building_098`，阈值 `1.60 m`）**：先后以 `min_floor_guard_m = 2.8 m` 和 `1.1 m` 两次实飞，无人机均精准水平对准中间格并开始垂直下降，但分别在屋面上方 `15.95 m` 和 `14.17 m` 处因两侧水平钢檩条进入中央 40% 安全视锥触发 `floor_depth_guard`（`guard_depth = 1.02 m`）安全拦停，**0 碰撞**，但**未进入室内**（VLM 独立核验 `inside_building = false`）。
- **结论与未验证项**：实飞证实原状 `ShedTornRoof` 因两根钢檩条横跨确属物理受阻样本；待后续离线导出移除 `Building_098` 缺失格中间两根钢檩条的网格变体（净空半径扩至 `2.51 m`）后，再补跑同阈值下 `Building_098`（真入室+室内取证+退出）与 `Building_095`（原状受阻）物理双例对照。[测量记录](../runs/harbor/20261125-task2-measure/summary.json) · [098 俯视叠加](../runs/harbor/20261125-task2-measure/Building_098/nadir_breach_overlay.jpg) · [098 首次入室尝试](../runs/harbor/20261125-task2-entry-passable/summary.json) · [098 入室 v2 核查](../runs/harbor/20261125-task2-entry-v2/summary.json) · [095 受阻记录](../runs/harbor/20261125-task2-entry-blocked/summary.json)

### 任务 3：沿线连续巡检（`Building_098` 32.2 m 连续仓库屋檐）

**方法与真值**　沿 `Building_098` 临码头侧 32.2 m 连续屋檐线（完好段 `0.00–16.10 m` → 受损掀翻/缺失段 `16.10–26.83 m` → 恢复完好段 `26.83–32.20 m`），无人机在距屋檐法向 `16.5 m`、高度 `22.0 m`、相机下俯 `-38°` 处以 `2.2 m/s` 从 `s = 0` 连续飞至 `s = 32.2 m`，每 `2.3 m` 采集同步 RGB-D（共 15 帧），由 `qwen3.5:27b --no-think` 逐帧输出屋面受损水平起止比例并经相机外参反投影至参考线融合投票。[任务配置](../configs/missions/coastal_harbor_line_inspection.json) · [执行脚本](../src/harbor/line_inspection.py)

**结果（一次通过、0 碰撞）**：飞行耗时 **`34.28 s`**，采集 `15` 帧，空中碰撞 **`0`**；**有效覆盖率 `100.0%`**（每 0.5 m 区段平均被 `9.27` 帧覆盖），**漏拍区段数 `0`**；损伤区间真值 `[16.10, 26.83] m`，实测估计 **`[15.50, 27.00] m`**，**起点误差 `0.60 m`、终点误差 `0.17 m`、损伤区间 IoU `0.9565`**。未验证项：跨多栋连续建筑更长线路及含多段离散损伤的巡检线。[巡检报告](../runs/harbor/20261125-task3-line-qwen27b/report.md) · [汇总 JSON](../runs/harbor/20261125-task3-line-qwen27b/summary.json) · [15 帧追踪拼图](../runs/harbor/20261125-task3-line-qwen27b/line_inspection_grid.jpg)

### 机载独立模式（零服务器依赖，UE 联机实飞验收）

为支持后期真机部署（不依赖 237 大显存服务器评估，保留原有 VLM 模式不变），三类海港任务已统一补齐纯机载独立模式，并在 Windows UE `CoastalHarbor` 中完成零服务器联机实飞验收：

- **任务 1（`--planner template --vlm none`，16 次实飞、0 碰撞、零服务器）**：用本机 22M 冻结 DINOv2（显存 `< 0.5 GB`）做多尺度空间特征格互相关 + 水平翻转自检（`TemplateViewpointPlanner`，单次视角决策仅 `48–161 ms`）。16 次视觉伺服 **16/16（100%）`range_confirmed` 收敛**：
  - **有遮挡（`occluded`）**：本机模板规划 **4/4（100%）** 准确触发无真值 180° 绕飞换位，指定部位拍清率从不换位直飞的 **`0/4 (0%)`** 跃升至 **`3/4 (75%)`**（追平 27B 服务器版，且平均耗时由 `69.7 s` 降至 **`58.0 s`**）；
  - **无遮挡（`direct`）**：`preset_recheck` `4/4 (100%)`，`active_viewpoint` `3/4 (75%)`；全工况合计 **`6/8 (75%)`** vs `preset_recheck` `4/8 (50%)`。
  - **诚实记录 2 例失败**：`Vessel_037 occluded` 绕飞后中继距船 `27.7 m` 致末段未锁船；`Building_078 direct` 因 `Shop4` 正立面本身左右完全对称，悬停微抖下水平翻转优势略低于 `0.012` 阈值误触发翻面（后续对左右对称建筑宜加中继前后分数对比复核，且当前阈值 `0.58 / 0.012` 仅在首轮 4 目标标定，留出泛化待测）。参考模板在 `data/templates/coastal_harbor_pilot/`。[独立实飞报告](../runs/harbor/20261125-task1-standalone-template/report.md) · [汇总 JSON](../runs/harbor/20261125-task1-standalone-template/summary.json) · [近景拼图](../runs/harbor/20261125-task1-standalone-template/final_closeups_grid.jpg)
- **任务 2（`--vlm none`）**：纯 RGB-D 深度几何完成破口测量、净空判定、对准穿洞与退出，零服务器调用（详见上文任务 2 实测）。
- **任务 3（`--detector rgbd`，在线实飞、0 碰撞、零服务器、0 GPU 显存）**：沿 `Building_098` 32.2 m 屋檐线实时采集 15 帧 RGB-D，飞行耗时 **`32.84 s`**，单帧深度平面残差连通段检测仅 **`3.9–8.1 ms`**。**有效覆盖率 `100.0%`**，**漏拍 `0`**，损伤区间估计 **`[16.00, 26.00] m`**（真值 `[16.10, 26.83] m`），**起点误差 `0.10 m`、终点误差 `0.83 m`、IoU `0.9091`**（与离线回放逐项一致）。回归测试共 **15/15 通过**。[实时独立报告](../runs/harbor/20261125-task3-standalone-live/report.md) · [15 帧追踪拼图](../runs/harbor/20261125-task3-standalone-live/line_inspection_grid.jpg) · [离线回放对照](../runs/harbor/20261125-task3-line-standalone-rgbd/report.md)

### 试点收尾修复、任务 2 真入室闭环与全任务双视角录屏（2026-11-25 晚）

针对首轮试点暴露的 3 项缺口完成修复、场景补建与联机实飞复验，并新增轻量单机双视角录屏模块 [recorder.py](../src/harbor/recorder.py)（1280×416 `Chase + FrontCamera` 20 fps H.264，实时叠加 DINOv2 热力图/破口掩码/屋面破损红框与中文状态栏），三个渔港任务脚本均已支持 `--record-video`。

- **任务 1 两个试点缺口修复与复验（全工况 8/8 = 100% 拍清、0 碰撞）**：
  - **修复内容**：（a）`TemplateViewpointPlanner` 默认翻转优势阈值从 `0.012` 改为 `0.0`（仅当水平镜像相关性高于原图即 `sc_flip > sc_orig` 时判为对侧镜像，消除 `Building_078` 左右对称正立面的误翻面）；（b）`estimate_target_center_from_depth` 优先按目标 `template_roi` 采样前景深度，为小船设置 `depth_center_offset_m = 1.5 m` 与 `max_orbit_standoff_m = 20.0 m`，并按 `orbit_r * tan(|pitch|)` 对齐绕飞落点高度，消除 `Vessel_037 occluded` 中继点偏远致小船落出 `template_roi` 上方的问题。
  - **单机无服务器版全 8 工况复飞（`--planner template --vlm none --record-video`）**：`direct` **`4/4 (100%)`**、`occluded` **`4/4 (100%)`**，总计 **`8/8 (100%)` 拍清、0 碰撞**（平均部位距 `8.60 m`，平均停距误差 `1.65 m`）；其中 `Vessel_037 occluded` 中继点精准居中（`v = 538.8 / 1080 px`）、末帧抵近至 `5.84 m`，`Building_078 direct` 直接抵近 `14.6 s` 拍清，全部 8 次飞行均生成可播 H.264 MP4。[复飞报告](../runs/harbor/20261125-task1-standalone-v2-video/report.md) · [Vessel_037 绕后录屏](../runs/harbor/20261125-task1-standalone-v2-video/captures/Vessel_037/occluded/active_viewpoint/flight.mp4)
  - **远端 27B VLM 版单点复测（`Vessel_037 occluded`，`--record-video`）**：绕飞中继距 `19.42 m`、末帧抵近至 `6.32 m`，**`clear = True`、0 碰撞**（同时仍诚实保留其终评对白色船壳微凹陷漏报的原始记录 `verdict = ruled_out`）。[27B 补测报告](../runs/harbor/20261125-task1-v2-qwen27b-vessel037-fix/report.md) · [27B 绕后录屏](../runs/harbor/20261125-task1-v2-qwen27b-vessel037-fix/captures/Vessel_037/occluded/active_viewpoint/flight.mp4)
- **任务 2 物理可通行 vs 受阻同阈值对照与真入室闭环（`min_passable_clearance_radius_m = 1.85 m`，0 碰撞）**：
  - **场景补建**：基于 `harbor_meshes.py` 新增 `clear_passable_bay=True` 变体，用 Blender 3.6 导出移除 `j=4` 缺失格室两根水平钢檩条的 `ShedTornRoof_Passable.fbx`（保留 `j=3` 掀翻板与檩条，[harbor_roof_passable.py](../src/scene_build/harbor_roof_passable.py)），经 UE 5.2 离线命令 [harbor_apply_passable_shed.py](../src/scene_build/harbor_apply_passable_shed.py) 仅赋给 `Building_098`，`Building_095` 保持原状 `ShedTornRoof` 作受阻对照。
  - **实飞结果（同一 `1.85 m` 阈值）**：
    - `Building_098`（可通行正样本）：顶视测得最大净空半径 **`2.484 m`**（理论真值 `2.51 m`，误差 `2.4 cm`）、面积 `44.41 m²`（误差 `6.88%`）→ 判定 `enter` → RGB-D 实时对准并垂直穿顶入室至 **`alt = 6.04 m`（`stop = interior_reached`）** → 室内前视取证（`qwen3.5:27b` 独立确认 `inside_building = true`, `roof_damage_confirmed = true`，识别出内墙、地面与屋顶破口直射光线）→ **原路垂直安全退出（`exited = true`）**，全程 `53.11 s`、**0 碰撞**，生成 `36.75 s` H.264 双视角录屏。
    - `Building_095`（原状受阻对照）：同阈值下测得最大净空半径 **`1.775 m < 1.85 m`**（面积 `47.56 m²`，误差 `0.27%`）→ 安全触发 **`blocked_abort`** 拒飞，**0 碰撞**。
    - [对照汇总 JSON](../runs/harbor/20261125-task2-passable-vs-blocked-video/summary.json) · [098 破口叠加图](../runs/harbor/20261125-task2-passable-vs-blocked-video/Building_098/nadir_breach_overlay.jpg) · [098 室内实拍](../runs/harbor/20261125-task2-passable-vs-blocked-video/Building_098/interior_forward.jpg) · [098 穿顶入室录屏](../runs/harbor/20261125-task2-passable-vs-blocked-video/Building_098/indoor_entry.mp4)
- **任务 3 实飞录屏复验（`--detector rgbd --record-video`）**：`35.99 s`、15 帧、**0 碰撞**，有效覆盖率 **`100.0%`**、漏拍 **`0`**、估计区间 `[16.00, 26.00] m`（起点误差 `0.10 m`、终点误差 `0.83 m`、**IoU `0.9091`**），同步生成带实时红色破损框的双视角录屏。[报告](../runs/harbor/20261125-task3-standalone-video/report.md) · [沿线巡检录屏](../runs/harbor/20261125-task3-standalone-video/line_inspection.mp4)
- **回归测试与未验证项**：单元测试 `test_harbor_part_recheck.py`、`test_harbor_indoor_entry.py`、`test_harbor_line_inspection.py`、`test_coordination.py` 共 **17/17 全部通过**。未验证项：留出集 26 个重点目标（含 12 辆车与多栋连排仓库沿线）的跨类别泛化评测，以及微损细粒度 VLM 召回率提升。

## 2026-11-26 实验文件盘点与空间清理

**目的**：按用户确认的 A + B + C 清单清理历史中断目录、废弃重试子目录及无回放依赖的新版重型输入缓存，保留全部正式实验结果、RViz 模板切换/入室回放数据、`.mp4` 录屏、多机协同及沿海渔港二场景数据。本次未运行新实验或修改控制代码。

**清理范围与实测结果（共删除 29,601 个文件、61.67 GiB / 66,216,870,651 字节，磁盘使用率由 26% 降至 20%）**：
- **A 组（零散残余与半截中断目录，3,174 个文件、6.45 GiB）**：清理根目录遗留 `projectairsim_client.log`、空目录 `runs/waypoint/waypoint-flight`、`data/captures/aircraft_a_tail/20261114-exposure-check-2`、`runs/scenes/20261124-boundary-concept`、30 个空目录与 17 个 0 字节日志，以及未被引用的中断目录 `20261119-p1-revalidation/10-hangar-209/repeat-01-incomplete-1795100991`、`20261120-p26-graded-active/10-hangar-209/repeat-01-incomplete-1795144684` 与 `1795148791`（保留被引用的 `18-tower-7889/repeat-01-incomplete-1795151552`）。
- **B 组（旧版过程重试与作废批跑，13,452 个文件、16.91 GiB）**：清理 `runs/approach/20261116-coarse-to-fine/` 中未被 `summary.json` 收录的 57 个过程重试子目录（`892.2 MiB`）、`runs/opening-validation/20261112-alignment-preflight`，以及被 11-19 复验取代的 `runs/evaluation/20261118-airport-handoff-revalidation/` 下 19 个目标原始子目录（保留顶层 `summary.json`、`targets.csv`、`shutdown.json` 等摘要文件）。
- **C 组（新版过程调试与无回放依赖缓存，12,975 个文件、38.31 GiB）**：清理 11-18 未入报告的调试目录（`20261118-aircraft-template-e2e`、`20261118-aircraft-b-instance-verification`、`20261118-aircraft-b-staged-shadow`、`20261118-identity-persistence`、`20261118-new-model-e2e`、`20261118-new-model-e2e-repeat`、`20261118-wsl-gateway-retest/attempt-01`、`20261118-shadow-stability`、`evaluation/20261118-{shadow-stability,active-replay,visible-part-evidence}`），并清理 `runs/evaluation/20261120-p26-graded-active/` 下的 58 份 `.h5`（`active-inputs.h5` 等）与 19 个 `viz_dump/` 目录（保留报告、日志与全套 `.mp4` 录像）。

**保留与限制**：`runs/approach/20261116-coarse-to-fine/` 全部 28 个验收目录（含 `viz_dump/`、`frames/`、`run.h5`、`.mp4`）、`runs/approach/20261119-active-remaining-targets/`、`runs/opening-*`、`runs/indoor/`、`runs/coordination/`、`runs/scenes/` 与 `runs/harbor/` 均完整保留；同时在 [review.py](../runs/visualization/20261119-airport-target-review/review.py) 补齐 `f14`/`f35` 到 `aircraft-a`/`aircraft-b` 的旧档案兼容映射，经 `--batch passed --verify` 与 `--batch active --verify` 对全 19 目标核验通过；已清理目录不再支持离线逐帧回放或重算。

## 2026-11-27 DOB-CBF-QP 安全视觉伺服实现与沿海渔港原生气动风场实飞对照

**目的**：在保持原解耦比例控制器（`baseline_p`）零破坏兼容的前提下，新增统一的 **DOB-CBF-QP**（速度扰动观测器 + 相机交互矩阵 FOV-CBF + RGB-D 障碍与短时盲区记忆 Depth-CBF）安全滤波模块 [cbf_qp_servo.py](../src/control/cbf_qp_servo.py)，完成离线数学/动力学验证，并在 Windows UE `CoastalHarbor` 中启用 ProjectAirSim 原生空气动力学风场（`world.set_wind_velocity`），对比 `baseline_p` 与 `DOB-CBF-QP` 在正常无风及 `18 m/s`（8 级大风）原生风场下的实飞表现。

**环境与参数**：Windows UE 5.2（`CoastalHarbor` Play 模式），WSL `airsim`（NumPy 2.2.6 + SciPy 1.15.3，零新增外部依赖）经 `172.21.192.1:8989/8990` 连接 ProjectAirSim；本地 RTX 4060 Ti 运行冻结 DINOv2 ViT-S/14；`part_recheck.py` 与 `indoor_entry.py` 新增 `--controller baseline_p|cbf_qp`、消融开关及 `--wind-north-mps / --wind-east-mps`（直接调用引擎原生 `world.set_wind_velocity` 并在退出时归零，风速真值不进入控制器），全部实飞同步开启 `--record-video`。

**原生气动风场标定探针（出生点 15 m 悬停，每档 2.6 s 零速指令）**：东向风速 `0 / 6 / 10 / 15 / 20 m/s` 对应的实测漂移为 `0.000 / 0.146 / 0.455 / 1.057 / 2.030 m`（近似风速平方阻力律），真实机身倾角达 `0.00° / -0.53° / -1.48° / -3.36° / -6.41°`。

**离线验证（`python -W error::RuntimeWarning -m unittest`，29/29 通过、0 警告）**：
- `-30° / -45° / -90°` 解析图像雅可比与 3D 针孔投影有限差分误差 `< 1e-4`；
- 斜视激进前飞下，无 FOV-CBF 丢锁、开启 FOV-CBF 全程稳锁画内（最大偏离 `< 0.88`）；
- 斜视大偏角 + 侧方障碍立柱 + 侧风对照：`baseline_p` 逼近障碍至 `0.71 m`，`cbf_qp` 保持 `> 2.2 m` 安全净距到达停机（单步中位 `0.088 ms`）；
- 穿顶入室盲区记忆消融（关闭 DOB 隔离测试）：洞缘移出下视 FOV 后，`baseline_p` 与 `cbf_no_memory` 均撞壁，完整 `cbf_qp`（含 `LocalObstacleMemory`）凭 6 s 体素记忆 0 碰撞安全穿入；
- 独立 DOB 单测验证 `0.55 m/s` 恒定侧风下稳态位置误差由 `~1.96 m` 收敛至 `< 0.15 m`。

**UE5 联机实飞结果（全部走原生 `world.set_wind_velocity`，全程 0 碰撞）**：

| 试验与原生风场工况 | 控制器 | 实飞结果与停机原因 | 关键指标（耗时 / 误差 / QP 延迟 / DOB 估计） | 记录与录屏 |
|---|---|---|---|---|
| 任务 2 无风（`Building_098` + `095`） | `cbf_qp` | `098` 穿顶入室+退出（`interior_reached`, `alt=6.04 m`）；`095` 受阻拒飞（`1.777 < 1.85 m`） | `28.27 s` / 317 步；QP 中位 **`0.354 ms`**（P95 `0.967 ms`）；0 碰撞 | [汇总](../runs/harbor/20261127-task2-cbf-normal/summary.json) · [098 录屏](../runs/harbor/20261127-task2-cbf-normal/Building_098/indoor_entry.mp4) |
| 任务 2 原生东向 `18 m/s` 大风（`Building_098`） | `baseline_p` | **未入室**（`opening_lost`, `alt=34.55 m`） | 最大水平误差被吹至 **`14.753 m`**，505 步 0 下降，最终破口被吹出视野 | [汇总](../runs/harbor/20261127-task2-nativewind18-baseline/summary.json) · [录屏](../runs/harbor/20261127-task2-nativewind18-baseline/Building_098/indoor_entry.mp4) |
| 任务 2 原生东向 `18 m/s` 大风（`Building_098`） | **`DOB-CBF-QP`** | **成功穿顶入室**（`interior_reached`, `alt=6.04 m`，`entered=True, exited=False`） | `34.62 s` / 335 步（87 步逆风对准 + 248 步穿洞）；DOB 在线估出等效漂移 **`[-0.001, 0.948, -0.030] m/s`**；QP 中位 **`0.510 ms`**（P95 `1.918 ms`）；0 碰撞 | [汇总](../runs/harbor/20261127-task2-nativewind18-dob-cbf/summary.json) · [录屏](../runs/harbor/20261127-task2-nativewind18-dob-cbf/Building_098/indoor_entry.mp4) |
| 任务 1 原生东向 `18 m/s` 大风（`Building_071 / direct`，直飞对照） | `baseline_p` | **未收敛**（`duration_limit`, `42.42 s`） | 从 `24.41 m` 被大风**倒吹至 `28.80 m`**（`stop_d = 26.27 m`），0 碰撞 | [报告](../runs/harbor/20261127-task1-nativewind18-baseline/report.md) · [录屏](../runs/harbor/20261127-task1-nativewind18-baseline/captures/Building_071/direct/preset_recheck/flight.mp4) |
| 任务 1 原生东向 `18 m/s` 大风（`Building_071 / direct`，直飞对照） | **`DOB-CBF-QP`** | **逆风收敛停机**（`range_confirmed`, `21.43 s`） | 推进至 **`10.63 m`**（`stop_d = 9.66 m`），DOB 估出 **`[-0.007, 0.665, -0.023] m/s`**，QP 中位 **`0.258 ms`**（P95 `0.916 ms`），0 碰撞 | [报告](../runs/harbor/20261127-task1-nativewind18-dob-cbf/report.md) · [录屏](../runs/harbor/20261127-task1-nativewind18-dob-cbf/captures/Building_071/direct/preset_recheck/flight.mp4) |
| 任务 1 原生东向 `18 m/s` 大风（`Vessel_037 / direct`，直飞对照） | `baseline_p` / `DOB-CBF-QP` | 两者均未达近距拍清（0 碰撞） | `baseline_p` 被吹远至 `36.67 m`；`DOB-CBF-QP` 推进至 `17.26 m`（DOB 估风 `+0.803 m/s`，QP 中位 `0.704 ms`），8 级大风下 `~6°` 持续滚转倾角导致小船 DINO 偏移 | [基线报告](../runs/harbor/20261127-task1-nativewind18-baseline/report.md) · [CBF 报告](../runs/harbor/20261127-task1-nativewind18-dob-cbf/report.md) |

**过程排查、诚实限制与未验证项**：
1. 首次实现 FISTA 对偶求解器时用全 1 向量做幂迭代低估了正负盒约束矩阵的最大特征值，触发数值溢出警告；改为小矩阵精确最大特征值 `eigvalsh(G)[-1] * 1.05` 后在 `-W error::RuntimeWarning` 下零警告通过。
2. 原生 `18 m/s` 首轮实飞中发现 `speed_cap_mps`（`0.7 m/s`）误截断了含抗风前馈的总输出速度，导致进入 `13 m` 近距限速后顶不动 `~0.95 m/s` 气动风偏；改为限制**对地净速度** `v_ground = u_safe + d_hat_body` 后，`Building_071` 立刻在 `21.43 s` 内逆风收敛至 `9.66 m`。
3. 诚实限制：（a）`Building_098` 在 `18 m/s` 原生大风下穿顶入室成功但退出阶段调用未挂 DOB 的 `move_to_position_async` 报 `exited=False`；（b）`Building_071` 停机后切回 `hover_async()` 拍照的 `0.5 s` 内被 8 级大风略微吹偏，使雨棚参考点滑至右画边外 `15 px`（`u=1935.8`）；（c）早期用指令叠加模拟风的 5 组草稿目录已清理，本节仅保留原生风场与无风实测。未验证项：时变阵风、真实 UE 窄通道障碍绕避与海港 26 个留出目标批跑。

## 2026-11-28 Archify 架构诊断、跨包循环依赖解耦与双场景架构图试点

**目的**：使用 Archify v3.3.2 对 `src/` 执行静态架构体检并生成试点交互式架构图；针对体检检出的唯一错误级包循环依赖（`COUP-6ac7e0b75e73`：`control → perception → evaluation → control`，涉及 `training`）完成最小解耦与全量单测回归。

**环境与参数**：本机 WSL（Ubuntu 22.04），原生 Linux Node `v24.22.1` + Archify v3.3.2（`explainer` 设为 `on-demand`）；Python `airsim` 环境执行 `python -W error::RuntimeWarning -m unittest discover -s tests -v`。

**方法**：
- 首次静态体检分析 `src/` 下 144 个 Python 文件（0 解析错误），共检出 181 条诊断（`error: 1`、`warning: 85`、`info: 95`），其中文件级 `coupling/import-cycle = 0`，唯一 `error` 为 `perception/` 与 `training/` 反向导入 `evaluation/` 通用辅助函数导致的包层级闭环。
- 新增零内部依赖叶子模块 [artifacts.py](../src/common/artifacts.py)（迁移 `read`、`write`、`digest`）与 [proposal_utils.py](../src/perception/proposal_utils.py)（迁移 `SOURCES`、`ratio`、`bin_name`、`point_hit`、`crop_coverage`、`grouped`、`box_iou`、`peak_candidates`、`coverage_summary`、`fuse_candidates`）；改写 `active_verification.py`、`shadow_verification.py`、`multicandidate_verification.py`、`train_scene_proposals.py` 的导入，在 `eval_template_selection.py` 与 `eval_proposal_fusion.py` 保留同名重导出以兼容既有调用与单测，并同步更新 `docs/ARCHITECTURE.md` 与 `src/README.md`。
- 同时基于 [candidate.json](../runs/visualization/20261128-archify-demo/diagram/candidate.json) 运行 `archify finalize architecture` 生成双场景试点交互架构图。

**结果**：
- **全量单元测试**：`205/205` 全部通过（耗时 `7.643 s`，`-W error::RuntimeWarning` 零警告）。
- **Archify 代码体检复核**：共分析 `146` 个 Python 文件（0 解析错误），诊断总数由 `181` 条降至 **`176`** 条，其中 **`error` 由 `1` 降至 `0`**（`coupling/cycle = 0`、`coupling/import-cycle = 0`），`warning` 由 `85` 降至 `81`，`info` 为 `95`。
- **双场景试点架构图**：`validate`（`9/9` 通过，0 错误 0 警告）、`deliver`（36 处源码引用核验通过）、`check`（0 错误 0 警告，`properCrossings = 0`）均通过。

**结论、限制与未验证项**：
1. 跨包循环依赖已消除，`src/` 恢复严格单向层级且离线单测零回归。
2. **源码 SHA-256 变更影响**：修改 `eval_template_selection.py`、`eval_proposal_fusion.py`、`active_verification.py`、`multicandidate_verification.py`、`shadow_verification.py` 会改变其源文件哈希，后续若需加载早期机场已冻结验证产物（`frozen.json`）须重新冻结（本次未实测旧冻结产物回放）。
3. **试点架构图待修订与未验证项**：（a）试点架构图将仅用于机场 6 机的多机协同调度中心（`SurveyCenter` / `SurveyStore`）与海港单机任务放在同一任务调度框内，归属暂未拆分，待项目结构稳定后修订；（b）WSL 内未安装 Linux 版 Chrome/Chromium/Edge，`archify browser-check` 状态为 `skipped`（`viewer/chrome-unavailable`），真实浏览器视口渲染效果尚未经自动化实测。

[代码体检工作台](../runs/visualization/20261128-archify-demo/code-analysis/analysis/repo.analysis.html) · [体检原始 JSON](../runs/visualization/20261128-archify-demo/code-analysis/analysis/findings.json) · [试点架构图 HTML](../runs/visualization/20261128-archify-demo/diagram/research-project.architecture.html) · [架构图配置](../runs/visualization/20261128-archify-demo/diagram/candidate.json) · [工件读写工具](../src/common/artifacts.py) · [候选提案工具](../src/perception/proposal_utils.py)
