# 机场视觉抵近实验记录

运行命令与参数见[使用手册](../docs/USAGE.md)，模块、控制律与保护条件见[架构说明](../docs/ARCHITECTURE.md)，配置与回放检查见[项目入口与数据验证](VALIDATION.md)。逐帧数据和完整诊断保存在各实验目录，本文只记录目的、关键参数、结果、失败与限制，以及数据入口。

## 通用口径

- **视觉控制**：冻结DINOv2 ViT-S/14，448输入、32×32特征格、softmax温度0.005，不训练；统计帧只用于计算标准化均值和方差。速度由图像匹配中心与深度生成。
- **坐标准备不计为视觉闭环**：按已知坐标采集模板、布置起点和转场。目标坐标只用于准备和事后评价，无人机位姿另用于高度、范围等独立保护。全部记录都不能证明未知目标坐标下的自主搜索能力。
- **距离分别报告**：停车滤波深度、到参考点、到整机最近表面、到所选部位表面的距离互不替代；深度阈值确认只说明测距满足停车条件。
- **帧率**：录像按30FPS编码并含重复帧，实际独立帧率以`report.json`和`*-frames.csv`为准。
- **热力图**：红色为相似度或softmax相对权重的显示，绿色十字为滤波后的控制中心；两者都不是分割掩膜或校准置信度，见[热力图说明](../docs/USAGE.md#匹配热力图的显示与解读)。
- **成功计数**：部位抵近以`arrival_confirmed=true`、入室以`entered=true`计为完成；预检失败和中断目录保留实际状态，不计为成功。运行目录中各文件的含义见[运行文件](../docs/USAGE.md#运行文件)。
- **起飞接口**：多次准备中`takeoff`（个别为`move`）返回False，每次都由实际爬升与位姿检查确认后继续；原始返回值保留在各`prepare.log`，原因未查明，下文不再逐次重复。
- **收尾状态**：在线试验结束时无人机悬停、客户端断开，未降落或解除电机，另有说明的除外。
- **共同限制**：在线飞行均为同一静态UE机场开发场景中的单次试验，没有留出集、重复成功率、动态障碍或完整机体通道验证；下文的“未验证”只列各项特有的部分。

## 数据位置

| 内容 | 位置 |
|---|---|
| 部位抵近试验 | [runs/approach/](../runs/approach/) |
| 洞口匹配、对准与入室 | [runs/opening-validation/](../runs/opening-validation/)、[runs/opening-alignment/](../runs/opening-alignment/)、[runs/opening-entry/](../runs/opening-entry/)、[runs/indoor/](../runs/indoor/) |
| 采集素材与开发视图 | [data/captures/](../data/captures/) |
| 模板与统计帧 | [data/templates/airport_7890/](../data/templates/airport_7890/) |
| 飞机网格与占用图 | [data/maps/airport_scene/](../data/maps/airport_scene/) |
| 相机帧率（含720p、1080p、采集间隔）与网络诊断 | [debug/camera/](../debug/camera/)、[debug/network/](../debug/network/) |
| 航点飞行 | [runs/waypoint/](../runs/waypoint/) |

包围盒评价读取`data/metadata/targets/`，表面距离使用机场三角面；占用图上的距离不能代替部位的真实距离。`debug/`在`.gitignore`内，不经Git同步。

## 机场单机抵近（StaticMeshActor_7890）

斜俯视起点水平距离100米，相机下俯30°，远段水平速度上限3米/秒，12米内合速度上限0.35米/秒。表中深度为末帧滤波测距，未测的表面距离记为“—”。

| 试验 | 控制周期 | 末帧深度/m | 最近模型表面/m | 结果与记录 |
|---|---:|---:|---:|---|
| 俯视机头 | 451 | 11.95 | — | 深度阈值确认；[报告](../runs/approach/near-nose-dual_1106/report.json) |
| 俯视机头与栅格同步 | 614 | 11.97 | — | 深度阈值确认；[报告](../runs/approach/map-session-1793950632689/report.json) |
| 100米斜俯视机头 | 228 | 11.75 | — | 深度阈值确认；[报告](../runs/approach/oblique100-1793952243221/report.json) |
| 机头近距 | 184 | 3.76 | 3.14 | 4米阈值连续独立帧确认；[记录](../runs/approach/close4-1793954736186/README.md) |
| 正前方机翼 | 606 | 8.06 | 3.87 | 碰撞体接触后停滞，由操作中断；[记录](../runs/approach/wing4-1793956203311/README.md) |
| 翼外侧机翼 | — | — | — | 远、近景校验未通过，未执行闭环；[记录](../runs/approach/wing4-outboard-20261106/README.md) |

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

## F-14部位抵近（StaticMeshActor_7888）

以下五项共用的设置：WSL经`172.21.192.1`连接，相机下俯35°，远段前向上限1.5米/秒，12米内合速度上限0.35米/秒，6米深度停车并要求3张独立深度帧确认。终点评价在实际相机姿态下采一张新图，从DINO中心发出射线，与飞机三角面求交，判断是否命中所选部位，这项几何只用于事后评价。共同起点的状态准入容差为位置0.35米、航向0.75°、速度0.08米/秒、倾斜1°。表中起点一致性是仿真记录的一致性，不是定位精度。部位选取与两级匹配见[架构说明](../docs/ARCHITECTURE.md#f-14部位抵近)，命令见[使用手册](../docs/USAGE.md#同一f-14起点抵近不同部位)。

### 2026-11-12：按部位分别准备

每个部位单独采集35、8、60米视角的模板与预检帧，从各自的60米起点出发，时限600秒，主翼与垂尾的ROI预检容差为0.02。三个部位都以`range_threshold`完成，没有新碰撞。这三项是分别重载和准备的，不是连续的自主转场。

| 部位 | 控制观测 | 停车滤波深度/m | 到所选部位表面/m | 实际独立帧率/FPS |
|---|---:|---:|---:|---:|
| 机头 | 318 | 5.839 | 5.591 | 12.33 |
| 主翼 | 366 | 5.817 | 4.865（任意表面4.861） | 12.11 |
| 垂尾 | 358 | 5.704 | 5.457 | 14.13 |

RGB与深度最大时差为84—110毫秒，39项测试通过。离线复核命令：`python runs/approach/20261112-f14-parts/evaluate_part.py nose --offline`（`wing`、`tail`同理）。见[汇总](../runs/approach/20261112-f14-parts/summary.json)、[三处核验图](../runs/approach/20261112-f14-parts/summary.png)、[轨迹图](../runs/approach/20261112-f14-parts/trajectories.png)、[部位选区](../runs/approach/20261112-f14-parts/part-definitions.json)、[场景核验](../runs/approach/20261112-f14-parts/preflight.json)、[测试日志](../runs/approach/20261112-f14-parts/tests.log)；各部位的报告、终点核验与录像在`nose/`、`wing/`、`tail/`子目录中。

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

见[汇总](../runs/approach/20261113-f14-common-start/summary.json)、[轨迹图](../runs/approach/20261113-f14-common-start/summary.png)、[三列DINO视频](../runs/approach/20261113-f14-common-start/matching-comparison.mp4)、[几何预检](../runs/approach/20261113-f14-common-start/geometry.json)、[参考ROI](../runs/approach/20261113-f14-common-start/reference/template-location.png)、[参数退出日志](../runs/approach/20261113-f14-common-start/nose/preflight-argument-failed.log)、[匹配重建核验](../runs/approach/20261113-f14-common-start/matching-reconstruction.json)、[测试日志](../runs/approach/20261113-f14-common-start/tests.log)；各部位的报告与终点核验在子目录中。

### 2026-11-13：正前上方起点

后面三项共用正前上方起点NED`(556.8,178.88,-32.0)`、航向−90°（从机头方向的斜上方看向飞机，不是低空平视），时限180秒。三项的变化依次是：参考图来源、窄ROI的原型方法。

| 试验 | 参考 | 机头 | 近侧主翼 | 一侧垂尾 |
|---|---|---|---|---|
| A 跨视角 | 侧前方参考，逐字节复制 | 完成：151观测，5.783 / 5.419米 | 完成：216观测，5.793 / 4.890米 | 起始落点错误，预检拒绝 |
| B 同视角 | 正前上方新采参考，统计不变 | 完成：132观测，5.773 / 5.321米 | 完成：192观测，5.522 / 3.700米 | 参考原型为0格，预检拒绝 |
| C 窄ROI精修 | 同B | — | — | 完成：193观测，5.739 / 5.342米 |

表中“a / b米”为停车滤波深度和到所选部位表面的距离。完成的各项均以`range_threshold`停止，正式控制中没有新碰撞或其他保护介入。被拒绝的垂尾没有抵近轨迹、终点或录像，但准备阶段的起飞和复位飞行已经执行。

**A 跨视角**。默认的全部位预检中，垂尾落点错误，导致整次被拒绝。为了独立评价每个部位，新增`--preflight-selected-part-only`：三个部位的诊断都保存，但只要求所选部位通过，阈值不变；44项测试通过。在同一张起始图上，机头、主翼、垂尾的峰值分别约为0.649、0.694、0.610，垂尾的响应点落在选区下方，可见单看峰值不能判断匹配是否正确。垂尾投影宽约6.31像素，只占约0.285个特征格。这里的困难与正面薄边成像和跨视角外观变化都相符，两者的影响未分离。实际独立帧率为19.55和18.37FPS。见[汇总](../runs/approach/20261113-f14-front-start/summary.json)、[轨迹图](../runs/approach/20261113-f14-front-start/summary.png)、[起始响应](../runs/approach/20261113-f14-front-start/start-matching.png)、[全部位拒绝日志](../runs/approach/20261113-f14-front-start/nose/preflight-all-parts-rejected.log)、[垂尾预检](../runs/approach/20261113-f14-front-start/tail/matching-preflight.json)、[输入来源](../runs/approach/20261113-f14-front-start/experiment.json)、[测试日志](../runs/approach/20261113-f14-front-start/tests.log)。

**B 同视角模板**。在起点悬停时采6帧参考，过程中未发飞行指令。整机裁剪`[797,385,1123,726]`；统计文件与A逐字节相同，避免同时改变两个因素。机头、主翼、垂尾的ROI分别选中18、24、0个特征格：垂尾框太窄，没有包含任何格中心，原实现在构造它的原型时会抛异常，连带阻断其他部位。修改为把这种部位记为`no_reference_patches`，单独拒绝，其他部位照常运行；未扩大ROI或降低阈值。45项测试通过，用旧输入回归的结果差为0。主翼到所选表面3.700米、到任意表面3.546米，都小于6米，说明测距阈值不等于机体到所有表面的净距。实际独立帧率只有约2.92FPS，重复帧约90%，原因未定位，所以不能比较模板视角对收敛速度的影响。见[同视角模板](../runs/approach/20261113-f14-front-matched-template/reference/aircraft-template.png)、[ROI与特征格](../runs/approach/20261113-f14-front-matched-template/reference-feature-grid.png)、[汇总](../runs/approach/20261113-f14-front-matched-template/summary.json)、[轨迹图](../runs/approach/20261113-f14-front-matched-template/summary.png)、[垂尾预检](../runs/approach/20261113-f14-front-matched-template/tail/matching-preflight.json)、[数值回归](../runs/approach/20261113-f14-front-matched-template/baseline-regression.json)、[测试日志](../runs/approach/20261113-f14-front-matched-template/tests.log)。

**C 窄ROI原型与局部配准**。离线比较了五种原型方法，用6张同场景图。其中“补一格”只有在额外加0.02框容差时才全部通过，但这个容差约19像素，远大于约6像素的垂尾宽度，而且没有一张图的粗DINO中心落在原始框内，因此不采用。最终方案是：以DINO候选为中心，在1.5格半径内做原图相关配准。6张图的精修点都落在原始框内；擦除目标的合成图全部被拒绝；8项缩放与旋转变换中7项通过（+5°因峰差0.146被拒）。阈值（相关分数0.85、峰差0.15）是用这批开发图选的，未独立校准。实现为显式选项`--thin-roi-refinement`，只作用于中心选格为空的部位；50项测试通过，不启用时回归差为0。

在线结果：初次客户端话题握手阻塞，被中断，未发指令。正式控制193组观测、62.29秒、轨迹50.01米；配准最低分数0.8621，单次感知最长0.155秒，实际独立帧率约18.06FPS。录像统一起点设在启动后0.2秒，所以第一条记录的时间约为−0.113秒，原时间保留。

**终点核验（冷启动失败）**：重新初始化匹配器时，粗DINO候选落在垂尾下方，配准分数0.796，低于0.85，被拒绝。用第192步的配准状态继续跟踪（`--offline --continue-tracking`）时，分数0.869，射线命中所选垂尾。延续跟踪通过不能代替冷启动重新截获。

见[原型比较](../runs/approach/20261113-f14-tail-roi-study/pooling-comparison.json)、[配准初验](../runs/approach/20261113-f14-tail-roi-study/refinement-verification.json)、[精修示意](../runs/approach/20261113-f14-tail-roi-study/proposal.png)、[汇总](../runs/approach/20261113-f14-tail-local-refinement/summary.json)、[轨迹图](../runs/approach/20261113-f14-tail-local-refinement/summary.png)、[控制报告](../runs/approach/20261113-f14-tail-local-refinement/tail/report.json)、[配准诊断](../runs/approach/20261113-f14-tail-local-refinement/tail/control-signals.json)、[延续跟踪核验](../runs/approach/20261113-f14-tail-local-refinement/tail/evaluation/part-surface-validation.json)、[冷启动失败](../runs/approach/20261113-f14-tail-local-refinement/tail/evaluation/cold-start-validation.json)、[冷启动日志](../runs/approach/20261113-f14-tail-local-refinement/tail/evaluation-cold-start.log)、[协议检查](../runs/approach/20261113-f14-tail-local-refinement/protocol-check.json)、[测试日志](../runs/approach/20261113-f14-tail-local-refinement/tests.log)。

**三项结论**：在正前上方起点，机头和主翼在跨视角和同视角模板下都完成了抵近。垂尾先因跨视角落点错误被拒，再因特征格为空被拒，最后通过局部配准完成了一次连续抵近，但近景冷启动重新截获没有通过。未测试的项：更高的特征分辨率、主动调整视角、低空平视正面、其他飞机。
