# 机场视觉抵近实验记录

## 方法与结果口径

抵近流程使用冻结DINO模板特征确定图像中的匹配中心，再用中心偏差与深度生成机体系速度指令。统计帧用于计算特征均值和方差，不更新编码器权重。

准备脚本使用目标几何选择机头或机翼ROI，并按已知坐标前往35、8、60、100米等采集位置，最后布置正式起点。这些坐标飞行属于模板准备，不计为视觉闭环抵近。正式闭环根据图像和深度运行；参考坐标用于评价，无人机位姿还用于高度保护和显示。因此，这些记录不能证明未知目标坐标条件下的完整自主搜索能力。

停车深度、到参考点的距离、到模型最近表面的距离和到指定部位表面的距离分别报告，不相互替代。深度阈值确认表示测距满足停车条件，是否达到具体部位需结合独立几何评价。

## 机场单机抵近

目标为`StaticMeshActor_7890`。下表的深度为最后一帧滤波测距；未测量的表面距离记为“—”。

| 试验 | 控制周期 | 末帧深度/m | 最近模型表面/m | 结果与记录 |
|---|---:|---:|---:|---|
| 俯视机头 | 451 | 11.95 | — | 深度阈值确认；[报告](../runs/approach/near-nose-dual_1106/report.json) |
| 俯视机头与栅格同步 | 614 | 11.97 | — | 深度阈值确认；[报告](../runs/approach/map-session-1793950632689/report.json) |
| 100米斜俯视机头 | 228 | 11.75 | — | 深度阈值确认；[报告](../runs/approach/oblique100-1793952243221/report.json) |
| 机头近距 | 184 | 3.76 | 3.14 | 4米阈值连续独立帧确认；[记录](../runs/approach/close4-1793954736186/README.md) |
| 正前方机翼 | 606 | 8.06 | 3.87 | 碰撞体接触后停滞，由操作中断；[记录](../runs/approach/wing4-1793956203311/README.md) |
| 翼外侧机翼 | — | — | — | 近、近景校验未通过，未执行闭环；[记录](../runs/approach/wing4-outboard-20261106/README.md) |

斜俯视起点水平距离100米，相机下俯30°，远段水平速度上限3米/秒；12米以内平移合速度上限0.35米/秒。机头近距试验的最终参考点距离约3.53米，网格表面距离约3.14米，详见[距离评价](../runs/approach/close4-1793954736186/distance-validation.json)。

正前方机翼试验到所选机翼区域表面仍约6.67米，不能用到任意机体表面的3.87米替代。碰撞对象及接触信息见[碰撞诊断](../runs/approach/wing4-1793956203311/collision-diagnostics.json)。翼外侧试验的准星偏到选定区域之外，校验失败不代表落点必然在整片机翼之外；[校验图](../runs/approach/wing4-outboard-20261106/matching-validation.png)保留实际落点。

## 匹配画面与采集质量

`CONTROL WEIGHTS`显示由原始相似度按温度0.005计算的softmax权重，平方根只用于叠色强度；绿色十字为实际控制使用的滤波中心。红区缩小不代表部位定位精度提高，该图不是分割掩膜。原始相似度与显示切换方法见[匹配显示说明](../docs/USAGE.md#匹配热力图的显示与解读)。

前摄和跟随录像为1920×1080、30FPS编码。实际独立图像帧率由对应`report.json`及`*-frames.csv`给出：机头近距约2.82FPS，正前方机翼约2.84FPS，录像含重复帧，不能将编码帧率当作实际采集帧率。

相机720p、1080p与采集间隔测试保存在[相机评价目录](../debug/camera)，每组包含自己的测量报告、视频和帧索引。

## 数据与回放

| 内容 | 位置 |
|---|---|
| 独立抵近试验 | [runs/approach/](../runs/approach/) |
| 采集与俯视素材 | [data/captures/](../data/captures/) |
| 可复用模板和统计帧 | [data/templates/airport_7890/](../data/templates/airport_7890/) |
| 飞机网格、全场与局部占用图 | [data/maps/airport_scene/](../data/maps/airport_scene/) |
| 相机帧率检查 | [debug/camera/](../debug/camera/) |
| 航点飞行 | [runs/waypoint/](../runs/waypoint/) |
| 模板、起点与控制运行 | [运行说明](../docs/USAGE.md#实时抵近) |
| RViz同步与回放 | [可视化说明](../docs/USAGE.md#rviz地图与匹配显示) |

包围盒评价读取`data/metadata/targets/`；模型表面距离使用机场三角面。几何评价不能用占用图上的距离代替指定部位的真实距离。

## 每次试验的记录与日志

选用空试验目录，通过`--out-dir`启动准备与控制。`projectairsim_client.log`保存本次客户端进程的连接日志，阶段终端输出分别重定向为`prepare.log`和`control.log`，与本次数据一起保存，具体命令见[运行方法](../docs/USAGE.md#实时抵近)。

| 文件 | 内容 |
|---|---|
| `report.json`或`test-summary.json` | 停车原因、校验状态、是否实际执行控制与采集统计 |
| `run.h5`、`poses.json`、`signals.csv` | 原始匹配、控制信号与位姿 |
| `front.mp4`、`chase.mp4`、`*-frames.csv` | 两路录像及源帧时间戳 |
| `viz_dump/` | 用于RViz的同步位姿、图像、相似度与匹配中心 |
| `distance-validation.json` | 离线几何距离及限速核对 |
| `matching-preflight.json` | 执行控制前的模板落点校验 |
| `collision-diagnostics.json` | 碰撞事件及停滞位姿，若本次发生碰撞 |
| `prepare.log`、`control.log`、`rviz.log` | 本次准备、控制和可视化终端输出 |

失败和中断记录保留其实际状态，不计入抵近成功；预检失败目录不应被当作已完成控制的录像试验。

[项目入口与数据验证](VALIDATION.md)记录配置、资产及回放检查方法。
