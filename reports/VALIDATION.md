# 项目离线验证

验证范围为配置、入口、机场数据、感知结果及ROS发布，不包含UE联机飞行。

| 检查 | 结果 |
|---|---|
| Python源码语法 | 51个文件可解析 |
| CLI入口 | 28个入口的`--help`可从项目目录之外执行 |
| 配置及运行集成测试 | 7项通过，含场景资产、相机设置、自定义档案、RGB/深度解码、fp16保存、控制反馈、轨迹与地图一致性 |
| 机场资产 | 2,345个保留文件均存在；2,226个图像、录像和数值文件的SHA256一致 |
| 编码器权重 | DINOv2、DINOv3各一份，SHA256一致 |
| 核心计算 | Fingerprint及9个控制、滤波、深度和解码定义的AST一致 |
| DINO保存帧复算 | 机翼远、近、近景3帧的匹配中心与峰值差异为0 |
| 机头预检 | 近距记录3帧匹配校验通过；仅运行离线验证 |
| 机翼预检 | 保存记录仍为近、近景部位校验失败；不记作抵近成功 |
| ROS地图 | 3张OccupancyGrid及MarkerArray可发布；背景38,224、近目标9,055、远目标8,970个体素 |
| ROS回放 | 从机场记录读取3帧，图像、标记和3个位姿轨迹正常发布 |
| Launch | 单机、多机启动文件可加载参数 |
| 文档链接 | 主文档、实验索引及保留运行README的本地链接有效 |

## 复查

```bash
conda activate airsim
cd /home/hui/research-project
python -m unittest discover -s tests -v
python src/capture/prepare_scene.py --profile airport --validate-only
python src/control/run_approach.py --out-dir runs/approach/close4-1793954736186 --validate-only
```

ROS入口和实时试验命令见[可视化说明](../docs/VISUALIZATION.md)及[运行说明](../docs/RUNNING.md)。UE编辑器内的几何导出脚本完成语法检查，须在对应关卡Play后执行。
