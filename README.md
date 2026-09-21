# my — 纯视觉无人机世界模型

无人机只靠机载相机，从空中滑近并**接触地面目标的指定部位**。不用目标坐标、不建度量地图。
技术路线：冻结 DINOv2 潜空间 + action-conditioned 世界模型 + 目标条件 cost + MPPI / 视觉伺服。

## ⚠️ 先看这条：现在的主线是 `airbase/`

仓库里有**两个场景**，目录结构互为镜像，同名脚本十几个，内容已经分叉：

| | 场景 | 状态 |
|---|---|---|
| **`airbase/`** | 机场卫星图 + 3D 飞机模型，接触机头/机翼/机尾 | 🟢 **活的主线**，所有新工作都在这 |
| `src/` + `docs/` + `outputs/` | 老场景（绿门） | 📦 **只读存档**，结论仍有效，代码不再改 |

```bash
cd ~/my/airbase && python src/airsim/wm/collect_episode.py ...   # ✅
python src/airsim/collect_episode.py                             # ❌ 在 ~/my 下 = 老场景脚本
```

第二条会正常启动、连上仿真、按老场景参数飞，**产出一堆看起来正常但完全没用的数据**，
还会污染 `outputs/`。2026-07-31 实际踩到过。详见
[`airbase/src/airsim/README.md`](airbase/src/airsim/README.md)。

（例外：`src/mpc/plan_viz_node.py` 是**活的** —— airbase 的 rviz 可视化还在用它。）

## 文档去哪找

| 想知道 | 看 |
|---|---|
| airbase 每天的实验结论、踩过的坑 | [`airbase/EXPERIMENTS.md`](airbase/EXPERIMENTS.md) ← **主台账** |
| airbase 的 39 个脚本分别干什么、还能不能跑 | [`airbase/src/airsim/README.md`](airbase/src/airsim/README.md) |
| 别人做到哪了（抵近/视觉伺服/停在阈值） | [`airbase/RELATED_WORK.md`](airbase/RELATED_WORK.md) |
| 两个场景 20 个同名脚本的重构提案（**待定，未执行**） | [`airbase/REFACTOR_PROPOSAL.md`](airbase/REFACTOR_PROPOSAL.md) |
| airbase 场景配置、相机几何 | [`airbase/README.md`](airbase/README.md) |
| PX4 联调怎么起 | [`airbase/RUN_PX4.md`](airbase/RUN_PX4.md) |
| 方法本身（世界模型/cost/MPPI 怎么组织） | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| 老场景做出过什么结论 | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| 源码模块划分 / 输出文件分类 | [`src/README.md`](src/README.md)、[`outputs/README.md`](outputs/README.md) |

## 目录

| 路径 | 说明 |
| --- | --- |
| `airbase/` | **主线场景**，自包含一套（`src/ sim_config/ weights/ outputs/`），软链共享 `dinov2/` |
| `ProjectAirSim/` | 上游仿真引擎（外部依赖，**已 gitignore**，需单独获取/构建） |
| `dinov2/` | DINOv2 上游仓（外部依赖，提供 hubconf） |
| `sim_config/` | 老场景的仿真配置 |
| `src/` | 老场景代码：`airsim/`(采集/训练/闭环) `px4/` `ros2/` `mpc/` `trajectory/` `colmap/` |
| `outputs/` | 老场景产物：数据集、特征、评估、runs |
| `weights/` | DINOv2 权重 + 老场景的 predictor/value |
| `docs/` | 方法与老场景台账 |

## 环境

conda 环境 `airsim`（Python 3.10，含 `projectairsim`、`opencv`、`pynng` 等）。

```bash
conda env create -f environment.yml
conda activate airsim
```

**UE 必须在前台 Play**，否则采集帧率掉到 ~3Hz。

训练（DINO 抽特征 / 世界模型 / 价值函数）在服务器 `ryh-dinov2` 上跑，本地不建 wm 环境。

## 快速开始（airbase）

```bash
conda activate airsim
cd ~/my/airbase

python src/airsim/wm/decode_check.py                            # 1) 确认能连上、能解码
python src/airsim/wm/collect_episode.py --band near --num 20    # 2) 采一批近场数据
```

其余脚本的前置条件和当前状态，查 [`airbase/src/airsim/README.md`](airbase/src/airsim/README.md)
的状态表（🟢 能跑 / 🟡 有坑 / 🔴 缺前置）。
