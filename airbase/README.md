# airbase 场景（空中接触地面飞机目标）

新场景：机场卫星图 + 放入的 **3D 飞机模型**（卫星图里烘焙的飞机是背景干扰，不碰）。
任务：无人机从空中斜下滑近、接触 3D 目标飞机的不同部位（机头/机翼/机尾）。
纯视觉（目标坐标脏、不可稳定还原），沿用 DINOv2 潜空间世界模型 + 目标条件 cost + MPPI 主线。
方法与老场景（绿门）一致，见 `../docs/ARCHITECTURE.md`；本文件夹是**本场景自包含的一套**，老场景不受影响。

## 目录

```
airbase/
  src/airsim/            # 复制自 ../src/airsim 的流水线脚本(可针对本场景改) + tools/
  sim_config/            # 本场景配置(已针对性修改)
  weights/               # 本场景重训的 predictor/value(待训);
                         # dinov2_vits14_pretrain.pth 软链→共享冻结 DINO 权重
  outputs/               # 本场景所有过程产物(数据集/特征/评估/runs/模板, 边跑边生成)
  dinov2 -> ../dinov2    # 软链: 共享冻结 DINO 仓(hubconf)
```

**关键：脚本保持"场景根下第 2 层"(`src/airsim/`)，故 `PROJECT_ROOT=parents[2]` 自动指到本文件夹，
默认参数全落进 `airbase/`——路径代码无需改。** 直接 `python airbase/src/airsim/xxx.py` 即可。

## 已做的配置改动（相对老场景）

- `sim_config/scene_airbase.jsonc`：`id` 改为 `SceneAirbase`；出生点 `xyz` **仍是老场景遗留值，待重设**(见文件 TODO)。
- `sim_config/robot_quadrotor_fastphysics_sensors.jsonc`：FrontCamera 由前视改 **下倾 pitch=-35°**
  （空中俯冲时前视会丢机腹下方目标；-35° 兼顾"看得见目标 + 保留前进光流"，可调）。

## 待办（按顺序）

1. **定出生点**：把 drone 摆到某架 3D 目标飞机上方、相机能看到目标的位置，写进 `scene_airbase.jsonc`。
2. **离线生死判据**（少量帧即可，不训练）：给一架目标飞机做模板，跑
   `check_cost_monotonic.py` / `check_target_cost.py`，验证 ① 斜下滑近时指纹 cost 单调；
   ② 3D 目标 vs 贴图飞机、机头/机翼/机尾**部位间可分**。不可分则先改 cost/模板，别急着采集重训。
3. 可分成立 → 按"斜下滑近"重设采集轨迹，走脚本 1→6（collect→dino→predictor）重训世界模型。
4. 目标函数：先指纹(每部位一张模板, 两段式 handoff) 拿 baseline；价值函数 V 作泛化跟进(须配对新 predictor 重训)。

## 与老场景共享/不共享

- **不复制、软链共享**：冻结 DINO 仓 `dinov2/` 和权重 `dinov2_vits14_pretrain.pth`。
- **完全不动**：`../sim_config/`、`../weights/`、`../outputs/`、`../src/`。
- **rviz 可视化**：仍用共享的 `../src/mpc/plan_viz*`（本地 ROS2，靠 `--dump-dir` 读，无需 fork）。
