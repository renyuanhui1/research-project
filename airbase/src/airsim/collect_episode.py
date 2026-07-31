"""脚本 2：collect_episode.py —— airbase 近场采集（绕目标随机站位，near/far 两个距离带）

面向 B/C 条件（世界模型 + 规划）的训练数据。与老场景版本的区别：
  - 起点不再是"当前位置 ±3m"，而是**以目标为中心的球面随机站位**（见 sample_station、BANDS）
  - 动作走**机体系** move_by_velocity_body_frame_async（与 PX4 伺服下发一致，
    且世界模型不必再从动作里反解 yaw 这个隐变量）
  - 每条 100 步（标称 10Hz，实测约 7.6Hz → 真实时长约 13s；缩放一律用 measure_dt 的实测值）
  - **一次起飞串 N 条**：站位之间直接飞过去，不重载场景（重载起降 300 条要 5~7.5 小时）
  - 记录 **RGB + 深度 + 分割**：深度进控制/latent，分割只进 evaluator（不喂控制器）

⚠️ 三路并非严格同步：以 RGB 新帧为节拍，深度/分割取"当下最新"（各等各的会把节拍拖到最慢一路）。
   实测 skew_depth 中位 0ms（对齐良好），skew_seg 中位 −130ms（稳定滞后约一帧，因为分割要多走
   一遍 stencil pass）。逐帧偏差存在 `skew_depth`/`skew_seg` dataset，汇总存在 `qc_skew_*` attrs
   —— **深度若要进世界模型，先查这两个数**；分割只进 evaluator，滞后一帧影响很小。

时间对齐约定（关键，否则训练数据是错的）：
    每步先取当前帧 image_t / pose_t / time_t，再下发动作 a_t，记录 (image_t, a_t, pose_t, time_t)。
    即 a_t 是"导致 image_t → image_{t+1} 的那个动作"。整条序列保持一致。

HDF5 datasets：
    rgb    (T,224,224,3) uint8    # 场景相机
    depth  (T,224,224)   uint16   # 米制（07-30 实测原值即米、量化步长 1m）；0 = 无效/天空
    seg    (T,224,224,3) uint8    # 分割相机的**调色板 BGR**，非 ID；离线按颜色查表
    action (T,4)         float32  # v_forward, v_right, v_down, yaw_rate（**机体系**）
    pose   (T,7)         float32  # pos_xyz + quat_wxyz（调试/评价用，不喂模型）
    time   (T,)          int64    # time_stamp（纳秒）
attrs 里带目标 OOB（center/size/quat）→ evaluator 可离线补算真实贴近距离，闭环零开销。
"""

import argparse
import asyncio
import math
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import projectairsim
from projectairsim import Drone, World
from projectairsim.types import BoxAlignment


# ---- 配置（魔法数集中于此）----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADDRESS = "172.21.192.1"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "sim_config"
DEFAULT_SCENE = "scene_airbase.jsonc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/datasets/nearfield"
DEFAULT_CAMERA = "FrontCamera"

# 目标（07-30 probe_target_object 实测）。分割 ID 必须运行时覆盖：默认全撞 83，
# 因为哈希跳过数字和下划线，staticmeshactor_22 与 _14 参与哈希的字符完全相同。
DEFAULT_TARGET = "StaticMeshActor_22"      # 一号飞机；机头轴 = 局部 +Y（已投影验证）
SEG_IDS = {"StaticMeshActor_22": 200, "StaticMeshActor_18": 210,
           "StaticMeshActor_16": 220, "StaticMeshActor_20": 230,
           "StaticMeshActor_24": 240, "StaticMeshActor_14": 40}

STORE_HW = 224          # 存储分辨率（正方形；注意 16:9 被各向异性压扁，投影时 vFOV 要写死）
DEFAULT_DT = 0.1        # 控制步长（秒）→ 10Hz
DEFAULT_STEPS = 100     # 步数 → 10s
DEFAULT_ALTITUDE = 6.0  # 仅供 record_approach/plan_closed_loop 复用（近场采集用球内站位，不走它）
CMD_DURATION_FACTOR = 1.5  # 指令时长 = dt × 此系数；整步内速度持续有效、无滑行空档
DEPTH_INVALID = 0       # 天空/无命中（原始 65504 = float16 最大有限值）统一存成 0

# 球内站位随机化。相机 -50° 的几何（07-28 实测）：目标水平距离在 0.19×高度（下边缘）
# 到 2.66×高度（上边缘）之间才在画面里。所以 r 与 alt **不能独立采样** ——
# 独立采样时 r/alt 跨 0.27~3.13，r=25/alt=8 那类站位目标完全看不见，整条 episode 报废。
RATIO_RANGE = (0.3, 2.4)    # r/alt，两端各留余量（相机 -50°: 下边缘 0.19、上边缘 2.66）

# 接近类模板的**终点安全壳**。必须按**该目标的** OOB 现算，不能写死 ——
# 各目标半高差很多：一号 3.24m、二号 5.35m、三号 5.05m、汽车 4.37m。
# 写死 8.0（照一号定的）去采二号就直接撞进机身了。
#   safe_alt = 目标半高 + SAFE_MARGIN  → 终点高出目标顶部 SAFE_MARGIN 米
#   SAFE_R   = 终点水平距中心多少米。此时无人机在目标正上方，靠 safe_alt 保证不碰
# 采集不需要真接触（接触是闭环任务的目标），停在安全壳上足以覆盖接近过程的状态。
SAFE_MARGIN = 2.0
SAFE_R = 5.0
# SAFE_MARGIN 取 2 的依据（07-31 实测标定）：指令位移与实际位移有约 1.3m 的跟踪损耗
# （速度跟不满），所以设计 2m → 实际终点离目标顶约 3.3m，噪声累积仅 ±0.26m。
# 这样接近过程覆盖到"离顶 3m"，比设 5m（实际 6.3m）多覆盖 3 米关键区段。
# 再小就不划算了：撞一次会卡住仿真、整批采集中断，代价不对称。
# 注意：**最后约 5m 无论如何采不到** —— 目标水平距离小于 0.19×高度就滑出画面下沿
# （07-28 相机几何），那段没有视觉信息可学，得靠向下测距/开环末段解决，不是采集能补的。
YAW_JITTER_DEG = 25.0       # 初始朝向 = 朝目标 ± 此值；别给 0，否则把"找到目标"送掉了

# 两个起点距离带，各采一批 → 能画出"性能随起点距离退化"的曲线，比单点对比强。
# r = 水平距离，alt = 高出目标中心的高度（斜距 = hypot(r, alt)，即相机实际看到的距离）。
BANDS = {
    "near": {"r": (15.0, 25.0), "alt": (15.0, 30.0)},   # 斜距 ~21~39m，目标占画面 7~20%
    "far":  {"r": (30.0, 40.0), "alt": (15.0, 35.0)},   # 斜距 ~33~53m，目标占画面 4~11%
}

# 参数依据（07-31 实跑 + 07-28/29 台账）：
# ① 下限不能太小：一号飞机 OOB 23.73×24.73×6.48 → 水平半对角 17.1m、顶部高 6.18m。
#    最初写的 r=8/alt=8 会让无人机悬在机翼正上方、距机顶仅 4.8m（53% 的起点落在飞机投影内）——
#    那不是"近场起点"，是已经贴在目标上了。
# ② 更根本的是别把**起点分布**当成**状态分布**：闭环时无人机从球面进入，不会凭空出现在 8m 处；
#    近距离状态会在 episode 飞行途中自然经过。起点应匹配测试时的进入点。
# ③ alt 上限 30/35 由**目标还剩多大**决定，与地形无关（无人机想飞多高飞多高）：
#    斜距一过 50m，目标就只占画面 4% 出头 ≈ 11 个 patch，再高就是个位数，世界模型学不到东西。
#    （注意别把 07-30 的"地形最高 37.6m → 转场巡航要 45m 以上"当成高度上限，那是**最低**安全高度。）
# ④ far 带落在 07-29 证明"指纹全平"的区域（20m 外 peak 0.197~0.216 无梯度）→
#    **A 条件（纯伺服）在 far 带预期会失败**，这是对照组该有的结果，不是配置错误。
# ⑤ alt 下限由几何自动兜底（见 sample_station 的 alt_lo），r=40 时会自动抬到 15.04m。

# 动作噪声（叠加在标称动作上）
NOISE_V = 0.2           # 速度噪声 (m/s)
NOISE_YAW = 0.1         # 偏航率噪声 (rad/s)

# 近场轨迹模板：分段 (v_forward, v_right, v_down, yaw_rate, steps)，**机体系**。
# 覆盖"接近/绕飞/纠偏/飞离"四类行为——只采接近的话，模型没见过偏了怎么办。
TEMPLATES = {
    "approach":  [(2.5, 0.0, 1.0, 0.0, 60), (2.0, 0.0, 1.2, 0.0, 40)],
    "orbit":     [(1.0, 2.0, 0.3, -0.35, 50), (1.0, -2.0, 0.3, 0.35, 50)],
    "correct":   [(2.0, 1.5, 0.5, 0.0, 30), (2.0, -1.5, 0.5, 0.3, 40), (2.5, 0.0, 1.0, 0.0, 30)],
    "drift":     [(1.5, 0.0, -0.3, 0.5, 40), (2.0, 0.0, 0.5, -0.2, 60)],
    "scan":      [(0.5, 0.0, 0.2, 0.6, 50), (0.5, 0.0, 0.2, -0.6, 50)],
}


# ---- 共享缓存：三路相机各存最新一帧 ----
_lock = threading.Lock()
_msgs = {}


def make_cb(tag):
    def cb(_topic, msg):
        with _lock:
            _msgs[tag] = msg
    return cb


def get_latest(tag="rgb"):
    with _lock:
        return _msgs.get(tag)


# 兼容 record_approach.py / plan_closed_loop.py（它们只用单路 RGB）
image_callback = make_cb("rgb")


def reset_cache():
    """清空缓存（换站位后调用，避免读到上一条的残留帧）。"""
    with _lock:
        _msgs.clear()


def decode(msg):
    """相机消息 → ndarray[h,w,c]。深度可能是 16UC1 或 32FC1，都要认。"""
    h, w = msg["height"], msg["width"]
    enc = str(msg["encoding"]).upper()
    specs = {"BGR": (np.uint8, 3), "RGB": (np.uint8, 3), "BGRA": (np.uint8, 4),
             "RGBA": (np.uint8, 4), "8UC1": (np.uint8, 1), "16UC1": (np.uint16, 1),
             "32FC1": (np.float32, 1)}
    if enc not in specs:
        raise ValueError(f"不认识的 encoding={enc!r}")
    dt, ch = specs[enc]
    data = msg["data"]
    buf = (np.frombuffer(data, dtype=dt) if isinstance(data, (bytes, bytearray, memoryview))
           else np.asarray(data, dtype=dt))
    if buf.size != h * w * ch:
        raise ValueError(f"元素数 {buf.size} != 预期 {h*w*ch}")
    return buf.reshape(h, w, ch)


def decode_image(msg):
    """→ (rgb_hwc, encoding)。兼容旧接口，调用方按 [0] 取图。"""
    a = decode(msg)[:, :, :3]
    if str(msg["encoding"]).upper().startswith("BGR"):
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    return a, str(msg["encoding"])


def to_store_rgb(msg):
    return cv2.resize(decode_image(msg)[0], (STORE_HW, STORE_HW), interpolation=cv2.INTER_AREA)


def to_store_depth(msg):
    """→ uint16 米。**必须 INTER_NEAREST**：深度插值会在物体边缘造出不存在的中间距离。

    存 uint16 而非 float32：07-30 实测原值就是整数米、量化步长 1m，所以当前**无精度损失**，
    体积减半。若哪天传感器改成真正的浮点深度，这里会静默丢掉亚米精度 —— 届时要改回 float16/32。
    """
    d = decode(msg)[..., 0].astype(np.float32)
    d = cv2.resize(d, (STORE_HW, STORE_HW), interpolation=cv2.INTER_NEAREST)
    d[~np.isfinite(d)] = 0.0
    d[d >= 65000.0] = 0.0                       # 天空哨兵 65504 → 无效
    return np.clip(d, 0, 65535).astype(np.uint16)


def to_store_seg(msg):
    """→ uint8 BGR。**必须 INTER_NEAREST**：调色板颜色插值出来的中间色不对应任何 ID。"""
    s = decode(msg)[:, :, :3]
    return cv2.resize(s, (STORE_HW, STORE_HW), interpolation=cv2.INTER_NEAREST)


def get_ned(drone):
    pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    return float(pos["x"]), float(pos["y"]), float(pos["z"])


def extract_pose(msg):
    """从帧自带字段取 pose = [pos_xyz, quat_wxyz]（7 维）。"""
    return np.array(
        [msg["pos_x"], msg["pos_y"], msg["pos_z"],
         msg["rot_w"], msg["rot_x"], msg["rot_y"], msg["rot_z"]],
        dtype=np.float32,
    )


def build_nominal(trajectory, steps):
    """把分段轨迹展开成每步标称动作 (steps, 4)，不足则循环、超出则截断。"""
    seq = []
    for vf, vr, vd, yr, n in trajectory:
        seq.extend([(vf, vr, vd, yr)] * n)
    if not seq:
        raise ValueError("trajectory 为空")
    while len(seq) < steps:
        seq.extend(seq)
    return np.array(seq[:steps], dtype=np.float32)


def require_success(name, result):
    print(f"{name}: {result}")
    if result is not True:
        raise RuntimeError(f"{name} 失败: {result}")


def read_target(world, name):
    """查目标 OOB（静止目标，一次即可）→ dict。evaluator 靠它离线补算真实距离。"""
    box = world.get_3d_bounding_box(name, BoxAlignment.OBJECT_ORIENTED)
    if not box or "center" not in box:
        raise RuntimeError(f"取不到 {name} 的包围盒（名字错？该 actor 没网格？）")
    c, s, q = box["center"], box["size"], box["quaternion"]
    center = np.array([c["x"], c["y"], c["z"]], np.float32)
    size = np.array([s["x"], s["y"], s["z"]], np.float32)
    quat = np.array([q["w"], q["x"], q["y"], q["z"]], np.float32)
    print(f"目标 {name}: center={center.round(2)} size={size.round(2)}")
    return {"name": name, "center": center, "size": size, "quat": quat}


def setup_segmentation(world):
    """运行时覆盖分割 ID（**每次载入场景都要重做**，不写回配置）。

    默认 ID 全撞 83：哈希 seg_id = 5 + Σ(ASCII≥97 的字符)，数字和下划线被跳过，
    故 staticmeshactor_22 与 _14 参与哈希的字符完全相同。覆盖走字符串精确比较，绕开哈希。
    """
    ok = 0
    for name, sid in SEG_IDS.items():
        try:
            if world.set_segmentation_id_by_name(name, sid, False, True):
                ok += 1
        except Exception as err:
            print(f"  分割 ID 设置失败 {name}: {err}")
    print(f"分割 ID 已覆盖 {ok}/{len(SEG_IDS)}")


def safe_alt_of(target):
    """终点相对目标中心的最低高度 = 该目标半高 + 余量。按目标现算，别写死。"""
    return float(target["size"][2]) / 2.0 + SAFE_MARGIN


def fit_to_station(name, traj, r, alt, dt, target):
    """把模板按站位调准。固定速度的模板没法同时适配 15m 和 40m 的起点。

    - 接近类（approach/correct）：缩放速度，使**终点停在目标外的安全壳上**（见 SAFE_* 常量）。
      注意终点必须按目标**体积**留间隙，不能按起始距离的百分比 —— 07-31 首次实跑
      用 r×0.9 直接撞进机身：r=20/alt=25 时终点距中心仅 2m、高出中心 2.5m，
      而飞机水平半对角 17.1m、半高 3.24m，那个点就在机身里面。
    - orbit：yaw_rate 按 r 反比缩放。以 r=16.7、yaw_rate 0.35（实测目标全程在画面中）
      为基准，r 变大时机头必须转得更慢才跟得住目标。
    - drift/scan：不动（本来就不要求精确到达）。
    """
    if name in ("approach", "correct"):
        fwd = sum(vf * n for vf, _, _, _, n in traj) * dt
        dwn = sum(vd * n for _, _, vd, _, n in traj) * dt
        need_f = max(0.0, r - SAFE_R)          # 只飞到安全壳，不是飞到中心
        need_d = max(0.0, alt - safe_alt_of(target))
        kf = need_f / fwd if fwd > 1e-6 else 0.0
        kd = need_d / dwn if dwn > 1e-6 else 0.0
        return [(vf * kf, vr * kf, vd * kd, yr, n) for vf, vr, vd, yr, n in traj]
    if name == "orbit":
        k = 16.7 / max(r, 1e-6)
        return [(vf, vr, vd, yr * k, n) for vf, vr, vd, yr, n in traj]
    return traj


def sample_station(rng, target, band="near", az_frac=None):
    """球内随机站位 → (north, east, down, yaw)。

    先采 alt，再按**视线比** r/alt 采 r（不独立采 r），保证目标一定落在画面里。
    """
    tn, te, td = target["center"]
    rr, ar = BANDS[band]["r"], BANDS[band]["alt"]
    # 先采 r（起点距离是自变量，要均匀覆盖该带），再在"目标必在画面内"的约束区间里采 alt。
    # 反过来（先 alt 再 r=ratio*alt）会让近一半样本被 clip 到 r 上限、近距离几乎采不到。
    r = rng.uniform(*rr)
    alt_lo = max(ar[0], r / RATIO_RANGE[1])     # 保证 r/alt <= 2.66，目标不越过画面上沿
    alt_hi = min(ar[1], r / RATIO_RANGE[0])     # 保证 r/alt >= 0.30，目标不滑出画面下沿
    alt = rng.uniform(alt_lo, alt_hi)
    # 方位角：给了 az_frac 就走**分层采样**（把 360° 均分给这批的 N 条，每条落自己格子里）。
    # 纯随机采 20 条实测得到 8 扇区 [4,1,5,1,2,2,5,0] —— 有扇区一条都没有。
    # "从哪个方位切入"是 C 条件要决策的核心变量，某个方位没数据会直接拖垮那个视角的预测。
    az = rng.uniform(0.0, 2 * math.pi) if az_frac is None else az_frac * 2 * math.pi
    n = float(tn + r * math.cos(az))
    e = float(te + r * math.sin(az))
    d = float(td - alt)                                  # NED: 负=高
    yaw = math.atan2(te - e, tn - n)                     # 朝向目标
    yaw += math.radians(rng.uniform(-YAW_JITTER_DEG, YAW_JITTER_DEG))
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi      # 规范到 [-π,π]，别给飞控越界值
    return n, e, d, yaw, r, alt


async def wait_first_frames(tags, timeout=20.0):
    waited = 0.0
    while any(get_latest(t) is None for t in tags):
        if waited >= timeout:
            missing = [t for t in tags if get_latest(t) is None]
            raise TimeoutError(f"{timeout}s 内没收到 {missing} 的帧，检查 capture-enabled")
        await asyncio.sleep(0.2)
        waited += 0.2


async def wait_first_frame(timeout=15.0):
    """兼容单路 RGB 的旧接口。"""
    await wait_first_frames(["rgb"], timeout=timeout)


async def wait_new_frame(last_ts, timeout=2.0, poll=0.002):
    """等一张时间戳不同于 last_ts 的 RGB 新帧，返回 (msg, waited)。

    事件驱动节拍的核心：相机配为 10Hz 时 dt 自然 = 相机周期 ≈ 0.1s（均匀）。
    waited≈0 说明取帧时已有新帧排队，即 CPU/RPC 赶不上相机。
    """
    waited = 0.0
    while True:
        msg = get_latest("rgb")
        if msg is not None and int(msg["time_stamp"]) != last_ts:
            return msg, waited
        if waited >= timeout:
            raise TimeoutError("等待新相机帧超时（检查相机出帧 / capture-interval）")
        await asyncio.sleep(poll)
        waited += poll


async def measure_dt(n=12):
    """实测相机帧间隔（秒）。

    模板缩放和指令时长都必须用**真实**步长：标称 10Hz，实测常只有 7.6Hz（0.132s），
    按 0.1 反推速度会让实际位移多 32% —— approach 的终点余量就是这么从 5m 缩到 3.2m 的。
    这与"帧率为什么上不去"是两回事（那是渲染负载），代码只需诚实使用真值。
    顺带充当热身检测：数值明显偏大就说明相机还没热透。
    """
    ts, last = [], -1
    for _ in range(n):
        msg, _ = await wait_new_frame(last)
        last = int(msg["time_stamp"])
        ts.append(last)
    return float(np.median(np.diff(ts)) / 1e9)


async def goto_station(drone, station, speed=8.0):
    """飞到球内站位并转到指定朝向（站位是绝对坐标 → 跨 episode 漂移不累积）。"""
    n, e, d, yaw = station[:4]
    n0, e0, d0 = get_ned(drone)
    print(f"  转场 {math.dist((n0, e0, d0), (n, e, d)):.0f}m → ({n:.0f},{e:.0f},{-d:.0f}m) ...",
          end="", flush=True)                    # 转场最长 90s，没有提示容易被当成卡死
    task = await drone.move_to_position_async(north=n, east=e, down=d, velocity=speed)
    await asyncio.wait_for(task, timeout=90.0)
    print(" 到位", end="", flush=True)
    # margin 默认 5°，实测指令 -160° 只到 -155.7°，采集起点朝向要准就得显式给小值
    task = await drone.rotate_to_yaw_async(yaw=yaw, margin=math.radians(1.0))
    await asyncio.wait_for(task, timeout=30.0)
    print(" 已对准")
    await asyncio.sleep(0.8)  # settle


async def run_episode(drone, args, trajectory, seed=0, tags=("rgb",), dt_use=None):
    """飞一条轨迹并返回 (bufs, stats)；不负责存盘。tags 决定存哪几路。"""
    rng = np.random.default_rng(seed)
    nominal = build_nominal(trajectory, args.steps)
    await wait_first_frames(tags)

    T = args.steps
    bufs = {
        "rgb": np.empty((T, STORE_HW, STORE_HW, 3), np.uint8),
        "action": np.empty((T, 4), np.float32),
        "pose": np.empty((T, 7), np.float32),
        "time": np.empty((T,), np.int64),
    }
    # 只为订阅了的路分配 → 单路 RGB 的调用方（record_approach）存出来的 h5 与历史格式一致
    if "depth" in tags:
        bufs["depth"] = np.zeros((T, STORE_HW, STORE_HW), np.uint16)
        bufs["skew_depth"] = np.zeros((T,), np.int64)    # 与 RGB 帧的时间戳偏差(纳秒)
    if "seg" in tags:
        bufs["seg"] = np.zeros((T, STORE_HW, STORE_HW, 3), np.uint8)
        bufs["skew_seg"] = np.zeros((T,), np.int64)

    # 用实测步长：标称 0.1 而实际 0.132 时，0.1×1.5=0.15 只比一步长 13%，余量太薄；
    # 相机再慢一点指令就会在下一帧到来前过期，整步末尾变成无动力滑行。
    cmd_duration = (dt_use or args.dt) * CMD_DURATION_FACTOR
    pending, behind = [], 0
    init = get_latest("rgb")
    last_ts = int(init["time_stamp"]) if init is not None else -1
    t_start = time.monotonic()

    for t in range(T):
        # 1) 等下一张 RGB 新帧 = image_t（事件驱动，相机周期即 dt）
        msg, waited = await wait_new_frame(last_ts)
        if waited <= 0.0:
            behind += 1
        last_ts = int(msg["time_stamp"])

        # 2) 记录三路。深度/分割取"当下最新"，不各等各的新帧（那会把节拍拖到最慢的一路）。
        #    同时记下它们与 RGB 的时间戳偏差 —— 否则事后无从判断存进去的是不是上一时刻的旧帧。
        bufs["rgb"][t] = to_store_rgb(msg)
        for tag, conv in (("depth", to_store_depth), ("seg", to_store_seg)):
            if tag not in tags:
                continue
            m2 = get_latest(tag)
            if m2 is not None:
                bufs[tag][t] = conv(m2)
                bufs[f"skew_{tag}"][t] = int(m2["time_stamp"]) - last_ts

        # 3) 本步动作 a_t（标称 + 噪声）
        a = nominal[t] + rng.normal(0.0, [NOISE_V, NOISE_V, NOISE_V, NOISE_YAW]).astype(np.float32)
        bufs["action"][t] = a
        bufs["pose"][t] = extract_pose(msg)
        bufs["time"][t] = last_ts

        # 4) 派发 a_t（只发送、不等完成），驱动 image_t → image_{t+1}。**机体系**
        pending.append(await drone.move_by_velocity_body_frame_async(
            v_forward=float(a[0]), v_right=float(a[1]), v_down=float(a[2]),
            duration=cmd_duration, yaw=float(a[3]), yaw_is_rate=True,
        ))

    elapsed = time.monotonic() - t_start

    # 速度指令的异常不能吞：某条没真正下发的话，h5 里会留下"记了动作但没执行"的错误映射。
    # （ProjectAirSim 服务端是 SingleTaskCall，新指令抢占旧指令，所以取消是正常的、不算失败。）
    results = await asyncio.gather(*pending, return_exceptions=True)
    cmd_err = [r for r in results
               if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError)]

    # 跳帧检测：behind 只看"取帧时是否已有新帧排队"，看不出时间戳跨了几个相机周期。
    fdt = np.diff(bufs["time"]) / 1e9
    dt_exp = dt_use or args.dt
    stats = {
        "elapsed": elapsed, "avg_hz": T / elapsed, "behind": behind,
        "dt_med": float(np.median(fdt)), "dt_max": float(fdt.max()),
        "n_skip": int((fdt > 1.5 * dt_exp).sum()),          # 疑似跳帧的步数
        "monotonic": bool((fdt > 0).all()),                 # 时间戳是否严格递增
        "cmd_err": len(cmd_err),
    }
    warn = ""
    if stats["n_skip"]:
        warn += f"  ⚠️跳帧 {stats['n_skip']} 步(最大间隔 {stats['dt_max']:.3f}s)"
    if cmd_err:
        warn += f"  ⚠️速度指令异常 {len(cmd_err)} 次: {cmd_err[0]!r}"
    if not stats["monotonic"]:
        warn += "  ⚠️时间戳非单调"
    print(f"  采集完成：{elapsed:.1f}s，{stats['avg_hz']:.2f}Hz，"
          f"dt中位 {stats['dt_med']:.4f}s，赶不上相机 {behind} 步{warn}")
    return bufs, stats


def save_hdf5(out, args, bufs, target=None, extra_attrs=None):
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # 默认拒绝覆盖：`--start-index` 用错（比如忘了错开批次）会静默抹掉已采的数据，
    # 而采集是不可逆的（要重开 UE 重飞）。要覆盖必须显式 --overwrite。
    if out.exists() and not getattr(args, "overwrite", False):
        raise FileExistsError(f"已存在，拒绝覆盖: {out}（要覆盖加 --overwrite，或换 --start-index）")
    gz = {"compression": "gzip", "compression_opts": 4}
    with h5py.File(out, "w") as f:
        f.create_dataset("rgb", data=bufs["rgb"], **gz)
        for k in ("depth", "seg"):                # 大片同色，gzip 压缩率很高
            if k in bufs:
                f.create_dataset(k, data=bufs[k], **gz)
        for k in ("skew_depth", "skew_seg"):      # 与 RGB 的时间戳偏差(ns)，判断是否取到旧帧
            if k in bufs:
                f.create_dataset(k, data=bufs[k])
        f.create_dataset("action", data=bufs["action"])
        f.create_dataset("pose", data=bufs["pose"])
        f.create_dataset("time", data=bufs["time"])
        f.attrs["dt"] = args.dt                   # 标称（指令 duration 按它算）
        # 实测 dt：相机出帧常低于标称 10Hz（实跑 8.5Hz → 0.117s）。动作是速度指令，
        # 位移 = 速度 × **真实**时长，用标称值训世界模型会带 ~17% 的系统偏差。
        if len(bufs["time"]) > 1:
            f.attrs["dt_actual"] = float(np.median(np.diff(bufs["time"])) / 1e9)
        f.attrs["steps"] = args.steps
        f.attrs["store_hw"] = STORE_HW
        f.attrs["camera"] = args.camera
        f.attrs["scene"] = args.scene
        f.attrs["action_layout"] = "v_forward,v_right,v_down,yaw_rate"
        f.attrs["action_frame"] = "body"          # 与老场景的 world-NED 数据不可混用
        f.attrs["pose_layout"] = "px,py,pz,qw,qx,qy,qz"
        f.attrs["depth_unit"] = "meter"
        f.attrs["depth_invalid"] = DEPTH_INVALID
        f.attrs["seg_layout"] = "palette_bgr"     # 非 ID，离线按颜色查表
        if target is not None:                    # evaluator 靠这几条离线补算真实距离
            f.attrs["target_name"] = target["name"]
            f.attrs["target_center"] = target["center"]
            f.attrs["target_size"] = target["size"]
            f.attrs["target_quat"] = target["quat"]
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
    mb = out.stat().st_size / 1e6
    print(f"  已存 {out.name}  ({mb:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser(description="airbase 近场采集（球内随机站位）→ HDF5")
    p.add_argument("--address", default=DEFAULT_ADDRESS)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--camera", default=DEFAULT_CAMERA)
    p.add_argument("--target", default=DEFAULT_TARGET, help="目标 actor 名（决定球心）")
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--band", default="both", choices=["near", "far", "both"],
                   help="起点距离带：near r∈[15,25] / far r∈[30,40] / both 交替（默认）")
    p.add_argument("--num", type=int, default=1, help="本次采几条（一次起飞串完）")
    p.add_argument("--start-index", type=int, default=0, help="文件编号起始（续采用）")
    p.add_argument("--overwrite", action="store_true",
                   help="允许覆盖同名 episode（默认拒绝，防止 --start-index 用错抹掉已采数据）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--template", default=None, choices=list(TEMPLATES),
                   help="只用指定模板；默认轮询全部")
    p.add_argument("--no-depth", action="store_true", help="不订阅深度")
    p.add_argument("--no-seg", action="store_true", help="不订阅分割")
    return p.parse_args()


async def main(args):
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [args.template] if args.template else list(TEMPLATES)
    bands = list(BANDS) if args.band == "both" else [args.band]

    client = projectairsim.ProjectAirSimClient(address=args.address)
    drone = None
    api_on = False
    try:
        client.connect()
        print("已连接 ProjectAirSim")
        world = World(client, args.scene,
                      sim_config_path=str(args.sim_config_dir.expanduser().resolve()),
                      delay_after_load_sec=2)
        print(f"场景已加载: {args.scene}")
        drone = Drone(client, world, "Drone1")

        target = read_target(world, args.target)
        setup_segmentation(world)

        # 订阅三路（深度/分割没开就自动降级，不中断采集）
        sensors = drone.sensors.get(args.camera, {})
        tags = []
        for key, tag in (("scene_camera", "rgb"), ("depth_camera", "depth"),
                         ("segmentation_camera", "seg")):
            if tag == "depth" and args.no_depth:
                continue
            if tag == "seg" and args.no_seg:
                continue
            topic = sensors.get(key)
            if topic is None:
                print(f"  [warn] 没有 {key}，跳过（config 里 capture-enabled 开了吗）")
                continue
            client.subscribe(topic, make_cb(tag))
            tags.append(tag)
        if "rgb" not in tags:
            raise RuntimeError(f"相机 '{args.camera}' 未配置 scene_camera")
        print(f"已订阅: {tags}")

        require_success("enable_api_control", drone.enable_api_control())
        api_on = True
        require_success("arm", drone.arm())
        print("起飞中...")
        await (await drone.takeoff_async())

        # 实测步长，之后的模板缩放与指令时长全用它（不用标称 args.dt，见 measure_dt 的说明）
        await wait_first_frames(tags)
        dt_use = await measure_dt()
        print(f"实测相机步长 {dt_use:.4f}s ({1/dt_use:.2f}Hz)，标称 {args.dt}s"
              f"{'  ← 偏慢，模板已按实测缩放' if dt_use > args.dt * 1.1 else ''}")

        for k in range(args.num):
            idx = args.start_index + k
            # 用**全局编号 idx** 轮询，不是批内 k：分批续采时 k 每次都从 0 开始，
            # `--num 1 --start-index N` 连采就会条条都是 approach/near，样本严重失衡。
            name = names[idx % len(names)]
            band = bands[idx % len(bands)]      # both 时近/远交替，保证两带样本量相当
            seed = args.seed + idx
            # 一条 episode 一个 RNG，按顺序取值。原来用同 seed 建两个 RNG，
            # 它们的第一个随机数完全相同 → 方位格内抖动与起点距离 r 人为相关，不是独立采样。
            rng_ep = np.random.default_rng(seed)
            # 分层：本批把 360° 均分成 num 格，第 k 条落在第 k 格内随机一点 → 方位不留空洞。
            # 这里用批内 k（分层要针对"本批的 num 条"），与上面模板轮询用 idx 不冲突。
            az_frac = (k + rng_ep.uniform(0.0, 1.0)) / args.num
            station = sample_station(rng_ep, target, band, az_frac)
            n, e, d, yaw, r, alt = station
            traj = fit_to_station(name, TEMPLATES[name], r, alt, dt_use, target)
            print(f"\n=== [{k+1}/{args.num}] episode_{idx:04d} 模板={name} 带={band} "
                  f"站位 r={r:.1f}m alt={alt:.1f}m 斜距={math.hypot(r, alt):.1f}m "
                  f"yaw={math.degrees(yaw):+.0f}° ===")
            try:
                reset_cache()
                await goto_station(drone, station)
                bufs, stats = await run_episode(drone, args, traj, seed, tags, dt_use=dt_use)
                extra = {"seed": seed, "template": name, "band": band,
                         "station_r": r, "station_alt": alt, "station_yaw": yaw,
                         "station_slant": math.hypot(r, alt)}
                # 质量诊断一并落盘：事后不用重算就能筛掉跳帧/指令异常的 episode
                extra.update({f"qc_{k2}": v for k2, v in stats.items() if k2 != "elapsed"})
                for tag in ("depth", "seg"):      # 跨流时间偏差的汇总（逐帧值另存 dataset）
                    if f"skew_{tag}" in bufs:
                        sk = bufs[f"skew_{tag}"] / 1e6
                        extra[f"qc_skew_{tag}_med_ms"] = float(np.median(sk))
                        extra[f"qc_skew_{tag}_absmax_ms"] = float(np.abs(sk).max())
                        extra[f"qc_skew_{tag}_over_half"] = float(
                            (np.abs(sk) > 0.5 * dt_use * 1e3).mean())
                save_hdf5(out_dir / f"episode_{idx:04d}.h5", args, bufs, target, extra)
            except FileExistsError as err:
                raise                            # 覆盖保护要立刻停，不能被单条容错吞掉
            except Exception as err:
                # 单条容错：站位飞不到 / 相机卡顿只跳过这条，下一条换站位常能恢复
                print(f"  episode_{idx:04d} 失败，跳过：{err}")
                continue
    finally:
        # 采完直接断开，**不等降落**（项目约定）：站位高达 30m，land 要磨很久，
        # 且 land 卡住时整个脚本会挂在这里。数据此时已全部落盘，降落没有意义。
        # ⚠️ 这条只适用于一次性的 FastPhysics 仿真采集。**真机/PX4 绝不可空中 disarm**。
        if api_on and drone is not None:
            try:
                drone.disarm()
                drone.disable_api_control()
            except Exception as err:
                print(f"释放控制异常（可忽略）: {err}")
        client.disconnect()
        print("已断开连接")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
