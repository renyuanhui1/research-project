"""脚本 7（阶段 B）：plan_closed_loop.py —— 仿真闭环 MPPI 规划（按目标图像飞/穿门）

每个控制步：FrontCamera 抓图 → DINO → z_t → MPPI 选动作 → 执行第一个 → 滚动重规划(MPC)。
- 目标 z_goal：旧模式取本地某条 episode 的末帧图像经 DINO 编码（与离线验证一致）。
- 新 target 模式：取目标模板图，经 DINO patch token 做目标指纹；MPPI 优化"目标居中 + 变大"。
- 为适配 4060 Ti(8G，仿真已占 ~7G)：DINO/世界模型用 fp16 autocast；MPPI 单次迭代 +
  跨步 warm-start（上一步规划平移作下一步起点），采样数少而靠闭环纠偏。
- --handoff 两段式：target 指纹 MPPI 远程领路，dist≤goal-thresh 后自动交接视觉伺服收尾（同一指纹）。
- Chase 相机与本脚本无关：只订阅/使用 FrontCamera。
- 全程在"训练集统计标准化"潜空间里算（z_mean/z_std 存在 ckpt）。

--dry-run：不连仿真，只加载模型+目标、跑几次规划计时与显存，验证本地可行性。
"""
import argparse
import asyncio
import time
from math import comb
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import collect_episode as ce
from extract_dino_features import load_model, to_input_tensor, IMAGENET_MEAN, IMAGENET_STD
from train_predictor import LatentPredictor
from train_value import ValueFn, pool_spatial

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    base = str(PROJECT_ROOT)
    p = argparse.ArgumentParser(description="仿真闭环 MPPI 规划")
    # 模型/权重
    p.add_argument("--repo-dir", default=f"{base}/dinov2")
    p.add_argument("--dino-weights", default=f"{base}/weights/dinov2_vits14_pretrain.pth")
    p.add_argument("--ckpt", default=f"{base}/weights/predictor_h5.pt")
    # 目标
    p.add_argument("--goal-episode", default=f"{base}/outputs/datasets/episodes_dataset/episode_0050.h5",
                   help="旧 poolcos/patchmse 模式：取该 episode 的末帧图像作为目标")
    p.add_argument("--goal-frame", type=int, default=-1, help="目标取第几帧（默认末帧 -1）")
    p.add_argument("--target-template", default=f"{base}/outputs/references/templates/tmpl_goal0050_green.png",
                   help="target 模式：目标紧裁剪模板图，用 DINO patch 均值作为目标指纹")
    # MPPI
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--sigma", type=float, default=0.25)
    p.add_argument("--sigma-min", type=float, default=0.05, help="退火时 sigma 下限")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--iters", type=int, default=2, help="每个控制步 MPPI 迭代次数(每次重心到mu并退火)")
    p.add_argument("--smooth-beta", type=float, default=0.7,
                   help="采样噪声的时间平滑系数(EMA)，越大动作序列越连贯，0=关闭(原抖动行为)")
    p.add_argument("--use-spline", action="store_true",
                   help="用B样条(Bézier)控制点参数化动作轨迹：搜控制点而非逐步动作，天生平滑、幅度不塌")
    p.add_argument("--n-ctrl", type=int, default=5, help="样条控制点数(< horizon)")
    p.add_argument("--ensemble-ckpts", nargs="+", default=None,
                   help="世界模型集成：多个 predictor ckpt。rollout 对模型间分歧大的动作降权(治 model exploitation)")
    p.add_argument("--disagree-weight", type=float, default=1.0,
                   help="集成分歧惩罚权重 λ：cost += λ·跨模型预测方差")
    p.add_argument("--init-vx", type=float, default=0.7,
                   help="mu 前进初始化速度(训练数据 vx 均值~1.2)，避免从零速起步搜索")
    p.add_argument("--vx-min", type=float, default=None,
                   help="调试用：硬性最小 v_north；默认关闭。打开后不再是纯 MPPI 目标驱动")
    p.add_argument("--v-max", type=float, default=1.2)
    p.add_argument("--vz-max", type=float, default=0.15, help="垂直速度(v_down)上限，压小防往下飘/撞地")
    p.add_argument("--yaw-max", type=float, default=0.35)
    # 闭环
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--cost-metric", choices=["value", "target", "poolcos", "patchmse"], default="target",
                   help="目标代价：value=时间距离价值函数(强尺子)；target=模板DINO指纹；poolcos/patchmse=旧整图目标")
    p.add_argument("--value-fn", default=f"{base}/weights/value_fn.pt",
                   help="value 模式的价值函数权重")
    p.add_argument("--goal-thresh", type=float, default=0.5,
                   help="到达判定阈值：target 默认 0.5；poolcos 可用 ~0.02；patchmse 需调大到 ~0.3")
    p.add_argument("--viz-dump", default=None,
                   help="每次重规划把采样/最优动作/响应图存该目录（npz），供 src/mpc/plan_viz_node.py 可视化")
    p.add_argument("--acquire", action="store_true",
                   help="截获段：起飞后原地旋转搜索目标（peak 过阈值且居中即停），出生朝向可任意")
    p.add_argument("--acquire-rate", type=float, default=0.3, help="截获段旋转速率 rad/s")
    p.add_argument("--acquire-peak", type=float, default=0.45, help="截获判定：指纹 peak 阈值")
    p.add_argument("--acquire-center", type=float, default=0.2, help="截获判定：|center_x| 上限")
    p.add_argument("--acquire-timeout", type=float, default=30.0, help="截获段超时（秒）")
    # 纯视觉伺服（不走 MPPI）：用指纹 center 直接比例控制，直线怼上目标，验证"接触"可达
    p.add_argument("--visual-servo", action="store_true",
                   help="纯视觉伺服模式：不用世界模型/MPPI，指纹 center 直接 P 控制 vy/vz、恒定前进撞上目标")
    p.add_argument("--handoff", action="store_true",
                   help="两段式：target MPPI 远程领路，dist≤goal-thresh 后自动交接视觉伺服收尾")
    p.add_argument("--servo-max-steps", type=int, default=80,
                   help="--handoff 交接后伺服段的步数预算（独立于 --max-steps）")
    p.add_argument("--servo-fwd", type=float, default=0.8, help="伺服恒定前进速度 v_north")
    p.add_argument("--servo-kp-lat", type=float, default=1.2, help="横向比例增益：vy = kp·center_x")
    p.add_argument("--servo-kp-vert", type=float, default=0.8, help="垂直比例增益：vz = kp·center_y")
    p.add_argument("--servo-vz-max", type=float, default=0.5, help="伺服垂直速度上限(比 MPPI 的 vz-max 放宽，好跟踪环高)")
    p.add_argument("--servo-stop-mass", type=float, default=0.0,
                   help=">0 时：目标响应面积 mass 超过它即判定接触在即并停(默认0=只跑满 max-steps)")
    p.add_argument("--target-conf-weight", type=float, default=1.0,
                   help="target cost: 目标匹配强度权重，越大越追求模板响应高")
    p.add_argument("--target-center-weight", type=float, default=0.8,
                   help="target cost: 目标居中权重")
    p.add_argument("--target-size-weight", type=float, default=0.4,
                   help="target cost: 目标面积/响应质量权重，正值表示鼓励目标变大")
    p.add_argument("--target-softmax-temp", type=float, default=0.08,
                   help="target cost: patch 响应转重心的 softmax 温度")
    p.add_argument("--target-topk-frac", type=float, default=0.08,
                   help="target cost: 用最高响应的前多少比例 patch 估计匹配强度")
    p.add_argument("--target-mass-thresh", type=float, default=0.35,
                   help="target cost: patch 响应超过该阈值开始计入面积")
    p.add_argument("--target-mass-sharpness", type=float, default=20.0,
                   help="target cost: 面积 sigmoid 的陡峭程度")
    p.add_argument("--target-terminal-weight", type=float, default=0.7,
                   help="target MPPI: 终点目标代价权重")
    p.add_argument("--target-path-weight", type=float, default=0.3,
                   help="target MPPI: rollout 过程中平均目标代价权重，抑制终点幻觉")
    p.add_argument("--action-prior-weight", type=float, default=0.15,
                   help="target MPPI: 动作先验惩罚权重，抑制乱横移/乱偏航")
    p.add_argument("--forward-bonus-weight", type=float, default=0.0,
                   help="调试用：正向 v_north 奖励；默认 0，避免把前进写死进 MPPI")
    p.add_argument("--dt", type=float, default=ce.DEFAULT_DT)
    p.add_argument("--replan-stride", type=int, default=1,
                   help="每次规划后连续执行几个动作再重规划（提交式控制，减颠簸）")
    p.add_argument("--dur-factor", type=float, default=1.5,
                   help="指令时长=dt×此值，拉长以盖住规划间隙、保持连续运动")
    p.add_argument("--no-fp16", action="store_true", help="关闭 fp16（默认开，省显存提速）")
    p.add_argument("--dry-run", action="store_true", help="不连仿真，只测加载/规划速度/显存")
    p.add_argument("--record-goal", default="",
                   help="录目标模式：起飞爬升后直飞 max-steps 步，把末帧存到此 png 路径后退出(生成保证可达的目标)")
    p.add_argument("--diag", action="store_true",
                   help="诊断模式：强制 stride=1，每步对比模型预测 z' 与仿真真实 z'")
    p.add_argument("--save-view", default="",
                   help="把每步无人机当前帧存到该目录(step_XXX.png)，用于对比目标图")
    p.add_argument("--sanity", action="store_true",
                   help="开环 sanity check：固定前进动作序列，对比世界模型 rollout 预测 vs 仿真真实 latent")
    p.add_argument("--sanity-vx", type=float, default=1.0, help="sanity 模式的固定前进速度")
    # 仿真连接（复用脚本2/4 的默认）
    p.add_argument("--address", default=ce.DEFAULT_ADDRESS)
    p.add_argument("--scene", default=ce.DEFAULT_SCENE)
    p.add_argument("--sim-config-dir", type=Path, default=ce.DEFAULT_CONFIG_DIR)
    p.add_argument("--camera", default=ce.DEFAULT_CAMERA)
    p.add_argument("--altitude", type=float, default=ce.DEFAULT_ALTITUDE)
    args = p.parse_args()
    if (args.visual_servo or args.handoff) and args.cost_metric != "target":
        raise SystemExit("--visual-servo/--handoff 需配合 --cost-metric target（要用模板指纹 center）")
    return args


class Planner:
    """封装 DINO 编码 + 世界模型 MPPI（fp16 autocast）。"""

    def __init__(self, args, device):
        self.args = args
        self.dev = device
        self.fp16 = not args.no_fp16
        self.dino = load_model("dinov2_vits14", device,
                               weights=args.dino_weights, repo_dir=args.repo_dir)
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if ck.get("z_mean") is None:
            raise SystemExit("ckpt 无 z_mean/z_std，无法在标准化潜空间规划")
        self.model = LatentPredictor(ck["dim"], ck["num_patches"]).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.zm = torch.tensor(np.asarray(ck["z_mean"]), device=device)
        self.zs = torch.tensor(np.asarray(ck["z_std"]), device=device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        vx_lo = -args.v_max if args.vx_min is None else args.vx_min
        if vx_lo > args.v_max:
            raise SystemExit(f"--vx-min({vx_lo}) 不能大于 --v-max({args.v_max})")
        self.a_lo = torch.tensor([vx_lo, -args.v_max, -args.vz_max, -args.yaw_max], device=device)
        self.a_hi = torch.tensor([args.v_max, args.v_max, args.vz_max, args.yaw_max], device=device)
        # B样条(Bernstein/Bézier)基矩阵 (H, C)：C 个控制点 → H 步平滑动作轨迹
        self.spline_B = None
        if args.use_spline:
            C, H = args.n_ctrl, args.horizon
            t = np.linspace(0.0, 1.0, H)
            Bm = np.stack([comb(C - 1, j) * t ** j * (1 - t) ** (C - 1 - j) for j in range(C)], axis=1)
            self.spline_B = torch.tensor(Bm, dtype=torch.float32, device=device)  # (H,C)
        self.grid = int(round(ck["num_patches"] ** 0.5))
        if self.grid * self.grid != ck["num_patches"]:
            raise SystemExit(f"暂只支持方形 patch grid: num_patches={ck['num_patches']}")
        xs = torch.linspace(-1.0, 1.0, self.grid, device=device)
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        self.patch_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (P,2)
        self.target_proto = None
        if args.cost_metric == "target":
            self.target_proto = self.load_target_proto(args.target_template)
        self.value_fn = None
        if args.cost_metric == "value":
            vck = torch.load(args.value_fn, map_location="cpu")
            self.value_fn = ValueFn(vck["d"]).to(device).eval()
            self.value_fn.load_state_dict(vck["model"])
            self.value_grid = vck.get("pool_grid", 4)
            print(f"value 价值函数已加载: {args.value_fn} (grid={self.value_grid}, kmax={vck['kmax']})")

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16,
                              enabled=self.fp16 and self.dev == "cuda")

    @torch.no_grad()
    def encode(self, rgb224):
        """rgb224:(224,224,3) uint8 → 标准化潜空间 z (P,D) float32。"""
        x = to_input_tensor(rgb224[None], 224, self.mean, self.std, self.dev)
        with self._autocast():
            feat = self.dino.forward_features(x)["x_norm_patchtokens"][0].float()
        return (feat - self.zm) / self.zs

    @torch.no_grad()
    def load_target_proto(self, template_path):
        """紧裁剪目标模板 → DINO patch 指纹。

        模板应尽量只包含目标本体。这里取所有 patch token 的 L2-normalized 均值作为单个目标描述子；
        后续在整帧/预测 latent 的 16x16 patch grid 上做余弦响应图。
        """
        path = str(template_path)
        bgr = cv2.imread(path)
        if bgr is None:
            raise SystemExit(f"target 模式需要有效模板图: {path}")
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        z = self.encode(rgb)
        proto = F.normalize(z.float(), dim=-1).mean(dim=0)
        proto = F.normalize(proto, dim=0)
        print(f"target 模板已加载: {path}  size={rgb.shape[1]}x{rgb.shape[0]}")
        return proto

    def target_components(self, z):
        """模板指纹响应分量。

        z: (..., P, D)。所有返回 tensor 的 batch 形状均为 z 的前缀维度。
        """
        if self.target_proto is None:
            raise RuntimeError("target_proto 未初始化")
        zn = F.normalize(z.float(), dim=-1)
        sim = torch.matmul(zn, self.target_proto)  # (...,P)

        k = max(1, int(sim.shape[-1] * self.args.target_topk_frac))
        topk_mean = sim.topk(k, dim=-1).values.mean(dim=-1)
        peak, peak_idx = sim.max(dim=-1)

        w = torch.softmax(sim / self.args.target_softmax_temp, dim=-1)
        center = torch.matmul(w, self.patch_xy)  # (...,2), 0 是图像中心
        center_penalty = (center ** 2).sum(dim=-1)

        mass = torch.sigmoid(
            (sim - self.args.target_mass_thresh) * self.args.target_mass_sharpness
        ).mean(dim=-1)

        cost = (
            self.args.target_conf_weight * (1.0 - topk_mean)
            + self.args.target_center_weight * center_penalty
            - self.args.target_size_weight * mass
        )
        return {
            "cost": cost,
            "peak": peak,
            "peak_idx": peak_idx,
            "topk_mean": topk_mean,
            "center": center,
            "center_penalty": center_penalty,
            "mass": mass,
            "sim": sim,
        }

    def target_cost(self, z):
        """模板指纹目标函数：匹配强 + 居中 + 变大。

        z: (..., P, D)，返回同 batch 形状的 cost。值越小越好。
        """
        return self.target_components(z)["cost"]

    @torch.no_grad()
    def target_stats(self, z):
        """打印当前帧 target 响应的可解释分量。"""
        if self.target_proto is None:
            return ""
        c = self.target_components(z)
        center = c["center"]
        peak_idx = int(c["peak_idx"].item())
        px = peak_idx % self.grid
        py = peak_idx // self.grid
        return (f" peak={c['peak'].item():.3f}"
                f" center=({center[0].item():+.2f},{center[1].item():+.2f})"
                f" mass={c['mass'].item():.3f}"
                f" patch=({px},{py})")

    def goal_cost(self, zb, z_goal):
        """到目标的代价。支持批量 (N,P,D) 或单个 (P,D)。
        poolcos: 先按 patch 池化再算余弦距离(1-cos)，比全局 patch 均方更单调。"""
        if self.args.cost_metric == "target":
            return self.target_cost(zb)
        if self.args.cost_metric == "value":  # 时间距离价值函数：粗空间池化后估"还差几步到目标"
            zbp = pool_spatial(zb.float(), self.value_grid)        # (...,grid²·D)
            zgp = pool_spatial(z_goal.float(), self.value_grid)    # (grid²·D,)
            return self.value_fn(zbp, zgp.expand_as(zbp))
        if self.args.cost_metric == "poolcos":
            zbp = zb.mean(dim=-2)       # (...,D)
            zgp = z_goal.mean(dim=-2)   # (D,)
            return 1 - torch.nn.functional.cosine_similarity(zbp, zgp.expand_as(zbp), dim=-1)
        return ((zb - z_goal) ** 2).mean(dim=(-2, -1))

    def action_prior_cost(self, acts):
        """target 模式的动作正则，防止 MPPI 为了骗目标函数采大横移/大偏航。

        acts: (N,H,4)。先验是温和前进，不横移、不升降、不偏航。
        """
        prior = torch.tensor([self.args.init_vx, 0.0, 0.0, 0.0], device=self.dev)
        scale = torch.tensor(
            [max(self.args.v_max, 1e-3),
             max(self.args.v_max, 1e-3),
             max(self.args.vz_max, 1e-3),
             max(self.args.yaw_max, 1e-3)],
            device=self.dev,
        )
        smooth_prior = (((acts - prior) / scale) ** 2).mean(dim=(1, 2))
        forward_bonus = -self.args.forward_bonus_weight * (acts[:, :, 0] / scale[0]).mean(dim=1)
        return smooth_prior + forward_bonus

    def to_actions(self, params):
        """把优化参数转成 H 步动作：样条模式=控制点(C,4)经基矩阵→(H,4)；否则限幅返回。
        注意必须 clamp：plan() 里 mu 由未限幅的 params 加权而来，rollout 评估用的是
        clamp 后的动作，不 clamp 就会"评估守规矩、执行越界"（曾实测执行出 vz=-0.73
        而 --vz-max 0.15）。"""
        if self.spline_B is not None:
            return (self.spline_B @ params).clamp(self.a_lo, self.a_hi)
        return params.clamp(self.a_lo, self.a_hi)

    @torch.no_grad()
    def plan(self, z0, z_goal, mu):
        """MPPI(CEM)：迭代 iters 次，每次在 mu 周围采样→rollout→加权更新 mu 并退火 sigma。
        mu 是"优化参数"：样条模式=控制点(C,4)，否则=逐步动作(H,4)。返回 (新mu, best_cost)。"""
        N = self.args.samples
        Hsteps = self.args.horizon
        spline = self.spline_B is not None
        P = mu.shape[0]                                  # 参数维：C 或 Hsteps
        sigma = torch.full((P, 4), self.args.sigma, device=self.dev)
        best = float("inf")
        b = self.args.smooth_beta
        for _ in range(self.args.iters):
            noise = torch.randn(N, P, 4, device=self.dev)
            if b > 0 and not spline:  # AR(1)：样条本身平滑，无需再平滑噪声
                out = noise.clone()
                c = (1 - b * b) ** 0.5
                for k in range(1, P):
                    out[:, k] = b * out[:, k - 1] + c * noise[:, k]
                noise = out
            params = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise      # (N,P,4)
            if spline:
                acts = torch.einsum("hc,ncd->nhd", self.spline_B, params).clamp(self.a_lo, self.a_hi)
            else:
                acts = params.clamp(self.a_lo, self.a_hi)
            with self._autocast():
                zb = z0.unsqueeze(0).expand(N, -1, -1).contiguous()
                path_cost = 0.0
                for k in range(Hsteps):
                    zb = self.model(zb, acts[:, k])
                    if self.args.cost_metric in ("target", "value"):
                        path_cost = path_cost + self.goal_cost(zb, z_goal)
                terminal_cost = self.goal_cost(zb, z_goal)  # (N,)
                if self.args.cost_metric in ("target", "value"):
                    path_cost = path_cost / Hsteps
                    action_cost = self.action_prior_cost(acts)
                    cost = (
                        self.args.target_terminal_weight * terminal_cost
                        + self.args.target_path_weight * path_cost
                        + self.args.action_prior_weight * action_cost
                    )
                else:
                    cost = terminal_cost
            w = torch.softmax(-cost / self.args.temperature, dim=0)
            mu = (w.view(-1, 1, 1) * params).sum(dim=0)                # 更新在参数空间(控制点)
            var = (w.view(-1, 1, 1) * (params - mu.unsqueeze(0)) ** 2).sum(dim=0)
            sigma = var.sqrt().clamp_min(self.args.sigma_min)  # 退火
            best = min(best, cost.min().item())
        # 供 --viz-dump 用：末轮采样快照（按 cost 排序后均匀抽 64 条，覆盖好中差）
        if getattr(self.args, "viz_dump", None):
            k = min(64, N)
            idx = torch.argsort(cost)[:: max(1, N // k)][:k]
            self.last_plan = {"acts": acts[idx].float().cpu().numpy(),
                              "cost": cost[idx].float().cpu().numpy()}
        return mu, best


def init_mu(args, device):
    """mu 前进初始化：vx 设为训练数据均值附近，其余 0，避免从零速起步搜索。
    样条模式下 mu 是 C 个控制点(每个4维)，否则是 horizon 步逐步动作。"""
    P = args.n_ctrl if args.use_spline else args.horizon
    mu = torch.zeros(P, 4, device=device)
    vx0 = args.init_vx
    if args.cost_metric == "target" and args.vx_min is not None:
        vx0 = max(vx0, args.vx_min)
    mu[:, 0] = min(vx0, args.v_max)
    return mu


def load_goal_rgb(args):
    """目标图：.png 直接读；否则从 episode h5 取指定帧。"""
    if str(args.goal_episode).endswith(".png"):
        bgr = cv2.imread(str(args.goal_episode))
        return np.ascontiguousarray(bgr[:, :, ::-1])  # BGR->RGB
    with h5py.File(args.goal_episode, "r") as h:
        rgb = h["rgb"][args.goal_frame]
    return np.ascontiguousarray(rgb)


def dry_run(planner, args):
    print("=== dry-run：不连仿真，测加载/规划速度/显存 ===")
    torch.cuda.reset_peak_memory_stats()
    goal_rgb = load_goal_rgb(args)
    z_goal = None if args.cost_metric == "target" else planner.encode(goal_rgb)
    z0 = planner.encode(goal_rgb)            # 用同一帧当起点，仅测速度
    mu = init_mu(args, planner.dev)
    # 预热
    for _ in range(2):
        planner.encode(goal_rgb); planner.plan(z0, z_goal, mu)
    torch.cuda.synchronize()
    t0 = time.time()
    K = 10
    for _ in range(K):
        z0 = planner.encode(goal_rgb)
        mu, best = planner.plan(z0, z_goal, mu)
    torch.cuda.synchronize()
    per = (time.time() - t0) / K * 1000
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"单步(编码+1次MPPI) ≈ {per:.1f} ms → 约 {1000/per:.1f} Hz")
    print(f"峰值显存 ≈ {peak:.0f} MB（需 < ~1000MB 才稳妥与仿真共存）")
    print(f"samples={args.samples} horizon={args.horizon} fp16={not args.no_fp16}")


async def servo_stage(planner, args, drone, n_steps, step0=0):
    """指纹视觉伺服段：center 比例控制对准目标、恒定前进（不用世界模型/MPPI）。
    纯伺服模式(--visual-servo)与两段式交接(--handoff)共用；step0 让 viz 步号接着 MPPI 段编。"""
    print(f"=== 视觉伺服段: fwd={args.servo_fwd} kp_lat={args.servo_kp_lat} "
          f"kp_vert={args.servo_kp_vert} ===（不用世界模型/MPPI）")
    servo_ts = -1
    for step in range(step0, step0 + n_steps):
        msg, _ = await ce.wait_new_frame(servo_ts)
        servo_ts = int(msg["time_stamp"])
        rgb = cv2.resize(ce.decode_image(msg)[0], (ce.STORE_HW, ce.STORE_HW),
                         interpolation=cv2.INTER_AREA)
        z_t = planner.encode(rgb)
        c = planner.target_components(z_t)
        cx, cy = c["center"][0].item(), c["center"][1].item()
        peak, mass = c["peak"].item(), c["mass"].item()
        # 环偏左(cx<0)→往西(vy<0)纠偏；环偏上(cy<0)→爬升(vz<0)。符号已离线验证。
        vy = float(np.clip(args.servo_kp_lat * cx, -args.v_max, args.v_max))
        vz = float(np.clip(args.servo_kp_vert * cy, -args.servo_vz_max, args.servo_vz_max))
        a = [args.servo_fwd, vy, vz, 0.0]
        pos = ce.extract_pose(msg)[:3]
        print(f"  step {step:3d}: peak={peak:.3f} mass={mass:.3f} "
              f"center=({cx:+.2f},{cy:+.2f}) a=[{a[0]:.2f},{vy:+.2f},{vz:+.2f}] "
              f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")
        if args.viz_dump:  # 复用 rviz 节点：把伺服动作当"最优线"铺满 horizon
            vd = Path(args.viz_dump); vd.mkdir(parents=True, exist_ok=True)
            act_best = np.tile(a, (args.horizon, 1)).astype(np.float32)
            np.savez_compressed(
                vd / f"step_{step:04d}.npz",
                step=step, pose=ce.extract_pose(msg),
                dist=float((cx * cx + cy * cy) ** 0.5), best=float(peak),
                dt=args.dt, grid=planner.grid, rgb=rgb, act_best=act_best,
                samp_acts=act_best[None], samp_cost=np.array([peak], np.float32),
                sim=c["sim"].float().cpu().numpy())
        if args.servo_stop_mass > 0 and mass >= args.servo_stop_mass:
            print(f"  目标充满视野(mass={mass:.3f} ≥ {args.servo_stop_mass})，接触在即，停")
            break
        await drone.move_by_velocity_async(
            v_north=a[0], v_east=vy, v_down=vz,
            duration=args.dt * args.dur_factor, yaw=0.0, yaw_is_rate=True)
    print("  伺服段结束")


async def closed_loop(planner, args):
    sim_cfg = str(args.sim_config_dir.expanduser().resolve())
    import projectairsim
    from projectairsim import Drone, World
    client = projectairsim.ProjectAirSimClient(address=args.address)
    client.connect()
    print("已连接仿真")
    try:
        world = World(client, args.scene, sim_config_path=sim_cfg, delay_after_load_sec=2)
        drone = Drone(client, world, "Drone1")
        topic = drone.sensors[args.camera]["scene_camera"]
        ce.reset_cache()
        client.subscribe(topic, ce.image_callback)
        z_goal = None if (args.record_goal or args.sanity or args.cost_metric == "target") \
            else planner.encode(load_goal_rgb(args))

        ce.require_success("enable_api_control", drone.enable_api_control())
        ce.require_success("arm", drone.arm())
        await asyncio.wait_for(await drone.takeoff_async(), timeout=30.0)
        # 复现采集时的初始状态：爬升到训练高度(~6m)，否则视角高度 OOD（采集 goto_start 会爬升）
        n, e, d = ce.get_ned(drone)
        climb = await drone.move_to_position_async(
            north=n, east=e, down=d - args.altitude, velocity=2.0)
        await asyncio.wait_for(climb, timeout=30.0)
        await asyncio.sleep(1.0)
        await ce.wait_first_frame()

        if args.acquire and args.cost_metric == "target":
            # 截获段：原地慢转扫描，指纹 peak 过阈值且大致居中即停转交给 MPPI。
            # 有了它出生朝向可以任意，不需要手工对准目标。
            print(f"=== 截获段：原地旋转搜索目标（rate={args.acquire_rate} rad/s）===")
            acq_ts, found = -1, False
            for _ in range(int(args.acquire_timeout / args.dt)):
                await drone.move_by_velocity_async(
                    v_north=0.0, v_east=0.0, v_down=0.0,
                    duration=args.dt * args.dur_factor,
                    yaw=args.acquire_rate, yaw_is_rate=True)
                msg, _ = await ce.wait_new_frame(acq_ts)
                acq_ts = int(msg["time_stamp"])
                rgb = cv2.resize(ce.decode_image(msg)[0], (ce.STORE_HW, ce.STORE_HW),
                                 interpolation=cv2.INTER_AREA)
                c = planner.target_components(planner.encode(rgb))
                peak, cx = c["peak"].item(), c["center"][0].item()
                if peak > args.acquire_peak and abs(cx) < args.acquire_center:
                    print(f"  已截获: peak={peak:.3f} center_x={cx:+.2f}")
                    found = True
                    break
            if not found:
                print("  截获超时，按当前朝向继续（目标可能不在有效距离内）")
            # 停转稳定
            await drone.move_by_velocity_async(v_north=0.0, v_east=0.0, v_down=0.0,
                                               duration=0.5, yaw=0.0, yaw_is_rate=True)
            await asyncio.sleep(0.5)

        if args.record_goal:  # 自飞录目标：直飞 max-steps 步，存末帧为保证可达的目标图
            rec_ts, msg = -1, None
            for _ in range(args.max_steps):
                await drone.move_by_velocity_async(
                    v_north=min(args.init_vx, args.v_max), v_east=0.0, v_down=0.0,
                    duration=args.dt * args.dur_factor, yaw=0.0, yaw_is_rate=True)
                msg, _ = await ce.wait_new_frame(rec_ts)
                rec_ts = int(msg["time_stamp"])
            rgb = cv2.resize(ce.decode_image(msg)[0], (ce.STORE_HW, ce.STORE_HW),
                             interpolation=cv2.INTER_AREA)
            cv2.imwrite(args.record_goal, rgb[:, :, ::-1])
            print(f"已保存自飞目标帧(直飞{args.max_steps}步): {args.record_goal}")
            return

        if args.sanity:  # 开环 sanity：固定前进，世界模型 rollout 预测 vs 仿真真实 latent 逐步对比
            fwd = args.sanity_vx
            a_t = torch.tensor([fwd, 0.0, 0.0, 0.0], device=planner.dev)
            msg, _ = await ce.wait_new_frame(-1)
            last_ts = int(msg["time_stamp"])
            rgb = cv2.resize(ce.decode_image(msg)[0], (ce.STORE_HW, ce.STORE_HW),
                             interpolation=cv2.INTER_AREA)
            z0 = planner.encode(rgb)
            z_pred = z0.clone()
            print(f"=== sanity 开环: 固定前进 vx={fwd}, 世界模型 rollout vs 真实 ===")
            print(f"{'step':>4}{'预测vs真实':>12}{'真实vs起点':>12}{'比值(<1好)':>12}")
            for k in range(args.max_steps):
                await drone.move_by_velocity_async(
                    v_north=fwd, v_east=0.0, v_down=0.0,
                    duration=args.dt * args.dur_factor, yaw=0.0, yaw_is_rate=True)
                msg, _ = await ce.wait_new_frame(last_ts)
                last_ts = int(msg["time_stamp"])
                rgb = cv2.resize(ce.decode_image(msg)[0], (ce.STORE_HW, ce.STORE_HW),
                                 interpolation=cv2.INTER_AREA)
                z_real = planner.encode(rgb)
                with torch.no_grad(), planner._autocast():
                    z_pred = planner.model(z_pred.unsqueeze(0), a_t.unsqueeze(0))[0].float()
                pred_err = ((z_pred - z_real) ** 2).mean().item()
                base = ((z0 - z_real) ** 2).mean().item()  # 真实离起点(identity 基线)
                print(f"{k:>4}{pred_err:>12.3f}{base:>12.3f}{pred_err / max(base, 1e-6):>12.2f}")
            return

        if args.visual_servo:  # 纯视觉伺服：指纹 center 直接比例控制，直线怼上目标（不走 MPPI）
            await servo_stage(planner, args, drone, args.max_steps)
            return

        mu = init_mu(args, planner.dev)
        last_ts = -1
        pending = []
        step = 0
        reached = False
        stride = 1 if args.diag else min(args.replan_stride, args.horizon)
        z_prev, a_prev = None, None  # 诊断用：上一步的观测与执行的动作
        pos_prev, pos_start = None, None  # 诊断用：世界坐标，量真实位移
        while step < args.max_steps:
            msg, _ = await ce.wait_new_frame(last_ts)
            last_ts = int(msg["time_stamp"])
            rgb = cv2.resize(ce.decode_image(msg)[0], (ce.STORE_HW, ce.STORE_HW),
                             interpolation=cv2.INTER_AREA)
            if args.save_view:
                Path(args.save_view).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{args.save_view}/step_{step:03d}.png", rgb[:, :, ::-1])
            z_t = planner.encode(rgb)
            dist = planner.goal_cost(z_t, z_goal).item()
            if args.diag and z_prev is not None:
                with torch.no_grad(), planner._autocast():
                    z_hat = planner.model(z_prev.unsqueeze(0), a_prev.unsqueeze(0))[0].float()
                pred_err = ((z_hat - z_t) ** 2).mean().item()    # 模型预测 vs 真实
                move_err = ((z_prev - z_t) ** 2).mean().item()   # 这一步 z 实际变化量
                print(f"    [diag] 模型预测误差={pred_err:.3f}  实际z变化={move_err:.3f}")
            pos = ce.extract_pose(msg)[:3]
            if pos_start is None:
                pos_start = pos
            dpos = 0.0 if pos_prev is None else float(np.linalg.norm(pos - pos_prev))
            dtot = float(np.linalg.norm(pos - pos_start))
            pos_prev = pos
            mu, best = planner.plan(z_t, z_goal, mu)
            acts_exec = planner.to_actions(mu)  # 样条:控制点→H步动作; 否则原样
            if args.viz_dump:  # 每次重规划落盘一份快照，供 rviz 节点可视化/离线复盘
                vd = Path(args.viz_dump)
                vd.mkdir(parents=True, exist_ok=True)
                sim = (planner.target_components(z_t)["sim"].float().cpu().numpy()
                       if args.cost_metric == "target" else np.zeros(1, np.float32))
                np.savez_compressed(
                    vd / f"step_{step:04d}.npz",
                    step=step, pose=ce.extract_pose(msg), dist=dist, best=best,
                    dt=args.dt, grid=planner.grid, rgb=rgb,
                    act_best=acts_exec.float().cpu().numpy(),
                    samp_acts=planner.last_plan["acts"],
                    samp_cost=planner.last_plan["cost"], sim=sim)
            tstat = planner.target_stats(z_t) if args.cost_metric == "target" else ""
            print(f"  step {step:3d}: dist={dist:.3f} best={best:.3f}{tstat} "
                  f"a0=[{acts_exec[0,0]:.2f},{acts_exec[0,1]:.2f},{acts_exec[0,2]:.2f},{acts_exec[0,3]:.2f}]"
                  f" pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) 步移={dpos:.2f} 累计={dtot:.1f}")
            if dist <= args.goal_thresh:
                reached = True
                if args.handoff:
                    print(f"  MPPI 段到位（dist={dist:.3f} ≤ {args.goal_thresh}），交接视觉伺服收尾")
                    await servo_stage(planner, args, drone, args.servo_max_steps, step0=step)
                else:
                    print(f"  到达目标（dist={dist:.3f} ≤ {args.goal_thresh}）")
                break
            z_prev, a_prev = z_t, acts_exec[0].clone()  # 记录本步起点与即将执行的第一个动作
            # 连续执行接下来 stride 个规划动作（不中途换向），指令时长拉长以保持连续运动
            for j in range(stride):
                a = acts_exec[j].tolist()
                pending.append(await drone.move_by_velocity_async(
                    v_north=a[0], v_east=a[1], v_down=a[2],
                    duration=args.dt * args.dur_factor, yaw=a[3], yaw_is_rate=True))
                step += 1
                if j < stride - 1:
                    m2, _ = await ce.wait_new_frame(last_ts)
                    last_ts = int(m2["time_stamp"])
            if planner.spline_B is None:  # 非样条：warm-start 平移 stride
                mu = torch.cat([mu[stride:], mu[-1:].repeat(stride, 1)], dim=0)
            # 样条：控制点不做逐步平移，直接保留上次控制点作为下一步热启动
        if not reached:
            print(f"  到达最大步数 {args.max_steps}，结束")
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        try:
            await asyncio.wait_for(await drone.land_async(), timeout=30.0)
            drone.disarm(); drone.disable_api_control()
        except Exception as e:
            print(f"降落/释放异常（可忽略）: {e}")
        client.unsubscribe(topic)
        client.disconnect()
        print("已断开仿真")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    planner = Planner(args, device)
    if args.dry_run:
        dry_run(planner, args)
    else:
        asyncio.run(closed_loop(planner, args))


if __name__ == "__main__":
    main()
