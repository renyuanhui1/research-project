# 重构提案：两个场景的 20 个同名脚本

**日期**：2026-09-15　**状态**：**待确认，未执行任何改动**

> 按 CLAUDE.md「做重大改动前先确认方案」，本文只做分析，不动代码。

---

## 一、现状（实测，非估计）

仓库里两个场景：`airbase/`（活的主线）与 `src/`（老场景，README 声明只读存档）。
两边各有 `src/airsim/`，**20 个脚本同名**。

用 `difflib.SequenceMatcher` 逐对比对（按行）：

| 相似度 | 脚本 | airbase | 老场景 | 判读 |
|---:|---|---:|---:|---|
| 0.29 | **`collect_episode.py`** | 684 | 297 | ⚠️ **同名但完全不同** |
| 0.76 | `record_approach.py` | 141 | 130 | 已明显分叉 |
| 0.81–0.94 | `verify_predictor` / `train_predictor` / `plan_closed_loop` / `eval_wm` / `decode_check` / `depth_map` / `consolidate` | — | — | 小改 |
| 0.95–1.00 | 其余 **11 个** | — | — | **只差引导块**（airbase 版多 9 行：`sim_config` 自定位取代 `parents[2]`） |

**结论**：分叉的主体不是逻辑，是**引导块**。11 对是纯重复，9 对是真分叉，1 对（`collect_episode.py`）
是彻底不同的两个程序。

## 二、为什么这是个真问题（不是洁癖）

1. **已经真实踩过坑**（`airbase/src/airsim/README.md` 记录）：
   2026-07-31 在 `~/my` 下敲 `python src/airsim/collect_episode.py`，命中了**老场景**那份 ——
   它会正常启动、连上仿真、按老参数飞，**产出一堆看起来正常但完全没用的数据**，还污染 `outputs/`。
   只是因为新脚本改了参数名（`--band/--num`）才被 argparse 拦住。
2. **修 bug 只修一份**：`depth_map.py` 的 `/1000` 单位 bug 两边都有，2026-07-30 只修了 airbase 那份。
3. **耦合极低**：实测 `airbase/` **不 import 老场景任何代码**
   （grep 到的 `parents[2]` 全是解释"为什么不用它"的注释）。
   唯一的实际联系是 `src/mpc/plan_viz_node.py` —— airbase 的 rviz 可视化还在用它。

## 三、三个方案

### 方案 A：删掉老场景的重复副本（最小、最彻底）

老场景按 README 已是"只读存档、代码不再改"。把那 11 对"只差引导块"里的**老场景副本删掉**，
`collect_episode.py` 这种真分叉的保留并**改名**（如 `collect_episode_legacy.py`）。

- ✅ 陷阱从根上消失；无需改任何逻辑
- ⚠️ 老场景的代码不再自洽（但它的结论已归档进 `docs/EXPERIMENTS.md`）
- **影响面**：删 11 个文件 + 改名 1 个

### 方案 B：airbase 成单一来源，老场景 import 它

老场景的副本改成 `from airbase... import` 的薄壳。

- ✅ 无重复
- ❌ 老场景从此依赖 airbase 的引导块与路径假设，**耦合度反而升高**
- ❌ 改动面大，违背"影响越小越好"

### 方案 C：不动，加护栏（零风险）

保留现状，但在老场景的入口脚本加**运行时拦截**：检测到 `airsim` 包来自 airbase 就报错退出。

- ✅ 零破坏，直接防住那个具体事故
- ❌ 重复代码仍在，`depth_map.py` 那类"修一份漏一份"的问题还在

## 四、建议

**方案 A**，理由：耦合实测为零，删除是纯收益；而 B 会引入本来不存在的耦合，C 治标不治本。
老场景的价值在 `docs/EXPERIMENTS.md` 的结论，不在能重跑。

**建议的执行顺序**（若同意）：
1. 先只做 `collect_episode.py` 改名 —— 这是唯一造成过实际损失的那个
2. 跑一遍 `grep -rn` 确认没有脚本 import 老场景的这些模块
3. 再删那 11 个重复副本
4. 更新 `README.md` / `src/airsim/README.md` 的措辞（现在写的是"十几个同名脚本"，可换成精确数字）

---

## 五、附带发现（与重构无关，独立记录）

| 发现 | 说明 |
|---|---|
| 仓库 **8.6 GB**（`.git` 仅 6.5 MB） | 其中 `outputs/` 3.3G + `airbase/outputs/` 4.9G，均已 gitignore。不是问题，但 `airbase/outputs/features/part_dist` 占 1.7G（09-15 抽的 DINO latent），若不需要可删 |
| `airbase/outputs/maps/voxel_probe` | 08-26 体素地图产物，与当前主线无关，可归档 |
| 老场景 `src/airsim/tools/` | 老场景有 `tools/` 子目录（airbase 已取消），两边结构不同；不影响使用，仅记录 |
