"""把数据集重编号成 (目标 × 距离带) 轮转的顺序。

为什么需要：`train_predictor.split_episodes` 和 `tools/eval_wm.py` 都是"排序后取最后 N 条"
当 held-out。分目标/分带采集时文件名天然按批聚集，最后 N 条会全来自同一个目标的同一个带 ——
留出集只验了那一角，泛化结论不成立。轮转编号后，最后 N 条自动跨目标跨带。

默认只打印计划（dry-run），确认无误再加 --apply。
"""
import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import h5py


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dir", type=Path, help="数据集目录（含 episode_*.h5）")
    p.add_argument("--apply", action="store_true", help="真改名；默认只打印计划")
    args = p.parse_args()

    files = sorted(args.data_dir.expanduser().resolve().glob("episode_*.h5"))
    if not files:
        raise SystemExit(f"没找到 episode_*.h5: {args.data_dir}")

    groups = defaultdict(list)
    for f in files:
        with h5py.File(f, "r") as h:
            key = (str(h.attrs.get("target_name", "?")), str(h.attrs.get("band", "?")))
        groups[key].append(f)

    print(f"{len(files)} 条，{len(groups)} 组:")
    for k in sorted(groups):
        print(f"  {k[0]:20s} {k[1]:5s}  {len(groups[k])} 条")

    # 轮转：各组依次取一条，直到取空（组大小不等时短组先耗尽，仍保持分散）
    order = [f for f in itertools.chain.from_iterable(
        itertools.zip_longest(*(groups[k] for k in sorted(groups)))) if f is not None]

    plan = [(f, args.data_dir / f"episode_{i:04d}.h5") for i, f in enumerate(order)]
    same = sum(1 for a, b in plan if a == b)
    print(f"\n计划: {len(plan)} 条重编号（其中 {same} 条名字不变）")
    for a, b in plan[:5]:
        print(f"  {a.name} → {b.name}")
    print("  ...")
    tail = [f"{groups_of(p, groups)}" for p, _ in plan[-6:]]
    print(f"末尾 6 条来自: {tail}   ← held-out 会取这些")

    if not args.apply:
        print("\n（dry-run，未改动。确认无误后加 --apply）")
        return

    # 两阶段改名：源和目标的名字空间重叠，直接改会互相覆盖
    for i, (src, _) in enumerate(plan):
        src.rename(src.with_name(f".tmp_{i:04d}"))
    for i, (_, dst) in enumerate(plan):
        (args.data_dir / f".tmp_{i:04d}").rename(dst)
    print(f"\n已重编号 {len(plan)} 条 → episode_0000 ~ episode_{len(plan)-1:04d}")


def groups_of(path, groups):
    for k, v in groups.items():
        if path in v:
            return f"{k[0].split('_')[-1]}/{k[1]}"
    return "?"


if __name__ == "__main__":
    main()
