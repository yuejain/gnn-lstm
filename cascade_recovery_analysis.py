#!/usr/bin/env python3
"""
cascade_recovery_analysis.py - P1.3 级联失效增强对比分析
对标水下无人机自恢复模型（Appl.Math.Model）：
对比 无恢复 vs 自恢复(ρ=0.1/0.2) vs 有向加权 的级联失效曲线。

用法:
    python cascade_recovery_analysis.py [--n_graphs 200]
输出:
    results/tables/cascade_recovery.csv
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import to_networkx

from src.utils import load_config, set_seed, save_csv, setup_logger, get_project_root
from src.resilience_labels import simulate_cascade_failure


def main():
    parser = argparse.ArgumentParser(description="级联恢复对比")
    parser.add_argument("--n_graphs", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(42)
    logger = setup_logger("cascade_rec", log_file="logs/cascade_recovery.log")
    root = get_project_root()
    os.chdir(root)

    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    n = len(data_list)
    idx = np.random.default_rng(42).permutation(n)[:args.n_graphs]
    n_steps = cfg["topology"]["cascade_steps"]

    configs = [
        ("无恢复（基线）", dict(self_recovery=0.0)),
        ("自恢复 ρ=0.1", dict(self_recovery=0.1)),
        ("自恢复 ρ=0.2", dict(self_recovery=0.2)),
        ("有向加权负载", dict(directed_weighted=True)),
    ]
    agg = {name: {"final": [], "peak": [], "area": []} for name, _ in configs}

    for i in idx:
        d = data_list[i]
        G = to_networkx(d, to_undirected=True)
        for name, kw in configs:
            ratios, _ = simulate_cascade_failure(
                G, failure_ratio=0.05, n_steps=n_steps, seed=42, **kw)
            agg[name]["final"].append(ratios[-1])
            agg[name]["peak"].append(ratios.max())
            agg[name]["area"].append(ratios.mean())  # 失效曲线下面积（越小越韧）

    rows = []
    for name, _ in configs:
        rows.append({
            "config": name,
            "final_failure_mean": round(float(np.mean(agg[name]["final"])), 4),
            "peak_failure_mean": round(float(np.mean(agg[name]["peak"])), 4),
            "failure_area_mean": round(float(np.mean(agg[name]["area"])), 4),
        })
    df = pd.DataFrame(rows)
    save_csv(df, "results/tables/cascade_recovery.csv")
    print("\n[级联失效增强对比]（200 图均值）")
    print(df.round(4).to_string(index=False))
    # 相对基线改善
    base_area = df.iloc[0]["failure_area_mean"]
    for r in df.iloc[1:]:
        imp = (base_area - r["failure_area_mean"]) / max(base_area, 1e-9) * 100
        print(f"{r['config']}: 失效面积降低 {imp:.1f}%（韧性提升）")


if __name__ == "__main__":
    main()
