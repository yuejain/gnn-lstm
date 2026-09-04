#!/usr/bin/env python3
"""
validate_label_accuracy.py - 标签精度校验（数据准确性保障）
验证 k=150 采样介数（数据工厂优化后）与全量介数计算的韧性标签相关性，
确认加速优化不引入显著标签漂移。

协议：
- 抽取 30 张 600 节点图
- 分别用 全量介数 / k=150 采样介数 计算 composite_score
- 输出 Pearson 相关、ΔR²、最大绝对偏差

用法:
    python validate_label_accuracy.py
输出:
    results/tables/label_accuracy_check.csv
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import networkx as nx
import pandas as pd

from src.utils import load_config, set_seed, save_csv
from src.resilience_labels import compute_all_metrics


def main():
    cfg = load_config("config.yaml")
    set_seed(42)
    os.chdir(ROOT)

    # 用冒烟生成的 909 图里的随机抽样（或重新生成 30 张 600 节点 random）
    from src.topology_factory import generate_sensor_positions, generate_variant
    from src.physical_layout import calibrate_obstacles

    rows = []
    for i in range(30):
        seed = 1000 + i
        set_seed(seed)
        positions, hazard, mapping, _ = generate_sensor_positions(cfg["park"], 600)
        G = generate_variant(positions, 150, "random", vparams={}, seed=seed)

        # 全量介数指标
        set_seed(seed)
        m_full = compute_all_metrics(G, cfg)  # 内部用 k=150（当前默认）
        # 强制全量介数重算
        set_seed(seed)
        G2 = G.copy()
        G2.graph.pop("betweenness", None)
        import src.resilience_labels as rl
        # 临时补丁：用全量介数
        orig = rl.nx.betweenness_centrality
        rl.nx.betweenness_centrality = lambda g, k=None, seed=None: orig(g)
        m_orig = compute_all_metrics(G2, cfg)
        rl.nx.betweenness_centrality = orig

        rows.append({
            "graph_id": i, "n_nodes": 600,
            "composite_full": m_orig["composite_score"],
            "composite_k150": m_full["composite_score"],
            "robustness_full": m_orig["robustness"],
            "robustness_k150": m_full["robustness"],
            "survivability_full": m_orig["survivability"],
            "survivability_k150": m_full["survivability"],
        })

    df = pd.DataFrame(rows)
    # 相关性
    corr_c = df["composite_full"].corr(df["composite_k150"])
    corr_r = df["robustness_full"].corr(df["robustness_k150"])
    corr_s = df["survivability_full"].corr(df["survivability_k150"])
    max_delta = (df["composite_full"] - df["composite_k150"]).abs().max()
    mean_delta = (df["composite_full"] - df["composite_k150"]).abs().mean()

    summary = pd.DataFrame([{
        "metric": "composite_score", "pearson": corr_c, "max_abs_delta": max_delta, "mean_abs_delta": mean_delta
    }, {
        "metric": "robustness", "pearson": corr_r, "max_abs_delta": np.nan, "mean_abs_delta": np.nan
    }, {
        "metric": "survivability", "pearson": corr_s, "max_abs_delta": np.nan, "mean_abs_delta": np.nan
    }])

    save_csv(summary, "results/tables/label_accuracy_check.csv")
    print("\n[标签精度校验：全量介数 vs k=150 采样]")
    print(summary.to_string(index=False))
    if corr_c > 0.99:
        print(f"✅ 相关性 {corr_c:.4f} > 0.99 → k 采样不引入显著标签漂移，优化安全")
    else:
        print(f"⚠️ 相关性 {corr_c:.4f} < 0.99 → 建议调高 k 采样数")


if __name__ == "__main__":
    main()
