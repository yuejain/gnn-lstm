#!/usr/bin/env python3
"""
fit_degree_prior.py - 从公开数据集拟合度分布先验
从 SNAP email-Eu-core（真实组织网络）拟合幂律指数 γ，
回写/输出到 results/tables/public_degree_prior.json，
供 config.yaml 的 geo_scale_free.gamma 标定参考。

用法:
    python tools/fit_degree_prior.py [--apply]   # --apply 回写 config
输出:
    results/tables/public_degree_prior.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import networkx as nx

from src.utils import load_config


def fit_powerlaw(degrees: np.ndarray) -> dict:
    """拟合幂律指数 γ（MLE：γ = 1 + n / Σ ln(k/k_min)）。"""
    deg = degrees[degrees >= 1]
    k_min = max(1, int(np.percentile(deg, 20)))  # 截断低度（避免噪声）
    deg_trunc = deg[deg >= k_min]
    if len(deg_trunc) < 10:
        return {"gamma": None, "k_min": k_min, "n_fit": len(deg_trunc)}
    gamma = 1 + len(deg_trunc) / np.sum(np.log(deg_trunc / k_min))
    return {"gamma": float(gamma), "k_min": int(k_min), "n_fit": int(len(deg_trunc))}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="幂律度分布先验拟合")
    parser.add_argument("--apply", action="store_true", help="回写 config.yaml 的 gamma")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    root = ROOT
    import os
    os.chdir(root)

    # 加载 email-Eu-core
    from src.public_data_loader import load_email_eu
    graphs = load_email_eu("data/public")
    if not graphs:
        print("email-Eu-core 缺失，跳过")
        sys.exit(0)
    G = graphs[0]
    degrees = np.array([d for _, d in G.degree()])
    result = fit_powerlaw(degrees)
    result["dataset"] = "email-Eu-core"
    result["n_nodes"] = G.number_of_nodes()
    result["n_edges"] = G.number_of_edges()
    result["mean_degree"] = float(degrees.mean())
    result["max_degree"] = int(degrees.max())

    # 也可参考 Topology Zoo 平均
    try:
        from src.public_data_loader import load_topology_zoo
        tz = load_topology_zoo("data/public", min_nodes=15)
        tz_degs = np.concatenate([np.array([d for _, d in g.degree()]) for g in tz])
        tz_fit = fit_powerlaw(tz_degs)
        result["topology_zoo"] = {"n_graphs": len(tz), **tz_fit}
    except Exception as e:
        result["topology_zoo"] = {"error": str(e)}

    out_path = Path("results/tables/public_degree_prior.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"拟合结果: γ={result['gamma']:.3f} (k_min={result['k_min']}, n_fit={result['n_fit']})")
    print(f"degree 范围: {degrees.min()}-{degrees.max()}, 均值 {result['mean_degree']:.2f}")
    if "topology_zoo" in result and "gamma" in result["topology_zoo"]:
        print(f"Topology Zoo: γ={result['topology_zoo']['gamma']:.3f} ({result['topology_zoo']['n_graphs']} 图)")
    print(f"已保存: {out_path}")

    # 可选回写 config
    if args.apply and result.get("gamma"):
        import re
        cfg_path = Path("config.yaml")
        text = cfg_path.read_text(encoding="utf-8")
        new_text = re.sub(r"gamma: [\d.]+", f"gamma: {result['gamma']:.2f}", text, count=1)
        if new_text != text:
            cfg_path.write_text(new_text, encoding="utf-8")
            print(f"已回写 config.yaml: gamma={result['gamma']:.2f}")
        else:
            print("config.yaml 未找到 gamma 字段（或已在合适值），未回写")


if __name__ == "__main__":
    main()
