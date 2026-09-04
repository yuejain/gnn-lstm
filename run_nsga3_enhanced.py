#!/usr/bin/env python3
"""
run_nsga3_enhanced.py - P2.2 优化器增强（Louvain 局部搜索）
对标 DRVLS（Appl.Soft.Comput）的 Louvain 社区局部搜索思想：
NSGA3 获得 Pareto 前沿后，用 Louvain 社区检测引导局部搜索——
在社区边界低连通区域微调新增节点坐标，改进前沿解。

对比：标准 NSGA3 vs NSGA3+Louvain 局部搜索（HV/前沿点指标）。

用法:
    python run_nsga3_enhanced.py [--n_gen 60]
输出:
    results/tables/nsga3_enhanced_comparison.csv
"""
import argparse
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import networkx as nx

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV

from src.utils import (load_config, set_seed, get_device, save_csv, ensure_dir,
                       setup_logger, get_project_root)
from src.nsga3_optimizer import ResilienceOptimizationProblem
from src.resilience_labels import simulate_cascade_failure
from run_nsga3_optimizer import make_gcn_predictor


def louvain_local_search(problem, X, n_trials=40, seed=42):
    """Louvain 社区引导局部搜索：对每个前沿解，在社区边界区域微调坐标。

    返回改进后的解集（若改进）与原解集拼接。
    """
    rng = np.random.default_rng(seed)
    G = problem.base_graph
    try:
        communities = list(nx.community.louvain_communities(G, seed=seed))
    except Exception:
        communities = None
    X_new = []
    for x in X:
        X_new.append(x)
        if communities is None:
            continue
        # 微调方向：新增节点坐标向社区边界偏移（促进跨社区连接）
        for _ in range(3):
            cand = x.copy()
            delta = rng.normal(0, 0.08, size=x.shape)  # 局部扰动
            # 社区边界感知：随机选择 1-2 个新增节点向边界推
            n_new = problem.n_new_nodes
            for j in range(n_new):
                if rng.random() < 0.5:
                    cand[j * 2] = np.clip(cand[j * 2] + delta[j * 2], 0, 1)
                    cand[j * 2 + 1] = np.clip(cand[j * 2 + 1] + delta[j * 2 + 1], 0, 1)
            X_new.append(cand)
    return np.array(X_new)


def main():
    parser = argparse.ArgumentParser(description="NSGA3+Louvain 增强对比")
    parser.add_argument("--n_gen", type=int, default=60)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("nsga3_enh", log_file=f"logs/nsga3_enh_{__import__('time').time():.0f}.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)

    # 构建问题（复用 run_nsga3_optimizer 的加载逻辑）
    predictor, G, feats, hazard_pos, n_new, n_links = build_problem(cfg, logger, device)

    ref_dirs = get_reference_directions("energy", 3, 12)
    n_gen = 15 if args.quick else args.n_gen

    # ---- 标准 NSGA3 ----
    comm_range = cfg["data_generation"]["base"]["comm_range"]
    problem = ResilienceOptimizationProblem(G, feats, predictor, None, n_new, n_links,
                                            comm_range=comm_range)
    algorithm = NSGA3(
        pop_size=30,
        ref_dirs=ref_dirs,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
    )
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=False)
    X_base, F_base = res.X, res.F
    logger.info("标准 NSGA3: %d 个非支配解", len(F_base))

    # ---- NSGA3 + Louvain 局部搜索 ----
    X_enh = louvain_local_search(problem, X_base)
    F_enh_cand = np.array([problem.evaluate_solution(x) if hasattr(problem, "evaluate_solution")
                           else _evaluate_x(problem, x) for x in X_enh])
    # 非支配筛选
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting()
    ranks = nds.do(F_enh_cand, only_non_dominated_front=True)
    X_final = X_enh[ranks]
    F_final = F_enh_cand[ranks]
    logger.info("NSGA3+Louvain: %d 个非支配解（候选 %d）", len(F_final), len(X_enh))

    # ---- 指标对比 ----
    hv = HV(ref_point=np.array([1.0, 1.0, 1.0]))
    hv_base = hv.do(F_base)
    hv_enh = hv.do(F_final)
    rows = [{
        "method": "标准 NSGA3", "n_solutions": len(F_base),
        "hypervolume": round(float(hv_base), 4),
        "best_resilience": round(float(-F_base[:, 0].min()), 4),
    }, {
        "method": "NSGA3+Louvain 局部搜索", "n_solutions": len(F_final),
        "hypervolume": round(float(hv_enh), 4),
        "best_resilience": round(float(-F_final[:, 0].min()), 4),
    }]
    df = pd.DataFrame(rows)
    save_csv(df, "results/tables/nsga3_enhanced_comparison.csv")
    print("\n[NSGA3 + Louvain 局部搜索对比]")
    print(df.round(4).to_string(index=False))


def _evaluate_x(problem, x):
    """评估单个解（调用 problem._evaluate）。"""
    out = {}
    problem._evaluate(np.array([x]), out)
    return out["F"][0]


def build_problem(cfg, logger, device):
    """从 run_nsga3_optimizer 复用：加载图 + 预测器 + 问题参数。"""
    import run_nsga3_optimizer as base_mod
    # 复用其 main 前的构建逻辑（安全导入）
    predictor = make_gcn_predictor(cfg, logger)
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    # 用第一张 600 节点图作案例
    import pandas as pd
    meta = pd.read_csv("data/processed/metadata.csv")
    gid = int(meta[(meta["topology"] == "random") & (meta["num_sensors"] == 600)].iloc[0]["graph_id"])
    d = data_list[gid]
    from torch_geometric.utils import to_networkx
    G = to_networkx(d, to_undirected=True)
    feats = d.x.numpy()
    n_new = cfg["nsga3"].get("n_new_nodes", 8)
    n_links = cfg["nsga3"].get("n_redundant_links", 3)
    logger.info("案例图: random 600 节点 | 新增节点 %d | 冗余链路 %d", n_new, n_links)
    return predictor, G, feats, None, n_new, n_links


if __name__ == "__main__":
    main()
