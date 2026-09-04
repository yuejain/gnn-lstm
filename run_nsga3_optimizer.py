#!/usr/bin/env python3
"""
run_nsga3_optimizer.py - 阶段四：NSGA-III 韧性增强器
基于 GCN 韧性预测 + LSTM 级联风险约束，寻找最优拓扑加固方案。

用法:
    python run_nsga3_optimizer.py --config config.yaml [--generations 50] [--cores 4]
    python run_nsga3_optimizer.py --config config.yaml --quick

输出:
    results/tables/pareto_front.csv          帕累托解集
    results/tables/compromise_solution.json  折中解（距离理想点最近）
    results/figures/nsga3_pareto.png         帕累托前沿 3D 图
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from src.utils import (load_config, set_seed, get_device, save_csv, ensure_dir,
                       setup_logger, get_project_root)
from src.nsga3_optimizer import ResilienceOptimizationProblem
from src.resilience_labels import simulate_cascade_failure


# ============================================================
# 模型代理（作为 NSGA-III 适应度函数）
# ============================================================

def make_gcn_predictor(cfg, logger):
    """加载 GCN/GAT 模型作为韧性评估代理（优先 5000 图新 WideGAT 模型）。"""
    from src.advanced_models import build_model
    # 优先新模型（5000 图 WideGAT），否则回退 gcn_resilience.pth
    import os as _os
    ckpt_path = "models/widegat_5000.pth" if _os.path.exists("models/widegat_5000.pth") \
        else "models/gcn_resilience.pth"
    device = get_device(cfg["general"]["device"], force_cuda=True)

    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        mcfg = ckpt["config"]
        model_name = ckpt.get("model_name", "GCN")
        model = build_model(model_name,
                            in_channels=mcfg["in_channels"],
                            hidden_dim=mcfg["hidden_dim"],
                            dropout=mcfg["dropout"],
                            num_layers=mcfg.get("num_layers", 3))
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        logger.info("已加载 %s 韧性评估模型: %s (R²=%.4f)",
                    model_name, ckpt_path, ckpt.get("metrics", {}).get("R2", float("nan")))

        def predict_batch(items):
            """批量前向：一次 GPU 调用评估整代种群（求解加速核心）。"""
            from torch_geometric.utils import from_networkx
            from torch_geometric.data import Batch
            data_list = []
            for G, feats in items:
                data = from_networkx(G)
                data.x = torch.tensor(feats, dtype=torch.float)
                data_list.append(data)
            batch = Batch.from_data_list(data_list).to(device)
            with torch.no_grad():
                out = model(batch).cpu().numpy()
            return list(out)

        # 同时支持批量与单图接口（batch_supported 标志驱动 pymoo 批量路径）
        def predict(*args):
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                return predict_batch(args[0])
            G, feats = args
            from torch_geometric.utils import from_networkx
            data = from_networkx(G)
            data.x = torch.tensor(feats, dtype=torch.float)
            data = data.to(device)
            with torch.no_grad():
                return float(model(data).cpu().item())

        predict.batch_supported = True
        return predict

    # 兜底：用图论指标近似
    logger.warning("GCN 模型缺失，使用图论指标（k连通/鲁棒性）作为韧性代理")
    from src.resilience_labels import compute_composite_score

    def predict(G, feats):
        if G.number_of_nodes() <= 1:
            return 0.5
        largest = len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()
        k = max(0, min(d for _, d in G.degree()))
        return compute_composite_score(largest, largest, k, G.number_of_nodes())
    return predict


def make_cascade_predictor(cfg, logger, fast_mode=True):
    """
    加载级联风险评估代理。
    fast_mode=True: 用拓扑结构快速代理（连通度+度分布，微秒级、有区分度，
                    支持批量评估——NSGA-III 求解加速核心）
    fast_mode=False: 用 GCN-LSTM 模型逐图精确评估（慢但精度高）
    """
    if fast_mode:
        import numpy as _np
        from scipy.sparse import coo_matrix

        def _fast_risk(G):
            """基于连通性与度分布的快速风险代理：风险随弱连通/低度节点比例上升。"""
            n = G.number_of_nodes()
            if n <= 1:
                return 1.0
            # 连通性比例（最大连通分量）
            try:
                largest = len(max(nx.connected_components(G), key=len)) / n
            except Exception:
                largest = 1.0
            # 低度节点比例（度<=1 视为脆弱点）
            degs = [d for _, d in G.degree()]
            low_deg = sum(1 for d in degs if d <= 1) / n
            # 度分布偏斜（高偏斜→中心化风险）
            mean_d = _np.mean(degs) + 1e-9
            skew = _np.mean((_np.array(degs) - mean_d) ** 3) / (mean_d ** 3 + 1e-9)
            skew_norm = float(_np.clip(skew, 0, 1))
            risk = (1 - largest) * 0.6 + low_deg * 0.3 + skew_norm * 0.1
            return float(_np.clip(risk, 0.0, 1.0))

        def risk(G_or_list):
            if isinstance(G_or_list, (list, tuple)):
                return [_fast_risk(G) for G in G_or_list]
            return _fast_risk(G_or_list)

        risk.batch_supported = True
        logger.info("级联风险代理: 拓扑结构快速模式（批量评估，求解加速）")
        return risk

    # ---- 精确模式：GCN-LSTM 模型 ----
    ckpt_path = "models/gcn_lstm_cascade.pth"
    if Path(ckpt_path).exists():
        from src.gcn_lstm import GCN_LSTM_Cascade
        ckpt = torch.load(ckpt_path, weights_only=False)
        device = get_device(cfg["general"]["device"], force_cuda=True)
        # 特征维度与训练一致（35 维含注意力增强）
        in_dim = ckpt.get("in_channels", 32)
        model = GCN_LSTM_Cascade(in_channels=in_dim,
                                 hidden_dim=cfg["gcn_lstm"]["hidden_dim"],
                                 lstm_layers=cfg["gcn_lstm"]["lstm_layers"],
                                 dropout=cfg["gcn_lstm"]["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        logger.info("已加载 GCN-LSTM 级联风险模型: %s", ckpt_path)

        def risk(G):
            from torch_geometric.utils import from_networkx
            data = from_networkx(G)
            n = G.number_of_nodes()
            F = in_dim
            T = cfg["topology"]["cascade_steps"]
            T = min(T, 10)  # 特征列 [13:23] 最多 10 步
            feats = torch.zeros(n, F, dtype=torch.float)
            pos = nx.get_node_attributes(G, "pos")
            degrees = torch.tensor([d for _, d in G.degree()], dtype=torch.float)
            for i in range(n):
                feats[i, 1] = 1.0                       # 类型: 监测点
                p = pos.get(i, (0.0, 0.0))
                feats[i, 3:5] = torch.tensor(p) / torch.tensor(cfg["park"]["area_size"], dtype=torch.float)
                feats[i, 7] = degrees[i] / max(n - 1, 1)
                feats[i, 31] = 0.5
                if F >= 35:
                    feats[i, 32] = degrees[i] / max(n, 1)
                    feats[i, 33] = 0.5
                    feats[i, 34] = 0.5
            # 浓度序列列 [13:23]：随时间步的负载演化代理
            for t in range(T):
                feats[:, 13 + t] = degrees / max(n, 1) * (t + 1) / T
            # 组装 (1, T, N, F)
            x = feats.unsqueeze(0).unsqueeze(0).repeat(1, T, 1, 1)  # (1,T,N,F)
            for t in range(T):
                x[0, t, :, 13 + t] = feats[:, 13 + t]
            x = x.to(device)
            with torch.no_grad():
                out = model(x, data.edge_index.to(device))
                return float(out[0, -1, :, 0].mean().cpu().item())
        return risk

    # 兜底：级联仿真
    logger.warning("GCN-LSTM 模型缺失，使用级联仿真作为风险代理")

    def risk(G):
        if G.number_of_nodes() <= 1:
            return 1.0
        ratios, _ = simulate_cascade_failure(G, failure_ratio=0.05,
                                             n_steps=cfg["topology"]["cascade_steps"],
                                             seed=cfg["general"]["seed"])
        return float(ratios[-1])
    return risk


def select_compromise_solution(F, X):
    """选择距离理想点（各目标最小值）最近的折中解。"""
    ideal = F.min(axis=0)
    dists = np.linalg.norm((F - ideal) / (F.std(axis=0) + 1e-9), axis=1)
    idx = int(np.argmin(dists))
    return idx, F[idx], X[idx], dists[idx]


def run_optimization(cfg, logger, quick=False, generations=None, cores=1):
    """执行 NSGA-III 优化。"""
    set_seed(cfg["general"]["seed"])
    nsga_cfg = cfg["nsga3"]
    gen = 5 if quick else (generations or nsga_cfg["generations"])
    pop_size = min(30 if quick else nsga_cfg["population"], 100)

    # 基础拓扑：从数据集中取一张图作为"当前园区拓扑（待优化）"
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    base_data = data_list[0]
    n_base = base_data.x.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(n_base))
    ei = base_data.edge_index.numpy()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    pos_arr = base_data.x[:, 3:5].numpy() * cfg["park"]["area_size"][0]
    for i in range(n_base):
        G.nodes[i]["pos"] = (float(pos_arr[i, 0]), float(pos_arr[i, 1]))
    base_features = base_data.x.numpy()

    logger.info("基础拓扑: %d 节点, %d 边 (取数据集首图)", n_base, G.number_of_edges())

    gcn_predict = make_gcn_predictor(cfg, logger)
    cascade_risk = make_cascade_predictor(cfg, logger)

    problem = ResilienceOptimizationProblem(
        base_graph=G,
        base_features=base_features,
        gcn_predict=gcn_predict,
        cascade_risk=cascade_risk,
        n_new_nodes=nsga_cfg["n_new_nodes"],
        n_redundant_links=nsga_cfg["n_redundant_links"],
        comm_range=cfg["park"]["communication_range"],
        area_size=cfg["park"]["area_size"],
    )

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=8)  # 3目标8分区=45参考方向
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=nsga_cfg["crossover_eta"]),
        mutation=PM(prob=nsga_cfg["mutation_prob"], eta=nsga_cfg["mutation_eta"]),
        eliminate_duplicates=True,
    )

    logger.info("启动 NSGA-III: pop=%d, gen=%d, 变异概率=%.2f",
                pop_size, gen, nsga_cfg["mutation_prob"])

    res = minimize(problem, algorithm, ("n_gen", gen), seed=cfg["general"]["seed"],
                   verbose=False, save_history=False)

    F, X = res.F, res.X
    logger.info("优化完成: 非支配解 %d 个", len(F))

    # 异常回滚：若非支配解过少，调高变异概率重启
    if len(F) < max(3, pop_size // 10) and not quick:
        logger.warning("非支配解过少 (%d)，按异常回滚策略调高变异概率至 %.2f 并重启",
                       len(F), nsga_cfg["fallback_mutation_prob"])
        algorithm.mutation = PM(prob=nsga_cfg["fallback_mutation_prob"],
                                eta=nsga_cfg["mutation_eta"])
        res = minimize(problem, algorithm, ("n_gen", max(gen // 2, 10)),
                       seed=cfg["general"]["seed"] + 1, verbose=False)
        F, X = res.F, res.X
        logger.info("重启完成: 非支配解 %d 个", len(F))

    # 折中解
    comp_idx, comp_F, comp_X, comp_dist = select_compromise_solution(F, X)
    logger.info("折中解: index=%d, 目标值=%s", comp_idx, np.round(comp_F, 4).tolist())

    # 解码折中解描述
    coords = comp_X[:nsga_cfg["n_new_nodes"] * 2].reshape(-1, 2) * np.array(cfg["park"]["area_size"])
    links = (comp_X[nsga_cfg["n_new_nodes"] * 2:] > 0.5).astype(int)
    compromise = {
        "objective_resilience": round(-float(comp_F[0]), 4),
        "objective_cascade_risk": round(float(comp_F[1]), 4),
        "objective_cost": round(float(comp_F[2]), 4),
        "n_added_nodes": nsga_cfg["n_new_nodes"],
        "n_added_links": int(links.sum()),
        "new_node_coords": np.round(coords, 1).tolist(),
        "link_switches": links.tolist(),
        "distance_to_ideal": round(float(comp_dist), 4),
    }

    # 保存结果
    pareto_df = pd.DataFrame(F, columns=["neg_resilience", "cascade_risk", "cost"])
    pareto_df["resilience"] = -pareto_df["neg_resilience"]
    pareto_df = pareto_df[["resilience", "cascade_risk", "cost"]]
    pareto_df["is_compromise"] = (np.arange(len(F)) == comp_idx)
    save_csv(pareto_df, "results/tables/pareto_front.csv")

    with open("results/tables/compromise_solution.json", "w", encoding="utf-8") as f:
        json.dump(compromise, f, ensure_ascii=False, indent=2)
    logger.info("已保存: pareto_front.csv + compromise_solution.json")

    # 绘制帕累托 3D 图
    plot_pareto(F, comp_idx, "results/figures/nsga3_pareto.png")

    return pareto_df, compromise


def plot_pareto(F, comp_idx, out_path):
    """绘制帕累托前沿 3D 散点图。"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(-F[:, 0], F[:, 1], F[:, 2], c="#1f77b4", s=30, alpha=0.7,
               label="Pareto Solutions")
    ax.scatter(-F[comp_idx, 0], F[comp_idx, 1], F[comp_idx, 2],
               c="#d62728", s=120, marker="*", label="Compromise Solution")
    ax.set_xlabel("Resilience (GCN)")
    ax.set_ylabel("Cascade Risk")
    ax.set_zlabel("Cost")
    ax.set_title("NSGA-III Pareto Front - Resilience Enhancement")
    ax.legend()
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"帕累托图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="阶段四：NSGA-III 韧性增强器")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["general"]["seed"])
    # 日志文件名带时间戳，避免沙箱覆盖已存在文件（PermissionError）
    from src.utils import timestamp_str
    logger = setup_logger("nsga3", log_file=f"logs/nsga3_{timestamp_str()}.log")

    root = get_project_root()
    os.chdir(root)

    pareto_df, compromise = run_optimization(cfg, logger, quick=args.quick,
                                             generations=args.generations,
                                             cores=args.cores)

    print("\n[阶段四完成] 帕累托前沿 (前 10 个解):")
    print(pareto_df.head(10).round(4).to_string())
    print("\n折中解:")
    print(json.dumps({k: v for k, v in compromise.items() if k != "new_node_coords"},
                     ensure_ascii=False, indent=2))
    print(f"\n新增节点坐标: {compromise['new_node_coords']}")


if __name__ == "__main__":
    main()
