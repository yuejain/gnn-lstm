#!/usr/bin/env python3
"""
generalization_test.py - 扩大测试集验证范围
用 600 节点训练的 GAT/WideGAT 模型，验证：
1. 跨规模泛化：900 / 1200 节点新图（未见规模）
2. 跨拓扑泛化：grid（网格）/ hierarchical（分层）新变体（未见拓扑）
3. 数据分布外（OOD）韧性预测误差对比

用法:
    python generalization_test.py [--quick]
输出:
    results/tables/generalization_test.csv
    results/figures/generalization_bar.png
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
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from src.utils import load_config, set_seed, get_device, save_csv, ensure_dir, setup_logger, get_project_root
from src.gcn_regressor import evaluate_regression
from src.advanced_models import build_model
from src.topology_factory import generate_sensor_positions, compute_node_features
from src.resilience_labels import compute_all_metrics


def build_grid_graph(positions, comm_range, rows=25):
    """网格拓扑：按空间行列连接（4 邻域），并保证全图连通。"""
    n = len(positions)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # 按 x 排序分列
    order = np.argsort(positions[:, 0])
    col = np.zeros(n, dtype=int)
    col[order] = np.arange(n) // rows
    n_cols = col.max() + 1
    # 列内按 y 排序连接（链式）
    for c in range(n_cols):
        nodes = np.where(col == c)[0]
        sorted_by_y = sorted(nodes, key=lambda i: positions[i, 1])
        for a, b in zip(sorted_by_y, sorted_by_y[1:]):
            G.add_edge(a, b)
    # 相邻列之间：按 y 排序后交错连接（每对相邻列连 max(rows, 5) 条边）
    for c in range(n_cols - 1):
        left = np.where(col == c)[0]
        right = np.where(col == c + 1)[0]
        left_s = sorted(left, key=lambda i: positions[i, 1])
        right_s = sorted(right, key=lambda i: positions[i, 1])
        k = max(min(len(left_s), len(right_s), rows), 5)
        for j in range(k):
            li = left_s[int(j * len(left_s) / k)]
            ri = right_s[int(j * len(right_s) / k)]
            G.add_edge(li, ri)
    # 兜底连通
    from src.topology_factory import _ensure_connected
    G = _ensure_connected(G, positions)
    return G


def build_hierarchical_graph(positions, comm_range):
    """分层拓扑：按危险源分簇，簇内密集+簇间稀疏。"""
    n = len(positions)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=comm_range)
    G.add_edges_from(pairs)
    # 簇间桥接（随机 2% 远距边）
    rng = np.random.default_rng(42)
    for _ in range(int(n * 0.02)):
        a, b = rng.choice(n, 2, replace=False)
        if not G.has_edge(a, b):
            G.add_edge(a, b)
    return G


def make_test_graphs(cfg, num_sensors, variant="grid", n_graphs=10, seed=42):
    """生成未见分布的测试图（与训练分布不同的规模/拓扑）。"""
    set_seed(seed)
    from src.wsn_data_loader import get_wsn_node_features
    wsn_dir = (get_project_root() / cfg["data_factory"]["wsn_data_dir"]).resolve()
    wsn_feats = None
    if wsn_dir.exists():
        wsn_feats = get_wsn_node_features(str(wsn_dir), n_nodes=num_sensors,
                                          seed=seed, feature_dim=5)

    graphs = []
    for g in range(n_graphs):
        set_seed(seed + g * 1000)
        positions, hazard_pos, mapping, _ = generate_sensor_positions(cfg["park"], num_sensors)
        if variant in ("grid", "hierarchical"):
            # 从 topology_factory 使用统一实现（已迁入主工厂）
            from src.topology_factory import generate_variant, build_grid_graph, build_hierarchical_graph
            vparams = cfg["topology"].get("variant_params", {}).get(variant, {})
            G = generate_variant(positions, cfg["park"]["communication_range"],
                                 variant, vparams=vparams, seed=seed + g)
        else:
            from src.topology_factory import generate_variant
            G = generate_variant(positions, cfg["park"]["communication_range"], variant,
                                 seed=seed + g)
        for i, (x, y) in enumerate(positions):
            G.nodes[i]["pos"] = (float(x), float(y))
        G.graph["hazard_pos"] = hazard_pos
        G.graph["sensor_to_hazard"] = mapping
        G.graph["features"] = compute_node_features(
            G, hazard_pos, mapping, wsn_features=wsn_feats,
            target_dim=cfg["data_factory"]["node_feature_dim"])
        G.graph["variant"] = variant
        G.graph["num_sensors"] = num_sensors
        graphs.append(G)
    return graphs


def to_pyg(G, cfg):
    """nx → PyG Data（与数据工厂一致：白名单字段）。"""
    from torch_geometric.utils import from_networkx
    data = from_networkx(G)
    data.x = torch.tensor(G.graph["features"], dtype=torch.float)
    m = compute_all_metrics(G, cfg)
    data.y = torch.tensor([m["composite_score"]], dtype=torch.float)
    data.graph_id = 0
    for attr in list(data.keys()):
        if attr not in ("edge_index", "x", "y"):
            delattr(data, attr)
    return data


def main():
    parser = argparse.ArgumentParser(description="泛化验证测试")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("generalization", log_file="logs/generalization.log")
    root = get_project_root()
    os.chdir(root)
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = get_device(cfg["general"]["device"], force_cuda=True)
    logger.info("设备: %s", device)

    # 加载主模型（优先 5000 图新模型 widegat_5000.pth，否则回退 gcn_resilience.pth）
    import os as _os
    ckpt_path = "models/widegat_5000.pth" if _os.path.exists("models/widegat_5000.pth") \
        else "models/gcn_resilience.pth"
    ckpt = torch.load(ckpt_path, weights_only=False)
    mcfg = ckpt["config"]
    model_name = ckpt.get("model_name", "GCN")
    model = build_model(model_name, mcfg["in_channels"], mcfg["hidden_dim"],
                        mcfg["dropout"], mcfg.get("num_layers", 3))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    logger.info("主模型: %s from %s (R²=%.4f)", model_name, ckpt_path,
                ckpt["metrics"].get("R2", float("nan")))

    # 场景矩阵：同分布（8 变体）+ 跨规模 + 真实网络 OOD
    variants_all = ["random", "geo_scale_free", "geo_small_world", "grid",
                    "hierarchical", "sbm", "ring", "tree"]
    scenarios = [
        # 同分布（600 节点各变体）
        *[(f"同分布 {v}", 600, v, 10) for v in variants_all],
        # 跨规模（random 变体）
        ("跨规模 300节点", 300, "random", 10),
        ("跨规模 900节点", 900, "random", 10),
        ("跨规模 1200节点", 1200, "random", 5),
        ("跨规模外推 1500节点", 1500, "random", 5),
    ]
    if args.quick:
        scenarios = [("同分布 random", 600, "random", 3),
                     ("同分布 grid", 600, "grid", 3),
                     ("跨规模 900节点", 900, "random", 3)]

    rows = []
    for label, n_nodes, variant, n_graphs in scenarios:
        logger.info("生成测试集: %s (%d 图)", label, n_graphs)
        graphs = make_test_graphs(cfg, n_nodes, variant, n_graphs)
        data_list = [to_pyg(G, cfg) for G in graphs]

        y_true, y_pred = [], []
        with torch.no_grad():
            for data in data_list:
                data = data.to(device)
                pred = model(data).cpu().item()
                y_true.append(data.y.item())
                y_pred.append(pred)
        m = evaluate_regression(np.array(y_true), np.array(y_pred))
        rows.append({"scenario": label, "n_nodes": n_nodes, "variant": variant,
                     "n_graphs": n_graphs, "MSE": m["MSE"], "RMSE": m["RMSE"],
                     "MAE": m["MAE"], "R2": m["R2"], "group": "synth"})
        logger.info("%s → R²=%.4f RMSE=%.4f", label, m["R2"], m["RMSE"])

    # ---- 真实网络 OOD 测试组 ----
    try:
        from src.public_data_loader import load_public_dataset, standardize_real_graph, to_pyg_standard
        real_sets = []
        email = load_public_dataset("email_eu")
        tz = load_public_dataset("topology_zoo", min_nodes=15, limit=20)
        haggle = load_public_dataset("haggle")
        real_sets = [("email-Eu-core", g) for g in email]
        real_sets += [(f"TopologyZoo-{g.graph.get('name','?')}", g) for g in tz]
        real_sets += [(f"Haggle-{g.graph.get('name','?')}", g) for g in haggle]
        logger.info("真实网络 OOD: %d 图（email+TopologyZoo+Haggle）", len(real_sets))
        for name, g in real_sets:
            try:
                gs = standardize_real_graph(g, cfg, seed=42)
                d = to_pyg_standard(gs, cfg).to(device)
                with torch.no_grad():
                    pred = model(d).cpu().item()
                rows.append({"scenario": f"真实网络-{name}", "n_nodes": gs.number_of_nodes(),
                             "variant": "real", "n_graphs": 1,
                             "MSE": (pred - d.y.item()) ** 2, "RMSE": abs(pred - d.y.item()),
                             "MAE": abs(pred - d.y.item()), "R2": float("nan"), "group": "real"})
            except Exception as e:
                logger.warning("真实网络 %s 评估失败: %s", name, e)
    except ImportError:
        logger.warning("public_data_loader 不可用，跳过真实网络 OOD 组")

    df = pd.DataFrame(rows)
    save_csv(df, "results/tables/generalization_test.csv")
    print("\n[泛化验证结果]")
    print(df.round(4).to_string())

    # 可视化：合成组柱状 + 真实网络组散点
    df_synth = df[df["group"] == "synth"].copy()
    df_real = df[df["group"] == "real"].copy()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(df_synth))
    colors = ["#2ca02c" if "同分布" in s else "#1f77b4" for s in df_synth["scenario"]]
    ax.bar(x, df_synth["R2"], color=colors, label="合成测试集")
    ax.axhline(0.8, color="red", ls="--", lw=1, label="R²=0.8 参考线")
    # 真实网络：MAE 散点（右侧轴，独立横坐标避免广播错误）
    ax2 = ax.twinx()
    if len(df_real):
        x_real = len(df_synth) + np.arange(len(df_real)) + 0.5
        ax2.scatter(x_real, df_real["MAE"], color="purple", marker="x",
                    s=40, label="真实网络 MAE")
        ax2.set_ylabel("MAE (真实网络)", color="purple")
    else:
        ax2.set_ylabel("MAE (真实网络)", color="purple")
    ax.set_xticks(list(x) + ([len(df_synth)] if len(df_real) else []))
    labels = list(df_synth["scenario"]) + (["真实网络 OOD"] if len(df_real) else [])
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title("Generalization Test (v2: 8 variants + real-world OOD)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    ensure_dir("results/figures")
    plt.savefig("results/figures/generalization_bar.png", dpi=300)
    plt.close()
    print(f"泛化图: results/figures/generalization_bar.png")

    # 真实网络统计摘要
    if len(df_real):
        real_mae = df_real["MAE"].dropna()
        print(f"真实网络 OOD: {len(df_real)} 图 | MAE 均值 {real_mae.mean():.4f} | "
              f"RMSE 均值 {df_real['RMSE'].dropna().mean():.4f}")


if __name__ == "__main__":
    main()
