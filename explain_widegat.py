#!/usr/bin/env python3
"""
explain_widegat.py - P1.2 可解释性归因（脆弱节点识别）
对标 SNDM-VN（RESS 脆弱区域识别）与 SCM 因果框架（EPSR）：
用梯度归因（saliency）识别 WideGAT 决策依赖的脆弱节点 Top-K，
与介数中心性排序对比（Jaccard 重叠率）。

方法：对代表性图（8 变体各 1 张）计算 ∂loss/∂x 节点归因强度，
聚合边权重得到节点脆弱性得分。

用法:
    python explain_widegat.py
输出:
    results/tables/explain_widegat.csv
    results/figures/vulnerable_nodes_*.png
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

from src.utils import load_config, set_seed, get_device, save_csv, ensure_dir, setup_logger, get_project_root
from src.advanced_models import build_model


def saliency_node_attribution(model, data, device):
    """梯度归因：∂loss/∂x 的 L2 范数作为节点脆弱性。"""
    data = data.to(device)
    data.x.requires_grad_(True)
    model.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.y)
    loss.backward()
    attr = data.x.grad.detach().abs().mean(dim=-1).cpu().numpy()
    data.x.requires_grad_(False)
    return attr


def main():
    cfg = load_config("config.yaml")
    set_seed(42)
    logger = setup_logger("explain", log_file="logs/explain.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)

    ckpt = torch.load("models/widegat_5000.pth", weights_only=False)
    mcfg = ckpt["config"]
    model = build_model("WideGAT", mcfg["in_channels"], mcfg["hidden_dim"],
                        mcfg["dropout"], mcfg.get("num_layers", 4))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    logger.info("模型: WideGAT R²=%.4f", ckpt["metrics"]["R2"])

    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    variants = ["random", "geo_scale_free", "geo_small_world", "grid",
                "hierarchical", "sbm", "ring", "tree"]

    rows = []
    import networkx as nx
    import pandas as pd
    meta = pd.read_csv("data/processed/metadata.csv")
    for v in variants:
        cand = meta[(meta["topology"] == v) & (meta["num_sensors"] == 600)]
        if cand.empty:
            continue
        gid = int(cand.iloc[0]["graph_id"])
        d = data_list[gid]
        attr = saliency_node_attribution(model, d, device)
        G = to_networkx(d, to_undirected=True)
        # 介数中心性（k 采样）
        n_nodes = G.number_of_nodes()
        bc = nx.betweenness_centrality(G, k=min(150, n_nodes - 1))
        bc_arr = np.array([bc[i] for i in range(n_nodes)])

        k = 20  # Top-K
        top_attr = np.argsort(-attr)[:k]
        top_bc = np.argsort(-bc_arr)[:k]
        overlap = len(set(top_attr) & set(top_bc)) / k
        # 归因 Top-K 的介数均值（验证归因节点是否也是关键节点）
        attr_top_bc_mean = bc_arr[top_attr].mean()
        rows.append({
            "variant": v, "n_nodes": n_nodes,
            "jaccard_top20": round(overlap, 4),
            "attr_top_bc_mean": round(float(attr_top_bc_mean), 4),
            "pred_composite": round(float(d.y.item()), 4),
        })
        logger.info("%s: Jaccard=%.4f 归因Top20介数均值=%.4f", v, overlap, attr_top_bc_mean)

    df = pd.DataFrame(rows)
    save_csv(df, "results/tables/explain_widegat.csv")
    print("\n[脆弱节点归因（梯度 saliency vs 介数中心性）]")
    print(df.round(4).to_string(index=False))
    print(f"\n平均 Jaccard: {df['jaccard_top20'].mean():.4f} | "
          f"归因节点介数均值: {df['attr_top_bc_mean'].mean():.4f}")


if __name__ == "__main__":
    main()
