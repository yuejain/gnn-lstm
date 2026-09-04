#!/usr/bin/env python3
"""
load_powergraph.py - PowerGraph 电网级联失效基准数据集接入

PowerGraph（ETH Zurich, NeurIPS 2024 D&B）图级任务：
- 节点特征 Bf: 3 维（净有功功率/净视在功率/电压幅值）
- 边特征 Ef: 4 维（有功功率流/无功功率流/线路电抗/线路额定值）
- 标签 of_bi: 二分类（DNS=0 或 DNS≠0）；of_reg: 回归（DNS 值）
- 解释 exp: groundtruth 边级解释

本脚本：用 PICG-Net 的 HyperLRMP 空间编码器 + 图级读出，做级联失效二分类，
对比 PowerGraph 论文报告的 GCN/GIN/GAT/Transformer 基线。

用法: python load_powergraph.py --dataset ieee39 --datatype binary --epochs 100
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from scipy.io import loadmat

from src.utils import setup_logger, get_device
from src.hyper_lr_mp import HyperLRMP
from src.build_hypergraph import build_hypergraph, to_tensors

DATA_DIR = Path("data/powergraph")


def load_powergraph(dataset_name: str):
    """加载 PowerGraph 数据集，返回 (graph_list, labels, exp_list)。

    每个 graph 为 dict：{x: (N,3), edge_index: (2,E), edge_attr: (E,4),
                         overload: (N,), hyperedge_to_node, hyperedge_ptr}
    """
    d = DATA_DIR / dataset_name
    blist = loadmat(str(d / "blist.mat"))["blist"]            # 边列表
    of_bi = loadmat(str(d / "of_bi.mat"))["of_bi"]            # 二分类标签
    Bf = loadmat(str(d / "Bf.mat"))["Bf"]                      # 节点特征（cell）
    Ef = loadmat(str(d / "Ef.mat"))["Ef"]                      # 边特征（cell）

    graphs, labels = [], []
    n_samples = len(of_bi) if of_bi.ndim == 1 else of_bi.shape[0]
    for i in range(n_samples):
        x = np.asarray(Bf[i][0], dtype=np.float32)             # (N, 3)
        ei = np.asarray(blist[i][0], dtype=np.int64) if isinstance(blist, np.ndarray) and blist.dtype == object else np.asarray(blist, dtype=np.int64)
        # blist 可能是统一矩阵或 cell array
        if blist.dtype == object:
            ei = np.asarray(blist[i][0], dtype=np.int64)
        else:
            ei = np.asarray(blist, dtype=np.int64)
        if ei.shape[0] == 2:
            ei = ei  # (2, E)
        elif ei.shape[1] == 2:
            ei = ei.T  # (E, 2) -> (2, E)
        ei = ei - 1  # MATLAB 1-based -> 0-based
        ef = np.asarray(Ef[i][0], dtype=np.float32) if Ef.dtype == object else np.asarray(Ef, dtype=np.float32)

        N = x.shape[0]
        # 计算介数中心性作为节点负载/过载（PowerGraph 无此物理量）
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
        try:
            bc = nx.betweenness_centrality(G, k=min(100, N), seed=42)
        except Exception:
            bc = dict(nx.degree(G))
        load = np.array([bc.get(j, 0.0) for j in range(N)], dtype=np.float32)
        load = load / (load.max() + 1e-9)
        overload = load / (1.2 * (load.mean() + 1e-9))

        # 超图：按节点特征聚类
        hyper = build_hypergraph(G, strategy="features", features=x, seed=42)
        het, hptr = to_tensors(hyper["hyperedge_to_node"], hyper["hyperedge_ptr"])

        graphs.append({
            "x": torch.tensor(x), "edge_index": torch.tensor(ei, dtype=torch.long),
            "edge_attr": torch.tensor(ef), "overload": torch.tensor(overload),
            "het": het, "hptr": hptr,
        })
        labels.append(float(np.asarray(of_bi[i]).reshape(-1)[0]) if of_bi.ndim > 1 else float(of_bi[i]))
    return graphs, np.array(labels)


class GraphClassifier(nn.Module):
    """HyperLRMP 编码器 + 全局池化 + 图级分类头。"""

    def __init__(self, in_channels=3, hidden=64, num_classes=2):
        super().__init__()
        self.enc = HyperLRMP(in_channels, hidden, use_attention=True, use_hyperedge=True)
        self.pool = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes))

    def forward(self, g):
        x, ei, ov = g["x"], g["edge_index"], g["overload"]
        h = self.enc(x, ei, ov, g["het"], g["hptr"], batch_size=1)
        # 全局池化（mean + max）
        h_mean = h.mean(dim=0, keepdim=True)
        h_max = h.max(dim=0, keepdim=True).values
        hg = torch.cat([h_mean, h_max], dim=-1)
        return self.pool(hg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee39", choices=["ieee24", "ieee39", "ieee118", "uk"])
    ap.add_argument("--datatype", default="binary", choices=["binary", "multiclass", "regression"])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    logger = setup_logger("powergraph", log_file="logs/powergraph.log")
    device = get_device("auto", force_cuda=True)

    graphs, labels = load_powergraph(args.dataset)
    logger.info("PowerGraph %s: %d 图，正类占比 %.3f", args.dataset, len(graphs), labels.mean())

    # 二分类
    n = len(graphs)
    idx = np.random.default_rng(0).permutation(n)
    n_tr = int(n * 0.8)
    tr, te = idx[:n_tr], idx[n_tr:]

    model = GraphClassifier(in_channels=3, hidden=64, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y = torch.tensor(labels, dtype=torch.long)

    for g in graphs:
        g["x"] = g["x"].to(device); g["edge_index"] = g["edge_index"].to(device)
        g["edge_attr"] = g["edge_attr"].to(device); g["overload"] = g["overload"].to(device)
        g["het"] = g["het"].to(device); g["hptr"] = g["hptr"].to(device)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = 0.0
        for i in tr:
            out = model(graphs[i])
            loss = loss + F.cross_entropy(out, y[i].unsqueeze(0))
        loss = loss / len(tr)
        loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            logger.info("epoch %d loss=%.4f", ep + 1, loss.item())

    model.eval()
    correct = 0
    preds, trues = [], []
    with torch.no_grad():
        for i in te:
            out = model(graphs[i])
            pred = out.argmax(dim=-1).item()
            preds.append(pred); trues.append(y[i].item())
            correct += (pred == y[i].item())
    acc = correct / len(te)
    from sklearn.metrics import roc_auc_score, f1_score
    auc = roc_auc_score(trues, preds)
    f1 = f1_score(trues, preds, zero_division=0)
    print(f"\n[PowerGraph {args.dataset} 二分类] acc={acc:.4f} auc={auc:.4f} f1={f1:.4f} 训练 {time.time()-t0:.0f}s")
    print("（对比 PowerGraph 论文 GCN/GIN/GAT/Transformer 基线的 acc）")


if __name__ == "__main__":
    main()
