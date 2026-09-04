#!/usr/bin/env python3
"""
train_curve.py - P1.1 韧性时间曲线预测（韧性内核三参数）
对标韧性内核卷积模型（JNLSSR）的"扰动-响应"视角：
每张图模拟"随机移除→逐步恢复"曲线，提取 3 个标签：
  absorb_depth : 吸收深度（性能最大损失比例）
  recovery_time: 恢复时间（恢复到 90% 性能的步数）
  kernel_auc   : 韧性内核 AUC（曲线下面积 / 理想面积）
WideGAT 共享主干 + 3 输出头回归。

用法:
    python train_curve.py [--epochs 100] [--quick]
输出:
    results/tables/resilience_curve.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_networkx

from src.utils import load_config, set_seed, get_device, save_csv, setup_logger, get_project_root
from src.advanced_models import WideGATNet
from src.gcn_regressor import evaluate_regression


def simulate_resilience_curve(G, remove_ratio=0.3, n_steps=20, seed=42):
    """模拟韧性时间曲线：随机移除 → 逐步恢复。

    返回 (absorb_depth, recovery_time, kernel_auc)。
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    import networkx as nx
    # 基准性能 = 完整图最大连通子团比例
    p0 = len(max(nx.connected_components(G), key=len)) / n
    # 吸收：随机移除
    n_remove = max(1, int(n * remove_ratio))
    nodes = list(G.nodes())
    removed = set(rng.choice(nodes, size=n_remove, replace=False))
    G_d = G.copy()
    G_d.remove_nodes_from(removed)
    s_min = (len(max(nx.connected_components(G_d), key=len)) / n) if G_d.number_of_nodes() else 0.0
    absorb_depth = (p0 - s_min) / max(p0, 1e-9)
    # 恢复：每步恢复 5% 被移除节点（随机）
    to_restore = list(removed)
    rng.shuffle(to_restore)
    step_size = max(1, len(to_restore) // (n_steps // 2))
    S = [s_min]
    recovered = set()
    for t in range(1, n_steps):
        for node in to_restore[(t - 1) * step_size: t * step_size]:
            recovered.add(node)
        G_r = G_d.copy()
        G_r.add_nodes_from(recovered)
        # 恢复边的近似：重新连接（简单模型：新加节点与原邻居连接，模拟链路修复）
        for u in recovered:
            for v in G.neighbors(u):
                if v in G_r.nodes():
                    G_r.add_edge(u, v)
        s_t = (len(max(nx.connected_components(G_r), key=len)) / n) if G_r.number_of_nodes() else 0.0
        S.append(s_t)
        if s_t >= 0.9 * p0:
            recovery_time = t
            break
    else:
        recovery_time = n_steps - 1
    # 韧性内核 AUC（归一化）
    S = np.array(S[:n_steps])
    kernel_auc = np.trapezoid(S) / max(n_steps - 1, 1) / max(p0, 1e-9)
    return float(np.clip(absorb_depth, 0, 1)), float(recovery_time) / max(n_steps, 1), float(np.clip(kernel_auc, 0, 1))


def main():
    parser = argparse.ArgumentParser(description="韧性时间曲线预测")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(42)
    logger = setup_logger("curve", log_file="logs/curve.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)

    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    n = len(data_list)
    idx = np.random.default_rng(42).permutation(n)
    n_use = min(n, 800 if args.quick else 2000)
    use = [data_list[i] for i in idx[:n_use]]
    logger.info("曲线标签生成: %d 图（每图 20 步恢复仿真）", len(use))

    # 生成 3 维曲线标签（缓存到 pt 避免重复计算）
    curve_path = Path("data/processed/resilience_curves.pt")
    if curve_path.exists():
        labels = torch.load(curve_path, weights_only=False)
        labels = labels[:n_use]
        logger.info("复用缓存曲线标签")
    else:
        t0 = time.time()
        labels = []
        for i, d in enumerate(use):
            G = to_networkx(d, to_undirected=True)
            labels.append(simulate_resilience_curve(G))
            if (i + 1) % 200 == 0:
                logger.info("曲线仿真 %d/%d", i + 1, n_use)
        labels = torch.tensor(labels, dtype=torch.float)
        torch.save(labels, curve_path)
        logger.info("曲线标签生成完成: %.0fs", time.time() - t0)

    # 划分
    n_tr, n_va = int(n_use * 0.7), int(n_use * 0.15)
    tr_i, va_i, te_i = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:n_use]
    train_d = [data_list[i] for i in tr_i]
    val_d = [data_list[i] for i in va_i]
    test_d = [data_list[i] for i in te_i]
    lab_tr = labels[:n_tr]
    lab_va = labels[n_tr:n_tr + n_va]
    lab_te = labels[n_tr + n_va:n_use]
    in_c = train_d[0].x.shape[1]

    model = WideGATNet(in_c, 256, 4, 0.15).to(device)
    # 替换输出层为 3 维（须在同一 device）
    model.lin2 = nn.Linear(128, 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=12)
    epochs = 5 if args.quick else args.epochs

    # 标签对齐：use 局部索引 → graph_id 映射（DataLoader batch 后 graph_id 是 (B,) tensor）
    gid_to_local = {int(d.graph_id): i for i, d in enumerate(use)}

    def to_batches(ds, labs, shuffle, offset=0):
        dl = DataLoader(ds, batch_size=32, shuffle=shuffle)
        out = []
        for d in dl:
            d = d.to(device)
            gids = d.graph_id.cpu().numpy() if hasattr(d, "graph_id") else None
            if gids is not None:
                # 全局 use 索引 → 切分子集局部索引（减偏移）
                local_idx = [gid_to_local[int(g)] - offset for g in gids]
                lab = labs[local_idx].to(device)
            else:
                lab = labs[:len(d.y)].to(device)
            out.append((d, lab))
        return out

    tr_b = to_batches(train_d, lab_tr, True, offset=0)
    va_b = to_batches(val_d, lab_va, False, offset=n_tr)
    te_b = to_batches(test_d, lab_te, False, offset=n_tr + n_va)

    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for d, lab in tr_b:
            opt.zero_grad()
            out = model(d)
            loss = F.mse_loss(out, lab)
            loss.backward()
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for d, lab in va_b:
                vloss += F.mse_loss(model(d), lab).item() * len(d.y)
        vloss /= max(len(val_d), 1)
        sched.step(vloss)
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 40:
            break
    train_s = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for d, lab in te_b:
            ys.append(lab.cpu())
            ps.append(model(d).cpu())
    ys, ps = torch.cat(ys), torch.cat(ps)

    names = ["absorb_depth", "recovery_time", "kernel_auc"]
    rows = []
    for k, name in enumerate(names):
        m = evaluate_regression(ys[:, k].numpy(), ps[:, k].numpy())
        rows.append({"label": name, "R2": round(m["R2"], 4), "RMSE": round(m["RMSE"], 4),
                     "MAE": round(m["MAE"], 4)})
        logger.info("曲线参数 %s → R²=%.4f", name, m["R2"])
    df = pd.DataFrame(rows)
    save_csv(df, "results/tables/resilience_curve.csv")
    print("\n[韧性时间曲线预测（韧性内核三参数）]")
    print(df.round(4).to_string(index=False))
    print(f"训练 {train_s:.0f}s")


if __name__ == "__main__":
    main()
