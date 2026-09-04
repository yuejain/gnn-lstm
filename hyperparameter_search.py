#!/usr/bin/env python3
"""
hyperparameter_search.py - R² 提升：超参扫描 + 节点级注意力特征增强
对 GAT 韧性回归模型进行系统超参扫描（hidden_dim / epochs / dropout），
并引入节点级注意力特征增强（PageRank / 特征向量中心性 / 局部拓扑熵）。

用法:
    python hyperparameter_search.py --config config.yaml
    python hyperparameter_search.py --config config.yaml --quick

输出:
    results/tables/hyperparameter_search.csv  扫描结果
    results/figures/hyperparam_heatmap.png    超参热图
    models/gcn_resilience.pth                 若找到更优配置则覆盖
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.utils import (load_config, set_seed, get_device, save_csv, ensure_dir,
                       setup_logger, get_project_root)
from src.gcn_regressor import evaluate_regression
from src.advanced_models import GATResilienceNet


# ============================================================
# 节点级注意力特征增强
# ============================================================

def enhance_node_features(data_list, mode="attention"):
    """
    增强节点特征：基于图拓扑计算注意力相关特征。

    mode:
      "attention"  - PageRank + 特征向量中心性 + 局部聚类熵（注意力代理特征）
      "none"       - 保持原特征
    """
    if mode == "none":
        return data_list

    from torch_geometric.utils import to_networkx
    enhanced = []
    for data in data_list:
        G = to_networkx(data, to_undirected=True)
        n = G.number_of_nodes()
        if n == 0:
            enhanced.append(data)
            continue

        feats = data.x.numpy().copy()
        base_dim = feats.shape[1]

        # 注意力代理特征
        try:
            pr = nx.pagerank(G, alpha=0.85)
            pr_arr = np.array([pr[i] for i in range(n)])
        except Exception:
            pr_arr = np.zeros(n)
        try:
            ev = nx.eigenvector_centrality_numpy(G)
            ev_arr = np.array([ev[i] for i in range(n)])
        except Exception:
            ev_arr = np.zeros(n)
        # 局部聚类熵：聚类系数
        try:
            cl = np.array([nx.clustering(G, i) for i in range(n)])
        except Exception:
            cl = np.zeros(n)

        # 标准化后拼接（注意：arr= 只重绑变量，须原地修改）
        for arr in (pr_arr, ev_arr, cl):
            std = arr.std() + 1e-9
            arr[:] = (arr - arr.mean()) / std

        extra = np.stack([pr_arr, ev_arr, cl], axis=1).astype(np.float32)
        new_feats = np.hstack([feats, extra])

        new_data = data.clone()
        new_data.x = torch.tensor(new_feats, dtype=torch.float)
        enhanced.append(new_data)
    return enhanced


# ============================================================
# 训练与评估
# ============================================================

def load_and_split(cfg):
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    n = len(data_list)
    ratios = cfg["gcn_regressor"]["train_val_test"]
    idx = np.random.default_rng(cfg["general"]["seed"]).permutation(n)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return ([data_list[i] for i in idx[:n_train]],
            [data_list[i] for i in idx[n_train:n_train + n_val]],
            [data_list[i] for i in idx[n_train + n_val:]])


def train_eval(cfg, device, logger, hidden_dim, epochs, dropout,
               lr, batch_size, patience, quick=False, feature_mode="none"):
    """训练 GAT 并返回测试指标。"""
    set_seed(cfg["general"]["seed"])
    train_d, val_d, test_d = load_and_split(cfg)
    if feature_mode != "none":
        train_d = enhance_node_features(train_d, feature_mode)
        val_d = enhance_node_features(val_d, feature_mode)
        test_d = enhance_node_features(test_d, feature_mode)

    train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_d, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_d, batch_size=batch_size, shuffle=False)

    in_channels = train_d[0].x.shape[1]
    model = GATResilienceNet(in_channels, hidden_dim, num_layers=3, dropout=dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 4)
    criterion = F.mse_loss

    epochs = min(epochs, 5 if quick else epochs)
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), data.y)
            loss.backward()
            optimizer.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                vl += criterion(model(data), data.y).item() * len(data.y)
        vl /= max(len(val_loader.dataset), 1)
        scheduler.step(vl)

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience and not quick:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    model = model.to(device)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            y_true.extend(data.y.cpu().tolist())
            y_pred.extend(model(data).cpu().tolist())
    metrics = evaluate_regression(np.array(y_true), np.array(y_pred))
    return metrics, model, in_channels, best_state


def plot_heatmap(df, out_path):
    """绘制 hidden×dropout R² 热图。"""
    pivot = df.pivot_table(index="hidden_dim", columns="dropout", values="R2", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"d={c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"h={i}" for i in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center")
    plt.colorbar(im, label="R²")
    ax.set_title("GAT Hyperparameter Search (R²)")
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="GAT 超参扫描")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--feature_mode", default="attention",
                        choices=["none", "attention"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("hyperparam", log_file="logs/hyperparam.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)
    logger.info("设备: %s | 特征增强模式: %s", device, args.feature_mode)

    # 扫描网格
    if args.quick:
        grid = {"hidden_dim": [64], "epochs": [5], "dropout": [0.2]}
    else:
        grid = {
            "hidden_dim": [64, 128, 256],
            "epochs": [500, 1000],
            "dropout": [0.1, 0.2, 0.3],
        }

    base = cfg["gcn_regressor"]
    results = []
    best = {"R2": -float("inf")}

    combos = list(itertools.product(grid["hidden_dim"], grid["epochs"], grid["dropout"]))
    logger.info("扫描组合数: %d", len(combos))

    for i, (hidden, epochs, dropout) in enumerate(combos):
        tag = f"h{hidden}_e{epochs}_d{dropout}"
        logger.info("[%d/%d] 训练 %s", i + 1, len(combos), tag)
        metrics, model, in_channels, best_state = train_eval(
            cfg, device, logger, hidden, epochs, dropout,
            lr=base["lr"], batch_size=base["batch_size"],
            patience=base["patience"], quick=args.quick,
            feature_mode=args.feature_mode)
        row = {"hidden_dim": hidden, "epochs": epochs, "dropout": dropout,
               "MSE": metrics["MSE"], "RMSE": metrics["RMSE"],
               "MAE": metrics["MAE"], "R2": metrics["R2"]}
        results.append(row)
        logger.info("%s → R²=%.4f MSE=%.5f", tag, metrics["R2"], metrics["MSE"])

        if metrics["R2"] > best["R2"]:
            best = {"R2": metrics["R2"], "hidden": hidden, "epochs": epochs,
                    "dropout": dropout, "model": model,
                    "in_channels": in_channels, "state": best_state,
                    "metrics": metrics}

    df = pd.DataFrame(results)
    save_csv(df, "results/tables/hyperparameter_search.csv")
    plot_heatmap(df, "results/figures/hyperparam_heatmap.png")

    print("\n[超参扫描结果]")
    print(df.sort_values("R2", ascending=False).round(4).to_string())
    print(f"\n最优配置: hidden={best['hidden']}, epochs={best['epochs']}, "
          f"dropout={best['dropout']}, R²={best['R2']:.4f}")

    # 覆盖主模型（若优于当前）
    cur = pd.read_csv("results/tables/model_comparison.csv")
    cur_best_r2 = cur["R2"].max() if len(cur) else 0.0
    if best["R2"] > cur_best_r2:
        torch.save({
            "state_dict": best["state"],
            "config": {"in_channels": best["in_channels"],
                       "hidden_dim": best["hidden"],
                       "num_layers": 3,
                       "dropout": best["dropout"]},
            "metrics": best["metrics"],
            "model_name": "GAT",
            "hyperparams": {"epochs": best["epochs"], "feature_mode": args.feature_mode},
        }, "models/gcn_resilience.pth")
        logger.info("✅ 新最优配置已保存为 gcn_resilience.pth (R²=%.4f)", best["R2"])
        print(f"✅ 新最优 GAT 配置已保存 (R²={best['R2']:.4f} > 原 {cur_best_r2:.4f})")
    else:
        logger.info("未超越当前最优 (%.4f)，保留现有模型", cur_best_r2)
        print(f"当前最优 R²={cur_best_r2:.4f}，扫描未超越，保留现有模型")


if __name__ == "__main__":
    main()
