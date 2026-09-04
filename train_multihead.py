#!/usr/bin/env python3
"""
train_multihead.py - P0.2 多指标联合预测（多任务学习）
对标 MIRA-GCN（Neurocomputing）的多指标融合思想：
WideGAT 多头输出 robustness/survivability/composite_score，
多任务损失 L = Σ λₖ·MSE(yₖ, ŷₖ)，与单任务基线对比。

用法:
    python train_multihead.py [--epochs 150]
输出:
    results/tables/multihead_comparison.csv
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

from src.utils import load_config, set_seed, get_device, save_csv, setup_logger, get_project_root
from src.advanced_models import WideGATNet
from src.gcn_regressor import evaluate_regression


class WideGATMultiHead(nn.Module):
    """WideGAT 多任务版：共享 GNN 主干 + 3 个回归头。"""

    def __init__(self, in_channels=35, hidden_dim=256, num_layers=4,
                 dropout=0.15, n_tasks=3):
        super().__init__()
        self.base = WideGATNet(in_channels, hidden_dim, num_layers, dropout)
        # 替换最后一层为多头
        shared = self.base.lin2
        self.base.lin2 = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(n_tasks)
        ])

    def forward(self, data, return_all=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.gelu(self.base.input_proj(x))
        x = F.dropout(x, p=self.base.dropout, training=self.training)
        for i in range(self.base.num_layers):
            h = self.base.convs[i](x, edge_index)
            h = self.base.norms[i](h)
            x = F.gelu(h + x)
            if i < self.base.num_layers - 1:
                x = F.dropout(x, p=self.base.dropout, training=self.training)
        x = torch_global_mean_pool(x, batch)
        x = F.gelu(self.base.lin1(x))
        x = F.dropout(x, p=self.base.dropout, training=self.training)
        outs = [head(x).squeeze(-1) for head in self.heads]
        return torch.stack(outs, dim=-1)  # (B, n_tasks)


def torch_global_mean_pool(x, batch):
    from torch_geometric.nn import global_mean_pool
    return global_mean_pool(x, batch)


def load_data(data_list, seed=42):
    n = len(data_list)
    idx = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    return ([data_list[i] for i in idx[:n_train]],
            [data_list[i] for i in idx[n_train:n_train + n_val]],
            [data_list[i] for i in idx[n_train + n_val:]])


def main():
    parser = argparse.ArgumentParser(description="多指标联合预测")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("multihead", log_file="logs/multihead.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    train_d, val_d, test_d = load_data(data_list)
    logger.info("数据集: train=%d val=%d test=%d", len(train_d), len(val_d), len(test_d))
    in_c = train_d[0].x.shape[1]

    tl = DataLoader(train_d, batch_size=32, shuffle=True)
    vl = DataLoader(val_d, batch_size=32)
    tel = DataLoader(test_d, batch_size=32)
    epochs = 5 if args.quick else args.epochs

    results = []
    # ---- 单任务基线（composite only）----
    model = WideGATNet(in_c, 256, 4, 0.15).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for d in tl:
            d = d.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(d), d.y)
            loss.backward()
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for d in vl:
                d = d.to(device)
                vloss += F.mse_loss(model(d), d.y).item() * len(d.y)
        vloss /= max(len(vl.dataset), 1)
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 40:
            break
    t_single = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for d in tel:
            d = d.to(device)
            yt.extend(d.y.cpu().tolist())
            yp.extend(model(d).cpu().tolist())
    m = evaluate_regression(np.array(yt), np.array(yp))
    results.append({"method": "单任务 (composite only)", "composite_R2": m["R2"],
                    "robustness_R2": float("nan"), "survivability_R2": float("nan"),
                    "train_s": round(t_single)})
    logger.info("单任务 → composite R²=%.4f", m["R2"])
    print(f"单任务: composite R²={m['R2']:.4f} | {t_single:.0f}s")

    # ---- 多任务（3 头联合）----
    model = WideGATMultiHead(in_c, 256, 4, 0.15, n_tasks=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for d in tl:
            d = d.to(device)
            opt.zero_grad()
            out = model(d)  # (B, 3)
            yt = torch.stack([d.y, d.robustness, d.survivability], dim=-1).to(device)
            # 多任务损失（权重 λ: composite 0.4, robustness 0.3, survivability 0.3）
            lam = torch.tensor([0.4, 0.3, 0.3], device=device)
            loss = (lam * (out - yt) ** 2).sum(dim=-1).mean()
            loss.backward()
            opt.step()
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for d in vl:
                d = d.to(device)
                out = model(d)
                yt = torch.stack([d.y, d.robustness, d.survivability], dim=-1).to(device)
                vloss += ((out - yt) ** 2).mean().item() * len(d.y)
        vloss /= max(len(vl.dataset), 1)
        sched.step(vloss)
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 50:
            break
    t_multi = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for d in tel:
            d = d.to(device)
            preds.append(model(d).cpu())
            ys.append(torch.stack([d.y, d.robustness, d.survivability], dim=-1).cpu())
    preds = torch.cat(preds); ys = torch.cat(ys)
    names = ["composite", "robustness", "survivability"]
    row = {"method": "多任务 3 头联合", "train_s": round(t_multi)}
    for k, name in enumerate(names):
        mm = evaluate_regression(ys[:, k].numpy(), preds[:, k].numpy())
        row[f"{name}_R2"] = mm["R2"]
        logger.info("多任务 %s → R²=%.4f", name, mm["R2"])
    results.append(row)
    print(f"多任务: composite R²={row['composite_R2']:.4f} | "
          f"robustness R²={row['robustness_R2']:.4f} | "
          f"survivability R²={row['survivability_R2']:.4f} | {t_multi:.0f}s")

    df = pd.DataFrame(results)
    save_csv(df, "results/tables/multihead_comparison.csv")
    print("\n[多指标联合预测对比]")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
