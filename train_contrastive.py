#!/usr/bin/env python3
"""
train_contrastive.py - P2.1 对比学习韧性表示（InfoNCE）
对标 MIRA-GCN（Neurocomputing）的 InfoNCE 对比损失：
L = -1/N Σ log[ exp(hᵢᵀhᵢ⁺/τ) / Σⱼ exp(hᵢᵀhⱼ⁻/τ) ]

正样本对：同变体同规模图（扰动前/后视图：特征 dropout + 边 dropout 增强）
负样本对：异变体图
预训练图编码器 → 微调 WideGAT 回归 → 对比无预训练基线。

用法:
    python train_contrastive.py [--epochs 60] [--quick]
输出:
    results/tables/contrastive_comparison.csv
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

from src.utils import load_config, set_seed, get_device, save_csv, setup_logger, get_project_root
from src.advanced_models import WideGATNet
from src.gcn_regressor import evaluate_regression


class GraphEncoder(nn.Module):
    """图编码器（预训练用）：GAT 卷积 → 图池化 → 投影头。"""

    def __init__(self, in_channels=35, hidden=256):
        super().__init__()
        self.base = WideGATNet(in_channels, hidden, 4, 0.15)
        # 投影头
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 128))

    def embed(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.gelu(self.base.input_proj(x))
        for i in range(self.base.num_layers):
            h = self.base.convs[i](x, edge_index)
            h = self.base.norms[i](h)
            x = F.gelu(h + x)
        x = global_mean_pool(x, batch)
        return x  # (B, hidden)

    def forward(self, data):
        return self.proj(self.embed(data))  # (B, 128)


def augment(data, feat_drop=0.2, edge_drop=0.15):
    """图增强：特征 dropout + 边 dropout。"""
    d = data.clone()
    mask = torch.rand(d.x.shape[0], 1, device=d.x.device) > feat_drop
    d.x = d.x * mask
    keep = torch.rand(d.edge_index.shape[1], device=d.edge_index.device) > edge_drop
    d.edge_index = d.edge_index[:, keep]
    return d


def infonce_loss(z1, z2, tau=0.5):
    """InfoNCE（对称版本）：L = -log(exp(z1·z2/τ) / Σ exp(z1·zj/τ))。"""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = z1 @ z2.T / tau  # (B, B) 对角为正对
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
    return loss / 2


def main():
    parser = argparse.ArgumentParser(description="对比学习韧性表示")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(42)
    logger = setup_logger("contrastive", log_file="logs/contrastive.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)

    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    meta = pd.read_csv("data/processed/metadata.csv")
    # 变体 → graph_id 映射（600 节点子集）
    gid_to_var = {int(r.graph_id): r.topology for r in meta.itertuples()}
    n = len(data_list)
    idx = np.random.default_rng(42).permutation(n)
    n_use = min(n, 400 if args.quick else 2400)
    use = [data_list[i] for i in idx[:n_use]]
    in_c = use[0].x.shape[1]
    logger.info("对比学习数据集: %d 图", len(use))

    # ---- 预训练编码器（InfoNCE）----
    enc = GraphEncoder(in_c).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=0.001)
    epochs = 5 if args.quick else args.epochs
    t0 = time.time()
    for epoch in range(epochs):
        enc.train()
        rng = np.random.default_rng(epoch)
        perm = rng.permutation(len(use))
        tot = 0.0
        for start in range(0, len(perm), 64):
            batch_idx = perm[start:start + 64]
            # 组 batch（相同节点数才能 stack：600 节点为主）
            batch = [use[i] for i in batch_idx]
            batch = [b for b in batch if b.num_nodes == 600]
            if len(batch) < 4:
                continue
            d = DataLoader_join(batch, device)
            z1 = enc(d)
            z2 = enc(augment(d))
            loss = infonce_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info("InfoNCE epoch %d loss=%.4f", epoch + 1, tot / max(len(perm) // 64, 1))
    pretrain_s = time.time() - t0
    logger.info("预训练完成: %.0fs", pretrain_s)
    print(f"InfoNCE 预训练: {pretrain_s:.0f}s")

    # ---- 微调评估（对比无预训练）----
    def finetune(pretrained):
        model = WideGATNet(in_c, 256, 4, 0.15).to(device)
        if pretrained:
            model.load_state_dict(enc.base.state_dict(), strict=False)
        n_tr, n_va = int(n_use * 0.7), int(n_use * 0.15)
        # use 局部索引（idx 是全局索引，不能直接索引 use）
        local_perm = np.random.default_rng(42).permutation(n_use)
        train_d = [use[i] for i in local_perm[:n_tr]]
        val_d = [use[i] for i in local_perm[n_tr:n_tr + n_va]]
        test_d = [use[i] for i in local_perm[n_tr + n_va:n_use]]
        tl = DataLoader(train_d, batch_size=32, shuffle=True)
        vl = DataLoader(val_d, batch_size=32)
        tel = DataLoader(test_d, batch_size=32)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        best_val, best_state, bad = float("inf"), None, 0
        t0 = time.time()
        for epoch in range(60):
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
            if bad >= 25:
                break
        train_s = time.time() - t0
        model.load_state_dict(best_state)
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for d in tel:
                d = d.to(device)
                yt.extend(d.y.cpu().tolist())
                yp.extend(model(d).cpu().tolist())
        m = evaluate_regression(np.array(yt), np.array(yp))
        return m, train_s

    results = []
    for pretrained, name in [(False, "无预训练（基线）"), (True, "InfoNCE 预训练+微调")]:
        m, ts = finetune(pretrained)
        results.append({"method": name, "R2": round(m["R2"], 4),
                        "RMSE": round(m["RMSE"], 4), "MAE": round(m["MAE"], 4),
                        "train_s": round(ts)})
        logger.info("%s → R²=%.4f", name, m["R2"])
        print(f"{name}: R²={m['R2']:.4f} | {ts:.0f}s")

    df = pd.DataFrame(results)
    save_csv(df, "results/tables/contrastive_comparison.csv")
    print("\n[InfoNCE 对比学习]")
    print(df.round(4).to_string(index=False))


def DataLoader_join(batch, device):
    """手动 join 同拓扑图（600 节点）。"""
    import torch_geometric.data as gd
    from torch_geometric.utils import to_dense_batch
    xs, ys, es, offs = [], [], [], []
    n_total = 0
    for b in batch:
        xs.append(b.x)
        ys.append(b.y)
        e = b.edge_index.clone()
        e = e + n_total
        es.append(e)
        n_total += b.num_nodes
    x = torch.cat(xs).to(device)
    y = torch.stack(ys).to(device)
    edge_index = torch.cat(es, dim=1).to(device)
    batch_vec = torch.arange(len(batch), device=device).repeat_interleave(
        torch.tensor([b.num_nodes for b in batch], device=device))
    return gd.Data(x=x, y=y, edge_index=edge_index, batch=batch_vec)


if __name__ == "__main__":
    main()
