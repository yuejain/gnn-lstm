#!/usr/bin/env python3
"""
train_compare.py - PICG-Net 消融对比实验（P0 核心机制验证）

一次加载数据，训练多个模型变体，输出对比表：
  GCN-LSTM          : 现有基线（纯 BCE）
  PICG-Net          : 完整版（超图 LR-MP + 物理感知注意力 + 物理约束损失）
  PICG-Net -phys    : 无物理约束（--no_load --no_phys）消融
  PICG-Net -hyper   : 无超图（--no_hyperedge）消融
  PICG-Net -attn    : 无物理感知注意力（--no_attention）消融

用法:
    python train_compare.py --epochs 60 --num_graphs 100
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
import torch.nn.functional as F

from src.utils import load_config, set_seed, get_device, setup_logger, get_project_root
from src.gcn_lstm import GCN_LSTM_Cascade, PICGNet
from train_picgn import build_data, picgn_loss, evaluate


def train_gnn_lstm(model, X_tr, Y_tr, X_te, edge_index, epochs, bs, device, lr=1e-3):
    """基线 GCN-LSTM（纯 BCE）。训练用 X_tr，评估返回 X_te 上预测。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B = X_tr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(B)
        for bi in range(0, B, bs):
            idx = perm[bi:bi + bs]
            xb, yb = X_tr[idx].to(device), Y_tr[idx].to(device)
            opt.zero_grad()
            out = model(xb, edge_index)
            pred = out[..., 0]
            loss = F.binary_cross_entropy(pred[:, 2], yb[:, 2])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(X_te.to(device), edge_index)
    return out.cpu()


def train_picgn(model, X_tr, Y_tr, X_te, edge_index, node_load_tr, overload_tr, omask_tr,
                overload_te, het, hptr, epochs, bs, device,
                lambda_load, lambda_phys, use_load, use_phys, lr=1e-3):
    """PICG-Net（物理约束损失）。训练用 X_tr，评估返回 X_te 上预测。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B = X_tr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(B)
        for bi in range(0, B, bs):
            idx = perm[bi:bi + bs]
            xb = X_tr[idx].to(device); yb = Y_tr[idx].to(device)
            lb = node_load_tr[idx].to(device); ob = overload_tr[idx].to(device)
            mb = omask_tr[idx].to(device)
            opt.zero_grad()
            probs, load_pred = model(xb, edge_index, overload=ob,
                                     hyperedge_to_node=het, hyperedge_ptr=hptr)
            loss, _, _, _ = picgn_loss(probs, load_pred, yb, lb, mb,
                                       lambda_load, lambda_phys, use_load, use_phys)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        probs, _ = model(X_te.to(device), edge_index, overload=overload_te.to(device),
                         hyperedge_to_node=het, hyperedge_ptr=hptr)
    return probs.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--num_graphs", type=int, default=100)
    ap.add_argument("--strategy", default="features")
    ap.add_argument("--lambda_load", type=float, default=0.1)
    ap.add_argument("--lambda_phys", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("train_compare", log_file="logs/train_compare.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)

    n_steps = cfg["topology"]["cascade_steps"]
    X, Y, edge_index, node_load, overload, omask, het, hptr, hyper = build_data(
        torch.load("data/raw/topologies.pt", weights_only=False),
        n_steps, args.num_graphs, args.strategy, cfg["general"]["seed"], device)
    B = X.shape[0]
    idx = np.random.default_rng(0).permutation(B)
    n_train = int(B * 0.7)
    n_val = int(B * 0.15)
    tr, te = idx[:n_train], idx[n_train + n_val:]
    edge_index = edge_index.to(device)
    in_c = X.shape[-1]
    bs = cfg["gcn_lstm"]["batch_size"]
    hd = cfg["gcn_lstm"]["hidden_dim"]
    ll = cfg["gcn_lstm"]["lstm_layers"]
    dr = cfg["gcn_lstm"]["dropout"]
    logger.info("数据: B=%d train=%d test=%d 超边=%d", B, n_train, len(te), hyper["num_hyperedges"])

    rows = []

    # 1) GCN-LSTM 基线
    set_seed(cfg["general"]["seed"])
    m = GCN_LSTM_Cascade(in_c, hd, ll, dr).to(device)
    t0 = time.time()
    out = train_gnn_lstm(m, X[tr], Y[tr], X[te], edge_index, args.epochs, bs, device)
    met = evaluate(out, Y[te])
    met["variant"] = "GCN-LSTM(基线)"
    met["train_s"] = round(time.time() - t0, 1)
    rows.append(met)

    # 2) PICG-Net 完整版
    set_seed(cfg["general"]["seed"])
    m = PICGNet(in_c, hd, ll, dr, use_attention=True, use_hyperedge=True).to(device)
    t0 = time.time()
    out = train_picgn(m, X[tr], Y[tr], X[te], edge_index, node_load[tr], overload[tr], omask[tr],
                      overload[te], het, hptr, args.epochs, bs, device,
                      args.lambda_load, args.lambda_phys, True, True)
    met = evaluate(out, Y[te])
    p = (out[:, 2, :, 0] > 0.5).numpy(); om = omask[te].numpy()
    met["physics_hit"] = float((p * om).sum() / max(1.0, p.sum()))
    met["variant"] = "PICG-Net(完整)"
    met["train_s"] = round(time.time() - t0, 1)
    rows.append(met)

    # 3) PICG-Net 无物理约束
    set_seed(cfg["general"]["seed"])
    m = PICGNet(in_c, hd, ll, dr, use_attention=True, use_hyperedge=True).to(device)
    t0 = time.time()
    out = train_picgn(m, X[tr], Y[tr], X[te], edge_index, node_load[tr], overload[tr], omask[tr],
                      overload[te], het, hptr, args.epochs, bs, device, 0.0, 0.0, False, False)
    met = evaluate(out, Y[te])
    p = (out[:, 2, :, 0] > 0.5).numpy(); om = omask[te].numpy()
    met["physics_hit"] = float((p * om).sum() / max(1.0, p.sum()))
    met["variant"] = "PICG-Net(-物理)"
    met["train_s"] = round(time.time() - t0, 1)
    rows.append(met)

    # 4) PICG-Net 无超图
    set_seed(cfg["general"]["seed"])
    m = PICGNet(in_c, hd, ll, dr, use_attention=True, use_hyperedge=False).to(device)
    t0 = time.time()
    out = train_picgn(m, X[tr], Y[tr], X[te], edge_index, node_load[tr], overload[tr], omask[tr],
                      overload[te], het, hptr, args.epochs, bs, device,
                      args.lambda_load, args.lambda_phys, True, True)
    met = evaluate(out, Y[te])
    p = (out[:, 2, :, 0] > 0.5).numpy(); om = omask[te].numpy()
    met["physics_hit"] = float((p * om).sum() / max(1.0, p.sum()))
    met["variant"] = "PICG-Net(-超图)"
    met["train_s"] = round(time.time() - t0, 1)
    rows.append(met)

    # 5) PICG-Net 无注意力
    set_seed(cfg["general"]["seed"])
    m = PICGNet(in_c, hd, ll, dr, use_attention=False, use_hyperedge=True).to(device)
    t0 = time.time()
    out = train_picgn(m, X[tr], Y[tr], X[te], edge_index, node_load[tr], overload[tr], omask[tr],
                      overload[te], het, hptr, args.epochs, bs, device,
                      args.lambda_load, args.lambda_phys, True, True)
    met = evaluate(out, Y[te])
    p = (out[:, 2, :, 0] > 0.5).numpy(); om = omask[te].numpy()
    met["physics_hit"] = float((p * om).sum() / max(1.0, p.sum()))
    met["variant"] = "PICG-Net(-注意力)"
    met["train_s"] = round(time.time() - t0, 1)
    rows.append(met)

    import pandas as pd
    df = pd.DataFrame(rows).set_index("variant")
    cols = ["accuracy", "macro_f1", "auc", "precision", "recall", "physics_hit", "train_s"]
    df = df[[c for c in cols if c in df.columns]]
    print("\n========== 消融对比结果 ==========")
    print(df.round(4).to_string())
    df.round(4).to_csv("results/tables/picgn_ablation.csv")
    logger.info("对比完成，已保存 results/tables/picgn_ablation.csv")


if __name__ == "__main__":
    main()
