#!/usr/bin/env python3
"""
train_picgn.py - 物理信息级联图网络（PICG-Net）训练脚本

损失 = L_cls + λ_load·L_load + λ_phys·L_phys
  L_cls  : 最后时间步失效概率 BCE（分类）
  L_load : LoadHead 负载预测 MSE（物理量监督）
  L_phys : 失效概率 vs 过载指示器 BCE（过载一致性物理约束）

消融开关：--no_attention / --no_hyperedge / --no_phys / --no_load

用法:
    python train_picgn.py --quick --num_graphs 60
    python train_picgn.py --epochs 100 --num_graphs 200
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
import networkx as nx

from src.utils import load_config, set_seed, get_device, setup_logger, get_project_root
from src.gcn_lstm import PICGNet
from src.resilience_labels import simulate_cascade_failure, compute_node_load_overload
from src.build_hypergraph import build_hypergraph, to_tensors


def build_data(data_list, n_steps, num_graphs, strategy, seed, device):
    """构建 PICG-Net 训练数据：序列 + 失效标签 + 负载/过载 + 超图结构。"""
    seqs, labels = [], []
    loads, overloads, omasks = [], [], []
    edge_index = None
    hyper = None
    n_used = 0

    for data in data_list:
        n = data.x.shape[0]
        feats = data.x.numpy()
        if n != 600:            # 固定拓扑要求（沿用现有 GCN-LSTM 做法）
            continue
        # 重建 nx.Graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        ei = data.edge_index.numpy()
        G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
        # 缓存介数（simulate_cascade_failure / compute_node_load_overload 复用，避免重复计算）
        if n > 1:
            try:
                G.graph["betweenness"] = nx.betweenness_centrality(G, k=150, seed=seed)
            except Exception:
                G.graph["betweenness"] = dict(nx.degree(G))

        # 级联标签
        _, failed_sets = simulate_cascade_failure(
            G, failure_ratio=0.05, n_steps=n_steps, seed=seed)

        # 序列：静态特征（所有时间步相同，不随失效衰减，避免标签泄漏）
        # 级联失效应从"拓扑 + 静态特征"递推预测，而非从输入特征的衰减直接读出
        seq = [feats.copy() for _ in range(n_steps)]

        # 标签：每步失效
        lab = np.zeros((n_steps, n), dtype=np.float32)
        for t in range(min(n_steps, len(failed_sets))):
            for node in failed_sets[t]:
                if node < n:
                    lab[t, node] = 1.0

        # 负载 / 过载 / 过载指示器（静态，基于全网络负载，不泄漏失效标签）
        node_load, overload, omask = compute_node_load_overload(G, theta=1.2, seed=seed)

        seqs.append(np.stack(seq, axis=0))
        labels.append(lab)
        loads.append(node_load)
        overloads.append(overload)   # (N,) 静态
        omasks.append(omask)

        if edge_index is None:
            edge_index = torch.tensor(ei, dtype=torch.long)
            # 超图（取第一个图的拓扑构建，batch 共享）
            hyper = build_hypergraph(G, strategy=strategy, features=feats, seed=seed)

        n_used += 1
        if n_used >= num_graphs:
            break

    if n_used == 0:
        raise RuntimeError("未找到 600 节点图，请检查 topologies.pt")

    X = torch.tensor(np.stack(seqs), dtype=torch.float)
    Y = torch.tensor(np.stack(labels), dtype=torch.float)
    node_load = torch.tensor(np.stack(loads), dtype=torch.float)        # (B, N)
    overload = torch.tensor(np.stack(overloads), dtype=torch.float)     # (B, T, N) 动态
    omask = torch.tensor(np.stack(omasks), dtype=torch.float)           # (B, N)
    het, hptr = to_tensors(hyper["hyperedge_to_node"], hyper["hyperedge_ptr"], device)
    return X, Y, edge_index, node_load, overload, omask, het, hptr, hyper


def picgn_loss(probs, load_pred, labels, node_load, omask,
               lambda_load=0.1, lambda_phys=0.1, use_load=True, use_phys=True, step=2):
    """PICG-Net 三损失（评估第 step 步，默认第3步使标签相对平衡）。
    probs/load_pred: (B,T,N,1)；labels: (B,T,N)；node_load/omask: (B,N)。"""
    p = probs[:, step, :, 0]            # (B, N) 第 step 步失效概率
    y = labels[:, step]                 # (B, N)
    loss_cls = F.binary_cross_entropy(p, y)
    lp = load_pred[:, step, :, 0]       # (B, N) 负载预测
    loss_load = F.mse_loss(lp, node_load) if use_load else torch.zeros_like(loss_cls)
    loss_phys = F.binary_cross_entropy(p, omask) if use_phys else torch.zeros_like(loss_cls)
    total = loss_cls + lambda_load * loss_load + lambda_phys * loss_phys
    return total, loss_cls, loss_load, loss_phys


def evaluate(probs, labels, step=2):
    """节点级失效预测指标（第 step 步，默认第3步）。"""
    p = probs[:, step, :, 0].cpu().numpy()
    y = labels[:, step].cpu().numpy()
    pred = (p > 0.5).astype(np.float32)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    return {
        "accuracy": float(accuracy_score(y.ravel(), pred.ravel())),
        "macro_f1": float(f1_score(y.ravel(), pred.ravel(), average="macro", zero_division=0)),
        "precision": float(precision_score(y.ravel(), pred.ravel(), zero_division=0)),
        "recall": float(recall_score(y.ravel(), pred.ravel(), zero_division=0)),
        "auc": float(roc_auc_score(y.ravel(), p.ravel())),
    }


def main():
    parser = argparse.ArgumentParser(description="PICG-Net 训练")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num_graphs", type=int, default=120, help="训练用图数量")
    parser.add_argument("--strategy", default="features", help="超边生成策略: community/features/kclique/none")
    parser.add_argument("--lambda_load", type=float, default=0.1)
    parser.add_argument("--lambda_phys", type=float, default=0.1)
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--no_hyperedge", action="store_true")
    parser.add_argument("--no_load", action="store_true")
    parser.add_argument("--no_phys", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("train_picgn", log_file="logs/train_picgn.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)
    logger.info("设备: %s", device)

    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    n_steps = cfg["topology"]["cascade_steps"]
    num_graphs = 24 if args.quick else args.num_graphs
    X, Y, edge_index, node_load, overload, omask, het, hptr, hyper = build_data(
        data_list, n_steps, num_graphs, args.strategy, cfg["general"]["seed"], device)
    logger.info("数据: X=%s Y=%s 超边=%d", tuple(X.shape), tuple(Y.shape),
                hyper["num_hyperedges"])

    # 划分 train/val/test
    B = X.shape[0]
    idx = np.random.default_rng(0).permutation(B)
    n_train = int(B * 0.7)
    n_val = int(B * 0.15)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    in_c = X.shape[-1]
    model = PICGNet(in_channels=in_c, hidden_dim=cfg["gcn_lstm"]["hidden_dim"],
                    lstm_layers=cfg["gcn_lstm"]["lstm_layers"],
                    dropout=cfg["gcn_lstm"]["dropout"],
                    use_attention=not args.no_attention,
                    use_hyperedge=not args.no_hyperedge).to(device)
    edge_index = edge_index.to(device)

    epochs = 5 if args.quick else (args.epochs or cfg["gcn_lstm"]["epochs"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["gcn_lstm"]["lr"])
    bs = cfg["gcn_lstm"]["batch_size"]

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        perm = np.random.permutation(n_train)
        for bi in range(0, n_train, bs):
            bidx = perm[bi:bi + bs]
            xb = X[bidx].to(device); yb = Y[bidx].to(device)
            lb = node_load[bidx].to(device); ob = overload[bidx].to(device)
            mb = omask[bidx].to(device)
            opt.zero_grad()
            probs, load_pred = model(xb, edge_index, overload=ob,
                                     hyperedge_to_node=het, hyperedge_ptr=hptr)
            loss, lc, ll, lp = picgn_loss(probs, load_pred, yb, lb, mb,
                                          args.lambda_load, args.lambda_phys,
                                          use_load=not args.no_load, use_phys=not args.no_phys)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, (n_train + bs - 1) // bs)
        if (epoch + 1) % max(1, epochs // 5) == 0 or args.quick:
            logger.info("Epoch %d/%d loss=%.5f (cls=%.4f load=%.4f phys=%.4f)",
                        epoch + 1, epochs, epoch_loss, lc.item(), ll.item(), lp.item())
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        probs_t, load_pred_t = model(X[test_idx].to(device), edge_index,
                                     overload=overload[test_idx].to(device),
                                     hyperedge_to_node=het, hyperedge_ptr=hptr)
        probs_t = probs_t.cpu(); load_pred_t = load_pred_t.cpu()
    metrics = evaluate(probs_t, Y[test_idx])
    # 物理一致性：预测失效节点与真实过载节点的命中率
    p = (probs_t[:, -1, :, 0] > 0.5).numpy()
    om = omask[test_idx].numpy()
    hit = (p * om).sum() / max(1.0, p.sum())
    metrics["physics_hit"] = float(hit)
    metrics["train_time_s"] = round(train_time, 1)
    logger.info("测试指标: %s", {k: round(v, 4) for k, v in metrics.items()})

    print("\n[PICG-Net 训练完成]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    # 保存
    torch.save({"state_dict": model.state_dict(), "metrics": metrics,
                "in_channels": in_c}, "models/picgn_cascade.pth")
    print("模型已保存: models/picgn_cascade.pth")


if __name__ == "__main__":
    main()
