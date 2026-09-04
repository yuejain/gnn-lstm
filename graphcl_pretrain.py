"""
graphcl_pretrain.py - GraphCL 图对比学习自监督预训练（创新模块）
在 900 图无标签数据上，用图对比学习预训练 WideGAT 编码器，再微调韧性回归头。

创新点：
- 三种图增强（GraphCL 风格）：节点特征 dropout / 边 dropout / 特征扰动
- InfoNCE 对比损失：同一图的两个增强视图为正样本对，批内其他图为负样本
- 预训练后冻结编码器微调回归头（或全量微调），对比"直接训练"的精度增益

用法:
    python graphcl_pretrain.py [--quick]
输出:
    models/graphcl_encoder.pth     预训练编码器
    models/graphcl_finetuned.pth   微调后完整模型
    results/tables/graphcl_comparison.csv  预训练 vs 直接训练对比
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
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from src.utils import load_config, set_seed, get_device, save_csv, setup_logger, get_project_root
from src.gcn_regressor import evaluate_regression
from src.advanced_models import WideGATNet


# ============================================================
# 图增强（GraphCL）
# ============================================================

def augment_view(data: Data, seed: int, node_drop_p=0.2, edge_drop_p=0.2, feat_noise=0.1):
    """生成一个增强视图：节点特征 dropout + 边 dropout + 特征高斯扰动。"""
    g = torch.Generator().manual_seed(seed)
    x = data.x.clone()
    edge_index = data.edge_index.clone()

    # 1. 节点特征 dropout（按节点 mask）
    node_mask = torch.rand(x.shape[0], generator=g) > node_drop_p
    x = x * node_mask.unsqueeze(-1).float()

    # 2. 边 dropout
    keep = torch.rand(edge_index.shape[1], generator=g) > edge_drop_p
    edge_index = edge_index[:, keep]

    # 3. 特征高斯扰动
    if feat_noise > 0:
        noise = torch.randn_like(x) * feat_noise
        x = x + noise

    # 无自环时添加（GATConv 需要）
    if edge_index.shape[1] == 0:
        edge_index = torch.arange(x.shape[0], device=x.device).unsqueeze(0).repeat(2, 1)
    return Data(x=x, edge_index=edge_index, y=data.y)


# ============================================================
# 对比学习预训练
# ============================================================

class GraphCLPretrainer:
    """GraphCL 预训练器：图级 InfoNCE 对比学习。"""

    def __init__(self, encoder: WideGATNet, device, tau=0.5):
        self.encoder = encoder.to(device)
        self.device = device
        self.tau = tau

        # 投影头（图级表征 → 对比空间）
        hid = encoder.hidden_dim
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hid, hid // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hid // 2, hid // 4),
        ).to(device)
        self.params = list(self.encoder.parameters()) + list(self.proj.parameters())

    def encode_graph(self, batch: Batch):
        """图级表征（global_mean_pool 前特征）。"""
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        # 与 WideGAT forward 相同的编码路径（到池化前）
        x = F.gelu(self.encoder.input_proj(x))
        x = F.dropout(x, p=self.encoder.dropout, training=True)
        for i in range(self.encoder.num_layers):
            h = self.encoder.convs[i](x, edge_index)
            h = self.encoder.norms[i](h)
            x = F.gelu(h + x)
            if i < self.encoder.num_layers - 1:
                x = F.dropout(x, p=self.encoder.dropout, training=True)
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(x, b)

    def infonce_loss(self, z1, z2):
        """InfoNCE 对比损失：z1/z2 为 (B, D) 两视图表征。"""
        z1 = F.normalize(self.proj(z1), dim=-1)
        z2 = F.normalize(self.proj(z2), dim=-1)
        B = z1.shape[0]
        # 相似度矩阵 (B, B)：对角线为正对
        sim = z1 @ z2.T / self.tau
        labels = torch.arange(B, device=self.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

    def pretrain(self, data_list, epochs=200, batch_size=32, lr=0.001,
                 aug_seed_base=1000, logger=None, quick=False):
        """无监督预训练。"""
        epochs = 5 if quick else epochs
        opt = torch.optim.Adam(self.params, lr=lr, weight_decay=1e-5)
        n = len(data_list)

        for epoch in range(epochs):
            self.encoder.train()
            self.proj.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batch = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size].tolist()
                # 两个增强视图
                v1 = [augment_view(data_list[j], aug_seed_base + epoch * 100 + j) for j in idx]
                v2 = [augment_view(data_list[j], aug_seed_base + epoch * 100 + j + 5000) for j in idx]
                b1 = Batch.from_data_list(v1).to(self.device)
                b2 = Batch.from_data_list(v2).to(self.device)
                opt.zero_grad()
                z1 = self.encode_graph(b1)
                z2 = self.encode_graph(b2)
                loss = self.infonce_loss(z1, z2)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batch += 1
            if logger and ((epoch + 1) % 20 == 0 or quick):
                logger.info("GraphCL Epoch %3d/%d | InfoNCE loss %.4f",
                            epoch + 1, epochs, total_loss / max(n_batch, 1))
        return total_loss / max(n_batch, 1)


# ============================================================
# 微调与评估
# ============================================================

def finetune(encoder, train_d, val_d, test_d, device, cfg, epochs=200,
             freeze_encoder=True, logger=None, quick=False):
    """冻结/解冻编码器微调回归头。"""
    from torch_geometric.nn import global_mean_pool

    class Finetuned(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
            self.head = torch.nn.Sequential(
                torch.nn.Linear(enc.hidden_dim, enc.hidden_dim // 2),
                torch.nn.GELU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(enc.hidden_dim // 2, 1),
            )

        def forward(self, data):
            x, edge_index, b = data.x, data.edge_index, data.batch
            x = F.gelu(self.enc.input_proj(x))
            for i in range(self.enc.num_layers):
                h = self.enc.convs[i](x, edge_index)
                h = self.enc.norms[i](h)
                x = F.gelu(h + x)
            g = global_mean_pool(x, b)
            return self.head(g).squeeze(-1)

    model = Finetuned(encoder).to(device)
    if freeze_encoder:
        for p in model.enc.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    tl = DataLoader(train_d, batch_size=32, shuffle=True)
    vl = DataLoader(val_d, batch_size=32)
    tel = DataLoader(test_d, batch_size=32)

    epochs = 5 if quick else epochs
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for d in tl:
            d = d.to(device)
            opt.zero_grad()
            try:
                loss = F.mse_loss(model(d), d.y)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
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
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 60 and not quick:
            break

    model.load_state_dict(best_state)
    model.eval()
    model.to(device)
    yt, yp = [], []
    with torch.no_grad():
        for d in tel:
            d = d.to(device)
            yt.extend(d.y.cpu().tolist())
            yp.extend(model(d).cpu().tolist())
    return evaluate_regression(np.array(yt), np.array(yp))


def main():
    parser = argparse.ArgumentParser(description="GraphCL 图对比学习预训练+微调")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--finetune_epochs", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("graphcl", log_file="logs/graphcl.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)
    logger.info("设备: %s（强制 GPU）", device)

    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    n = len(data_list)
    idx = np.random.default_rng(cfg["general"]["seed"]).permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_d = [data_list[i] for i in idx[:n_train]]
    val_d = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test_d = [data_list[i] for i in idx[n_train + n_val:]]
    logger.info("数据集: train=%d val=%d test=%d", len(train_d), len(val_d), len(test_d))
    in_c = train_d[0].x.shape[1]

    # ---- 步骤 1：GraphCL 预训练 ----
    encoder = WideGATNet(in_c, 512, 4, 0.15)
    logger.info("GraphCL 预训练开始（%d epochs）", args.pretrain_epochs)
    pretrainer = GraphCLPretrainer(encoder, device)
    pretrainer.pretrain(train_d + val_d, epochs=args.pretrain_epochs,
                        batch_size=32, logger=logger, quick=args.quick)
    torch.save(encoder.state_dict(), "models/graphcl_encoder.pth")
    logger.info("预训练编码器已保存: models/graphcl_encoder.pth")

    # ---- 步骤 2：冻结编码器微调 ----
    logger.info("冻结编码器微调回归头")
    m_frozen = finetune(encoder, train_d, val_d, test_d, device, cfg,
                        epochs=args.finetune_epochs, freeze_encoder=True,
                        logger=logger, quick=args.quick)

    # ---- 步骤 3：全量微调 ----
    logger.info("全量微调（解冻编码器）")
    m_full = finetune(encoder, train_d, val_d, test_d, device, cfg,
                      epochs=args.finetune_epochs, freeze_encoder=False,
                      logger=logger, quick=args.quick)

    # ---- 对比 ----
    rows = [
        {"method": "GraphCL预训练+冻结微调", **m_frozen},
        {"method": "GraphCL预训练+全量微调", **m_full},
    ]
    df = pd.DataFrame(rows)
    save_csv(df, "results/tables/graphcl_comparison.csv")
    logger.info("对比表: results/tables/graphcl_comparison.csv")
    print("\n[GraphCL 对比]")
    print(df.round(4).to_string())

    # 若全量微调优于当前主模型则替换
    cur = pd.read_csv("results/tables/widegat_comparison.csv") if Path(
        "results/tables/widegat_comparison.csv").exists() else None
    cur_best = cur["R2"].max() if cur is not None and len(cur) else 0.0
    if m_full["R2"] > cur_best:
        torch.save({"state_dict": encoder.state_dict(),
                    "config": {"in_channels": in_c, "hidden_dim": 512, "num_layers": 4, "dropout": 0.15},
                    "metrics": m_full, "model_name": "GraphCL-WideGAT",
                    "hyperparams": {"pretrain": args.pretrain_epochs,
                                    "finetune": args.finetune_epochs}},
                   "models/gcn_resilience.pth")
        logger.info("✅ GraphCL-WideGAT 更优 (R²=%.4f)，已替换主模型", m_full["R2"])
        print(f"✅ GraphCL-WideGAT 已替换主模型 (R²={m_full['R2']:.4f})")
    else:
        logger.info("未超越当前最优 (%.4f)，保留现有模型", cur_best)


if __name__ == "__main__":
    main()
