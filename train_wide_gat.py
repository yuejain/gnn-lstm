#!/usr/bin/env python3
"""
train_wide_gat.py - 模型加宽对比：WideGAT vs GAT（900 图数据集）
在扩大量级后的训练集上对比原 GAT（h128/8heads/3层）与 WideGAT（h512/16heads/4层，
参数量提升 20 倍），验证模型加宽对韧性回归精度的增益。

用法:
    python train_wide_gat.py [--quick]
输出:
    results/tables/widegat_comparison.csv
    models/gcn_resilience.pth   若 WideGAT 更优则替换主模型（含 model_name=WideGAT）
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
from torch_geometric.loader import DataLoader

from src.utils import load_config, set_seed, get_device, save_csv, ensure_dir, setup_logger, get_project_root
from src.gcn_regressor import evaluate_regression
from src.advanced_models import GATResilienceNet, WideGATNet


def load_split(cfg):
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    n = len(data_list)
    idx = np.random.default_rng(cfg["general"]["seed"]).permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    return ([data_list[i] for i in idx[:n_train]],
            [data_list[i] for i in idx[n_train:n_train + n_val]],
            [data_list[i] for i in idx[n_train + n_val:]])


def train_eval(model, train_d, val_d, test_d, cfg, device, quick=False, epochs=None, patience=None):
    model = model.to(device)
    bs = cfg["gcn_regressor"]["batch_size"]
    tl = DataLoader(train_d, batch_size=bs, shuffle=True)
    vl = DataLoader(val_d, batch_size=bs)
    tel = DataLoader(test_d, batch_size=bs)

    epochs = 5 if quick else (epochs or 400)
    patience = 2 if quick else (patience or 80)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=patience // 4)

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for d in tl:
            d = d.to(device)
            opt.zero_grad()
            try:
                loss = F.mse_loss(model(d), d.y)
            except torch.cuda.OutOfMemoryError:
                # 显存自适应：OOM 时清空缓存并减半 batch 重试
                torch.cuda.empty_cache()
                bs_new = max(1, bs // 2)
                tl = DataLoader(train_d, batch_size=bs_new, shuffle=True)
                vl = DataLoader(val_d, batch_size=bs_new)
                tel = DataLoader(test_d, batch_size=bs_new)
                print(f"[OOM] batch_size 自适应: {bs} → {bs_new}")
                bs = bs_new
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
        sched.step(vloss)
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for d in tel:
            d = d.to(device)
            yt.extend(d.y.cpu().tolist())
            yp.extend(model(d).cpu().tolist())
    return evaluate_regression(np.array(yt), np.array(yp))


def main():
    parser = argparse.ArgumentParser(description="WideGAT 模型加宽对比")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("widegat", log_file="logs/widegat.log")
    root = get_project_root()
    os.chdir(root)
    # 强制 GPU 训练（CUDA 不可用直接报错，绝不静默降级 CPU）
    device = get_device(cfg["general"]["device"], force_cuda=True)
    logger.info("设备: %s（强制 GPU）", device)

    train_d, val_d, test_d = load_split(cfg)
    logger.info("数据集: train=%d val=%d test=%d (总计 %d)",
                len(train_d), len(val_d), len(test_d), len(train_d) + len(val_d) + len(test_d))
    in_c = train_d[0].x.shape[1]

    models = {
        "GAT (h128/8h/3L, 0.06M)": GATResilienceNet(in_c, 128, 3, 0.2),
        "WideGAT (h512/16h/4L, 1.21M)": WideGATNet(in_c, 512, 4, 0.15),
    }

    results = []
    for name, model in models.items():
        logger.info("训练: %s", name)
        m = train_eval(model, train_d, val_d, test_d, cfg, device, quick=args.quick)
        m["model"] = name
        results.append(m)
        logger.info("%s → %s", name, {k: round(v, 4) for k, v in m.items() if k != "model"})

    df = pd.DataFrame(results)
    save_csv(df, "results/tables/widegat_comparison.csv")
    print("\n[模型加宽对比]")
    print(df.round(4).to_string())

    # WideGAT 更优则替换主模型
    if "WideGAT" in df["model"].values:
        wg = df[df["model"].str.contains("WideGAT")].iloc[0]
        gat = df[df["model"].str.contains("GAT (h128")].iloc[0]
        if wg["R2"] > gat["R2"]:
            best_model = models["WideGAT (h512/16h/4L, 1.21M)"]
            torch.save({
                "state_dict": best_model.state_dict(),
                "config": {"in_channels": in_c, "hidden_dim": 512, "num_layers": 4, "dropout": 0.15},
                "metrics": {"MSE": wg["MSE"], "RMSE": wg["RMSE"], "MAE": wg["MAE"], "R2": wg["R2"]},
                "model_name": "WideGAT",
                "hyperparams": {"heads": 16, "epochs": 400, "feature_mode": "embedded35"},
            }, "models/gcn_resilience.pth")
            logger.info("✅ WideGAT 更优 (R²=%.4f vs %.4f)，已替换主模型", wg["R2"], gat["R2"])
            print(f"✅ WideGAT 已替换主模型 (R²={wg['R2']:.4f} > GAT {gat['R2']:.4f})")
        else:
            logger.info("GAT 仍更优，保留现有模型")
            print("GAT 仍更优，未替换")


if __name__ == "__main__":
    main()
