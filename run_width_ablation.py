#!/usr/bin/env python3
"""
run_width_ablation.py - 宽度消融实验（WideGAT h128/256/512）
论文 Table：注意力宽度对韧性回归精度的影响。
512 维结果复用 models/widegat_5000.pth（R²=0.9316），补齐 h128/h256。

用法:
    python run_width_ablation.py [--epochs 150]
输出:
    results/tables/width_ablation.csv
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

from src.utils import load_config, set_seed, get_device, save_csv, setup_logger, get_project_root
from src.advanced_models import WideGATNet
from src.gcn_regressor import evaluate_regression


def train_one(hidden, epochs, data_list, cfg, device, seed=42):
    set_seed(seed)
    n = len(data_list)
    idx = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_d = [data_list[i] for i in idx[:n_train]]
    val_d = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test_d = [data_list[i] for i in idx[n_train + n_val:]]
    tl = DataLoader(train_d, batch_size=32, shuffle=True)
    vl = DataLoader(val_d, batch_size=32)
    tel = DataLoader(test_d, batch_size=32)

    in_c = train_d[0].x.shape[1]
    model = WideGATNet(in_c, hidden, 4, 0.15).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=15)
    best_val, best_state, bad = float("inf"), None, 0
    import time
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
        sched.step(vloss)
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 50:
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


def main():
    parser = argparse.ArgumentParser(description="宽度消融")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    logger = setup_logger("width_ablation", log_file="logs/width_ablation.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    logger.info("数据集: %d 图", len(data_list))

    rows = []
    for h in [128, 256]:
        m, ts = train_one(h, args.epochs, data_list, cfg, device)
        rows.append({"hidden_dim": h, **{k: m[k] for k in ("MSE", "RMSE", "MAE", "R2")},
                     "train_s": round(ts)})
        logger.info("h%d → R²=%.4f RMSE=%.4f (%.0fs)", h, m["R2"], m["RMSE"], ts)
        print(f"h{h}: R²={m['R2']:.4f} | RMSE={m['RMSE']:.4f} | {ts:.0f}s")

    # 复用 512 已训模型
    ckpt = torch.load("models/widegat_5000.pth", weights_only=False)
    rows.append({"hidden_dim": 512, **{k: ckpt["metrics"][k] for k in ("MSE", "RMSE", "MAE", "R2")},
                 "train_s": round(ckpt["train_s"])})
    logger.info("h512 → R²=%.4f（复用 widegat_5000.pth）", ckpt["metrics"]["R2"])

    df = pd.DataFrame(rows).sort_values("hidden_dim")
    save_csv(df, "results/tables/width_ablation.csv")
    print("\n[宽度消融]")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
