#!/usr/bin/env python3
"""
train_widegat_save.py - 训练 WideGAT（5000 图）并保存 checkpoint
论文核心基线模型（GATConv 路线，MLA 已舍弃）。
训练后保存 models/widegat_5000.pth，供泛化测试/推理复用。

用法:
    python train_widegat_save.py [--epochs 250] [--quick]
输出:
    models/widegat_5000.pth  (state_dict + config + metrics)
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.utils import load_config, set_seed, get_device, setup_logger, get_project_root
from src.advanced_models import WideGATNet
from src.gcn_regressor import evaluate_regression


def main():
    parser = argparse.ArgumentParser(description="WideGAT 5000 图训练（保存 checkpoint）")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=256,
                        help="注意力宽度（默认 256 = config 超参调优值，与 baseline 公平对比）")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--out", type=str, default=None, help="checkpoint 输出名（默认 widegat_5000.pth）")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("widegat_save", log_file="logs/widegat_save.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)
    logger.info("设备: %s | hidden=%d layers=%d", device, args.hidden, args.layers)

    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    n = len(data_list)
    idx = np.random.default_rng(cfg["general"]["seed"]).permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_d = [data_list[i] for i in idx[:n_train]]
    val_d = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test_d = [data_list[i] for i in idx[n_train + n_val:]]
    logger.info("数据集: train=%d val=%d test=%d", len(train_d), len(val_d), len(test_d))
    in_c = train_d[0].x.shape[1]

    tl = DataLoader(train_d, batch_size=32, shuffle=True)
    vl = DataLoader(val_d, batch_size=32)
    tel = DataLoader(test_d, batch_size=32)

    model = WideGATNet(in_c, args.hidden, args.layers, 0.15).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=25)
    best_val, best_state, bad = float("inf"), None, 0
    epochs = 5 if args.quick else args.epochs

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
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 80:
            logger.info("早停 @ epoch %d (best_val=%.6f)", epoch + 1, best_val)
            break
        if (epoch + 1) % 10 == 0:
            logger.info("epoch %d/%d val_loss=%.6f", epoch + 1, epochs, vloss)
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

    # 保存 checkpoint
    os.makedirs("models", exist_ok=True)
    out_path = args.out or f"models/widegat_h{args.hidden}.pth"
    torch.save({
        "state_dict": best_state,
        "config": {"in_channels": in_c, "hidden_dim": args.hidden, "dropout": 0.15,
                   "num_layers": args.layers},
        "metrics": m,
        "train_s": train_s,
        "epochs_done": epochs,
        "model_name": "WideGAT",
    }, out_path)
    logger.info("已保存 %s | R²=%.4f RMSE=%.4f (%.0fs)",
                out_path, m["R2"], m["RMSE"], train_s)
    print(f"[WideGAT h{args.hidden}] R²={m['R2']:.4f} | RMSE={m['RMSE']:.4f} | 训练 {train_s:.0f}s | "
          f"已保存 {out_path}")


if __name__ == "__main__":
    main()
