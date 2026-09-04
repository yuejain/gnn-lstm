#!/usr/bin/env python3
"""
train_latent_widegat.py - MLA 低秩注意力 + 混合精度训练（DeepSeek-V3 技术应用）
在 5000 图数据集上训练 LatentWideGAT（MLA 低秩压缩图注意力），
并与 WideGAT 对比精度/显存/速度。

DeepSeek-V3 技术映射：
- MLA（低秩 K/V 压缩）→ LatentGATConv（latent_ratio=0.25）
- FP8 低精度训练思想 → BF16 autocast 混合精度

用法:
    python train_latent_widegat.py [--quick]
输出:
    results/tables/latent_widegat_comparison.csv
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
from src.gcn_regressor import evaluate_regression
from src.advanced_models import WideGATNet
from src.deepseek_tech import LatentWideGAT, enable_mixed_precision


def main():
    parser = argparse.ArgumentParser(description="LatentWideGAT MLA 训练对比")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("latent_widegat", log_file="logs/latent_widegat.log")
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

    tl = DataLoader(train_d, batch_size=32, shuffle=True)
    vl = DataLoader(val_d, batch_size=32)
    tel = DataLoader(test_d, batch_size=32)

    models = {
        "WideGAT (GATConv)": WideGATNet(in_c, 512, 4, 0.15),
        "LatentWideGAT (MLA)": LatentWideGAT(in_c, 512, 4, 0.15, latent_ratio=0.25),
    }
    results = []
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)
        best_val, best_state, bad = float("inf"), None, 0
        epochs = 5 if args.quick else args.epochs

        import time
        t0 = time.time()
        for epoch in range(epochs):
            model.train()
            for d in tl:
                d = d.to(device)
                opt.zero_grad()
                with enable_mixed_precision(model, device):
                    loss = F.mse_loss(model(d), d.y)
                loss.backward()
                opt.step()
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                with enable_mixed_precision(model, device):
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
            if bad >= 60:
                break
        train_time = time.time() - t0

        model.load_state_dict(best_state)
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for d in tel:
                d = d.to(device)
                yt.extend(d.y.cpu().tolist())
                yp.extend(model(d).cpu().tolist())
        m = evaluate_regression(np.array(yt), np.array(yp))
        m.update({"model": name, "params_M": round(n_params, 2), "train_s": round(train_time)})
        results.append(m)
        logger.info("%s → R²=%.4f (%dM 参数, %.0fs)", name, m["R2"], n_params, train_time)
        print(f"{name}: R²={m['R2']:.4f} | {n_params:.2f}M 参数 | 训练 {train_time:.0f}s")

    df = pd.DataFrame(results)
    save_csv(df, "results/tables/latent_widegat_comparison.csv")
    print("\n[MLA 低秩注意力对比]")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()
