#!/usr/bin/env python3
"""save_widegat_main.py - 将 WideGAT 训练为最终主模型并保存"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.utils import load_config, set_seed, get_device
from src.gcn_regressor import evaluate_regression
from src.advanced_models import WideGATNet


def main():
    cfg = load_config("config.yaml")
    device = get_device(cfg["general"]["device"], force_cuda=True)
    print("设备:", device)

    set_seed(cfg["general"]["seed"])
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    n = len(data_list)
    idx = np.random.default_rng(cfg["general"]["seed"]).permutation(n)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_d = [data_list[i] for i in idx[:n_train]]
    val_d = [data_list[i] for i in idx[n_train:n_train + n_val]]
    test_d = [data_list[i] for i in idx[n_train + n_val:]]
    print(f"数据集: train={len(train_d)} val={len(val_d)} test={len(test_d)}")

    tl = DataLoader(train_d, batch_size=32, shuffle=True)
    vl = DataLoader(val_d, batch_size=32)
    tel = DataLoader(test_d, batch_size=32)
    in_c = train_d[0].x.shape[1]

    model = WideGATNet(in_c, 512, 4, 0.15).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20)

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(400):
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
        sched.step(vloss)
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 80:
            print(f"早停 @ epoch {epoch+1}, best_val={best_val:.5f}")
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
    m = evaluate_regression(np.array(yt), np.array(yp))
    print("WideGAT 测试集:", {k: round(v, 6) for k, v in m.items()})

    torch.save({
        "state_dict": best_state,
        "config": {"in_channels": in_c, "hidden_dim": 512, "num_layers": 4, "dropout": 0.15},
        "metrics": m,
        "model_name": "WideGAT",
        "hyperparams": {"heads": 16, "epochs": 400, "feature_mode": "embedded35",
                        "residual": True, "layernorm": True},
    }, "models/gcn_resilience.pth")
    print(f"✅ WideGAT 已保存为主模型 (R²={m['R2']:.4f})")


if __name__ == "__main__":
    main()
