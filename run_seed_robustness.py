#!/usr/bin/env python3
"""
run_seed_robustness.py - 统计稳健性实验（WideGAT 3 seeds）
论文 Table：均值±标准差报告（审稿人必查项）。
seed ∈ {0, 42, 2024}，各 150 epochs。

用法:
    python run_seed_robustness.py [--epochs 150]
输出:
    results/tables/seed_robustness.csv
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
from torch_geometric.loader import DataLoader

from src.utils import load_config, set_seed, get_device, save_csv, setup_logger, get_project_root
from src.advanced_models import WideGATNet
from run_width_ablation import train_one


def main():
    parser = argparse.ArgumentParser(description="3-seed 稳健性")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    logger = setup_logger("seed_robust", log_file="logs/seed_robustness.log")
    root = get_project_root()
    os.chdir(root)
    device = get_device(cfg["general"]["device"], force_cuda=True)
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)

    rows = []
    for seed in [0, 42, 2024]:
        m, ts = train_one(512, args.epochs, data_list, cfg, device, seed=seed)
        rows.append({"seed": seed, "R2": m["R2"], "RMSE": m["RMSE"],
                     "MAE": m["MAE"], "train_s": round(ts)})
        logger.info("seed=%d → R²=%.4f RMSE=%.4f", seed, m["R2"], m["RMSE"])
        print(f"seed={seed}: R²={m['R2']:.4f} | RMSE={m['RMSE']:.4f}")

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "seed": "mean±std",
        "R2": f"{df['R2'].mean():.4f}±{df['R2'].std():.4f}",
        "RMSE": f"{df['RMSE'].mean():.4f}±{df['RMSE'].std():.4f}",
        "MAE": f"{df['MAE'].mean():.4f}±{df['MAE'].std():.4f}",
    }])
    df = pd.concat([df, summary], ignore_index=True)
    save_csv(df, "results/tables/seed_robustness.csv")
    print("\n[3-seed 稳健性]")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
