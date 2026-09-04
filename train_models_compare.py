#!/usr/bin/env python3
"""
train_models_compare.py - 新算法调优：多模型对比择优
在相同数据/超参下对比 GCN / GAT / GraphSAGE / GCN-GAT 集成 的韧性回归性能，
选出最优模型并保存（若优于基线则替换 gcn_resilience.pth 供 NSGA-III 使用）。

用法:
    python train_models_compare.py --config config.yaml [--models GCN,GAT,GraphSAGE,GCN_GAT_Mix]
    python train_models_compare.py --config config.yaml --quick

输出:
    results/tables/model_comparison.csv      多模型对比表
    results/figures/model_comparison.png     对比柱状图
    models/gcn_resilience.pth                若最优模型优于基线则覆盖（含 model_name 标记）
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

from src.utils import load_config, set_seed, get_device, save_csv, ensure_dir, setup_logger, get_project_root
from src.gcn_regressor import ResilienceGCN, evaluate_regression
from src.advanced_models import build_model


def load_dataset():
    data_path = "data/raw/topologies.pt"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"{data_path} 不存在，请先运行 data_factory.py")
    return torch.load(data_path, weights_only=False)


def split_dataset(data_list, ratios=(0.7, 0.15, 0.15), seed=42):
    n = len(data_list)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    idx = np.random.default_rng(seed).permutation(n)
    return ([data_list[i] for i in idx[:n_train]],
            [data_list[i] for i in idx[n_train:n_train + n_val]],
            [data_list[i] for i in idx[n_train + n_val:]])


def train_eval(model, train_loader, val_loader, test_loader, cfg, device, logger,
               epochs=500, quick=False):
    """训练并评估单个模型，返回 (metrics, history)。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["gcn_regressor"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["gcn_regressor"]["patience"] // 4)
    criterion = F.mse_loss

    epochs = min(epochs, 5 if quick else cfg["gcn_regressor"]["epochs"])
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(epochs):
        model.train()
        tl = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(data.y)
        tl /= max(len(train_loader.dataset), 1)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                vl += criterion(model(data), data.y).item() * len(data.y)
        vl /= max(len(val_loader.dataset), 1)
        scheduler.step(vl)

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= cfg["gcn_regressor"]["patience"] and not quick:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")

    # 测试
    model = model.to(device)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            y_true.extend(data.y.cpu().tolist())
            y_pred.extend(model(data).cpu().tolist())
    metrics = evaluate_regression(np.array(y_true), np.array(y_pred))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="多模型对比调优")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", default="GCN,GAT,GraphSAGE,GCN_GAT_Mix")
    parser.add_argument("--epochs", type=int, default=0,
                        help="训练 epochs（0=取 config，默认 1000 可覆盖为 250）")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs > 0:
        cfg["gcn_regressor"]["epochs"] = args.epochs
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("model_compare", log_file="logs/model_compare.log")

    root = get_project_root()
    os.chdir(root)

    data_list = load_dataset()
    train_data, val_data, test_data = split_dataset(
        data_list, cfg["gcn_regressor"]["train_val_test"], cfg["general"]["seed"])
    logger.info("数据集: train=%d val=%d test=%d", len(train_data), len(val_data), len(test_data))

    bs = cfg["gcn_regressor"]["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)

    device = get_device(cfg["general"]["device"], force_cuda=True)
    logger.info("设备: %s", device)
    in_channels = data_list[0].x.shape[1]

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results = []
    best_model = None
    best_metrics = None

    for name in model_names:
        logger.info("=" * 50)
        logger.info("训练模型: %s", name)
        model = build_model(name, in_channels,
                            cfg["gcn_regressor"]["hidden_dim"],
                            cfg["gcn_regressor"]["dropout"],
                            cfg["gcn_regressor"]["num_layers"])
        metrics = train_eval(model, train_loader, val_loader, test_loader,
                             cfg, device, logger, quick=args.quick)
        metrics["model"] = name
        results.append(metrics)
        logger.info("%s 测试集: %s", name, {k: round(v, 4) for k, v in metrics.items() if k != "model"})

        if best_metrics is None or metrics["R2"] > best_metrics["R2"]:
            best_metrics = metrics
            best_model = model

    df = pd.DataFrame(results)
    save_csv(df, "results/tables/model_comparison.csv")
    logger.info("模型对比表: results/tables/model_comparison.csv")

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = df["model"].tolist()
    axes[0].bar(names, df["R2"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(names)])
    axes[0].set_title("R² Score (higher better)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(alpha=0.3)
    axes[1].bar(names, df["MSE"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(names)])
    axes[1].set_title("MSE (lower better)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(alpha=0.3)
    plt.suptitle("Model Comparison - Resilience Regression")
    plt.tight_layout()
    ensure_dir("results/figures")
    plt.savefig("results/figures/model_comparison.png", dpi=300)
    plt.close()
    print(f"对比图已保存: results/figures/model_comparison.png")

    # 若最优模型非基线 GCN 且 R² 更优，覆盖主模型供下游使用
    print("\n[模型对比结果]")
    print(df.round(4).to_string())

    if best_model is not None and best_metrics["model"] != "GCN":
        logger.info("最优模型 %s (R²=%.4f) 优于基线 GCN，已替换 gcn_resilience.pth",
                    best_metrics["model"], best_metrics["R2"])
        torch.save({
            "state_dict": best_model.state_dict(),
            "config": {"in_channels": in_channels,
                       "hidden_dim": cfg["gcn_regressor"]["hidden_dim"],
                       "num_layers": cfg["gcn_regressor"]["num_layers"],
                       "dropout": cfg["gcn_regressor"]["dropout"]},
            "metrics": best_metrics,
            "model_name": best_metrics["model"],
        }, "models/gcn_resilience.pth")
        print(f"✅ 最优模型 {best_metrics['model']} 已保存为 gcn_resilience.pth (R²={best_metrics['R2']:.4f})")
    else:
        logger.info("基线 GCN 仍为最优，保留现有模型")
        print("基线 GCN 仍为最优，未替换模型")


if __name__ == "__main__":
    main()
