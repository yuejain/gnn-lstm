#!/usr/bin/env python3
"""
benchmark_experiments.py - 论文基准数据集对比实验
在 CiteSeer / BlogCatalog / Flickr 三个数据集上（论文表4-1）运行
GCN / GAT 半监督节点分类，标记数 40/60/80 每类（论文表4-2/4-3协议），
输出 ACC 与 Macro-F1 与论文结果对比。

用法:
    python benchmark_experiments.py [--quick]
输出:
    results/tables/benchmark_comparison.csv
    results/figures/benchmark_acc.png
    results/paper_draft/benchmark_results.tex
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
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import accuracy_score, f1_score

from src.utils import set_seed, get_device, save_csv, ensure_dir, setup_logger, get_project_root

DEVICE = None


class GCNNodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=0.5)
        return self.conv2(x, edge_index)


class GATNodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden // heads, heads=heads)
        self.conv2 = GATConv(hidden, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=0.5)
        return self.conv2(x, edge_index)


def load_dataset(name):
    """加载基准数据集（复用已下载的 processed 文件）。"""
    if name == "CiteSeer":
        from torch_geometric.datasets import Planetoid
        ds = Planetoid(root="data/benchmark", name="CiteSeer")
        return ds[0], 6, 3703
    if name == "BlogCatalog":
        data = torch.load("data/benchmark/BlogCatalog/processed/blogcatalog.pt", weights_only=False)[0]
        return data, 39, 743
    if name == "Flickr":
        data = torch.load("data/benchmark/Flickr/processed/flickr.pt", weights_only=False)[0]
        return data, 7, 500
    raise ValueError(name)


def make_splits(data, n_classes, n_per_class, seed):
    """每类 n_per_class 标记的 train/val/test 划分（论文标记数 40/60/80 协议）。"""
    rng = np.random.default_rng(seed)
    n = data.num_nodes
    y = data.y.numpy()
    train_idx, val_idx = [], []
    for c in range(n_classes):
        cand = np.where(y == c)[0]
        if len(cand) == 0:
            continue
        rng.shuffle(cand)
        train_idx.extend(cand[:n_per_class].tolist())
        val_idx.extend(cand[n_per_class:n_per_class + n_per_class // 2].tolist())

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.ones(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[train_idx] = False
    test_mask[val_idx] = False
    return train_mask, val_mask, test_mask


def train_eval(model_name, data, n_classes, n_per_class, seed, epochs=200, quick=False):
    """训练节点分类器并返回 (acc, macro_f1)。"""
    set_seed(seed)
    n = data.num_nodes
    in_c = data.x.shape[1]

    model = (GCNNodeClassifier(in_c, 64, n_classes) if model_name == "GCN"
             else GATNodeClassifier(in_c, 64, n_classes)).to(DEVICE)

    if data.x.shape[0] > 200000:
        x = data.x.to(DEVICE)
    else:
        x = data.x.to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_mask, val_mask, test_mask = make_splits(data, n_classes, n_per_class, seed)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)
    y = data.y.to(DEVICE)

    epochs = min(epochs, 5 if quick else epochs)
    best_val = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.nll_loss(F.log_softmax(out, dim=1)[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=1).cpu().numpy()
    y_true = y.cpu().numpy()
    test_idx = test_mask.cpu().numpy()

    acc = accuracy_score(y_true[test_idx], pred[test_idx])
    f1 = f1_score(y_true[test_idx], pred[test_idx], average="macro", zero_division=0)
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="论文基准数据集对比实验")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--datasets", default="CiteSeer,BlogCatalog,Flickr")
    parser.add_argument("--labels", default="40,60,80")
    args = parser.parse_args()

    global DEVICE
    DEVICE = get_device("auto", force_cuda=True)
    logger = setup_logger("benchmark", log_file="logs/benchmark.log")
    logger.info("设备: %s", DEVICE)

    root = get_project_root()
    os.chdir(root)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    label_list = [int(l) for l in args.labels.split(",") if l.strip()]

    # 论文基线（表4-2 ACC / 表4-3 Macro F1）
    paper_acc = {
        "CiteSeer": {40: {"GCN": 69.60, "GAT": 70.32, "GCN-NSGA3": 71.23},
                     60: {"GCN": 71.34, "GAT": 71.72, "GCN-NSGA3": 74.38},
                     80: {"GCN": 73.25, "GAT": 74.44, "GCN-NSGA3": 77.63}},
        "BlogCatalog": {40: {"GCN": 70.71, "GAT": 66.43, "GCN-NSGA3": 69.93},
                        60: {"GCN": 71.80, "GAT": 69.04, "GCN-NSGA3": 70.46},
                        80: {"GCN": 72.94, "GAT": 73.01, "GCN-NSGA3": 73.80}},
        "Flickr": {40: {"GCN": 43.27, "GAT": 36.97, "GCN-NSGA3": 55.27},
                   60: {"GCN": 46.58, "GAT": 37.35, "GCN-NSGA3": 64.03},
                   80: {"GCN": 47.77, "GAT": 40.03, "GCN-NSGA3": 71.06}},
    }
    paper_f1 = {
        "CiteSeer": {40: {"GCN": 69.10, "GAT": 69.37, "GCN-NSGA3": 71.23},
                     60: {"GCN": 70.48, "GAT": 71.76, "GCN-NSGA3": 72.38},
                     80: {"GCN": 72.25, "GAT": 74.56, "GCN-NSGA3": 73.63}},
        "BlogCatalog": {40: {"GCN": 70.28, "GAT": 67.04, "GCN-NSGA3": 74.92},
                        60: {"GCN": 71.60, "GAT": 70.43, "GCN-NSGA3": 75.63},
                        80: {"GCN": 72.94, "GAT": 73.01, "GCN-NSGA3": 76.80}},
        "Flickr": {40: {"GCN": 43.48, "GAT": 38.44, "GCN-NSGA3": 54.27},
                   60: {"GCN": 46.96, "GAT": 38.96, "GCN-NSGA3": 65.03},
                   80: {"GCN": 48.73, "GAT": 39.04, "GCN-NSGA3": 65.77}},
    }

    results = []
    for dname in datasets:
        logger.info("=" * 50)
        logger.info("数据集: %s", dname)
        data, n_classes, _ = load_dataset(dname)
        for npl in label_list:
            for model_name in ["GCN", "GAT"]:
                acc, f1 = train_eval(model_name, data, n_classes, npl,
                                     seed=42, quick=args.quick)
                row = {"dataset": dname, "n_per_class": npl, "model": model_name,
                       "ACC": acc * 100, "MacroF1": f1 * 100}
                # 论文对照
                row["paper_GCN_ACC"] = paper_acc[dname][npl]["GCN"]
                row["paper_GAT_ACC"] = paper_acc[dname][npl]["GAT"]
                row["paper_GCN_F1"] = paper_f1[dname][npl]["GCN"]
                row["paper_GAT_F1"] = paper_f1[dname][npl]["GAT"]
                results.append(row)
                logger.info("%s n=%d %s → ACC=%.2f F1=%.2f (论文GCN=%.2f GAT=%.2f)",
                            dname, npl, model_name, acc * 100, f1 * 100,
                            paper_acc[dname][npl]["GCN"], paper_acc[dname][npl]["GAT"])

    df = pd.DataFrame(results)
    save_csv(df, "results/tables/benchmark_comparison.csv")
    logger.info("对比表: results/tables/benchmark_comparison.csv")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, dname in zip(axes, datasets):
        sub = df[df["dataset"] == dname]
        # 每个 n_per_class 取一行（去重）
        npl_unique = sorted(sub["n_per_class"].unique())
        labels = [str(n) + "/class" for n in npl_unique]
        x = np.arange(len(npl_unique))
        w = 0.18
        for j, (col, color, model) in enumerate([
            ("ACC", "#1f77b4", "GAT"), ("paper_GCN_ACC", "#7f7f7f", "GCN"),
            ("paper_GAT_ACC", "#ff7f0e", "GAT")]):
            vals = []
            for npl in npl_unique:
                row = sub[(sub["n_per_class"] == npl) & (sub["model"] == model)]
                vals.append(row[col].values[0] if len(row) else np.nan)
            ax.bar(x + (j - 1) * w, vals, w, label=col.replace("_", " "), color=color)
        ax.set_title(f"{dname}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle("Benchmark Node Classification (GAT vs Paper Baseline)")
    plt.tight_layout()
    ensure_dir("results/figures")
    plt.savefig("results/figures/benchmark_acc.png", dpi=300)
    plt.close()
    print(f"对比图: results/figures/benchmark_acc.png")

    # LaTeX 表
    lines = ["% 论文基准数据集对比（标记数 40/60/80，ACC%）",
             "\\begin{table}[htbp]", "\\centering",
             "\\caption{Node classification benchmark vs paper (Table 4-2)}",
             "\\label{tab:benchmark}", "\\begin{tabular}{llccccc}",
             "\\hline", "Dataset & N/class & GCN & GAT & Paper-GCN & Paper-GAT & Paper-NSGA3 \\\\", "\\hline"]
    for dname in datasets:
        for npl in label_list:
            sub = df[(df["dataset"] == dname) & (df["n_per_class"] == npl)]
            gcn = sub[sub["model"] == "GCN"]["ACC"].values[0]
            gat = sub[sub["model"] == "GAT"]["ACC"].values[0]
            p_gcn = sub[sub["model"] == "GCN"]["paper_GCN_ACC"].values[0]
            p_gat = sub[sub["model"] == "GAT"]["paper_GAT_ACC"].values[0]
            p_nsga = paper_acc[dname][npl]["GCN-NSGA3"]
            lines.append(f"{dname} & {npl} & {gcn:.2f} & {gat:.2f} & {p_gcn:.2f} & {p_gat:.2f} & {p_nsga:.2f} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    ensure_dir("outputs/paper_draft")
    with open("outputs/paper_draft/benchmark_results.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"LaTeX 表: outputs/paper_draft/benchmark_results.tex")

    print("\n[基准实验完成]")
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
