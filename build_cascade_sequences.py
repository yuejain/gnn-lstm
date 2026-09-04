#!/usr/bin/env python3
"""
build_cascade_sequences.py - 级联失效时序序列独立生成器
与数据工厂解耦：从 topologies.pt 读取 PyG 图 → 重建 nx.Graph →
生成 10 步级联失效序列（供 GCN-LSTM / MTP 级联预测训练）。

为什么独立：
- 数据工厂 5000 图主流程不再被级联拖慢（每图 10 步介数传播很重）
- 此脚本可单独运行、可并行（--parallel 多进程 chunk 处理）、可 --resume

用法:
    python build_cascade_sequences.py [--parallel 14]
输出:
    data/processed/cascade_sequences.pt  list[dict{node_feats, failure_labels, n_nodes}]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch_geometric.utils import to_networkx

from src.utils import load_config, set_seed, setup_logger, get_project_root


def _worker(seqs_chunk):
    """模块级 worker：处理一批 (idx, seq) 重建级联序列（可 pickle）。"""
    cfg, data_list, n_steps = seqs_chunk
    from src.resilience_labels import simulate_cascade_failure
    import networkx as nx

    results = []
    for data in data_list:
        G = to_networkx(data, to_undirected=True)
        # 坐标（用于特征对齐，非必需）
        pos = data.pos if hasattr(data, "pos") else None
        feats = data.x.numpy()
        try:
            _, failed_sets = simulate_cascade_failure(
                G, failure_ratio=0.05, n_steps=n_steps, seed=42)
            n_nodes = G.number_of_nodes()
            labels = np.zeros((n_steps, n_nodes))
            for t, failed in enumerate(failed_sets[:n_steps]):
                for node in failed:
                    if node < n_nodes:
                        labels[t, node] = 1.0
            results.append({
                "node_feats": feats.astype(np.float32),
                "failure_labels": labels.astype(np.float32),
                "n_nodes": n_nodes,
            })
        except Exception as e:
            print(f"[cascade] 图跳过: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="级联序列独立生成器")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--parallel", type=int, default=0,
                        help="并行进程数（0=自动全核心）")
    parser.add_argument("--steps", type=int, default=None, help="级联步数（默认取 config）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("cascade_seq", log_file="logs/cascade_seq.log")
    root = get_project_root()
    os.chdir(root)

    n_steps = args.steps or cfg["topology"]["cascade_steps"]
    data_list = torch.load("data/raw/topologies.pt", weights_only=False)
    logger.info("加载数据集: %d 图, 级联 %d 步", len(data_list), n_steps)

    if args.parallel <= 0:
        import os as _os
        args.parallel = max(4, min((_os.cpu_count() or 8) - 2, 14))

    import time as _time
    t0 = _time.time()
    if args.parallel > 1 and len(data_list) >= 40:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        multiprocessing.freeze_support()
        # 均分 20 个 chunk
        chunk_size = max(10, len(data_list) // (args.parallel * 4))
        chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
        all_seqs = []
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            pending = {ex.submit(_worker, (cfg, ch, n_steps)): i for i, ch in enumerate(chunks[:args.parallel * 2])}
            it = iter(chunks[args.parallel * 2:])
            done = 0
            while pending:
                fut = next(as_completed(pending))
                pending.pop(fut)
                all_seqs.extend(fut.result())
                done += 1
                if done % 5 == 0:
                    logger.info("级联进度: %d/%d chunks", done, len(chunks))
                try:
                    ch_next = next(it)
                    pending[ex.submit(_worker, (cfg, ch_next, n_steps))] = 0
                except StopIteration:
                    pass
    else:
        all_seqs = _worker((cfg, data_list, n_steps))

    elapsed = _time.time() - t0
    torch.save(all_seqs, "data/processed/cascade_sequences.pt")
    logger.info("已保存: data/processed/cascade_sequences.pt (%d 序列, %.1fs, %.2f 图/s)",
                len(all_seqs), elapsed, len(all_seqs) / max(elapsed, 1e-9))
    print(f"[级联序列完成] {len(all_seqs)} 序列 / {elapsed:.1f}s")


if __name__ == "__main__":
    main()
