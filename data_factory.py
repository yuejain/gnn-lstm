#!/usr/bin/env python3
"""
data_factory.py - 阶段一：多层拓扑数据工厂
生成带韧性标签的合成图数据集，复用程洋论文的节点映射流程与 CFD-GAN 参数，
并集成 4 个 WSN 数据集作为节点特征增强。

用法:
    python data_factory.py --config config.yaml [--num_sensors 600] [--num_graphs 30]
    python data_factory.py --config config.yaml --quick   # 快速冒烟（少量图）

输出:
    data/raw/topologies.pt        PyG 图数据对象（x 节点特征 / edge_index / y 韧性标签）
    data/processed/metadata.csv   图元数据（图ID、拓扑类型、节点数、鲁棒性值）
    data/processed/cascade_sequences.pt  级联失效时序序列（供 GCN-LSTM）
"""
import argparse
import os
import sys
from pathlib import Path

# 关键：限制 BLAS/OpenMP 线程数（每个进程单线程）
# 14 个 worker 若各自用默认多线程（OpenBLAS 8线程）→ 112 线程抢核 → 上下文切换风暴互相拖死
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# 确保可从项目根导入 src
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.utils import load_config, set_seed, save_csv, ensure_dir, setup_logger, get_project_root
from src.topology_factory import generate_dataset, compute_node_features, generate_variant, generate_sensor_positions
from src.resilience_labels import compute_all_metrics, simulate_cascade_failure
from src.wsn_data_loader import get_wsn_node_features


# ============================================================
# 参数网格展开（5000 图方案）
# ============================================================

def expand_grid(cfg) -> list[dict]:
    """
    将 config.data_generation 展开为 cell 列表（确定性顺序）。
    每 cell: {variant, n_nodes, comm_range, n_hazard, cluster_r, occlusion, n_graphs, seed_base}
    """
    dg = cfg.get("data_generation", {})
    if not dg:
        # 兼容旧配置：直接按 variants × num_graphs_per_variant
        return [{"variant": v, "n_nodes": cfg["park"]["num_sensors"][0],
                 "comm_range": cfg["park"]["communication_range"],
                 "n_hazard": cfg["park"]["num_hazard_sources"],
                 "cluster_r": cfg["park"]["hazard_cluster_radius"],
                 "occlusion": cfg["park"].get("occlusion_target", 0.0),
                 "n_graphs": cfg["topology"]["num_graphs_per_variant"],
                 "seed_base": cfg["general"]["seed"]}
                for v in cfg["topology"]["variants"]]

    base = dg.get("base", {})
    base_per_variant = dg.get("base_per_variant", 150)
    sweeps = dg.get("sweeps", {})
    extra = dg.get("extra_per_variant", {})
    variants = cfg["topology"]["variants"]

    cells = []
    for v in variants:
        vp = cfg["topology"].get("variant_params", {}).get(v, {})
        # 锚点档（基准参数）
        cells.append({
            "variant": v, "n_nodes": base.get("n_nodes", 600),
            "comm_range": base.get("comm_range", 150),
            "n_hazard": base.get("n_hazard", 50),
            "cluster_r": base.get("cluster_r", 200),
            "occlusion": base.get("occlusion", 0.15),
            "n_graphs": base_per_variant, "seed_base": 0, "vparams": vp,
        })
        # 各扫描维度（OFAT）
        for dim, spec in sweeps.items():
            per = spec.get("per_variant", 30)
            if isinstance(per, list):
                per_list = per
            else:
                per_list = [per] * len(spec["values"])
            for val, n_g in zip(spec["values"], per_list):
                cell = {
                    "variant": v, "n_nodes": base.get("n_nodes", 600),
                    "comm_range": base.get("comm_range", 150),
                    "n_hazard": base.get("n_hazard", 50),
                    "cluster_r": base.get("cluster_r", 200),
                    "occlusion": base.get("occlusion", 0.15),
                    "n_graphs": n_g, "seed_base": 0, "vparams": vp,
                }
                cell[dim] = val
                cells.append(cell)
        # 额外分配
        if v in extra:
            cells.append({
                "variant": v, "n_nodes": base.get("n_nodes", 600),
                "comm_range": base.get("comm_range", 150),
                "n_hazard": base.get("n_hazard", 50),
                "cluster_r": base.get("cluster_r", 200),
                "occlusion": base.get("occlusion", 0.15),
                "n_graphs": extra[v], "seed_base": 100000, "vparams": vp,
            })
    return cells


def _gen_graph_chunk_worker(cfg, chunk, wsn_feats, return_nx: bool = False):
    """
    模块级 worker：生成一批图的 (pyg_data, metadata)（可 pickle，Windows 兼容）。
    chunk: list[cell]，每 cell 含 variant/n_nodes/comm_range/n_hazard/cluster_r/
           occlusion/n_graphs/seed_base/vparams
    wsn_feats: None 或 data_dir 字符串
    return_nx=False（默认）时不返回 nx_graphs —— 跨进程传输量降 90%，防管道死锁；
    级联序列由独立脚本 build_cascade_sequences.py 生成。
    """
    import networkx as nx
    from src.physical_layout import calibrate_obstacles
    from src.topology_factory import generate_sensor_positions, generate_variant

    wsn_dir_str = None
    wsn_bank = None
    if wsn_feats is not None:
        wsn_dir_str = wsn_feats  # main 传入 wsn_data_dir 字符串

    def _wsn_for(n_nodes):
        """从 feature_bank 快速采样（bank 只加载一次，避免每图重读 4 个 CSV）。"""
        nonlocal wsn_bank
        if wsn_dir_str is None:
            return None
        if wsn_bank is None:
            from src.wsn_data_loader import WSNDataLoader
            loader = WSNDataLoader(wsn_dir_str)
            wsn_bank = loader.build_node_feature_bank()["feature_bank"]
        rng = np.random.default_rng(42)
        idx = rng.choice(len(wsn_bank), n_nodes, replace=True)
        return wsn_bank[idx][:, :5]

    pyg_list = []
    meta_rows = []
    nx_graphs = []
    job_idx = 0
    for cell in chunk:
        variant = cell["variant"]
        n_nodes = cell["n_nodes"]
        comm_range = cell["comm_range"]
        n_hazard = cell["n_hazard"]
        cluster_r = cell["cluster_r"]
        occlusion = cell["occlusion"]
        n_graphs = cell["n_graphs"]
        seed_base = cell.get("seed_base", 0) + job_idx * 100000
        vparams = dict(cell.get("vparams") or {})

        for g in range(n_graphs):
            # 槽位展开后 n_graphs=1；保留 g 偏移兼容旧调用（多图 cell）
            seed = seed_base + g * 1000 + (sum(ord(c) for c in variant) % 1000) + job_idx * 7
            try:
                set_seed(seed)
                # 临时替换危险源/聚集参数
                park_cfg = dict(cfg["park"])
                park_cfg["num_hazard_sources"] = n_hazard
                park_cfg["hazard_cluster_radius"] = cluster_r
                positions, hazard_pos, sensor_to_hazard, _ = generate_sensor_positions(
                    park_cfg, n_nodes, layout=park_cfg.get("layout", "road"))

                # 障碍物（random 变体使用）
                obstacles = None
                if occlusion > 0.01 and variant == "random":
                    obstacles = calibrate_obstacles(
                        positions, comm_range, occlusion, park_cfg["area_size"],
                        max_obstacles=park_cfg.get("max_obstacles", 60), seed=seed)
                    vparams["obstacles"] = obstacles
                    vparams["atten_prob"] = park_cfg.get("atten_prob", 0.3)

                G = generate_variant(positions, comm_range, variant,
                                     vparams=vparams, seed=seed)
                G.graph["variant"] = variant
                G.graph["num_sensors"] = n_nodes
                G.graph["hazard_pos"] = hazard_pos
                G.graph["sensor_to_hazard"] = sensor_to_hazard
                G.graph["features"] = compute_node_features(
                    G, hazard_pos, sensor_to_hazard, wsn_features=_wsn_for(n_nodes),
                    target_dim=cfg["data_factory"]["node_feature_dim"])
                # 指标计算一次并缓存
                m = compute_all_metrics(G, cfg)
                G.graph["metrics"] = m

                # 单图转 PyG（复用 build_pyg_graphs）
                from torch_geometric.utils import from_networkx
                data = from_networkx(G)
                data.x = torch.tensor(np.asarray(G.graph["features"]), dtype=torch.float)
                data.y = torch.tensor([m["composite_score"]], dtype=torch.float)
                data.robustness = torch.tensor([m["robustness"]], dtype=torch.float)
                data.survivability = torch.tensor([m["survivability"]], dtype=torch.float)
                data.graph_id = 0  # 合并时重新编号
                data.variant = variant
                data.num_sensors = n_nodes
                for attr in list(data.keys()):
                    if attr not in ("edge_index", "x", "y", "robustness", "survivability"):
                        delattr(data, attr)
                pyg_list.append(data)
                G.graph.pop("metrics", None)  # 减小传输体积
                if return_nx:
                    nx_graphs.append(G)
                meta_rows.append({
                    "graph_id": 0, "topology": variant, "num_sensors": n_nodes,
                    "num_nodes": m["num_nodes"], "num_edges": m["num_edges"],
                    "robustness": m["robustness"], "survivability": m["survivability"],
                    "k_connectivity": m["k_connectivity"], "composite_score": m["composite_score"],
                    "avg_degree": m["avg_degree"], "comm_range": comm_range,
                    "n_hazard": n_hazard, "cluster_r": cluster_r, "occlusion": occlusion,
                })
            except Exception as e:
                # 单图失败不崩整个 chunk（跳过并记录到文件，避免 worker stdout 阻塞）
                # 注意：worker 内禁止 print/写 stdout（Start-Process Hidden 窗口下会死锁）
                pass
            job_idx += 1
    return pyg_list, meta_rows, nx_graphs


def build_pyg_graphs(graphs, cfg):
    """
    将 NetworkX 图转换为 PyG Data 对象（x/edge_index/y）。
    from_networkx 会拷贝节点与图级属性，这里手动构建以确保仅含必要字段。
    指标优先读 G.graph["metrics"] 缓存（worker 已算），避免重复计算。
    """
    from torch_geometric.utils import from_networkx

    data_list = []
    for G in graphs:
        data = from_networkx(G)
        # 特征矩阵（优先使用已缓存的 35 维特征）
        feats = G.graph.get("features")
        if feats is None:
            feats = compute_node_features(
                G, G.graph.get("hazard_pos"), G.graph.get("sensor_to_hazard"),
                wsn_features=None, target_dim=cfg["data_factory"]["node_feature_dim"])
        data.x = torch.tensor(np.asarray(feats), dtype=torch.float)

        # 韧性标签（优先读缓存）
        metrics = G.graph.get("metrics")
        if metrics is None:
            metrics = compute_all_metrics(G, cfg)
        data.y = torch.tensor([metrics["composite_score"]], dtype=torch.float)
        data.robustness = torch.tensor([metrics["robustness"]], dtype=torch.float)
        data.survivability = torch.tensor([metrics["survivability"]], dtype=torch.float)
        data.graph_id = len(data_list)
        data.variant = G.graph.get("variant", "unknown")
        data.num_sensors = G.graph.get("num_sensors", 0)

        # 剥离多余属性（白名单：仅保留 edge_index/x/y + 标签字段）
        for attr in list(data.keys()):
            if attr not in ("edge_index", "x", "y", "robustness", "survivability"):
                delattr(data, attr)
        # 释放缓存减小 pickle 体积
        G.graph.pop("metrics", None)
        data_list.append(data)
    return data_list


def build_cascade_sequences(graphs, cfg):
    """
    生成级联失效时序序列（10 时间步拓扑快照），供 GCN-LSTM 训练。
    每步的"节点失效概率"作为目标标签。

    Returns:
        sequences: list[dict]，含 node_feats (T, N, F)、failure_labels (T, N)
    """
    n_steps = cfg["topology"]["cascade_steps"]
    sequences = []
    for G in graphs:
        feats = G.graph["features"]
        _, failed_sets = simulate_cascade_failure(
            G, failure_ratio=0.05, n_steps=n_steps, seed=cfg["general"]["seed"])

        # 每步标签：该节点是否已失效（0/1）
        labels = np.zeros((n_steps, len(G.nodes())))
        n_nodes = len(G.nodes())
        for t, failed in enumerate(failed_sets[:n_steps]):
            for node in failed:
                if node < n_nodes:
                    labels[t, node] = 1.0

        sequences.append({
            "node_feats": feats.astype(np.float32),
            "failure_labels": labels.astype(np.float32),
            "n_nodes": n_nodes,
        })
    return sequences


def main():
    parser = argparse.ArgumentParser(description="阶段一：多层拓扑数据工厂")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--num_sensors", type=int, default=None,
                        help="传感器节点数（默认取配置第一档 300）")
    parser.add_argument("--num_graphs", type=int, default=None,
                        help="每变体图数量（默认取配置值）")
    parser.add_argument("--topology", choices=["all", "random", "scale_free", "small_world"],
                        default="all")
    parser.add_argument("--quick", action="store_true", help="快速冒烟模式")
    parser.add_argument("--parallel", type=int, default=0,
                        help="并行进程数（0=自动检测全核心，默认自动）")
    parser.add_argument("--skip-cascade", action="store_true",
                        help="跳过级联时序序列生成（5000 图时显著提速）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传：跳过已完成 shards（崩溃/中断后恢复）")
    parser.add_argument("--benchmark", action="store_true",
                        help="输出吞吐量/延迟量化指标（重构前后对比用）")
    args = parser.parse_args()

    # 全核心自动并行：取物理核心数（留 2 核给系统），上限 14
    _cores = None
    if args.parallel <= 0:
        import os as _os
        try:
            _cores = _os.cpu_count() or 8
        except Exception:
            _cores = 8
        args.parallel = max(4, min(_cores - 2, 14))

    cfg = load_config(args.config)
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("data_factory", log_file="logs/data_factory.log")
    if _cores:
        logger.info("[并行] 自动检测到 %d 逻辑核心，使用 %d 进程", _cores, args.parallel)

    root = get_project_root()
    os.chdir(root)

    num_sensors = args.num_sensors or cfg["park"]["num_sensors"][0]
    num_graphs = args.num_graphs or cfg["topology"]["num_graphs_per_variant"]
    if args.quick:
        num_graphs = min(num_graphs, 5)
        logger.info("快速冒烟模式: %d 图/变体", num_graphs)

    variants = cfg["topology"]["variants"] if args.topology == "all" else [args.topology]

    # ---- WSN 特征注入（worker 内按节点数采样） ----
    wsn_dir = (root / cfg["data_factory"]["wsn_data_dir"]).resolve()
    wsn_feats = None
    if cfg["data_factory"]["use_wsn_datasets"] and wsn_dir.exists():
        logger.info("集成 WSN 数据集（来自 %s）", wsn_dir)
        wsn_feats = str(wsn_dir)  # 传路径，worker 内按需采样
    else:
        logger.warning("WSN 数据集目录不可用（%s），使用纯拓扑特征", wsn_dir)

    # ---- 批量生成拓扑（参数网格 + chunk 并行） ----
    all_pyg = []
    all_graphs = []
    metadata = []

    # 展开参数网格（--quick 时每 cell 限 3 图）
    cells = expand_grid(cfg)
    if args.topology != "all":
        cells = [c for c in cells if c["variant"] == args.topology]
    if args.num_graphs:
        for c in cells:
            c["n_graphs"] = args.num_graphs
    if args.quick:
        for c in cells:
            c["n_graphs"] = min(c["n_graphs"], 3)
        logger.info("快速冒烟模式: ≤3 图/cell (%d cells)", len(cells))

    total_graphs = sum(c["n_graphs"] for c in cells)
    logger.info("参数网格: %d cells → 预计 %d 图 (变体=%s)",
                len(cells), total_graphs, sorted(set(c["variant"] for c in cells)))

    # ---- shard 目录与断点续传 ----
    shard_dir = Path("data/raw/shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path("data/raw/shards/progress.json")
    done_shards = set()
    if args.resume and progress_path.exists():
        import json
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                done_shards = set(json.load(f).get("done_shards", []))
            logger.info("断点续传: %d 个已完成 shards 将跳过", len(done_shards))
        except Exception:
            done_shards = set()

    # 按图数均分 chunks（chunk ~30 图：进程多时更细粒度，负载均衡 + 进度及时）
    chunk_size = max(15, min(40, total_graphs // (max(args.parallel, 1) * 12)))
    chunks = []
    cur = []
    cur_n = 0
    for c in cells:
        for _ in range(c["n_graphs"]):
            # 展开为单图槽位（n_graphs=1）：worker 对每个 cell 恰好生成 1 图
            # 避免双重展开（旧 bug：完整 cell(n_graphs=150) 入 chunk 导致 4350 图/块）
            slot = dict(c)
            slot["n_graphs"] = 1
            cur.append(slot)
            cur_n += 1
            if cur_n >= chunk_size:
                chunks.append(cur)
                cur, cur_n = [], 0
    if cur:
        chunks.append(cur)
    # 保留原始索引用于 shard 命名与断点续传；过滤已完成
    chunks = [(i, ch) for i, ch in enumerate(chunks) if str(i) not in done_shards]
    logger.info("并行生成: %d 进程 × %d chunks 待处理 (chunk_size=%d)",
                args.parallel, len(chunks), chunk_size)

    import time as _time
    t_start = _time.time()
    n_ok = 0

    def _save_shard(chunk_idx, pyg_list, rows):
        """每 chunk 完成立即落盘 shard + 更新 progress（崩溃不丢已产图）。"""
        nonlocal n_ok
        if pyg_list:
            shard_path = shard_dir / f"shard_{chunk_idx:04d}.pt"
            torch.save(pyg_list, shard_path)
            # 元数据行持久化（append CSV）
            csv_path = Path("data/raw/shards/meta.csv")
            write_header = not csv_path.exists()
            rows_df = pd.DataFrame(rows)
            rows_df.to_csv(csv_path, mode="a", header=write_header, index=False)
            n_ok += len(pyg_list)
        # 更新进度
        import json
        done = set(done_shards)
        done.add(str(chunk_idx))
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"done_shards": sorted(done, key=int)}, f)

    if args.parallel > 1 and total_graphs >= 20:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        multiprocessing.freeze_support()
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            # 限制 in-flight 任务数 = 进程数×2，防跨进程管道缓冲积压死锁
            return_nx = False
            pending = {}
            it = iter(chunks)
            for _ in range(min(len(chunks), args.parallel * 2)):
                ch_idx, ch = next(it)
                pending[ex.submit(_gen_graph_chunk_worker, cfg, ch, wsn_feats, return_nx)] = ch_idx
            done = 0
            while pending:
                fut = next(as_completed(pending))
                ch_idx = pending.pop(fut)
                pyg_list, rows, _nx_list = fut.result()
                _save_shard(ch_idx, pyg_list, rows)
                done += 1
                if done == 1 or done % 5 == 0 or done == len(chunks):
                    logger.info("并行进度: %d/%d chunks 完成 (累计 %d 图, %.1f 图/min)",
                                done, len(chunks), n_ok, n_ok / max((_time.time() - t_start) / 60, 1e-9))
                # 补充下一个任务（保持 in-flight 恒定）
                try:
                    ch_next = next(it)
                    pending[ex.submit(_gen_graph_chunk_worker, cfg, ch_next[1], wsn_feats, return_nx)] = ch_next[0]
                except StopIteration:
                    pass
    else:
        logger.info("串行生成: %d chunks", len(chunks))
        for ch_idx, ch in chunks:
            pyg_list, rows, _nx_list = _gen_graph_chunk_worker(cfg, ch, wsn_feats, return_nx=False)
            _save_shard(ch_idx, pyg_list, rows)

    elapsed = _time.time() - t_start
    logger.info("生成完成: %d 图, 耗时 %.1fs (吞吐 %.2f 图/s)", n_ok, elapsed, n_ok / max(elapsed, 1e-9))

    # ---- 合并 shards 为最终数据集 ----
    all_pyg = []
    for shard_path in sorted(shard_dir.glob("shard_*.pt")):
        all_pyg.extend(torch.load(shard_path, weights_only=False))
    for i, d in enumerate(all_pyg):
        d.graph_id = i
    torch.save(all_pyg, "data/raw/topologies.pt")
    logger.info("已保存 PyG 数据集: data/raw/topologies.pt (%d 图)", len(all_pyg))

    # ---- 元数据（合并 meta.csv 行） ----
    meta_csv = Path("data/raw/shards/meta.csv")
    if meta_csv.exists():
        df = pd.read_csv(meta_csv)
        df["graph_id"] = range(len(df))
        save_csv(df, "data/processed/metadata.csv")
        logger.info("已保存元数据: data/processed/metadata.csv (%d 行)", len(df))

        # 摘要
        summary = df.groupby("topology")[["robustness", "survivability", "composite_score"]].mean()
        logger.info("\n数据集韧性摘要:\n%s", summary.round(4).to_string())
        logger.info("\n[阶段一完成] 数据集概览:\n%s\n韧性指标均值:\n%s",
                    df.groupby("topology").size().to_string(), summary.round(4).to_string())

    # ---- benchmark 量化输出 ----
    if args.benchmark:
        bench = {
            "num_graphs": n_ok, "elapsed_s": round(elapsed, 2),
            "throughput_gps": round(n_ok / max(elapsed, 1e-9), 4),
            "avg_latency_s_per_graph": round(elapsed / max(n_ok, 1), 4),
            "parallel_workers": args.parallel,
            "chunk_size": chunk_size,
            "total_chunks": len(chunks) + len(done_shards),
        }
        import json as _json
        with open("results/tables/data_factory_benchmark.json", "w", encoding="utf-8") as f:
            _json.dump(bench, f, indent=2, ensure_ascii=False)
        logger.info("[Benchmark] 吞吐 %.4f 图/s | 单图延迟 %.3fs | %d 图 / %.1fs",
                    bench["throughput_gps"], bench["avg_latency_s_per_graph"],
                    bench["num_graphs"], bench["elapsed_s"])


if __name__ == "__main__":
    main()
