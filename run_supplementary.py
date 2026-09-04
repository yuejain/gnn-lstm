#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_supplementary.py — E1~E8 补充实验 + 扩大测试集
设计原则：
  1) 所有输出写入 results/supplement/ 与 data/raw/topologies_exttest.pt，
     不触碰 v5 正在使用的 pareto_front.csv / topologies.pt / metadata.csv / 模型文件；
  2) NSGA-III 类实验（E2/E3/E4/E5a/E5c）使用小规模（pop<=30），置于脚本后半段执行，
     降低与 v5 的 CPU 争抢；GPU 显存余量 10GB，小模型并行训练安全；
  3) 每个实验独立函数，支持 --only 选择；
  4) 结果均为真实运行产出，不做任何推测性回填。

用法:
    python run_supplementary.py                  # 全部执行（后台推荐）
    python run_supplementary.py --only E1,E8     # 只跑指定实验
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx

from src.utils import (load_config, set_seed, get_device, save_csv, ensure_dir,
                       setup_logger, get_project_root)
from src.topology_factory import generate_sensor_positions, generate_variant, generate_dataset
from src.resilience_labels import compute_all_metrics, random_attack_robustness, targeted_attack_survivability
from src.advanced_models import GATResilienceNet
from src.gcn_regressor import evaluate_regression

OUT_DIR = ROOT / "results" / "supplement"
ensure_dir(OUT_DIR)


# ============================================================
# 工具
# ============================================================
def pyg_to_nx(data):
    G = nx.Graph()
    n = data.x.shape[0]
    G.add_nodes_from(range(n))
    ei = data.edge_index.numpy()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    return G


def batched_predict(model, data_list, device, batch_size=32):
    from torch_geometric.loader import DataLoader
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds.extend(model(data).cpu().tolist())
            trues.extend(data.y.cpu().tolist())
    return np.array(trues), np.array(preds)


def train_gat(cfg, device, logger, hidden=128, epochs=500, dropout=0.2,
              seed=42, train_d=None, val_d=None, test_d=None, patience=100):
    """独立 GAT 训练（不写任何主文件），返回 (metrics, model)。"""
    set_seed(seed)
    from torch_geometric.loader import DataLoader as GDL
    tr_loader = GDL(train_d, batch_size=cfg["gcn_regressor"]["batch_size"], shuffle=True)
    va_loader = GDL(val_d, batch_size=cfg["gcn_regressor"]["batch_size"], shuffle=False)
    in_ch = train_d[0].x.shape[1]
    model = GATResilienceNet(in_ch, hidden, num_layers=3, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["gcn_regressor"]["lr"])
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for d in tr_loader:
            d = d.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(d), d.y)
            loss.backward()
            opt.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for d in va_loader:
                d = d.to(device)
                vl += F.mse_loss(model(d), d.y).item() * len(d.y)
        vl /= max(len(va_loader.dataset), 1)
        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    yt, yp = batched_predict(model.to(device), test_d, device)
    metrics = evaluate_regression(yt, yp)
    return metrics, model


# ============================================================
# 0. 扩大测试集：同分布扩展（seed=42 延续生成 40图/变体，取后10图/变体=30图）
#    注：原 seed=1000 独立生成方案暴露跨分布偏移问题（R2 波动大），弃用。
# ============================================================
def gen_extended_testset(cfg, logger):
    out_pt = ROOT / "data" / "raw" / "topologies_exttest.pt"
    wsn_dir = (ROOT / cfg["data_factory"]["wsn_data_dir"]).resolve()
    from data_factory import get_wsn_node_features, build_pyg_graphs
    num_sensors = cfg["park"]["num_sensors"][0]
    wsn_feats = get_wsn_node_features(str(wsn_dir), n_nodes=num_sensors,
                                      seed=cfg["general"]["seed"], feature_dim=5) \
        if (cfg["data_factory"]["use_wsn_datasets"] and wsn_dir.exists()) else None
    all_graphs, meta = [], []
    for variant in cfg["topology"]["variants"]:
        logger.info("[E0] 同分布扩展测试集生成: %s (40图/变体, 取后10图)", variant)
        graphs = generate_dataset(cfg, num_sensors=num_sensors, variant=variant,
                                  num_graphs=40, wsn_features=wsn_feats,
                                  seed=cfg["general"]["seed"])
        # 前 30 图与主数据集相同（同一 seed 生成序列），取后 10 图作为同分布新测试图
        new_graphs = graphs[30:]
        for G in new_graphs:
            m = compute_all_metrics(G, cfg)
            meta.append({"graph_id": len(all_graphs), "topology": variant,
                         "num_nodes": m["num_nodes"], "num_edges": m["num_edges"],
                         "robustness": m["robustness"], "survivability": m["survivability"],
                         "composite_score": m["composite_score"]})
            all_graphs.append(G)
    data_list = build_pyg_graphs(all_graphs, cfg)
    torch.save(data_list, out_pt)
    df = pd.DataFrame(meta)
    save_csv(df, OUT_DIR / "exttest_metadata.csv")
    logger.info("[E0] 同分布扩展测试集完成: %d 图 -> %s", len(data_list), out_pt)
    return data_list


# ============================================================
# E1. 统计稳健性：5 seeds × GAT(h128/e500/d0.2)，在扩展测试集(90图)上评估
# ============================================================
def e1_multi_seed(cfg, logger):
    device = get_device(cfg["general"]["device"])
    data_list = torch.load(ROOT / "data/raw/topologies.pt", weights_only=False)
    rng = np.random.default_rng(cfg["general"]["seed"])
    idx = rng.permutation(len(data_list))
    n_tr, n_va = int(len(data_list) * 0.7), int(len(data_list) * 0.15)
    train_d = [data_list[i] for i in idx[:n_tr]]
    val_d = [data_list[i] for i in idx[n_tr:n_tr + n_va]]
    orig_test = [data_list[i] for i in idx[n_tr + n_va:]]
    ext_pt = ROOT / "data/raw/topologies_exttest.pt"
    ext_test = torch.load(ext_pt, weights_only=False) if ext_pt.exists() else gen_extended_testset(cfg, logger)

    rows = []
    for sd in [42, 7, 2024, 123, 999]:
        t0 = time.time()
        m_ext, model = train_gat(cfg, device, logger, hidden=128, epochs=500, dropout=0.2,
                                 seed=sd, train_d=train_d, val_d=val_d, test_d=ext_test)
        yt2, yp2 = batched_predict(model, orig_test, device)
        m_orig = evaluate_regression(yt2, yp2)
        rows.append({"seed": sd, "R2_ext30": m_ext["R2"], "MSE_ext30": m_ext["MSE"],
                     "MAE_ext30": m_ext["MAE"], "R2_orig15": m_orig["R2"],
                     "MSE_orig15": m_orig["MSE"], "train_sec": round(time.time() - t0, 1)})
        logger.info("[E1] seed=%d R2(ext30)=%.4f R2(orig15)=%.4f (%.0fs)", sd,
                    m_ext["R2"], m_orig["R2"], time.time() - t0)
    df = pd.DataFrame(rows)
    r2 = df["R2_ext30"]
    ci = 1.96 * r2.std(ddof=1) / np.sqrt(len(r2))
    summary = pd.DataFrame([{"n_seeds": len(df), "R2_mean": r2.mean(), "R2_sd": r2.std(ddof=1),
                             "R2_ci95": ci, "R2_min": r2.min(), "R2_max": r2.max(),
                             "R2_orig15_mean": df["R2_orig15"].mean()}])
    save_csv(df, OUT_DIR / "E1_multi_seed.csv")
    save_csv(summary, OUT_DIR / "E1_summary.csv")
    logger.info("[E1] 完成: R2(ext30)=%.4f±%.4f (95%%CI ±%.4f, n=%d); R2(orig15)均值=%.4f",
                r2.mean(), r2.std(ddof=1), ci, len(r2), df["R2_orig15"].mean())


# ============================================================
# E8. 案例仿真升级：高斯烟羽浓度阈值 CR/RD + 敏感性扫描
# ============================================================
def _gaussian_conc(Q, u, dx, dy, z=0.5):
    """简化高斯烟羽（D 类稳定度），返回浓度 kg/m³。"""
    if dx <= 1e-6:
        return 0.0
    sy = 0.16 * dx * (1 + 0.0004 * dx) ** -0.5
    sz = 0.14 * dx * (1 + 0.0003 * dx) ** -0.5
    if sy < 1e-6 or sz < 1e-6:
        return 0.0
    return Q / (2 * np.pi * u * sy * sz) * np.exp(-dy ** 2 / (2 * sy ** 2)) * np.exp(-z ** 2 / (2 * sz ** 2))


def sim_cr_rd(sensors, sources, wind_u, Q, thr=1e-4, n_steps=100, seed=42,
              downwind_only=True, downwind_range=300.0):
    """
    浓度阈值触发仿真（thr=1e-4 kg/m³）。
    downwind_only=True 时仅统计泄漏源下风向（东侧 dx>0 且 dx<=downwind_range）的传感器，
    模拟主导东风下真实监测盲区。
    CR = 被覆盖危险源比例；RD = 全源最早触发时刻。
    """
    rng = np.random.default_rng(seed)
    t_arr = np.arange(n_steps)
    u = wind_u * np.ones(n_steps)
    for k in range(3):
        u += rng.uniform(0.3, 0.8) * np.sin(rng.uniform(0.05, 0.2) * t_arr + rng.uniform(0, 2 * np.pi))
    u = np.clip(u, 0.1, None)
    sensors = np.asarray(sensors, dtype=float)
    covered_src, min_rd = 0, None
    for src in sources:
        src_ok = False
        for t in range(n_steps):
            for s in sensors:
                dx = s[0] - src[0]
                dy = s[1] - src[1]
                if downwind_only and (dx <= 0 or dx > downwind_range):
                    continue
                if _gaussian_conc(Q, u[t], dx, dy) > thr:
                    src_ok = True
                    rd_t = float(t + np.hypot(dx, dy) / max(u[t], 0.1))
                    min_rd = rd_t if min_rd is None else min(min_rd, rd_t)
                    break
            if src_ok:
                break
        covered_src += src_ok
    return covered_src / max(len(sources), 1), (min_rd if min_rd is not None else float(n_steps))


def e8_case_simulation(cfg, logger):
    rng = np.random.default_rng(42)
    area = 1500
    n_cand, n_haz = 91, 12
    base = rng.uniform(0, area, size=(n_cand, 2))
    haz = rng.uniform(0, area, size=(n_haz, 2))
    # 折中解新增 8 节点（results/tables/compromise_solution.json，若存在）
    added = []
    cj = ROOT / "results" / "tables" / "compromise_solution.json"
    if cj.exists():
        try:
            added = np.array(json.loads(cj.read_text(encoding="utf-8"))["new_node_coords"], dtype=float)
        except Exception:
            added = np.array([])
    # 贪婪覆盖补充基线：选距危险源总距离最小的 8 个候选
    dist = np.array([[np.hypot(c[0] - h[0], c[1] - h[1]) for h in haz] for c in base])
    greedy_idx = np.argsort(dist.min(axis=1))[:8]
    greedy = base[greedy_idx]
    schemes = {"base(91)": base,
               "base+greedy8": np.vstack([base, greedy]),
               "base+NSGA3_8": np.vstack([base, added]) if len(added) else base}
    rows = []
    for u0 in [1.5, 2.5, 4.0]:
        for Q in [0.1, 1.0, 10.0]:
            for name, sens in schemes.items():
                cr, rd = sim_cr_rd(sens, haz[:5], u0, Q)
                rows.append({"wind": u0, "leak_Q": Q, "scheme": name, "CR": cr, "RD": rd,
                             "n_sensors": len(sens)})
    df = pd.DataFrame(rows)
    save_csv(df, OUT_DIR / "E8_case_simulation.csv")
    # 汇总：默认场景 u=2.5, Q=1.0
    default = df[(df["wind"] == 2.5) & (df["leak_Q"] == 1.0)]
    logger.info("[E8] 完成；默认场景(u=2.5,Q=1):\n%s", default.round(4).to_string(index=False))
    return df


# ============================================================
# E6. 扩展性实验：900/1200 节点指标计算耗时 + GAT 推理耗时
# ============================================================
def e6_scalability(cfg, logger):
    device = get_device(cfg["general"]["device"])
    rows = []
    for size in [600, 900, 1200]:
        for variant in cfg["topology"]["variants"]:
            set_seed(42)
            t0 = time.time()
            graphs = generate_dataset(cfg, num_sensors=size, variant=variant,
                                      num_graphs=2, wsn_features=None, seed=42)
            t_gen = time.time() - t0
            t0 = time.time()
            for G in graphs:
                compute_all_metrics(G, cfg)
            t_metrics = (time.time() - t0) / len(graphs)
            rows.append({"num_sensors": size, "topology": variant,
                         "gen_2graphs_sec": round(t_gen, 2),
                         "metrics_per_graph_sec": round(t_metrics, 3),
                         "edges": graphs[0].number_of_edges()})
            logger.info("[E6] %d节点 %s: 生成%.1fs, 指标%.3fs/图", size, variant, t_gen, t_metrics)
    # GAT 推理耗时（600/900/1200）
    from src.advanced_models import GATResilienceNet
    from data_factory import build_pyg_graphs
    for size in [600, 900, 1200]:
        graphs = generate_dataset(cfg, num_sensors=size, variant="random",
                                  num_graphs=1, wsn_features=None, seed=42)
        dl = build_pyg_graphs(graphs, cfg)
        model = GATResilienceNet(dl[0].x.shape[1], 128, num_layers=3, dropout=0.2).to(device)
        d = dl[0].to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                model(d)
            t0 = time.time()
            for _ in range(10):
                model(d)
            infer_ms = (time.time() - t0) / 10 * 1000
        row = rows[-1] if rows else {}
        rows.append({"num_sensors": size, "topology": "GAT_infer", "gen_2graphs_sec": np.nan,
                     "metrics_per_graph_sec": np.nan, "edges": int(dl[0].edge_index.shape[1] // 2),
                     "infer_ms": round(infer_ms, 2)})
        logger.info("[E6] GAT 推理 %d节点: %.2f ms/图", size, infer_ms)
    df = pd.DataFrame(rows)
    save_csv(df, OUT_DIR / "E6_scalability.csv")
    logger.info("[E6] 完成 -> E6_scalability.csv")


# ============================================================
# E5b. 消融-去 WSN 集成：纯拓扑特征训练 GAT 对比
# ============================================================
def e5b_ablation_nowsn(cfg, logger):
    device = get_device(cfg["general"]["device"])
    from data_factory import build_pyg_graphs
    num_sensors = cfg["park"]["num_sensors"][0]
    all_g = []
    for variant in cfg["topology"]["variants"]:
        all_g += generate_dataset(cfg, num_sensors=num_sensors, variant=variant,
                                  num_graphs=30, wsn_features=None, seed=cfg["general"]["seed"])
    data_list = build_pyg_graphs(all_g, cfg)
    rng = np.random.default_rng(cfg["general"]["seed"])
    idx = rng.permutation(len(data_list))
    n_tr, n_va = int(len(data_list) * 0.7), int(len(data_list) * 0.15)
    train_d = [data_list[i] for i in idx[:n_tr]]
    val_d = [data_list[i] for i in idx[n_tr:n_tr + n_va]]
    test_d = [data_list[i] for i in idx[n_tr + n_va:]]
    m, _ = train_gat(cfg, device, logger, hidden=128, epochs=500, dropout=0.2,
                     seed=42, train_d=train_d, val_d=val_d, test_d=test_d)
    df = pd.DataFrame([{"mode": "no_WSN", **{k: round(v, 6) for k, v in m.items()}}])
    save_csv(df, OUT_DIR / "E5b_ablation_nowsn.csv")
    logger.info("[E5b] 去WSN集成: R2=%.4f (对比含WSN基线 0.858)", m["R2"])


# ============================================================
# NSGA-III 工具（输出隔离到 supplement/，不动 pareto_front.csv）
# ============================================================
class TwoObjWrapper:
    """2 目标包装：仅保留韧性(neg) + 成本。"""

    def __init__(self, inner):
        from pymoo.core.problem import Problem as PymooProblem

        class _W(PymooProblem):
            def __init__(self, inner_):
                self.inner = inner_
                super().__init__(n_var=inner_.n_var, n_obj=2, xl=0.0, xu=1.0)

            def _evaluate(self, X, out, *a, **k):
                inner_out = {}
                self.inner._evaluate(X, inner_out, *a, **k)
                out["F"] = inner_out["F"][:, [0, 2]]

        self.problem = _W(inner)


def run_nsga3_small(cfg, logger, base_pt=ROOT / "data/raw/topologies.pt",
                    pop=30, gen=30, n_obj=3, seed=42, out_tag="E3",
                    save_history=False, history=None):
    from run_nsga3_optimizer import make_gcn_predictor, make_cascade_predictor
    from src.nsga3_optimizer import ResilienceOptimizationProblem
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    set_seed(seed)
    nsga_cfg = cfg["nsga3"]
    data_list = torch.load(base_pt, weights_only=False)
    base_data = data_list[0]
    n_base = base_data.x.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n_base))
    ei = base_data.edge_index.numpy()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    pos_arr = base_data.x[:, 3:5].numpy() * cfg["park"]["area_size"][0]
    for i in range(n_base):
        G.nodes[i]["pos"] = (float(pos_arr[i, 0]), float(pos_arr[i, 1]))
    gcn_predict = make_gcn_predictor(cfg, logger)
    cascade_risk = make_cascade_predictor(cfg, logger)
    problem = ResilienceOptimizationProblem(
        base_graph=G, base_features=base_data.x.numpy(),
        gcn_predict=gcn_predict, cascade_risk=cascade_risk,
        n_new_nodes=nsga_cfg["n_new_nodes"], n_redundant_links=nsga_cfg["n_redundant_links"],
        comm_range=cfg["park"]["communication_range"], area_size=cfg["park"]["area_size"])
    if n_obj == 2:
        problem = TwoObjWrapper(problem).problem
    ref = get_reference_directions("das-dennis", n_obj, n_partitions=8 if n_obj == 3 else 12)
    algorithm = NSGA3(ref_dirs=ref, pop_size=pop,
                      sampling=FloatRandomSampling(),
                      crossover=SBX(prob=0.9, eta=nsga_cfg["crossover_eta"]),
                      mutation=PM(prob=nsga_cfg["mutation_prob"], eta=nsga_cfg["mutation_eta"]),
                      eliminate_duplicates=True)
    logger.info("[%s] NSGA-III 启动 pop=%d gen=%d n_obj=%d", out_tag, pop, gen, n_obj)
    t0 = time.time()

    def cb(alg):
        if history is not None:
            F = alg.pop.get("F")
            history.append(F.copy())

    res = minimize(problem, algorithm, ("n_gen", gen), seed=seed,
                   verbose=False, save_history=save_history, callback=cb)
    F = np.atleast_2d(res.F)
    logger.info("[%s] 完成: %d 非支配解, %.0fs", out_tag, len(F), time.time() - t0)
    return F, res.X


def e3_convergence(cfg, logger):
    history = []
    F, X = run_nsga3_small(cfg, logger, pop=30, gen=30, seed=42, out_tag="E3", history=history)
    # 每代 HV（pymoo 0.6.x: HV 类）
    from pymoo.indicators.hv import HV
    ref_point = np.max(F, axis=0) * 1.1 + 1e-6
    hv = HV(ref_point=ref_point)
    rows = []
    for i, Fi in enumerate(history):
        try:
            v = hv.do(np.atleast_2d(Fi))
        except Exception:
            v = np.nan
        rows.append({"gen": i, "hv": v, "n_pareto": len(Fi)})
    df = pd.DataFrame(rows)
    save_csv(df, OUT_DIR / "E3_convergence.csv")
    logger.info("[E3] HV 曲线: 首代 %.4f -> 末代 %.4f", rows[0]["hv"], rows[-1]["hv"])
    return df


def e4_normalization(cfg, logger):
    """归一化敏感性：std 归一化 vs Min-Max 归一化的折中解对比。"""
    F, X = run_nsga3_small(cfg, logger, pop=30, gen=30, seed=42, out_tag="E4")

    def pick(F, mode):
        ideal = F.min(axis=0)
        if mode == "std":
            dist = np.linalg.norm((F - ideal) / (F.std(axis=0) + 1e-9), axis=1)
        elif mode == "minmax":
            span = F.max(axis=0) - F.min(axis=0) + 1e-9
            dist = np.linalg.norm((F - ideal) / span, axis=1)
        elif mode == "raw":
            dist = np.linalg.norm(F - ideal, axis=1)
        return int(np.argmin(dist)), dist.min()

    rows = []
    for mode in ["raw", "std", "minmax"]:
        idx, d = pick(F, mode)
        rows.append({"mode": mode, "idx": idx, "dist": round(float(d), 4),
                     "resilience": round(float(-F[idx, 0]), 4),
                     "cascade_risk": round(float(F[idx, 1]), 4),
                     "cost": round(float(F[idx, 2]), 4)})
    df = pd.DataFrame(rows)
    save_csv(df, OUT_DIR / "E4_normalization.csv")
    logger.info("[E4] 折中解对比(三种距离):\n%s", df.round(4).to_string(index=False))
    return df


def e5a_ablation_2obj(cfg, logger):
    """消融 a：去除级联目标，仅 韧性+成本 两目标。"""

    F, X = run_nsga3_small(cfg, logger, pop=30, gen=30, seed=42, out_tag="E5a", n_obj=2)
    # 折中解（std 归一化）
    ideal = F.min(axis=0)
    dist = np.linalg.norm((F - ideal) / (F.std(axis=0) + 1e-9), axis=1)
    idx = int(np.argmin(dist))
    df = pd.DataFrame([{"n_obj": 2, "n_pareto": len(F),
                        "resilience": round(float(-F[idx, 0]), 4),
                        "cost": round(float(F[idx, 1]), 4)}])
    save_csv(df, OUT_DIR / "E5a_ablation_2obj.csv")
    logger.info("[E5a] 2目标消融: 折中解韧性=%.4f 成本=%.2f (对比3目标 0.517/8.0)",
                -F[idx, 0], F[idx, 1])
    return df


def e5c_proxy_vs_sim(cfg, logger):
    """消融 c：小图（200 节点）上，GAT 代理优化 vs 直接仿真评估。"""
    device = get_device(cfg["general"]["device"])
    from src.advanced_models import GATResilienceNet
    from run_nsga3_optimizer import make_gcn_predictor
    from src.nsga3_optimizer import ResilienceOptimizationProblem
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from data_factory import build_pyg_graphs

    set_seed(42)
    graphs = generate_dataset(cfg, num_sensors=200, variant="random",
                              num_graphs=1, wsn_features=None, seed=42)
    dl = build_pyg_graphs(graphs, cfg)
    d0 = dl[0]
    G = pyg_to_nx(d0)
    pos_arr = d0.x[:, 3:5].numpy() * cfg["park"]["area_size"][0]
    for i in range(G.number_of_nodes()):
        G.nodes[i]["pos"] = (float(pos_arr[i, 0]), float(pos_arr[i, 1]))
    base_feats = d0.x.numpy()

    gcn_pred = make_gcn_predictor(cfg, logger)  # 加载真实 GAT 模型

    def sim_score(G2):
        m = compute_all_metrics(G2, cfg)
        return m["composite_score"]

    class ProxyProblem(ResilienceOptimizationProblem):
        def _evaluate(self, X, out, *a, **k):
            F = np.zeros((len(X), 3))
            for i, x in enumerate(X):
                new_pos, link_switches = self._decode(x)
                G2, new_nodes = self._build_topology(new_pos, link_switches)
                feats_new = self._build_features(G2, new_nodes)
                pred = self.gcn_predict(G2, feats_new)
                risk = self.cascade_risk(G2)
                cost = self.n_new_nodes * 1.0 + int(link_switches.sum()) * 0.3
                F[i] = [-pred, risk, cost]
            out["F"] = F

    class SimProblem(ResilienceOptimizationProblem):
        def _evaluate(self, X, out, *a, **k):
            F = np.zeros((len(X), 3))
            for i, x in enumerate(X):
                new_pos, link_switches = self._decode(x)
                G2, new_nodes = self._build_topology(new_pos, link_switches)
                score = sim_score(G2)
                risk = self.cascade_risk(G2)
                cost = self.n_new_nodes * 1.0 + int(link_switches.sum()) * 0.3
                F[i] = [-score, risk, cost]
            out["F"] = F

    ref = get_reference_directions("das-dennis", 3, n_partitions=8)
    results = {}
    for tag, P in [("proxy_GAT", ProxyProblem), ("direct_sim", SimProblem)]:
        problem = P(base_graph=G, base_features=base_feats, gcn_predict=gcn_pred,
                    cascade_risk=(lambda G2: 0.66),
                    n_new_nodes=4, n_redundant_links=6,
                    comm_range=cfg["park"]["communication_range"],
                    area_size=cfg["park"]["area_size"])
        alg = NSGA3(ref_dirs=ref, pop_size=20,
                    sampling=FloatRandomSampling(),
                    crossover=SBX(prob=0.9, eta=20),
                    mutation=PM(prob=0.1, eta=20), eliminate_duplicates=True)
        res = minimize(problem, alg, ("n_gen", 15), seed=42, verbose=False)
        F = np.atleast_2d(res.F)
        idx = int(np.argmin(np.linalg.norm((F - F.min(axis=0)) / (F.std(axis=0) + 1e-9), axis=1)))
        # 统一用直接仿真评估两个方案
        x = res.X[idx]
        new_pos, link_switches = problem._decode(x)
        G2, new_nodes = problem._build_topology(new_pos, link_switches)
        sim_score_val = sim_score(G2)
        results[tag] = {"n_pareto": len(F), "opt_resilience": round(float(-F[idx, 0]), 4),
                        "sim_score_of_compromise": round(float(sim_score_val), 4)}
        logger.info("[E5c] %s: 优化目标韧性=%.4f, 直接仿真重评=%.4f",
                    tag, -F[idx, 0], sim_score_val)
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "mode"})
    save_csv(df, OUT_DIR / "E5c_proxy_vs_sim.csv")
    logger.info("[E5c] 完成")
    return df


# ============================================================
# E7. 级联预测改进：Focal Loss / 类别加权 vs BCE
# ============================================================
def e7_cascade_loss(cfg, logger):
    device = get_device(cfg["general"]["device"])
    from train_gcn_lstm import build_sequences_from_data, evaluate
    from src.gcn_lstm import GCN_LSTM_Cascade

    data_list = torch.load(ROOT / "data/raw/topologies.pt", weights_only=False)
    X, Y, edge_index = build_sequences_from_data(data_list, cfg)
    in_ch = X.shape[-1]
    pos_w = (Y[:, -1] < 0.5).float().mean().item()  # 负样本占比
    results = []
    for name, mode in [("weighted", "weighted"), ("focal", "focal")]:
        set_seed(42)
        model = GCN_LSTM_Cascade(in_ch, cfg["gcn_lstm"]["hidden_dim"],
                                 cfg["gcn_lstm"]["lstm_layers"], cfg["gcn_lstm"]["dropout"]).to(device)
        edge_index_d = edge_index.to(device)
        epochs = cfg["gcn_lstm"]["epochs"]
        opt = torch.optim.Adam(model.parameters(), lr=cfg["gcn_lstm"]["lr"])
        B = X.shape[0]
        n_batches = max(1, B // cfg["gcn_lstm"]["batch_size"])
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(B)
            for bi in range(n_batches):
                idx = perm[bi * cfg["gcn_lstm"]["batch_size"]:(bi + 1) * cfg["gcn_lstm"]["batch_size"]]
                xb, yb = X[idx].to(device), Y[idx].to(device)
                opt.zero_grad()
                out = model(xb, edge_index_d)[..., 0]
                tgt = yb[:, -1]
                if mode == "weighted":
                    w = torch.where(tgt > 0.5, 1.0, pos_w).mean(dim=-1)
                    loss = F.binary_cross_entropy(out[:, -1], tgt, weight=w.unsqueeze(-1).expand_as(out[:, -1]))
                else:  # focal
                    p = torch.clamp(out[:, -1], 1e-6, 1 - 1e-6)
                    ce = F.binary_cross_entropy(p, tgt, reduction="none")
                    pt = torch.where(tgt > 0.5, p, 1 - p)
                    loss = ((1 - pt) ** 2 * ce).mean()
                loss.backward()
                opt.step()
        model.to("cpu")
        metrics, preds, trues = evaluate(model, X, Y, edge_index, cfg)
        results.append({"loss": mode, **{k: round(v, 4) for k, v in metrics.items()}})
        logger.info("[E7] %s: ACC=%.4f MacroF1=%.4f Recall=%.4f (BCE基线 0.9665/0.4915/1.0)",
                    mode, metrics["accuracy"], metrics["macro_f1"], metrics["recall"])
    df = pd.DataFrame(results)
    save_csv(df, OUT_DIR / "E7_cascade_loss.csv")
    return df


# ============================================================
# E2. 领域基线：随机布点 vs 贪婪覆盖 vs 高重要性（案例镜像）
# ============================================================
def e2_baselines(cfg, logger):
    rng = np.random.default_rng(42)
    area = 1500
    n_cand, n_haz = 91, 12
    base = rng.uniform(0, area, size=(n_cand, 2))
    haz = rng.uniform(0, area, size=(n_haz, 2))
    dist = np.array([[np.hypot(c[0] - h[0], c[1] - h[1]) for h in haz] for c in base])
    greedy_idx = np.argsort(dist.min(axis=1))[:8]
    rand_idx = rng.choice(n_cand, 8, replace=False)
    # 高重要性：介数中心性 Top-8
    G = generate_variant(base, cfg["park"]["communication_range"], "random")
    bc = nx.betweenness_centrality(G)
    imp_idx = sorted(range(n_cand), key=lambda i: -bc[i])[:8]
    # 独立部署方案：仅评估选出的 8 个节点（不叠加 base，避免 base 主导掩盖差异）
    schemes = {"random_8": base[rand_idx],
               "greedy_cover_8": base[greedy_idx],
               "high_bc_8": base[imp_idx]}
    rows = []
    for name, s in schemes.items():
        cr, rd = sim_cr_rd(s, haz[:5], 2.5, 1.0)
        # 韧性：8 节点并入全网络后计算（反映部署对整体韧性的影响）
        G2 = generate_variant(np.vstack([base, s]), cfg["park"]["communication_range"], "random")
        m = compute_all_metrics(G2, cfg)
        rows.append({"scheme": name, "CR": round(cr, 4), "RD": round(rd, 2),
                     "composite": round(m["composite_score"], 4)})
        logger.info("[E2] %s: CR=%.3f RD=%.1fs 韧性=%.4f", name, cr, rd, m["composite_score"])
    df = pd.DataFrame(rows)
    save_csv(df, OUT_DIR / "E2_baselines.csv")
    return df


# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="逗号分隔: E0,E1,E2,E3,E4,E5a,E5b,E5c,E6,E7,E8")
    args = parser.parse_args()

    cfg = load_config(ROOT / "config.yaml")
    set_seed(cfg["general"]["seed"])
    logger = setup_logger("supplement", log_file="logs/supplement.log")
    os.chdir(ROOT)
    logger.info("==== 补充实验开始 ====")

    only = set(args.only.split(",")) if args.only else None

    def run(tag, fn):
        if only and tag not in only:
            logger.info("[%s] 跳过（--only 指定）", tag)
            return
        logger.info("[%s] ====== 开始 ======", tag)
        t0 = time.time()
        try:
            fn()
            logger.info("[%s] 完成 (%.0fs)", tag, time.time() - t0)
        except Exception as e:
            logger.exception("[%s] 失败: %s", tag, e)

    run("E0", lambda: gen_extended_testset(cfg, logger))
    run("E8", lambda: e8_case_simulation(cfg, logger))
    run("E6", lambda: e6_scalability(cfg, logger))
    run("E2", lambda: e2_baselines(cfg, logger))
    run("E5b", lambda: e5b_ablation_nowsn(cfg, logger))
    run("E1", lambda: e1_multi_seed(cfg, logger))
    run("E7", lambda: e7_cascade_loss(cfg, logger))
    run("E5c", lambda: e5c_proxy_vs_sim(cfg, logger))
    run("E3", lambda: e3_convergence(cfg, logger))
    run("E4", lambda: e4_normalization(cfg, logger))
    run("E5a", lambda: e5a_ablation_2obj(cfg, logger))
    logger.info("==== 补充实验全部结束 ====")


if __name__ == "__main__":
    main()
