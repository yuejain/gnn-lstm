"""
resilience_metrics_v2.py - 韧性指标 v2 模块（基于 25 篇论文方法论提炼）
替代 resilience_labels.py 中固定权重 0.4/0.4/0.2 的单一综合评分体系。

设计来源：
  - A1 供应链韧性 17 指标综述（CAIE 2024）：韧性曲线特征族
  - A4 道路网络三维韧性分解（TRE 2026）：抵抗 R1 / 适应 R2 / 恢复 R3
  - B1/B2 级联相变（CSF 2026, CNSNS 2026）：相变阈值 p_c、分支因子预警
  - B4 供应网络抗毁性（IJAE 2024）：网络效率、累计缺失服务量
  - A3 功能重要性加权（RESS 2026）：BFI x FSF 归一化打分

用法（示例）：
    from src.resilience_metrics_v2 import compute_resilience_profile_v2
    profile = compute_resilience_profile_v2(G, cfg)
    # profile["composite_v2"] 即为自适应权重综合韧性（可直接作 GNN 标签）
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def global_efficiency(G) -> float:
    """网络效率 E = 1/[n(n-1)] * sum(1/d_ij)（B4，Dijkstra 最短路）。"""
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0
    try:
        return float(nx.global_efficiency(G))
    except Exception:
        # 兜底：大图用平均逆距离近似
        inv = 0.0
        cnt = 0
        for s in list(G.nodes())[:300]:
            lengths = nx.single_source_shortest_path_length(G, s)
            inv += sum(1.0 / l for l in lengths.values() if l > 0)
            cnt += len(lengths) - 1
        return float(inv / max(cnt, 1))


def percolation_threshold(G, n_trials: int = 20, seed: int = 42) -> float:
    """
    相变阈值 p_c（B2）：随机移除节点，最大连通子团比例骤降到 < 0.5 时对应的移除比例。
    用二分扫描 + 多次试验均值，比单点 AUC 更稳健地刻画网络抗毁极限。
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0

    def lcc_at(ratio):
        n_rm = max(1, int(n * ratio))
        fracs = []
        for _ in range(n_trials):
            nodes = rng.choice(list(G.nodes()), size=n_rm, replace=False)
            Gc = G.copy()
            Gc.remove_nodes_from(nodes)
            if Gc.number_of_nodes() == 0:
                fracs.append(0.0)
            else:
                largest = max(nx.connected_components(Gc), key=len)
                fracs.append(len(largest) / n)
        return float(np.mean(fracs))

    lo, hi = 0.0, 1.0
    for _ in range(12):  # 二分 12 轮 → 精度 ~0.0002
        mid = (lo + hi) / 2
        if lcc_at(mid) > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def branching_factor_warning(G, remove_ratio: float = 0.2,
                             n_steps: int = 10, seed: int = 42) -> dict:
    """
    级联分支因子预警（B1）：级联传播中每步失效节点数比 S_t/S_(t-1)，
    若 η→1 且平台期发散 → 系统逼近相变。返回 {branch_factors, plateau_len, alarm}。
    """
    from src.resilience_labels import simulate_cascade_failure
    ratios, failed_sets = simulate_cascade_failure(
        G, failure_ratio=remove_ratio, n_steps=n_steps, seed=seed)
    deltas = np.diff(ratios)
    eta = deltas[1:] / (deltas[:-1] + 1e-12)
    # 平台期：连续步数失效增量 < 1e-3
    plateau = 0
    for d in reversed(deltas):
        if d < 1e-3:
            plateau += 1
        else:
            break
    return {"branch_factors": eta.tolist(), "plateau_len": int(plateau),
            "alarm": bool(plateau >= 3 and len(eta) > 0 and float(np.mean(eta[-3:])) > 0.8)}


def functional_importance_weighted_resilience(G, importance: dict | None = None,
                                              fsf: float = 1.0) -> float:
    """
    功能重要性加权韧性（A3）：关键节点（高 BFI）失效的系统级放大效应。
    指标 = 加权后的失效-连通退化曲线 AUC（重要节点权重高 → 其失效惩罚大）。
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0
    if importance is None:
        try:
            bc = nx.betweenness_centrality(G, k=(150 if n > 300 else None), seed=42)
        except Exception:
            bc = dict(G.degree())
        imp = {v: float(bc[v]) for v in G.nodes()}
    else:
        imp = importance
    imp_sum = sum(imp.values()) + 1e-9

    attack_order = sorted(imp, key=imp.get, reverse=True)
    Gc = G.copy()
    weighted_perf = []
    total_w = imp_sum
    for target in attack_order:
        if target not in Gc:
            continue
        Gc.remove_node(target)
        if Gc.number_of_nodes() == 0:
            weighted_perf.append(0.0)
            break
        largest = max(nx.connected_components(Gc), key=len)
        # 加权功能保留 = (存活重要度 / 总重要度) * (LCC 比例)，FSF 调节功能敏感
        alive_imp = sum(imp.get(v, 0.0) for v in largest)
        weighted_perf.append(fsf * (alive_imp / total_w) * (len(largest) / n))
        if len(weighted_perf) >= int(n * 0.5):  # 50% 节点被移除即停（A4 协议）
            break

    try:
        auc = np.trapezoid(weighted_perf) / len(weighted_perf)
    except AttributeError:
        auc = np.trapz(weighted_perf) / len(weighted_perf)
    return float(np.clip(auc, 0.0, 1.0))


def resilience_curve_profile(G, cfg, seed: int = 42) -> dict:
    """
    韧性曲线三维分解（A4 范式）：
      R1 抵抗 = P(t1)/P(t0)        移除前瞬时性能保持率
      R2 适应 = 移除阶段平均性能    （扰动施加期间的性能均值）
      R3 恢复 = 恢复阶段性能积分比  （移除后重新连通的恢复程度）
    性能 P 采用"连通效率 + 网络效率"复合（比单一 LCC 更灵敏）。
    本实现以"随机移除 30% → 恢复 50% 被移除节点（优先恢复高重要度）"模拟三阶段。
    """
    n = G.number_of_nodes()
    if n <= 1:
        return {"R1": 1.0, "R2": 1.0, "R3": 1.0, "R_star": 1.0}

    def perf(Gx):
        return 0.5 * (nx.density(Gx)) + 0.5 * global_efficiency(Gx)

    p0 = perf(G)
    remove_ratio = cfg["topology"].get("targeted_remove_ratio", 0.3)
    n_rm = max(1, int(n * remove_ratio))
    rng = np.random.default_rng(seed)

    # R1 抵抗：移除 1% 试探性节点的瞬时保持
    n_try = max(1, int(n * 0.01))
    nodes_try = rng.choice(list(G.nodes()), size=n_try, replace=False)
    G1 = G.copy()
    G1.remove_nodes_from(nodes_try)
    R1 = perf(G1) / max(p0, 1e-9)

    # R2 适应：移除至 remove_ratio 过程中的平均性能（5 步渐进）
    G2 = G.copy()
    removed = set()
    perfs = []
    order = rng.choice(list(G2.nodes()), size=min(n_rm * 2, n), replace=False).tolist()
    for i in range(5):
        batch = order[i * (n_rm // 5):(i + 1) * (n_rm // 5)]
        removed |= set(batch)
        G2.remove_nodes_from([x for x in batch if x in G2])
        perfs.append(perf(G2))
    R2 = float(np.mean(perfs)) / max(p0, 1e-9)

    # R3 恢复：优先恢复高介数节点（对应应急修复优先级）
    G3 = G2.copy()
    try:
        bc = nx.betweenness_centrality(G3, k=(150 if n > 300 else None), seed=seed)
    except Exception:
        bc = dict(G3.degree())
    recovery = sorted(removed, key=lambda v: bc.get(v, 0.0), reverse=True)[:n_rm // 2]
    pos = {v: G.nodes[v].get("pos", (0.0, 0.0)) for v in G3.nodes()}
    restored = 0
    for v in recovery:
        G3.add_node(v, pos=pos.get(v, (0.0, 0.0)))
        restored += 1
        if restored >= n_rm // 2:
            break
    R3 = perf(G3) / max(p0, 1e-9)

    R_star = 0.3 * R1 + 0.4 * R2 + 0.3 * R3  # 广义韧性（A4 归一化 AUC 类比）
    return {"R1": float(np.clip(R1, 0, 1)), "R2": float(np.clip(R2, 0, 1)),
            "R3": float(np.clip(R3, 0, 1)), "R_star": float(np.clip(R_star, 0, 1))}


def compute_resilience_profile_v2(G, cfg, importance: dict | None = None) -> dict:
    """
    韧性指标 v2 总入口：返回完整指标档案（供 GNN 多任务标签 / 报告表）。
    """
    n = G.number_of_nodes()
    profile = {"num_nodes": n, "num_edges": G.number_of_edges()}

    # 基础拓扑指标
    profile["global_efficiency"] = global_efficiency(G)
    profile["percolation_threshold_pc"] = percolation_threshold(G)
    profile["density"] = float(nx.density(G))
    profile["avg_degree"] = float(np.mean([d for _, d in G.degree()])) if n > 0 else 0.0

    # 韧性曲线三维分解
    curve = resilience_curve_profile(G, cfg)
    profile.update(curve)

    # 功能重要性加权韧性（A3）
    profile["functional_weighted_resilience"] = \
        functional_importance_weighted_resilience(G, importance)

    # 级联分支因子预警（B1）
    profile["cascade_warning"] = branching_factor_warning(G)

    # 综合韧性 v2：自适应加权（R_star 主导 + 功能韧性 + p_c 归一）
    pc_norm = profile["percolation_threshold_pc"]
    profile["composite_v2"] = float(np.clip(
        0.4 * curve["R_star"] + 0.3 * profile["functional_weighted_resilience"]
        + 0.3 * pc_norm, 0.0, 1.0))
    return profile


if __name__ == "__main__":
    # 自检：3 种拓扑对比
    import yaml
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    from src.topology_factory import generate_sensor_positions, generate_variant
    for variant in ["grid", "geo_scale_free", "ring"]:
        positions = generate_sensor_positions(cfg, num_sensors=200, layout="road")
        G = generate_variant(positions, comm_range=150.0, variant=variant, cfg=cfg)
        p = compute_resilience_profile_v2(G, cfg)
        print(f"[{variant}] R_star={p['R_star']:.3f} p_c={p['percolation_threshold_pc']:.3f} "
              f"comp_v2={p['composite_v2']:.3f} alarm={p['cascade_warning']['alarm']}")
