"""
resilience_labels.py - 韧性标签计算
指标定义（对应技术路线阶段二 + 论文韧性评估）：
- 鲁棒性 (Robustness): 随机失效 10%-50% 后最大连通子团比例（AUC 曲线下面积）
- 抗毁性 (Survivability): 蓄意攻击（介数中心性排序）30% 节点的性能退化 AUC
- k-连通度: 网络节点连通度（大图用最小度上界近似）
- 综合韧性评分: 0.4*鲁棒性 + 0.4*抗毁性 + 0.2*(k连通/n)
- 级联失效标签: 10 时间步失效比例曲线（供 GCN-LSTM 训练）
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def random_attack_robustness(G, remove_ratios=(0.1, 0.2, 0.3, 0.4, 0.5),
                             n_trials: int = 3, seed: int = 42) -> float:
    """
    随机失效鲁棒性：对多个移除比例求最大连通子团比例的均值，再对比例序列求 AUC。

    Returns:
        鲁棒性分数 ∈ [0, 1]
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0

    ratios = []
    for ratio in remove_ratios:
        n_remove = max(1, int(n * ratio))
        fracs = []
        for _ in range(n_trials):
            nodes = rng.choice(list(G.nodes()), size=min(n_remove, n), replace=False)
            G_removed = G.copy()
            G_removed.remove_nodes_from(nodes)
            if G_removed.number_of_nodes() == 0:
                fracs.append(0.0)
            else:
                largest = max(nx.connected_components(G_removed), key=len)
                fracs.append(len(largest) / n)
        ratios.append(np.mean(fracs))

    # 梯形积分 AUC（移除比例为横轴）; 兼容 numpy 2.x (trapz → trapezoid)
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz
    auc = _trapz(ratios) / max(len(ratios) - 1, 1)
    return float(np.clip(auc, 0.0, 1.0))


def targeted_attack_survivability(G, remove_ratio: float = 0.3,
                                  seed: int = 42) -> float:
    """
    蓄意攻击抗毁性：按介数中心性排序逐节点移除，计算性能退化曲线 AUC。

    优化：采用"静态攻击"策略——基于初始网络的介数中心性一次性排序，
    按固定顺序移除（论文与图论研究标准做法）。相比每步重算介数（O(N²·E)），
    计算复杂度降至 O(N·E)，600 节点规模加速 ~100 倍。

    Returns:
        抗毁性分数 ∈ [0, 1]（越大越抗攻击）
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 1.0

    # 复用集中计算的介数（compute_all_metrics 已缓存）；无缓存则 k 采样计算
    bc = G.graph.get("betweenness")
    if bc is None:
        try:
            bc = nx.betweenness_centrality(G, k=(150 if n > 300 else None), seed=seed)
        except Exception:
            bc = dict(nx.degree(G))
    attack_order = sorted(bc, key=bc.get, reverse=True)

    G_copy = G.copy()
    n_remove = int(n * remove_ratio)
    performance = []
    for i in range(n_remove):
        if G_copy.number_of_nodes() <= 1:
            performance.append(0.0)
            break
        target = attack_order[i] if i < len(attack_order) else list(G_copy.nodes())[0]
        if target not in G_copy:
            # 已移除，改取当前网络介数最高节点（兜底）
            try:
                bc_cur = nx.betweenness_centrality(G_copy)
            except Exception:
                bc_cur = dict(nx.degree(G_copy))
            target = max(bc_cur, key=bc_cur.get)
        G_copy.remove_node(target)
        if G_copy.number_of_nodes() > 0:
            largest = max(nx.connected_components(G_copy), key=len)
            performance.append(len(largest) / n)
        else:
            performance.append(0.0)

    if len(performance) == 0:
        return 1.0
    try:
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz
    auc = _trapz(performance) / len(performance)
    return float(np.clip(auc, 0.0, 1.0))


def compute_k_connectivity(G) -> int:
    """
    计算节点连通度 k。大图（n>100）用最小度上界近似以控制计算量。
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0
    if n > 100:
        # 上界近似：k <= 最小度
        return max(0, min(d for _, d in G.degree()))
    try:
        return int(nx.node_connectivity(G))
    except Exception:
        return 0


def compute_composite_score(robustness: float, survivability: float,
                            k_conn: int, n: int) -> float:
    """综合韧性评分 = 0.4*鲁棒性 + 0.4*抗毁性 + 0.2*(k连通/n)"""
    return 0.4 * robustness + 0.4 * survivability + 0.2 * (k_conn / max(n, 1))


def compute_all_metrics(G, cfg) -> dict:
    """
    计算图 G 的全部韧性指标。

    Returns:
        dict: robustness, survivability, k_connectivity, composite_score,
              largest_cc_ratio, avg_degree, density
    """
    n = G.number_of_nodes()
    if n == 0:
        return {
            "num_nodes": 0, "num_edges": 0, "robustness": 0.0,
            "survivability": 0.0, "k_connectivity": 0, "composite_score": 0.0,
            "avg_degree": 0.0, "density": 0.0,
        }

    # 集中计算一次介数中心性并缓存（survivability/cascade/features 全流程复用）
    if G.graph.get("betweenness") is None and n > 1:
        try:
            G.graph["betweenness"] = nx.betweenness_centrality(
                G, k=(150 if n > 300 else None), seed=cfg["general"]["seed"])
        except Exception:
            G.graph["betweenness"] = dict(nx.degree(G))

    robustness = random_attack_robustness(
        G, remove_ratios=cfg["topology"]["random_remove_ratio"], seed=cfg["general"]["seed"])
    survivability = targeted_attack_survivability(
        G, remove_ratio=cfg["topology"]["targeted_remove_ratio"], seed=cfg["general"]["seed"])
    k_conn = compute_k_connectivity(G)
    composite = compute_composite_score(robustness, survivability, k_conn, n)

    return {
        "num_nodes": n,
        "num_edges": G.number_of_edges(),
        "robustness": robustness,
        "survivability": survivability,
        "k_connectivity": k_conn,
        "composite_score": composite,
        "avg_degree": float(np.mean([d for _, d in G.degree()])),
        "density": float(nx.density(G)),
    }


def simulate_cascade_failure(G, failure_ratio: float = 0.1,
                             n_steps: int = 10, seed: int = 42,
                             self_recovery: float = 0.0,
                             directed_weighted: bool = False):
    """
    级联失效模拟：初始随机失效 failure_ratio 节点，后续每步按负载（介数中心性）超阈值传播。
    负载 = 介数中心性，若节点负载超过当前网络负载均值的 threshold 倍则失效。

    Args:
        self_recovery: 节点自恢复率 ρ ∈ [0,1]（对标水下无人机自恢复模型）。
            每步按 ρ 比例恢复已失效节点（连接其存活邻居），模拟节点/链路自愈。
        directed_weighted: 有向加权负载模式（用入度加权负载近似有向传播）。

    Returns:
        failure_ratios: (n_steps,) 每步累计失效比例（考虑恢复后净失效）
        failed_sets: list[set] 每步失效节点集合（当前净失效）
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    G_cur = G.copy()
    failed = set()

    # 初始随机失效
    n_init = max(1, int(n * failure_ratio))
    init_failed = set(rng.choice(list(G_cur.nodes()), size=min(n_init, n), replace=False))
    G_cur.remove_nodes_from(init_failed)
    failed |= init_failed

    failure_ratios = [len(failed) / n]
    failed_sets = [failed.copy()]

    threshold = 1.2  # 负载超均值 1.2 倍视为过载
    # 静态负载复用缓存介数（compute_all_metrics 已算一次，此处零成本）
    static_load = G.graph.get("betweenness")
    if static_load is None:
        try:
            static_load = nx.betweenness_centrality(G, k=(150 if n > 300 else None),
                                                     seed=seed)
        except Exception:
            static_load = dict(nx.degree(G))
    if directed_weighted:
        # 有向加权近似：负载 = 介数 × 入度权重（弱通信场景入向流量主导）
        in_deg = dict(G.in_degree()) if G.is_directed() else dict(G.degree())
        static_load = {node: static_load.get(node, 0.0) * (1 + in_deg.get(node, 0) / max(n, 1))
                       for node in G.nodes()}

    for step in range(1, n_steps):
        # 自恢复：每步按 ρ 恢复部分已失效节点
        if self_recovery > 0 and failed:
            n_rec = max(1, int(len(failed) * self_recovery))
            # 优先恢复度高的失效节点（重要节点优先自愈）
            rec_candidates = sorted(failed, key=lambda u: G.degree(u), reverse=True)[:n_rec]
            for u in rec_candidates:
                # 恢复节点及其到存活邻居的边
                G_cur.add_node(u)
                for v in G.neighbors(u):
                    if v in G_cur.nodes():
                        G_cur.add_edge(u, v)
                failed.discard(u)

        if G_cur.number_of_nodes() <= 1:
            failure_ratios.append(len(failed) / n)
            failed_sets.append(failed.copy())
            break
        # 基于当前存活节点的静态负载判断过载
        cur_load = {node: static_load.get(node, 0.0) for node in G_cur.nodes()}
        mean_load = np.mean(list(cur_load.values())) + 1e-9
        overloaded = {node for node, l in cur_load.items() if l > threshold * mean_load}
        G_cur.remove_nodes_from(overloaded)
        failed |= overloaded
        failure_ratios.append(len(failed) / n)
        failed_sets.append(failed.copy())
        if len(overloaded) == 0 and not (self_recovery > 0 and failed):
            # 无新失效且无恢复空间，剩余步骤持平
            for _ in range(step + 1, n_steps):
                failure_ratios.append(len(failed) / n)
                failed_sets.append(failed.copy())
            break

    return np.array(failure_ratios), failed_sets


def compute_node_load_overload(G, theta: float = 1.2, seed: int = 42):
    """计算节点负载（归一化介数）与过载指示器，供 PICG-Net 物理约束监督。

    Returns:
        node_load: (N,) 归一化介数负载 [0,1]（供 LoadHead 的 L_load 监督）
        overload: (N,) 过载程度 load/(θ·mean)（供 LR-MP 物理感知注意力）
        overload_mask: (N,) 过载指示器 {0,1}（load > θ·mean，供 L_phys 监督）
    """
    n = G.number_of_nodes()
    static_load = G.graph.get("betweenness")
    if static_load is None:
        try:
            static_load = nx.betweenness_centrality(G, k=(150 if n > 300 else None), seed=seed)
        except Exception:
            static_load = dict(nx.degree(G))
    node_load = np.array([static_load.get(i, 0.0) for i in range(n)], dtype=np.float32)
    max_load = node_load.max() + 1e-9
    node_load = node_load / max_load                      # 归一化 [0,1]
    mean_load = node_load.mean() + 1e-9
    overload = node_load / (theta * mean_load)             # 过载程度（>1 表示过载）
    overload_mask = (overload > 1.0).astype(np.float32)    # 过载指示器
    return node_load, overload, overload_mask
