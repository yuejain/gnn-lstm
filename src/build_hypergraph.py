"""
build_hypergraph.py - 超图构建工具（PICG-Net 数据侧）

超边 = 一组设施/节点对同一目标节点的联合依赖（高阶 AND 关系）。
在通用拓扑上提供多种超边生成策略，供消融实验（超图 vs 异构图 vs 多层图）使用；
校园场景（P2）再用"设施 -> 楼宇"的真实依赖定义超边。

输出 CSR 格式（供 HyperLRMP 消费）：
    hyperedge_to_node: (E_hyper,) 每条超边成员节点 id 平铺
    hyperedge_ptr: (M+1,)         每条超边的成员区间 [ptr[e], ptr[e+1])
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch


def build_hypergraph_from_communities(G, min_size: int = 3, max_size: int = 50,
                                      seed: int = 42) -> tuple[list[int], list[int], int]:
    """用 greedy_modularity 社区检测生成超边（每个社区 = 一条超边）。"""
    communities = nx.algorithms.community.greedy_modularity_communities(G, seed=seed)
    hyperedges = []
    for comm in communities:
        nodes = sorted(comm)
        if min_size <= len(nodes) <= max_size:
            hyperedges.append(nodes)
    return _pack_hyperedges(hyperedges, len(G))


def build_hypergraph_from_features(G, feature_matrix, n_clusters: int = 16,
                                   seed: int = 42) -> tuple[list[int], list[int], int]:
    """按节点特征聚类生成超边（同质节点联合支撑同一功能 = 一条超边）。"""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(feature_matrix)
    hyperedges = []
    for c in range(n_clusters):
        nodes = [i for i, lb in enumerate(labels) if lb == c]
        if len(nodes) >= 2:
            hyperedges.append(sorted(nodes))
    return _pack_hyperedges(hyperedges, feature_matrix.shape[0])


def build_hypergraph_kclique(G, k: int = 3, cap: int = 200) -> tuple[list[int], list[int], int]:
    """用 k-clique 生成超边（完全耦合的节点组 = 联合依赖）。"""
    from networkx.algorithms.clique import enumerate_all_cliques
    hyperedges = []
    for clique in enumerate_all_cliques(G):
        if len(clique) == k:
            hyperedges.append(sorted(clique))
            if len(hyperedges) >= cap:
                break
    return _pack_hyperedges(hyperedges, len(G))


def build_campus_hyperedges(building_to_facilities: dict[int, list[int]],
                            n_nodes: int) -> tuple[list[int], list[int], int]:
    """校园场景超边：building -> 依赖的设施节点列表（多设施联合支撑同一楼宇）。"""
    hyperedges = [sorted(fac) for fac in building_to_facilities.values() if len(fac) >= 2]
    return _pack_hyperedges(hyperedges, n_nodes)


def _pack_hyperedges(hyperedges: list[list[int]], n_nodes: int):
    """把超边列表打包为 CSR 格式。返回 (hyperedge_to_node, hyperedge_ptr, num_hyperedges)。"""
    hyperedge_to_node: list[int] = []
    ptr: list[int] = [0]
    for e in hyperedges:
        hyperedge_to_node.extend(e)
        ptr.append(len(hyperedge_to_node))
    if not hyperedges:
        return [], [], 0
    return hyperedge_to_node, ptr, len(hyperedges)


def to_tensors(hyperedge_to_node, hyperedge_ptr, device=None):
    """转 torch tensor（供 HyperLRMP 使用）。"""
    het = torch.tensor(hyperedge_to_node, dtype=torch.long, device=device)
    hptr = torch.tensor(hyperedge_ptr, dtype=torch.long, device=device)
    return het, hptr


def build_hypergraph(G, strategy: str = "community", features=None, seed: int = 42,
                     **kwargs):
    """统一入口：按策略生成超图 CSR 结构。

    Returns:
        dict: {hyperedge_to_node, hyperedge_ptr, num_hyperedges, strategy}
    """
    n = len(G)
    if strategy == "community":
        hn, hp, m = build_hypergraph_from_communities(G, seed=seed, **kwargs)
    elif strategy == "features" and features is not None:
        hn, hp, m = build_hypergraph_from_features(G, features, seed=seed, **kwargs)
    elif strategy == "kclique":
        hn, hp, m = build_hypergraph_kclique(G, **kwargs)
    else:  # 无超图（消融：纯普通图 LR-MP）
        hn, hp, m = [], [], 0
    return {"hyperedge_to_node": hn, "hyperedge_ptr": hp,
            "num_hyperedges": m, "strategy": strategy}
