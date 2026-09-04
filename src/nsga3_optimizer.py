"""
nsga3_optimizer.py - NSGA-III 韧性增强器核心模块
基于预测的韧性评分和级联风险，寻找最优拓扑加固方案（复用程洋录用论文 NSGA-III 框架）。

编码策略（论文 2.3 节）：实数编码 + 阈值 σ 转二进制
决策变量:
  [0 : n_new_nodes*2]        新增节点坐标（实数 [0,1]，缩放至园区尺寸）
  [n_new_nodes*2 : ...]      冗余链路开关（实数 [0,1]，>0.5 视为 1）
目标函数（3 个）:
  1. 最大化 GCN 韧性评分          (取负 → 最小化)
  2. 最小化 LSTM 级联失效风险     (无模型时用级联仿真代理)
  3. 最小化拓扑改动成本           (新增节点数 + 新增链路数)
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch

from pymoo.core.problem import Problem


class ResilienceOptimizationProblem(Problem):
    """NSGA-III 韧性增强优化问题（3 目标）。"""

    def __init__(self, base_graph, base_features, gcn_predict, cascade_risk,
                 n_new_nodes: int = 8, n_redundant_links: int = 12,
                 comm_range: float = 150.0, area_size=(2000, 2000),
                 node_cost: float = 1.0, link_cost: float = 0.3,
                 extra_cost_weight: float = 0.02):
        """
        Args:
            base_graph: nx.Graph 当前园区拓扑（待优化）
            base_features: (N, F) 当前节点特征
            gcn_predict: callable(nx.Graph, features) -> resilience_score
            cascade_risk: callable(nx.Graph) -> risk_score ∈ [0,1]
            n_new_nodes: 可新增节点数（决策变量：坐标）
            n_redundant_links: 可新增冗余链路数（决策变量：二进制开关）
        """
        self.base_graph = base_graph
        self.base_features = base_features
        self.gcn_predict = gcn_predict
        self.cascade_risk = cascade_risk
        self.n_new_nodes = n_new_nodes
        self.n_redundant_links = n_redundant_links
        self.comm_range = comm_range
        self.area_size = area_size
        self.node_cost = node_cost
        self.link_cost = link_cost
        self.extra_cost_weight = extra_cost_weight

        n_vars = n_new_nodes * 2 + n_redundant_links
        super().__init__(n_var=n_vars, n_obj=3, xl=0.0, xu=1.0)

        # 预计算基础图中可添加冗余链路的候选节点对（未直接相连的近邻）
        self._candidate_pairs = self._find_link_candidates()

    def _find_link_candidates(self):
        """选择距离 < 2*comm_range 且当前未直连的节点对作为冗余链路候选。"""
        G = self.base_graph
        pos = nx.get_node_attributes(G, "pos")
        nodes = list(G.nodes())[:self.n_redundant_links * 4]
        pairs = []
        nodes_list = list(G.nodes())
        for i, u in enumerate(nodes_list):
            for v in nodes_list[i + 1:]:
                if u in pos and v in pos:
                    d = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
                    if d < 2 * self.comm_range and not G.has_edge(u, v):
                        pairs.append((u, v))
                if len(pairs) >= self.n_redundant_links * 4:
                    break
            if len(pairs) >= self.n_redundant_links * 4:
                break
        if len(pairs) < self.n_redundant_links:
            # 不足时用任意节点对补足
            nodes_list = list(G.nodes())
            while len(pairs) < self.n_redundant_links and len(nodes_list) >= 2:
                u, v = nodes_list[0], nodes_list[-1]
                if not G.has_edge(u, v):
                    pairs.append((u, v))
                nodes_list.pop()
        return pairs[:self.n_redundant_links]

    def _decode(self, x):
        """将实数编码解码为: 新增节点坐标 + 新增链路开关。"""
        coords = x[:self.n_new_nodes * 2].reshape(-1, 2)
        new_pos = coords * np.array(self.area_size)
        link_switches = x[self.n_new_nodes * 2:]
        return new_pos, (link_switches > 0.5).astype(int)

    def _build_topology(self, new_pos, link_switches):
        """根据决策变量构建新拓扑图。"""
        G = self.base_graph.copy()
        pos = {n: np.array(p) for n, p in nx.get_node_attributes(G, "pos").items()}

        # 新增节点
        new_nodes = list(range(max(G.nodes()) + 1, max(G.nodes()) + 1 + self.n_new_nodes))
        for i, nid in enumerate(new_nodes):
            p = new_pos[i]
            pos[nid] = p
            G.add_node(nid, pos=(float(p[0]), float(p[1])))

        # 新增节点按通信半径连边
        all_pos = {**pos}
        for i, u in enumerate(new_nodes):
            for v, pv in all_pos.items():
                if v == u:
                    continue
                if np.linalg.norm(pos[u] - pv) <= self.comm_range:
                    G.add_edge(u, v)

        # 冗余链路开关
        for idx, sw in enumerate(link_switches):
            if sw == 1 and idx < len(self._candidate_pairs):
                u, v = self._candidate_pairs[idx]
                G.add_edge(u, v)

        # 记录新增信息
        G.graph["n_added_nodes"] = len(new_nodes)
        G.graph["n_added_links"] = int(link_switches.sum())
        return G, new_nodes

    def _build_features(self, G, new_nodes):
        """为新增节点构造特征（基于坐标和度），与既有特征对齐。"""
        F = self.base_features.shape[1]
        feats = np.zeros((len(G.nodes()), F), dtype=np.float32)
        base_n = self.base_features.shape[0]
        feats[:base_n] = self.base_features

        pos = nx.get_node_attributes(G, "pos")
        for i, nid in enumerate(new_nodes):
            p = pos[nid]
            feats[base_n + i, 1] = 1.0            # 类型: 监测点
            feats[base_n + i, 3:5] = np.array(p) / np.array(self.area_size)
            feats[base_n + i, 5] = 0.5            # 距离占位
            feats[base_n + i, 7] = G.degree(nid) / max(G.number_of_nodes() - 1, 1)
            feats[base_n + i, 31] = 0.5           # 风险占位
            # 注意力特征列 [32:35]：用度/连通性代理填充（新节点无历史拓扑）
            if F >= 35:
                feats[base_n + i, 32] = G.degree(nid) / max(G.number_of_nodes(), 1)
                feats[base_n + i, 33] = 0.5
                feats[base_n + i, 34] = 0.5
        return feats

    def _evaluate(self, X, out, *args, **kwargs):
        """pymoo 评估函数。
        加速：若 gcn_predict 支持 batch（callable 接受 Data 列表），则整代一次前向；
        否则逐个体评估（回退模式）。
        """
        n_pop = X.shape[0]
        F1 = np.zeros(n_pop)
        F2 = np.zeros(n_pop)
        F3 = np.zeros(n_pop)

        # 解码全部个体
        topologies = []
        costs = np.zeros(n_pop)
        for i in range(n_pop):
            new_pos, link_switches = self._decode(X[i])
            G_new, new_nodes = self._build_topology(new_pos, link_switches)
            feats_new = self._build_features(G_new, new_nodes)
            topologies.append((G_new, feats_new))
            costs[i] = (G_new.graph["n_added_nodes"] * self.node_cost +
                        G_new.graph["n_added_links"] * self.link_cost)

        # 目标 1: 韧性（批量前向）
        use_batch = getattr(self.gcn_predict, "batch_supported", False)
        if use_batch:
            try:
                res = self.gcn_predict([(G, f) for G, f in topologies])
                for i, r in enumerate(res):
                    F1[i] = -float(r)
            except Exception:
                use_batch = False
        if not use_batch:
            for i, (G_new, feats_new) in enumerate(topologies):
                try:
                    F1[i] = -float(self.gcn_predict(G_new, feats_new))
                except Exception:
                    F1[i] = -0.5

        # 目标 2: 级联风险（批量支持则批量，否则逐图）
        risk_batch = getattr(self.cascade_risk, "batch_supported", False)
        if risk_batch:
            try:
                risks = self.cascade_risk([G for G, _ in topologies])
                for i, r in enumerate(risks):
                    F2[i] = float(r)
            except Exception:
                risk_batch = False
        if not risk_batch:
            for i, (G_new, _) in enumerate(topologies):
                try:
                    F2[i] = float(self.cascade_risk(G_new))
                except Exception:
                    F2[i] = 0.5

        F3 = costs
        out["F"] = np.column_stack([F1, F2, F3])
