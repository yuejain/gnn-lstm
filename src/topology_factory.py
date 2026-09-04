"""
topology_factory.py - 多层拓扑数据工厂核心模块
复用程洋硕士论文 3.1.1 节"园区拓扑图构建与GCN特征嵌入"：
- 异构节点编码（32维混合表征：类型独热 + 归一化坐标 + 距泄漏源距离 + 设备密度 + 动态属性）
- 三类边构建（物理连接边 / 扩散路径边 / 风险关联边）
- 三种基础拓扑变体（随机 / 无标度 BA / 小世界 WS）
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from src.utils import set_seed


# ============================================================
# 1. 物理拓扑生成（复用论文 3.1.1 异构节点编码）
# ============================================================

def generate_sensor_positions(cfg, num_sensors: int, layout: str = "road"):
    """
    生成传感器节点坐标（支持道路锚点布局与纯随机布局）。

    layout="road"（默认）: 45% 危险源聚集 + 30% 道路锚点 + 25% 均匀覆盖
    layout="legacy":      沿用旧逻辑（危险源聚集 + 均匀覆盖）

    Returns:
        sensor_positions: (N, 2) 坐标
        hazard_pos: (H, 2) 危险源坐标
        sensor_to_hazard: (N,) 节点所属危险源索引（-1 表示非聚集）
        road_anchors: (M, 2) 道路锚点（road 布局；legacy 返回空数组）
    """
    from src.physical_layout import generate_road_network

    x_range, y_range = cfg["area_size"]
    n_hazard = cfg["num_hazard_sources"]
    cluster_r = cfg["hazard_cluster_radius"]
    rng = np.random.default_rng()

    # 1. 随机生成重大危险源位置（60% 靠近道路锚点 + 40% 均匀散布）
    hazard_pos = np.random.rand(n_hazard, 2) * [x_range, y_range]

    road_anchors = np.empty((0, 2))
    if layout == "road":
        _, road_anchors = generate_road_network(cfg["area_size"],
                                                n_roads=cfg.get("n_roads", 4), seed=42)
        if len(road_anchors) > 0:
            n_hazard_road = int(n_hazard * 0.6)
            idx = np.random.choice(len(road_anchors), min(n_hazard_road, len(road_anchors)),
                                   replace=False)
            for i, ai in enumerate(idx[:n_hazard]):
                if i < len(idx):
                    offset = np.random.randn(2) * 60.0
                    hazard_pos[i] = road_anchors[ai] + offset
                    hazard_pos[i] = np.clip(hazard_pos[i], [0, 0], [x_range, y_range])

    # 2. 为每个危险源分配传感器（聚集部署）
    sensors_per_hazard = num_sensors // n_hazard
    sensor_positions = []
    sensor_to_hazard = []

    for i, h_pos in enumerate(hazard_pos):
        for _ in range(sensors_per_hazard):
            offset = np.random.randn(2) * (cluster_r / 3.0)
            pos = h_pos + offset
            pos = np.clip(pos, [0, 0], [x_range, y_range])
            sensor_positions.append(pos)
            sensor_to_hazard.append(i)

    # 3. 剩余传感器：road 布局下 30% 道路锚点 + 70% 均匀覆盖
    remaining = num_sensors - len(sensor_positions)
    if remaining > 0:
        if layout == "road" and len(road_anchors) > 0:
            n_road = int(remaining * 0.5)  # 锚点部署一半
            n_road = min(n_road, remaining)
            for _ in range(n_road):
                a = road_anchors[np.random.randint(len(road_anchors))]
                pos = a + np.random.randn(2) * 40.0
                pos = np.clip(pos, [0, 0], [x_range, y_range])
                sensor_positions.append(pos)
                sensor_to_hazard.append(-1)
            remaining -= n_road
        random_pos = np.random.rand(remaining, 2) * [x_range, y_range]
        sensor_positions.extend(random_pos)
        sensor_to_hazard.extend([-1] * remaining)

    return np.array(sensor_positions), hazard_pos, np.array(sensor_to_hazard), road_anchors


def build_geo_graph(sensor_positions, comm_range: float, max_range_mult: float = 3.0,
                    obstacles=None, atten_prob: float = 0.3):
    """
    基于通信半径构建传感器网络拓扑图（带连通性自增保证 + LOS 遮挡过滤）。

    若提供 obstacles（AABB 数组），被 LOS 遮挡的边按 atten_prob 概率保留
    （模拟反射/多径），否则断开（物理遮挡约束，论文 ≤15% 遮挡率）。
    """
    from src.physical_layout import los_blocked

    n = len(sensor_positions)
    tree = cKDTree(sensor_positions)
    current_range = float(comm_range)

    while current_range <= comm_range * max_range_mult:
        pairs = tree.query_pairs(r=current_range)
        # LOS 过滤
        if obstacles is not None and len(obstacles) > 0 and pairs:
            p = np.array([sensor_positions[i] for i, _ in pairs])
            q = np.array([sensor_positions[j] for _, j in pairs])
            blocked = los_blocked(p, q, obstacles)
            keep = ~blocked
            # 被遮挡但距离较近的边按 atten_prob 保留
            dists = np.linalg.norm(p - q, axis=1)
            near_keep = blocked & (dists <= comm_range * 1.15)
            near_keep = near_keep & (np.random.rand(len(pairs)) < atten_prob)
            keep = keep | near_keep
            pairs = [pair for pair, k in zip(pairs, keep) if k]
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(pairs)
        if nx.is_connected(G) and n > 1:
            break
        current_range *= 1.1

    # 兜底：若仍不连通，人工补边（连接各连通分量的最近节点对）
    if not (nx.is_connected(G) and n > 1):
        G = _ensure_connected(G, sensor_positions)

    for i, (x, y) in enumerate(sensor_positions):
        G.nodes[i]["pos"] = (float(x), float(y))
    G.graph["comm_range"] = current_range
    return G


# ============================================================
# 2. 八种拓扑变体
# ============================================================

def _sample_powerlaw_degrees(n, gamma: float, k_min: int = 2, k_max: int = 30,
                             rng=None) -> np.ndarray:
    """采样幂律度序列（保证 Σdeg 为偶数）。"""
    if rng is None:
        rng = np.random.default_rng()
    # 逆 CDF 采样：k = k_min * (1 - u)^(-1/(gamma-1))
    u = rng.random(n)
    k = k_min * (1 - u) ** (-1.0 / (gamma - 1))
    k = np.clip(np.round(k), k_min, k_max).astype(int)
    if k.sum() % 2 != 0:
        k[0] += 1
    return k


def _spatial_configuration_model(positions, comm_range: float, gamma: float = 2.5,
                                 k_min: int = 2, k_max: int = 30,
                                 p_long: float = 0.02, rng=None) -> nx.Graph:
    """
    空间约束配置模型：幂律度序列 + 距离加权连边。
    候选边必须在通信半径内（低概率 p_long 放宽为长边）。
    """
    n = len(positions)
    if rng is None:
        rng = np.random.default_rng()
    tree = cKDTree(positions)
    in_range = {i: set(tree.query_ball_point(positions[i], r=comm_range)) - {i}
                for i in range(n)}
    degs = _sample_powerlaw_degrees(n, gamma, k_min, k_max, rng)
    order = np.argsort(-degs)
    adj = {i: set() for i in range(n)}

    for i in order:
        tries = 0
        while len(adj[i]) < degs[i] and tries < 200:
            tries += 1
            cands = [j for j in in_range[i] if j not in adj[i] and len(adj[j]) < degs[j]]
            if not cands and rng.random() < p_long:
                cands = [j for j in range(n) if j not in adj[i] and len(adj[j]) < degs[j]]
            if not cands:
                break
            dists = np.linalg.norm(positions[np.array(cands)] - positions[i], axis=1)
            w = (np.array([degs[j] - len(adj[j]) + 1 for j in cands])) / (dists + 1.0)
            w = w / w.sum()
            j = cands[int(rng.choice(len(cands), p=w))]
            adj[i].add(j)
            adj[j].add(i)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in adj[i]:
            if i < j:
                G.add_edge(i, j)
    return G


def _geo_small_world(positions, comm_range: float, k: int = 6, p: float = 0.1,
                     ring_mult: float = 1.5, rng=None) -> nx.Graph:
    """空间约束小世界：角度排序成环 + 重连限定通信半径。"""
    n = len(positions)
    if rng is None:
        rng = np.random.default_rng()
    center = positions.mean(axis=0)
    angles = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
    ring = np.argsort(angles)
    pos_of = {node: idx for idx, node in enumerate(ring)}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    k = min(k, n - 1)
    for i in range(n):
        for d in range(1, k // 2 + 1):
            j = ring[(pos_of[i] + d) % n]
            if np.linalg.norm(positions[i] - positions[j]) <= comm_range * ring_mult:
                G.add_edge(i, j)
    # 重连（限定通信半径）
    edges = list(G.edges())
    for u, v in edges:
        if rng.random() < p:
            cands = [j for j in range(n) if j != u and not G.has_edge(u, j)
                     and np.linalg.norm(positions[u] - positions[j]) <= comm_range]
            if cands:
                G.remove_edge(u, v)
                G.add_edge(u, int(cands[rng.integers(len(cands))]))
    return G


def build_grid_graph(positions, comm_range, rows=None) -> nx.Graph:
    """网格拓扑：按 x 排序分列，列内链式 + 列间交错桥接。"""
    n = len(positions)
    if rows is None:
        rows = max(5, int(np.sqrt(n)))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    order = np.argsort(positions[:, 0])
    col = np.zeros(n, dtype=int)
    col[order] = np.arange(n) // rows
    n_cols = col.max() + 1
    for c in range(n_cols):
        nodes = np.where(col == c)[0]
        sorted_by_y = sorted(nodes, key=lambda i: positions[i, 1])
        for a, b in zip(sorted_by_y, sorted_by_y[1:]):
            G.add_edge(a, b)
    for c in range(n_cols - 1):
        left = np.where(col == c)[0]
        right = np.where(col == c + 1)[0]
        left_s = sorted(left, key=lambda i: positions[i, 1])
        right_s = sorted(right, key=lambda i: positions[i, 1])
        k = max(min(len(left_s), len(right_s), rows), 5)
        for j in range(k):
            li = left_s[int(j * len(left_s) / k)]
            ri = right_s[int(j * len(right_s) / k)]
            if np.linalg.norm(positions[li] - positions[ri]) <= comm_range * 1.5:
                G.add_edge(li, ri)
    return G


def build_hierarchical_graph(positions, comm_range, n_zones: int = 6,
                             inter_ratio: float = 0.02, rng=None) -> nx.Graph:
    """分层拓扑：k-means 分区 + 区内地理稠密 + 区间稀疏桥接。"""
    n = len(positions)
    if rng is None:
        rng = np.random.default_rng()
    from scipy.cluster.vq import kmeans2
    n_zones = min(n_zones, max(2, n // 10))
    centroids, labels = kmeans2(positions, n_zones, seed=42)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    tree = cKDTree(positions)
    for z in range(n_zones):
        zone_nodes = np.where(labels == z)[0]
        if len(zone_nodes) < 2:
            continue
        sub_tree = cKDTree(positions[zone_nodes])
        pairs = sub_tree.query_pairs(r=comm_range * 0.9)
        for a, b in pairs:
            G.add_edge(zone_nodes[a], zone_nodes[b])
        # 区内连通（链式兜底）
        sorted_nodes = sorted(zone_nodes, key=lambda i: positions[i][1])
        for a, b in zip(sorted_nodes, sorted_nodes[1:]):
            G.add_edge(a, b)
    # 区间桥接（in-range 的跨区对 + 随机 2%）
    pairs = tree.query_pairs(r=comm_range)
    for a, b in pairs:
        if labels[a] != labels[b] and rng.random() < inter_ratio * 10:
            G.add_edge(a, b)
    # 随机远距边
    n_extra = max(1, int(n * inter_ratio))
    for _ in range(n_extra):
        a, b = rng.choice(n, 2, replace=False)
        G.add_edge(int(a), int(b))
    return G


def build_sbm_graph(positions, comm_range, n_blocks: int = 4,
                    p_in: float = 0.15, p_out: float = 0.008, rng=None) -> nx.Graph:
    """SBM 分区：块内地理稠密 + 块间稀疏骨干。"""
    n = len(positions)
    if rng is None:
        rng = np.random.default_rng()
    from scipy.cluster.vq import kmeans2
    n_blocks = min(n_blocks, max(2, n // 50))
    centroids, labels = kmeans2(positions, n_blocks, seed=42)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # 块内：地理图（半径缩小）
    for z in range(n_blocks):
        zone_nodes = np.where(labels == z)[0]
        if len(zone_nodes) < 2:
            continue
        sub_tree = cKDTree(positions[zone_nodes])
        pairs = sub_tree.query_pairs(r=comm_range * 0.85)
        for a, b in pairs:
            if rng.random() < p_in * 10:
                G.add_edge(zone_nodes[a], zone_nodes[b])
        # 块内链式连通兜底
        sorted_nodes = sorted(zone_nodes, key=lambda i: positions[i][1])
        for a, b in zip(sorted_nodes, sorted_nodes[1:]):
            G.add_edge(a, b)
    # 块间：in-range 跨块 + 质心间骨干
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=comm_range)
    for a, b in pairs:
        if labels[a] != labels[b] and rng.random() < p_out * 100:
            G.add_edge(a, b)
    return G


def build_ring_graph(positions, comm_range, n_backbone: int = 12,
                     backbone_mult: float = 2.0, rng=None) -> nx.Graph:
    """环形骨干 + 支线接入。"""
    n = len(positions)
    if rng is None:
        rng = np.random.default_rng()
    n_backbone = min(n_backbone, n // 5)
    center = positions.mean(axis=0)
    angles = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
    order = np.argsort(angles)
    # 均匀取骨干节点
    backbone_idx = [order[int(i * n / n_backbone)] for i in range(n_backbone)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # 骨干环
    for i in range(n_backbone):
        G.add_edge(backbone_idx[i], backbone_idx[(i + 1) % n_backbone])
    # 支线：其余节点连最近骨干
    backbone_pos = positions[np.array(backbone_idx)]
    for i in range(n):
        if i in backbone_idx:
            continue
        d = np.linalg.norm(positions[i] - backbone_pos, axis=1)
        j = backbone_idx[int(d.argmin())]
        if d.min() <= comm_range * backbone_mult:
            G.add_edge(i, j)
    # 相邻支线短边
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=comm_range)
    for a, b in pairs:
        if a not in backbone_idx and b not in backbone_idx and rng.random() < 0.3:
            G.add_edge(a, b)
    return G


def build_tree_graph(positions, comm_range, extra_leaf_edge: bool = True,
                     rng=None) -> nx.Graph:
    """支线树状：MST + 叶子冗余边。"""
    n = len(positions)
    if rng is None:
        rng = np.random.default_rng()
    root = int(np.linalg.norm(positions - positions.mean(axis=0), axis=1).argmin())
    # Prim MST（仅允许 ≤ comm_range；不足放宽 1.5x）
    G = nx.Graph()
    G.add_nodes_from(range(n))
    in_tree = {root}
    tree = cKDTree(positions)
    for _ in range(n - 1):
        best_d, best_i, best_j = float("inf"), None, None
        for i in in_tree:
            neigh = tree.query_ball_point(positions[i], r=comm_range * 1.5)
            for j in neigh:
                if j not in in_tree:
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < best_d:
                        best_d, best_i, best_j = d, i, j
        if best_i is None:
            break
        G.add_edge(best_i, best_j)
        in_tree.add(best_j)
    # 叶子冗余边
    if extra_leaf_edge:
        leaves = [i for i in G.nodes() if G.degree(i) == 1 and i != root]
        pairs = tree.query_pairs(r=comm_range)
        for a, b in pairs:
            if (a in leaves or b in leaves) and rng.random() < 0.2:
                G.add_edge(a, b)
    return G


def _ensure_connected(G, positions):
    """兜底连通：不连通时用 MST 边缝合。"""
    if nx.is_connected(G):
        return G
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return G
    tree = cKDTree(positions)
    for ci in range(len(comps) - 1):
        best_d, edge = float("inf"), None
        for a in comps[ci]:
            for b in comps[ci + 1]:
                d = np.linalg.norm(positions[a] - positions[b])
                if d < best_d:
                    best_d, edge = d, (a, b)
        if edge:
            G.add_edge(*edge)
    return G


def generate_variant(positions, comm_range: float, variant: str,
                     vparams: dict | None = None, **kwargs):
    """
    生成指定变体的拓扑图（8 变体）。

    - random:         基于通信半径的地理随机图（LOS 物理约束）
    - geo_scale_free: 空间约束配置模型（幂律度序列 + 距离加权连边）
    - geo_small_world:空间约束 WS（环形格 + 重连限定通信半径）
    - grid:           网格拓扑（列链式 + 交错桥接）
    - hierarchical:   分层拓扑（k-means 分区 + 骨干桥接）
    - sbm:            随机块模型（块内地理稠密 + 块间稀疏）
    - ring:           环形骨干 + 支线接入
    - tree:           MST 支线树 + 叶子冗余
    """
    vparams = vparams or {}
    n = len(positions)
    rng = np.random.default_rng(kwargs.get("seed", 42))

    if variant == "random":
        G = build_geo_graph(positions, comm_range,
                            max_range_mult=vparams.get("max_range_mult", 3.0),
                            obstacles=vparams.get("obstacles"),
                            atten_prob=vparams.get("atten_prob", 0.3))
    elif variant == "geo_scale_free":
        G = _spatial_configuration_model(
            positions, comm_range,
            gamma=vparams.get("gamma", 2.5),
            k_min=vparams.get("k_min", 2),
            k_max=vparams.get("k_max", int(comm_range / 30)),
            p_long=vparams.get("p_long", 0.02), rng=rng)
    elif variant == "geo_small_world":
        G = _geo_small_world(
            positions, comm_range,
            k=vparams.get("k", 6),
            p=vparams.get("p", 0.1),
            ring_mult=vparams.get("ring_mult", 1.5), rng=rng)
    elif variant == "grid":
        G = build_grid_graph(positions, comm_range, rows=vparams.get("rows"))
    elif variant == "hierarchical":
        G = build_hierarchical_graph(
            positions, comm_range,
            n_zones=vparams.get("n_zones", 6),
            inter_ratio=vparams.get("inter_ratio", 0.02), rng=rng)
    elif variant == "sbm":
        G = build_sbm_graph(
            positions, comm_range,
            n_blocks=vparams.get("n_blocks", 4),
            p_in=vparams.get("p_in", 0.15),
            p_out=vparams.get("p_out", 0.008), rng=rng)
    elif variant == "ring":
        G = build_ring_graph(
            positions, comm_range,
            n_backbone=vparams.get("n_backbone", 12),
            backbone_mult=vparams.get("backbone_mult", 2.0), rng=rng)
    elif variant == "tree":
        G = build_tree_graph(positions, comm_range,
                             extra_leaf_edge=vparams.get("extra_leaf_edge", True), rng=rng)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # 统一写入坐标（供特征计算）
    for i in range(n):
        G.nodes[i]["pos"] = (float(positions[i][0]), float(positions[i][1]))
    G = _ensure_connected(G, positions)
    G.graph["variant"] = variant
    return G


# ============================================================
# 3. 异构节点特征编码（论文 3.1.1：32 维混合表征）
# ============================================================

def compute_node_features(G, hazard_pos, sensor_to_hazard,
                          wsn_features: np.ndarray | None = None,
                          target_dim: int = 32) -> np.ndarray:
    """
    计算异构节点特征（32 维混合表征，复用论文 3.1.1 节编码规则）：

    静态属性（13 维）:
      [0:3]   节点类型独热编码（泄漏源=001, 监测点=010, 路径节点=100）
      [3:5]   空间坐标归一化 (x, y) ∈ [0,1]
      [5]     与最近危险源距离 d_i = ||x_i - x_source|| / d_max
      [6]     周边设备密度（半径 20m 内设备数量占比）
      [7:11]  图拓扑特征（度、聚类系数、介数中心性、接近中心性）
      [11:13] 电池/能量水平、链路质量（WSN 数据集成时填充，否则用图拓扑代理）

    动态属性（19 维）:
      [13:23] 10 个时间步的时序浓度序列（Z-score 标准化）
      [23]    风速 (m/s)
      [24]    温度梯度 ∇T
      [25]    湍流强度 I_u
      [26:31] WSN 网络性能特征（距离 CH / 能耗 / 数据发送 / 丢包率等）
      [31]    未来风险预估值（LSTM 输出占位）

    Args:
        wsn_features: (N, k) WSN 数据集提供的节点特征矩阵（可缺省）

    Returns:
        (N, target_dim) 特征矩阵
    """
    n = G.number_of_nodes()
    feat = np.zeros((n, target_dim), dtype=np.float32)

    # --- 节点类型独热编码 ---
    # 泄漏源节点 = 与危险源重合的传感器（sensor_to_hazard >= 0 且距离 < 聚集半径）
    for i in range(n):
        if sensor_to_hazard is not None and i < len(sensor_to_hazard) and sensor_to_hazard[i] >= 0:
            feat[i, 0] = 1.0   # 泄漏源
        else:
            feat[i, 1] = 1.0   # 监测点

    # --- 空间坐标归一化 ---
    pos_arr = np.array([G.nodes[i].get("pos", (0.0, 0.0)) for i in range(n)])
    max_dim = np.max(pos_arr, axis=0) - np.min(pos_arr, axis=0)
    max_dim[max_dim == 0] = 1.0
    feat[:, 3:5] = (pos_arr - np.min(pos_arr, axis=0)) / max_dim

    # --- 与危险源距离 ---
    if hazard_pos is not None and len(hazard_pos) > 0:
        d_max = np.max(np.linalg.norm(pos_arr - pos_arr[0], axis=1)) + 1e-9
        d_to_hazard = np.min(np.linalg.norm(pos_arr[:, None, :] - hazard_pos[None, :, :], axis=2), axis=1)
        feat[:, 5] = d_to_hazard / d_max

    # --- 周边设备密度（半径 20m 内） ---
    if len(pos_arr) > 1:
        tree = cKDTree(pos_arr)
        for i in range(n):
            neighbors = tree.query_ball_point(pos_arr[i], r=20.0)
            feat[i, 6] = (len(neighbors) - 1) / max(n - 1, 1)

    # --- 图拓扑特征（度/聚类/介数/接近中心性，一次性批量计算） ---
    degrees = np.array([d for _, d in G.degree()])
    if n > 1:
        feat[:, 7] = degrees / max(n - 1, 1)
        # 聚类系数批量一次调用（nx.clustering(G) 内部一次遍历，避免 n 次单独调用）
        try:
            clust = nx.clustering(G)
            feat[:, 8] = np.array([clust[i] for i in range(n)])
        except Exception:
            feat[:, 8] = 0.0
    else:
        feat[:, 7] = degrees

    # 中心性（介数/接近）——优先复用图缓存（数据工厂已算过介数）
    try:
        cached_bc = G.graph.get("betweenness")
        if cached_bc is not None:
            bc = cached_bc
        else:
            bc = nx.betweenness_centrality(G, k=(150 if n > 300 else None))
        # closeness 大图用 k 采样近似（O(N²) → O(k·E)，1200 节点 6s → 0.6s）
        if n > 300:
            k_close = 200
            cc = nx.closeness_centrality(G, k=k_close,
                                         seed=42 if "seed" not in globals() else globals().get("seed", 42))
        else:
            cc = nx.closeness_centrality(G)
        feat[:, 9] = np.array([bc[i] for i in range(n)])
        feat[:, 10] = np.array([cc[i] for i in range(n)])
    except Exception:
        pass

    # --- WSN 特征集成 ---
    if wsn_features is not None:
        k = min(wsn_features.shape[1], 5)
        feat[:, 11:11 + k] = wsn_features[:, :k]
        feat[:, 26:26 + k] = wsn_features[:, :k]

    # --- 动态属性：时序浓度序列（Z-score） ---
    # 简化模型：基于高斯烟羽分布生成 10 时间步浓度（作为 CFD-GAN 场景库的代理）
    conc_series = _simulate_concentration_series(pos_arr, hazard_pos, n_steps=10)
    for t in range(10):
        s = conc_series[:, t]
        mu, std = np.mean(s), np.std(s) + 1e-9
        feat[:, 13 + t] = (s - mu) / std

    # --- 气象参数（默认值，案例验证时注入真实值） ---
    feat[:, 23] = 2.5    # 风速 (m/s)
    feat[:, 24] = 0.05   # 温度梯度
    feat[:, 25] = 0.15   # 湍流强度

    # --- 未来风险预估值（占位，LSTM 阶段填充） ---
    feat[:, 31] = 0.5

    # --- 节点级注意力特征增强（[32:35]，提升 GAT 回归精度） ---
    # 基于图拓扑的注意力代理特征：PageRank / 特征向量中心性 / 局部聚类系数
    if target_dim >= 35 and n > 1:
        try:
            pr = nx.pagerank(G, alpha=0.85)
            pr_arr = np.array([pr[i] for i in range(n)])
        except Exception:
            pr_arr = np.zeros(n)
        if n <= 800:
            try:
                ev = nx.eigenvector_centrality_numpy(G)
                ev_arr = np.array([ev[i] for i in range(n)])
            except Exception:
                ev_arr = np.zeros(n)
        else:
            # 大图降级：用缓存介数/度作为特征向量中心性近似（避免 O(N³) 幂迭代）
            ev_arr = np.array([G.degree(i) for i in range(n)], dtype=float)
        # 聚类系数复用前面 batch 计算结果（若缓存）或重算
        try:
            if "clust" in locals() and clust is not None:
                cl_arr = np.array([clust[i] for i in range(n)])
            else:
                cl_arr = np.array(nx.clustering(G).values())
        except Exception:
            cl_arr = np.zeros(n)
        for arr in (pr_arr, ev_arr, cl_arr):
            std = arr.std() + 1e-9
            arr[:] = (arr - arr.mean()) / std  # 原地修改（arr= 只重绑变量不生效）
        feat[:, 32] = pr_arr
        feat[:, 33] = ev_arr
        feat[:, 34] = cl_arr

    return feat


def _simulate_concentration_series(pos_arr, hazard_pos, n_steps: int = 10):
    """基于高斯烟羽模型（论文 2.1.1 节）的简化浓度场模拟，作为时序动态属性。"""
    n = len(pos_arr)
    series = np.zeros((n, n_steps))
    if hazard_pos is None or len(hazard_pos) == 0:
        return series

    Q = 5.0          # 泄漏速率 (kg/s)
    u = 2.5          # 风速 (m/s)
    H = 0.5          # 有效高度
    wind_dir = np.array([1.0, 0.2])  # 偏东风
    wind_dir = wind_dir / np.linalg.norm(wind_dir)

    for t in range(n_steps):
        # 时间演化：泄漏强度先增后稳
        q_t = Q * min(1.0, t / 3.0 + 0.2)
        for src in hazard_pos[:5]:  # 取前 5 个危险源（计算效率）
            rel = pos_arr - src
            x_proj = rel @ wind_dir                     # 顺风距离
            y_perp = np.linalg.norm(rel - x_proj[:, None] * wind_dir, axis=1)  # 横向距离
            sig_y = 0.16 * np.maximum(x_proj, 0.01) * (1 + 0.0001 * x_proj) ** -0.5
            sig_z = 0.12 * np.maximum(x_proj, 0.01)
            mask = x_proj > 0
            contrib = np.zeros(n)
            contrib[mask] = (q_t / (2 * np.pi * u * sig_y[mask] * sig_z[mask])) * \
                np.exp(-y_perp[mask] ** 2 / (2 * sig_y[mask] ** 2)) * \
                np.exp(-H ** 2 / (2 * sig_z[mask] ** 2))
            series[:, t] += contrib

    # 归一化到 [0,1] 便于特征融合
    mx = np.max(series) + 1e-9
    if mx > 0:
        series = series / mx
    return series


# ============================================================
# 4. 数据集批量生成
# ============================================================

def generate_dataset(cfg, num_sensors: int = 600, variant: str = "random",
                     num_graphs: int = 30, wsn_features: np.ndarray | None = None,
                     seed: int = 42):
    """
    批量生成指定变体的拓扑图数据集。

    Returns:
        graphs: list[nx.Graph]，每张图带 pos 属性、graph['hazard_pos']、graph['sensor_to_hazard']
    """
    set_seed(seed)
    park_cfg = cfg["park"]
    topo_cfg = cfg["topology"]

    graphs = []
    for g in range(num_graphs):
        set_seed(seed + g * 1000 + (sum(ord(c) for c in variant) % 1000))
        positions, hazard_pos, sensor_to_hazard, _ = generate_sensor_positions(
            park_cfg, num_sensors, layout=park_cfg.get("layout", "road"))
        vparams = topo_cfg.get("variant_params", {}).get(variant, {})
        G = generate_variant(
            positions, park_cfg["communication_range"], variant,
            vparams=vparams, seed=seed + g)
        # 附加元数据
        G.graph["variant"] = variant
        G.graph["num_sensors"] = num_sensors
        G.graph["hazard_pos"] = hazard_pos
        G.graph["sensor_to_hazard"] = sensor_to_hazard
        G.graph["features"] = compute_node_features(
            G, hazard_pos, sensor_to_hazard, wsn_features,
            target_dim=cfg["data_factory"]["node_feature_dim"],
        )
        graphs.append(G)
    return graphs
