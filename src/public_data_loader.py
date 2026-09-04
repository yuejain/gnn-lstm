"""
public_data_loader.py - 公开数据集加载与标准化
加载三类公开真实网络数据集（仅供 OOD 泛化测试，绝不进训练集）：
1. SNAP email-Eu-core   — 组织部门分区网络（≈园区 SBM 分区）
2. Topology Zoo         — 276 个真实 ISP/骨干网拓扑（GraphML，带地理坐标）
3. CRAWDAD Haggle       — 7 个真实移动无线接触图（TSV）

标准化流程（与训练数据同口径）：
LCC → pos → 随机危险源 → compute_node_features(35维) → compute_all_metrics 打标签 → to_pyg
"""
from __future__ import annotations

import gzip
import glob
import os
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx

from src.topology_factory import compute_node_features, generate_sensor_positions
from src.resilience_labels import compute_all_metrics
from src.utils import set_seed


def load_email_eu(root="data/public") -> list[nx.Graph]:
    """加载 SNAP email-Eu-core（边列表 .txt.gz，0-based ID）。"""
    path = Path(root) / "email-Eu-core.txt.gz"
    if not path.exists():
        return []
    G = nx.Graph()
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            G.add_edge(int(a), int(b))
    # 取最大连通子图
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    G.graph["name"] = "email-Eu-core"
    return [G]


def load_topology_zoo(root="data/public", min_nodes: int = 15,
                      max_nodes: int = 800, limit: int | None = None) -> list[nx.Graph]:
    """加载 Topology Zoo GraphML 网络（过滤过小图，可选限制数量）。"""
    files = sorted(glob.glob(str(Path(root) / "graphml" / "*.graphml")))
    graphs = []
    for f in files:
        try:
            G = nx.read_graphml(f)
        except Exception:
            continue
        n = G.number_of_nodes()
        if n < min_nodes or n > max_nodes:
            continue
        # 清理无属性节点（GraphML 节点可能无 pos）
        name = Path(f).stem
        G = G.subgraph(list(G.nodes())).copy()
        G.graph["name"] = name
        graphs.append(G)
        if limit and len(graphs) >= limit:
            break
    return graphs


def load_haggle(root="data/public") -> list[nx.Graph]:
    """
    加载 CRAWDAD Haggle 移动接触图。
    TSV 格式: 时间戳<TAB>CONN<TAB>节点A<TAB>节点B<TAB>up/down
    取 CONN 行，去重为无向接触图。每个子目录一个图。
    """
    base = Path(root) / "haggle"
    if not base.exists():
        return []
    graphs = []
    # 递归扫描所有 *.tsv（排除 new-to-old-ids.tsv）
    for tsv in sorted(glob.glob(str(base / "**" / "*.tsv"), recursive=True)):
        if "new-to-old" in tsv or "old-to-new" in tsv:
            continue
        edges = set()
        with open(tsv, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and parts[1] == "CONN":
                    try:
                        a, b = int(parts[2]), int(parts[3])
                    except ValueError:
                        continue
                    if a == b:
                        continue
                    edges.add((min(a, b), max(a, b)))
        if len(edges) < 5:
            continue
        G = nx.Graph()
        G.add_edges_from(edges)
        sub = Path(tsv).parent.name
        G.graph["name"] = f"haggle-{sub}"
        graphs.append(G)
    return graphs


def load_public_dataset(name: str, root="data/public", **kwargs) -> list[nx.Graph]:
    """统一入口。"""
    if name == "email_eu":
        return load_email_eu(root)
    if name == "topology_zoo":
        return load_topology_zoo(root, **kwargs)
    if name == "haggle":
        return load_haggle(root)
    raise ValueError(f"Unknown public dataset: {name}")


def _layout_positions(G: nx.Graph, seed: int = 42) -> dict:
    """获取节点坐标：优先原数据坐标（Topology Zoo 有 lat/long），否则 spring_layout。"""
    pos = {}
    for node in G.nodes():
        attrs = G.nodes[node]
        if "x" in attrs and "y" in attrs:
            try:
                pos[node] = (float(attrs["x"]), float(attrs["y"]))
            except (TypeError, ValueError):
                continue
        elif "Longitude" in attrs and "Latitude" in attrs:
            try:
                pos[node] = (float(attrs["Longitude"]), float(attrs["Latitude"]))
            except (TypeError, ValueError):
                continue
    if len(pos) == G.number_of_nodes():
        # 归一化到园区尺度 [0, area]
        arr = np.array([pos[n] for n in G.nodes()])
        mn, mx = arr.min(axis=0), arr.max(axis=0)
        span = mx - mn
        span[span == 0] = 1.0
        arr = (arr - mn) / span * 1900.0 + 50.0  # 映射到 ~2000m 园区
        return {n: tuple(arr[i]) for i, n in enumerate(G.nodes())}
    # 兜底 spring_layout（带 seed 保证可复现）
    set_seed(seed)
    sp = nx.spring_layout(G, seed=seed, scale=1900, center=(1000, 1000))
    return {n: (float(sp[n][0]), float(sp[n][1])) for n in G.nodes()}


def standardize_real_graph(G: nx.Graph, cfg: dict, seed: int = 42) -> nx.Graph:
    """
    将真实网络标准化为与训练数据同口径的图：
    LCC → pos → 随机危险源 → 35 维特征 → 韧性标签（写入 G.graph）。
    """
    set_seed(seed)
    # 0. 多重图 → 简单图（Topology Zoo 部分网络为 MultiGraph）
    if G.is_multigraph():
        G = nx.Graph(G)

    # 1. 取最大连通子图
    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()

    # 1.5 重映射节点为连续 0..n-1（真实网络 ID 可能非连续）
    mapping = {old: i for i, old in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    # 2. 坐标（原地设置）
    pos = _layout_positions(G, seed)
    for node in G.nodes():
        G.nodes[node]["pos"] = pos[node]

    # 3. 随机放置危险源（数量与训练一致）
    park_cfg = cfg["park"]
    n_hazard = park_cfg["num_hazard_sources"]
    area = park_cfg["area_size"]
    rng = np.random.default_rng(seed)
    hazard_pos = rng.uniform([0, 0], area, size=(n_hazard, 2))

    # 4. 传感器-危险源映射（近似：最近危险源）
    pos_arr = np.array([pos[n] for n in G.nodes()])
    d = np.linalg.norm(pos_arr[:, None, :] - hazard_pos[None, :, :], axis=2)
    sensor_to_hazard = d.argmin(axis=1)

    # 5. 35 维特征（无 WSN 特征注入，纯拓扑+空间）
    feats = compute_node_features(G, hazard_pos, sensor_to_hazard,
                                  wsn_features=None, target_dim=cfg["data_factory"]["node_feature_dim"])
    G.graph["features"] = feats.astype(np.float32)
    G.graph["hazard_pos"] = hazard_pos
    G.graph["sensor_to_hazard"] = sensor_to_hazard

    # 6. 韧性标签
    m = compute_all_metrics(G, cfg)
    G.graph["metrics"] = m
    return G


def to_pyg_standard(G: nx.Graph, cfg: dict) -> torch.Tensor:
    """转 PyG Data（字段白名单与 generalization_test.to_pyg 一致）。"""
    from torch_geometric.utils import from_networkx
    # 清理边属性（GraphML 常带不一致的边属性，from_networkx 会报错）
    for u, v, attrs in G.edges(data=True):
        attrs.clear()
    # 清理节点属性（仅保留 pos）
    for node, attrs in G.nodes(data=True):
        pos = attrs.get("pos")
        attrs.clear()
        if pos is not None:
            attrs["pos"] = pos
    data = from_networkx(G)
    data.x = torch.tensor(G.graph["features"], dtype=torch.float)
    data.y = torch.tensor([G.graph["metrics"]["composite_score"]], dtype=torch.float)
    # 白名单清理
    for attr in list(data.keys()):
        if attr not in ("edge_index", "x", "y"):
            delattr(data, attr)
    return data
