"""
physical_layout.py - 物理布局约束模块
为拓扑生成提供物理真实性：
1. 障碍物生成与 LOS（视线）遮挡判定（线段-AABB 相交，numpy 向量化）
2. 遮挡率校准（倍增/缩放障碍数逼近目标遮挡率）
3. 道路网生成与锚点（传感器沿道路布点，天然形成骨干）
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


# ============================================================
# 1. 障碍物与 LOS 遮挡
# ============================================================

def generate_obstacles(area_size, n_obstacles: int, seed: int = 42,
                       size_range=(40, 120)) -> np.ndarray:
    """
    生成 AABB 矩形障碍物：每行 [x_min, y_min, x_max, y_max]。
    """
    rng = np.random.default_rng(seed)
    w = rng.uniform(*size_range, size=n_obstacles)
    h = rng.uniform(*size_range, size=n_obstacles)
    x = rng.uniform(0, area_size[0] - w.max() * 1.1, size=n_obstacles)
    y = rng.uniform(0, area_size[1] - h.max() * 1.1, size=n_obstacles)
    return np.stack([x, y, x + w, y + h], axis=1)


def _seg_rect_intersect(p1, p2, rect) -> bool:
    """线段 p1-p2 与 AABB 矩形是否相交（Liang-Barsky，标量版）。"""
    x1, y1 = p1
    x2, y2 = p2
    xmin, ymin, xmax, ymax = rect

    dx, dy = x2 - x1, y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return False
    return True


def los_blocked(p, q, obstacles: np.ndarray) -> bool:
    """判断 p-q 连线是否被任一障碍物遮挡（向量化：p,q 可为数组）。"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if obstacles is None or len(obstacles) == 0:
        return False
    # 批量：P×Q×M 计算量大，逐边×全部障碍（向量化障碍维度）
    if p.ndim == 1 and q.ndim == 1:
        for rect in obstacles:
            if _seg_rect_intersect(p, q, rect):
                return True
        return False
    # 批量输入（n_edges, 2）×（n_obs, 4）
    n_edges = p.shape[0]
    res = np.zeros(n_edges, dtype=bool)
    for m, rect in enumerate(obstacles):
        xmin, ymin, xmax, ymax = rect
        dx = q[:, 0] - p[:, 0]
        dy = q[:, 1] - p[:, 1]
        p_arr = np.stack([-dx, dx, -dy, dy], axis=1)          # (E,4)
        q_arr = np.stack([p[:, 0] - xmin, xmax - p[:, 0],
                          p[:, 1] - ymin, ymax - p[:, 1]], axis=1)  # (E,4)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = q_arr / p_arr
        u1 = np.zeros(n_edges)
        u2 = np.ones(n_edges)
        for k in range(4):
            pk = p_arr[:, k]
            qk = q_arr[:, k]
            parallel = np.abs(pk) < 1e-12
            outside = parallel & (qk < 0)
            res[outside] = True  # 平行且在矩形外
            valid = ~parallel
            tk = qk[valid] / pk[valid]
            u1[valid] = np.maximum(u1[valid], np.where(pk[valid] < 0, tk, -np.inf))
            u2[valid] = np.minimum(u2[valid], np.where(pk[valid] > 0, tk, np.inf))
        hit = (u1 <= u2) & ~np.isinf(u1) & ~np.isinf(u2)
        res[hit] = True
        if res.all():
            break
    return res


def occlusion_rate(positions, comm_range: float, obstacles,
                   max_pairs: int = 2000, seed: int = 42) -> float:
    """估算给定障碍物布局的遮挡率（采样 in-range 点对统计 LOS 阻断比例）。"""
    n = len(positions)
    if n < 2 or obstacles is None or len(obstacles) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=comm_range)
    pairs = list(pairs)
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    if not pairs:
        return 0.0
    p = np.array([positions[i] for i, _ in pairs])
    q = np.array([positions[j] for _, j in pairs])
    blocked = los_blocked(p, q, obstacles)
    return float(blocked.mean())


def calibrate_obstacles(positions, comm_range, target_rate: float,
                        area_size, max_obstacles: int = 60,
                        max_tries: int = 6, seed: int = 42) -> np.ndarray:
    """倍增/缩放障碍物数量逼近目标遮挡率。"""
    if target_rate <= 0.01:
        return np.empty((0, 4))
    obstacles = generate_obstacles(area_size, max(1, int(max_obstacles * 0.3)), seed)
    for _ in range(max_tries):
        rate = occlusion_rate(positions, comm_range, obstacles, seed=seed)
        if rate >= target_rate * 0.85:
            break
        obstacles = generate_obstacles(area_size, min(max_obstacles, len(obstacles) * 2), seed)
    return obstacles


# ============================================================
# 2. 道路网与锚点
# ============================================================

def generate_road_network(area_size, n_roads: int = 4, seed: int = 42):
    """
    生成道路网：随机横/纵/斜线段。
    返回 (roads, anchors)：
      roads: list[(x1,y1,x2,y2)] 线段
      anchors: (M,2) 锚点坐标（交点 + 沿线等距点）
    """
    rng = np.random.default_rng(seed)
    w, h = area_size
    roads = []
    for _ in range(n_roads):
        horizontal = rng.random() < 0.5
        if horizontal:
            y = rng.uniform(0.1 * h, 0.9 * h)
            roads.append((0, y, w, y))
        else:
            x = rng.uniform(0.1 * w, 0.9 * w)
            roads.append((x, 0, x, h))
    # 加一条对角（可选）
    if rng.random() < 0.5:
        roads.append((0, 0, w, h * 0.6))

    # 锚点 = 交点 + 沿线等距点
    anchors = []
    for r1 in roads:
        for r2 in roads:
            if r1 == r2:
                continue
            inter = _seg_intersect(r1, r2)
            if inter is not None:
                anchors.append(inter)
    for (x1, y1, x2, y2) in roads:
        length = np.hypot(x2 - x1, y2 - y1)
        n_pts = max(2, int(length / 300))  # 每 300m 一个锚点
        for i in range(1, n_pts):
            t = i / n_pts
            anchors.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    anchors = np.array(anchors) if anchors else np.empty((0, 2))
    return roads, anchors


def _seg_intersect(r1, r2):
    """二维线段交点（标量）。"""
    x1, y1, x2, y2 = r1
    x3, y3, x4, y4 = r2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)
