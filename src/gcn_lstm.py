"""
gcn_lstm.py - GCN-LSTM 动态级联预测器
双通道模型：GCN 提取空间结构 -> LSTM 提取时间演化 -> 融合预测下一时刻节点失效概率。
复用程洋硕士论文 3.1.2 节 LSTM 模块设计（3 层 LSTM，128 单元，T=10）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.hyper_lr_mp import HyperLRMP


class SpatialEncoder(nn.Module):
    """GCN 空间编码器：对每个时间步的图快照提取节点级空间特征。"""

    def __init__(self, in_channels: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


class GCN_LSTM_Cascade(nn.Module):
    """
    级联失效预测模型。

    输入: 连续 T 个时间步的拓扑快照序列 [batch, T, N, F]（同图拓扑，节点特征随时间演化）
    输出: 下一时刻节点失效概率 [batch, T, N, 1]（sigmoid）

    结构:
        GCN 空间编码（每时间步共享权重） -> LSTM 时间演化 -> 分类头
    """

    def __init__(self, in_channels: int = 32, hidden_dim: int = 64,
                 lstm_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial = SpatialEncoder(in_channels, hidden_dim)
        # LSTM: 输入为每步 GCN 节点特征（N, hidden），按节点维度独立演化
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq_x, edge_index):
        """
        Args:
            seq_x: (batch, T, N, F) 节点特征序列
            edge_index: (2, E) 图拓扑边（各时间步共享）
        Returns:
            (batch, T, N, 1) 每步失效概率
        """
        batch, T, N, F = seq_x.shape
        device = seq_x.device

        # GCN 空间编码（每时间步）
        spatial_out = []
        for t in range(T):
            xt = seq_x[:, t].reshape(batch * N, F)          # (B*N, F)
            # 批量图需要扩充边索引（同图复制 batch 份）
            eb = edge_index.clone()
            if batch > 1:
                offset = torch.arange(batch, device=device).repeat_interleave(edge_index.shape[1]) * N
                eb = edge_index.repeat(1, batch) + offset
            st = self.spatial(xt, eb)                        # (B*N, H)
            spatial_out.append(st.reshape(batch, N, -1))     # (B, N, H)

        spatial_seq = torch.stack(spatial_out, dim=1)        # (B, T, N, H)

        # LSTM 时间演化：将节点作为 batch，时间步作为序列
        # (B, T, N, H) -> (B*N, T, H)
        lstm_in = spatial_seq.permute(0, 2, 1, 3).reshape(batch * N, T, self.hidden_dim)
        lstm_out, _ = self.lstm(lstm_in)                     # (B*N, T, H)
        lstm_out = lstm_out.reshape(batch, N, T, self.hidden_dim)

        # 分类头：预测每步失效概率
        logits = self.classifier(lstm_out)                   # (B, N, T, 1)
        probs = torch.sigmoid(logits)
        # 目标标签是"累计已失效"，模型输出后，取最后一步为最终预测
        return probs.permute(0, 2, 1, 3)                     # (B, T, N, 1)


def build_synthetic_sequences(graphs, cfg):
    """
    构建训练样本：从真实生成的拓扑图 + 级联仿真标签构造 (T, N, F) 序列。

    Args:
        graphs: list[nx.Graph]（带 features 属性）
        cfg: 配置
    Returns:
        (tensor_seq, tensor_labels, edge_index)
    """
    n_steps = cfg["topology"]["cascade_steps"]
    n_nodes = len(graphs[0].nodes()) if graphs else 0

    seqs, labels = [], []
    edge_index = None
    from torch_geometric.utils import from_networkx

    for G in graphs:
        feats = G.graph["features"]  # (N, F) 含注意力增强特征
        # 用级联仿真生成每步失效掩码作为标签
        from src.resilience_labels import simulate_cascade_failure
        _, failed_sets = simulate_cascade_failure(
            G, failure_ratio=0.05, n_steps=n_steps, seed=cfg["general"]["seed"])

        # 构造序列：节点特征随失效状态演化（失效节点特征衰减）
        seq = []
        for t in range(n_steps):
            x_t = feats.copy()
            if t < len(failed_sets):
                for node in failed_sets[t]:
                    x_t[node] *= 0.1   # 失效节点特征衰减
            seq.append(x_t)
        seqs.append(np.stack(seq, axis=0))  # (T, N, F)

        # 标签：每步失效概率（累计）
        lab = np.zeros((n_steps, n_nodes), dtype=np.float32)
        for t in range(min(n_steps, len(failed_sets))):
            lab[t] = 0.0
            for node in failed_sets[t]:
                lab[t, node] = 1.0
        labels.append(lab)

    # edge_index（取第一张图的拓扑）
    data = from_networkx(graphs[0])
    edge_index = data.edge_index

    return (torch.tensor(np.stack(seqs), dtype=torch.float),   # (B, T, N, F)
            torch.tensor(np.stack(labels), dtype=torch.float), # (B, T, N)
            edge_index)


class PICGNet(nn.Module):
    """物理信息级联图网络（PICG-Net）—— 统一框架完整模型。

    级联失效三要素 -> 模型组件：
      ①高阶耦合      -> 超图（HyperLRMP 的 HyperedgeAggregate）
      ②过载动力学    -> 负载重分配消息传递 + 物理感知注意力（LoadRedistributionConv）
      ③拓扑演化      -> FADT（失效感知动态拓扑，可选开关）
    统一监督        -> 物理约束损失（在 train_picgn.py 中实现 L_load + L_phys）

    结构：HyperLRMP 空间编码（每时间步共享） -> LSTM 时间演化 -> 双头
          （失效概率头 + LoadHead 负载预测头）
    """

    def __init__(self, in_channels: int = 32, hidden_dim: int = 64,
                 lstm_layers: int = 2, dropout: float = 0.2, temp: float = 1.0,
                 use_attention: bool = True, use_hyperedge: bool = True,
                 use_overload: bool = True, use_fadt: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_fadt = use_fadt
        self.spatial = HyperLRMP(in_channels, hidden_dim, temp=temp,
                                 use_attention=use_attention,
                                 use_hyperedge=use_hyperedge,
                                 use_overload=use_overload)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                            num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))
        # LoadHead：从空间特征预测节点负载（归一化介数），供 L_load 监督 + 物理可解释
        self.load_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    def forward(self, seq_x, edge_index, overload=None,
                hyperedge_to_node=None, hyperedge_ptr=None, failed_masks=None):
        """seq_x: (B, T, N, F)；edge_index: (2, E)；
        overload: (B, T, N) 动态 或 (B, N) 静态 或 (N,)；
        failed_masks: (B, T, N) 每步失效掩码（FADT 用，可选）。

        Returns:
            probs: (B, T, N, 1) 每步失效概率
            load_pred: (B, T, N, 1) 每步节点负载预测
        """
        batch, T, N, F = seq_x.shape
        device = seq_x.device
        if overload is None:
            overload = torch.ones(N, device=device)
        if overload.dim() == 1:
            overload = overload.unsqueeze(0).expand(batch, -1)          # (B, N)

        spatial_out, load_preds = [], []
        for t in range(T):
            xt = seq_x[:, t].reshape(batch * N, F)          # (B*N, F)
            # 动态 overload：取第 t 步（若为 (B,T,N)），否则用静态
            ov = overload[:, t] if overload.dim() == 3 else overload
            ov = ov.reshape(batch * N)                      # (B*N,)
            eb = edge_index.clone()
            if batch > 1:
                offset = torch.arange(batch, device=device).repeat_interleave(
                    edge_index.shape[1]) * N
                eb = edge_index.repeat(1, batch) + offset
            # FADT：按上一时刻失效 mask 动态收缩拓扑
            if self.use_fadt and failed_masks is not None and t > 0:
                failed = failed_masks[:, t - 1].bool()          # (B, N)
                node_keep = ~failed.reshape(batch * N)          # (B*N,)
                keep = node_keep[eb[0]] & node_keep[eb[1]]      # 任一端失效则删边
                eb = eb[:, keep]
                ov = ov * node_keep.float()                     # 失效节点不再传导压力
            st = self.spatial(xt, eb, ov, hyperedge_to_node, hyperedge_ptr,
                              batch_size=batch)  # (B*N, H)
            spatial_out.append(st.reshape(batch, N, -1))
            load_preds.append(self.load_head(st).reshape(batch, N))

        spatial_seq = torch.stack(spatial_out, dim=1)        # (B, T, N, H)
        load_pred = torch.stack(load_preds, dim=1).unsqueeze(-1)  # (B, T, N, 1)

        # LSTM 时间演化
        lstm_in = spatial_seq.permute(0, 2, 1, 3).reshape(batch * N, T, self.hidden_dim)
        lstm_out, _ = self.lstm(lstm_in)                     # (B*N, T, H)
        lstm_out = lstm_out.reshape(batch, N, T, self.hidden_dim)
        logits = self.classifier(lstm_out)                   # (B, N, T, 1)
        probs = torch.sigmoid(logits).permute(0, 2, 1, 3)    # (B, T, N, 1)
        return probs, load_pred
