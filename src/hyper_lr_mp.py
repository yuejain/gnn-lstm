"""
hyper_lr_mp.py - 超图负载重分配消息传递层（PICG-Net 算法核心）

把级联失效的"过载 -> 负载重分配 -> 邻居过载"动力学写成消息传递核：

1. 普通图 LR-MP（LoadRedistributionConv）：
   物理感知注意力 alpha_ij = softmax_j(phi(overload_j))，phi = exp(overload/temp)。
   - 高过载节点向邻居传导更强的失效压力（物理感知）；
   - softmax 归一化保证权重是概率分布（流守恒：对每个目标节点 i，Σ_j alpha_ij = 1）。

2. 超边聚合（HyperedgeAggregate）：
   节点 -> 超边（聚合成员节点特征）-> 节点（过载超边回传失效压力），
   刻画"多设施对同一楼宇的联合依赖"的高阶 AND 耦合。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import softmax as pyg_softmax


class LoadRedistributionConv(MessagePassing):
    """负载重分配消息传递（普通图）。

    消息 m_ij = alpha_ij · W · h_j，注意力为 GAT 式可学习打分 + 物理过载偏置：
        alpha_ij = softmax_j( leaky_relu(a^T[W h_i ‖ W h_j]) + β·log1p(overload_j) )
    - 可学习注意力 a 让模型自适应学习重分配系数；
    - 物理偏置 β·log1p(overload) 引导高过载节点传导更强失效压力（对数压缩避免爆炸）。
    """

    def __init__(self, in_channels: int, out_channels: int, temp: float = 1.0,
                 bias: bool = True, use_attention: bool = True):
        super().__init__(aggr="add", flow="source_to_target")
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.att = nn.Linear(2 * out_channels, 1, bias=False)   # GAT 式打分
        self.beta = nn.Parameter(torch.tensor(0.5))             # 物理过载偏置强度
        self.use_attention = use_attention
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        with torch.no_grad():
            self.beta.fill_(0.5)

    def forward(self, x, edge_index, overload):
        """x: (N,in)；edge_index:(2,E)；overload:(N,) 过载程度(>=0)。"""
        return self.propagate(edge_index, x=x, overload=overload.unsqueeze(-1))

    def message(self, x_i, x_j, overload_j, index, size_i):
        overload_j = overload_j.squeeze(-1)  # (E,)
        msg = self.lin(x_j)                  # (E, out)
        if self.use_attention:
            score = self.att(torch.cat([self.lin(x_i), msg], dim=-1)).squeeze(-1)  # (E,)
            score = F.leaky_relu(score) + self.beta * torch.log1p(overload_j)      # 物理偏置
            alpha = pyg_softmax(score, index, num_nodes=size_i)
        else:
            alpha = pyg_softmax(torch.ones_like(overload_j), index, num_nodes=size_i)
        return alpha.unsqueeze(-1) * msg


class HyperedgeAggregate(nn.Module):
    """超边聚合：节点 -> 超边 -> 节点 的高阶耦合消息传递（支持 batch）。

    超边 = 一组设施对同一节点的联合依赖（高阶 AND 关系）。
    """

    def __init__(self, channels: int, bias: bool = True, use_overload: bool = True):
        super().__init__()
        self.use_overload = use_overload
        self.lin = nn.Linear(channels, channels, bias=bias)

    def forward(self, x, hyperedge_to_node, hyperedge_ptr, overload=None, batch_size=1):
        """x: (B*N, C)；hyperedge_to_node:(E_hyper,) 单图成员节点 id(0..N-1)；
        hyperedge_ptr:(M+1,)；overload:(B*N,)。返回 (B*N, C)。"""
        device = x.device
        N = x.shape[0] // batch_size
        C = x.shape[1]
        M = hyperedge_ptr.numel() - 1
        if M == 0:
            return x.new_zeros_like(x)

        # 每个成员所属的超边 id
        seg_lens = hyperedge_ptr[1:] - hyperedge_ptr[:-1]
        hyperedge_ids = torch.repeat_interleave(
            torch.arange(M, device=device), seg_lens)  # (E_hyper,)

        # batch 展开（超边结构对每个图复制）
        hn = torch.cat([hyperedge_to_node + b * N for b in range(batch_size)])
        hids = torch.cat([hyperedge_ids + b * M for b in range(batch_size)])
        B_M = batch_size * M

        # 节点 -> 超边：成员特征 mean 聚合
        member = x[hn]  # (B*E_hyper, C)
        x_hyper_sum = x.new_zeros(B_M, C).scatter_add(
            0, hids.unsqueeze(-1).expand(-1, C), member)
        cnt = x.new_zeros(B_M).scatter_add(
            0, hids, torch.ones_like(hids, dtype=x.dtype))
        x_hyper = x_hyper_sum / cnt.clamp(min=1).unsqueeze(-1)  # (B_M, C)

        # 超边过载态（回传加权）
        if self.use_overload and overload is not None:
            ov_member = overload[hn].unsqueeze(-1)
            ov_sum = x.new_zeros(B_M, 1).scatter_add(0, hids.unsqueeze(-1), ov_member)
            ov_hyper = ov_sum / cnt.clamp(min=1).unsqueeze(-1)
            weight = torch.sigmoid(ov_hyper)  # (B_M, 1)
        else:
            weight = 1.0

        # 超边 -> 节点：回传（scatter 到成员节点）
        msg = (weight * self.lin(x_hyper))[hids]  # (B*E_hyper, C)
        out = x.new_zeros(batch_size * N, C).scatter_add(
            0, hn.unsqueeze(-1).expand(-1, C), msg)
        return out


class HyperLRMP(nn.Module):
    """超图负载重分配消息传递层（PICG-Net 空间编码核心）。

    h_i' = ReLU( GCN(h_i) + LR-MP(h_i, overload) + Σ_e∋i W_e·h_e )
    - GCN 分支：度归一化提供稳定基础表示；
    - LR-MP 分支：过载加权的物理增强（残差，不破坏 GCN 稳定性）；
    - 超边聚合：高阶耦合回传。
    """

    def __init__(self, in_channels: int, out_channels: int, temp: float = 1.0,
                 use_attention: bool = True, use_hyperedge: bool = True,
                 use_overload: bool = True):
        super().__init__()
        self.use_hyperedge = use_hyperedge
        self.gcn = GCNConv(in_channels, out_channels)          # 稳定基础
        self.lr_conv = LoadRedistributionConv(in_channels, out_channels, temp,
                                              bias=True, use_attention=use_attention)
        self.hyper_agg = HyperedgeAggregate(out_channels, bias=True, use_overload=use_overload)

    def forward(self, x, edge_index, overload, hyperedge_to_node=None,
                hyperedge_ptr=None, batch_size=1):
        h = self.gcn(x, edge_index) + self.lr_conv(x, edge_index, overload)  # (B*N, out)
        if self.use_hyperedge and hyperedge_to_node is not None and hyperedge_ptr is not None:
            h = h + self.hyper_agg(h, hyperedge_to_node, hyperedge_ptr,
                                   overload=overload, batch_size=batch_size)
        return F.relu(h)
