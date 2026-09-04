"""
gcn_regressor.py - GCN 静态韧性评估器模型
复用程洋录用论文的 GCN 骨干网络（3 层 GCNConv），改造最后一层：
移除节点分类器，添加全局平均池化 + 线性回归层 → 输出图级韧性评分 (0-1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class ResilienceGCN(nn.Module):
    """
    图级韧性回归模型。

    结构:
        conv1(3层 GCNConv, hidden=64) + ReLU + Dropout
        global_mean_pool  →  图级表示
        MLP(64 → 64 → 1)  →  韧性分数 (回归)
    """

    def __init__(self, in_channels: int = 32, hidden_dim: int = 64,
                 num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # 图级读出: 平均池化 + 回归头（原论文为节点分类，此处改造为图级回归）
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Args:
            data: PyG Data 对象，含 x/edge_index/batch
        Returns:
            (B, 1) 韧性评分
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级读出（关键改造点）
        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x).squeeze(-1)


def evaluate_regression(y_true, y_pred):
    """计算回归指标: MSE / RMSE / MAE / R²。"""
    y_true = torch.as_tensor(y_true, dtype=torch.float)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float)
    mse = F.mse_loss(y_pred, y_true).item()
    rmse = mse ** 0.5
    mae = F.l1_loss(y_pred, y_true).item()
    var = torch.var(y_true)
    r2 = 1.0 - (mse / var.item()) if var.item() > 1e-9 else 0.0
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
