"""
wsn_data_loader.py - 4 个 WSN 数据集集成模块
将 WSN 运行数据（节点级属性/性能/攻击标签）集成到拓扑图节点特征中，
使 GCN 静态评估器与 GCN-LSTM 级联预测器使用真实传感器网络数据训练。

数据集:
1. WsnData (1).csv            - 传感器时空数据（SensorID/X/Y/SensorData/BatteryLife/Temperature/IsFaulty）
2. WSN-DS.csv                 - WSN 网络数据（id/Time/Is_CH/Dist_To_CH/Energy/Attack type）
3. WSN-DS2.csv                - WSN-DS 变体（label 列）
4. WSN_Latency_Categorical_Dataset.csv - 网络延迟分类数据（Hop/Delay/Buffer/Link_Quality/Latency）
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


class WSNDataLoader:
    """加载并标准化 4 个 WSN 数据集，输出统一的节点特征与标签。"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_all(self) -> dict:
        """加载全部 4 个数据集。"""
        result = {}
        for name, fname in [
            ("wsn_basic", "WsnData (1).csv"),
            ("wsn_ds1", "WSN-DS.csv"),
            ("wsn_ds2", "WSN-DS2.csv"),
            ("wsn_latency", "WSN_Latency_Categorical_Dataset.csv"),
        ]:
            path = self.data_dir / fname
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    # WSN-DS 系列列名可能带前导/尾随空格，统一 strip
                    df.columns = [str(c).strip() for c in df.columns]
                    result[name] = df
                except Exception as e:
                    print(f"[wsn] 加载 {fname} 失败: {e}")
            else:
                print(f"[wsn] 未找到 {fname}，跳过")
        return result

    # ----------------------------------------------------------
    # 特征工程：各数据集 → 统一的 (样本, 特征) 矩阵 + 标签
    # ----------------------------------------------------------

    def extract_basic_features(self, df) -> tuple[np.ndarray, np.ndarray]:
        """
        WsnData (1).csv → 节点时空特征
        特征: [SensorType编码, X, Y, SensorData, BatteryLife, Temperature]
        标签: IsFaulty (0/1)
        """
        df = df.copy()
        # SensorType 编码
        if "SensorType" in df.columns:
            df["SensorType_enc"] = df["SensorType"].astype("category").cat.codes
        feats = df[[
            "SensorType_enc", "X", "Y", "SensorData", "BatteryLife", "Temperature"
        ]].values.astype(np.float32) if "SensorType_enc" in df.columns else df[[
            "X", "Y", "SensorData", "BatteryLife", "Temperature"
        ]].values.astype(np.float32)
        labels = df["IsFaulty"].values.astype(np.float32) if "IsFaulty" in df.columns \
            else np.zeros(len(df))
        return feats, labels

    def extract_ds_features(self, df) -> tuple[np.ndarray, np.ndarray]:
        """
        WSN-DS / WSN-DS2 → 网络节点性能特征
        特征: [Is_CH, Dist_To_CH, ADV_S, JOIN_S, SCH_S, DATA_S, Rank, dist_CH_To_BS, Energy]
        标签: Attack type / label（攻击类型→0/1 异常）
        """
        df = df.copy()
        energy_col = "Expaned Energy" if "Expaned Energy" in df.columns else \
            ("Consumed Energy" if "Consumed Energy" in df.columns else None)
        label_col = "Attack type" if "Attack type" in df.columns else \
            ("label" if "label" in df.columns else None)

        cols = ["Is_CH", "Dist_To_CH", "ADV_S", "JOIN_S", "SCH_S", "DATA_S",
                "Rank", "dist_CH_To_BS"]
        if energy_col:
            cols.append(energy_col)
        feats = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

        if label_col:
            labels = df[label_col].astype(str).apply(
                lambda s: 0.0 if s.lower() in ("normal", "0") else 1.0).values.astype(np.float32)
        else:
            labels = np.zeros(len(df))
        return feats, labels

    def extract_latency_features(self, df) -> tuple[np.ndarray, np.ndarray]:
        """
        WSN_Latency → 网络质量特征
        特征: [Hop_Count, Transmission_Delay, Buffer_Occupancy, Channel_Utilization,
               Energy_Level, Link_Quality, Packet_Size, PDR, Traffic_Class]
        标签: Latency_Category（High/Medium/Low → 2/1/0 风险等级）
        """
        df = df.copy()
        feats = df[[
            "Hop_Count", "Transmission_Delay", "Buffer_Occupancy", "Channel_Utilization",
            "Energy_Level", "Link_Quality", "Packet_Size", "PDR"
        ]].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        # Traffic_Class 编码
        if "Traffic_Class" in df.columns:
            tc = df["Traffic_Class"].astype("category").cat.codes.values.astype(np.float32)
            feats = np.hstack([feats, tc.reshape(-1, 1)])
        # 标签：延迟等级 → 风险
        lat_map = {"Low": 0.0, "Medium": 1.0, "High": 2.0, "Low Latency": 0.0,
                   "Medium Latency": 1.0, "High Latency": 2.0}
        labels = df["Latency_Category"].map(lat_map).fillna(1.0).values.astype(np.float32)
        return feats, labels

    # ----------------------------------------------------------
    # 统一入口：聚合特征 → 节点级特征字典
    # ----------------------------------------------------------

    def build_node_feature_bank(self) -> dict:
        """
        构建节点特征库：将 4 个数据集标准化为可注入拓扑图的特征。

        Returns:
            {
              "feature_bank": np.ndarray (M, F) 聚合特征矩阵（各数据集采样拼接）
              "labels": np.ndarray (M,) 聚合标签（0=正常, 1=故障/攻击/高延迟）
              "sources": list[str] 每条样本来源
            }
        """
        datasets = self.load_all()
        bank_feats, bank_labels, bank_sources = [], [], []

        if "wsn_basic" in datasets:
            f, l = self.extract_basic_features(datasets["wsn_basic"])
            bank_feats.append(f); bank_labels.append(l)
            bank_sources += ["basic"] * len(f)

        if "wsn_ds1" in datasets:
            f, l = self.extract_ds_features(datasets["wsn_ds1"])
            bank_feats.append(f); bank_labels.append(l)
            bank_sources += ["ds1"] * len(f)

        if "wsn_ds2" in datasets:
            f, l = self.extract_ds_features(datasets["wsn_ds2"])
            bank_feats.append(f); bank_labels.append(l)
            bank_sources += ["ds2"] * len(f)

        if "wsn_latency" in datasets:
            f, l = self.extract_latency_features(datasets["wsn_latency"])
            bank_feats.append(f); bank_labels.append(l)
            bank_sources += ["latency"] * len(f)

        if not bank_feats:
            raise FileNotFoundError("未加载到任何 WSN 数据集")

        # 各数据集特征维度不同 → 分别标准化后按需截断/填充到统一维度
        target_dim = max(f.shape[1] for f in bank_feats)
        padded = []
        for f in bank_feats:
            p = np.zeros((len(f), target_dim), dtype=np.float32)
            p[:, :f.shape[1]] = f
            # 标准化
            mu, std = p.mean(axis=0), p.std(axis=0) + 1e-9
            padded.append((p - mu) / std)

        feature_bank = np.vstack(padded)
        labels = np.concatenate(bank_labels)
        return {
            "feature_bank": feature_bank,
            "labels": labels,
            "sources": bank_sources,
        }


def get_wsn_node_features(data_dir: str, n_nodes: int, seed: int = 42,
                          feature_dim: int = 5) -> np.ndarray | None:
    """
    从 WSN 特征库中为 n_nodes 个图节点采样对齐的特征（每节点 5 维）。

    Returns:
        (n_nodes, feature_dim) 特征矩阵；数据集不可用时返回 None
    """
    try:
        loader = WSNDataLoader(data_dir)
        bank = loader.build_node_feature_bank()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(bank["feature_bank"]), size=n_nodes, replace=True)
        feats = bank["feature_bank"][idx]
        # 截取前 feature_dim 维
        return feats[:, :feature_dim]
    except Exception as e:
        print(f"[wsn] 特征注入失败（降级为纯拓扑特征）: {e}")
        return None
