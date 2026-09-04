[README.md](https://github.com/user-attachments/files/31817855/README.md)
# 传感器网络韧性评估与增强模型（Resilience Agent）

基于GCN-NSGA-III 框架的算法整体优化与革新：构建"传感器网络韧性评估与增强"端到端模型。

## 核心革新点

| 维度 | 原框架 | 本模型革新 |
|------|--------------|-----------|
| 评估目标 | 传感器布点优化（节点分类） | **网络韧性评估**（图级回归：鲁棒性/抗毁性/综合韧性评分） |
| 空间建模 | 3 层 GCNConv + 池化 | 3 层 GCNConv(64) + 全局平均池化 + 回归头 |
| 动态预测 | 单 LSTM 预测浓度场 | **GCN-LSTM 双通道**预测级联失效概率 |
| 优化目标 | 成本/覆盖率 | **三目标**：max 韧性 + min 级联风险 + min 成本 |
| 数据来源 | CFD-GAN 合成 | 合成拓扑 + **4 个真实 WSN 数据集集成** |
| 约束处理 | 三维约束协同 | 保留（遮挡/安全距离/高度/可达性） |

## 五阶段流水线

```
run_pipeline.py (主控制器)
├── 阶段一  data_factory.py         多层拓扑数据工厂（3 变体 × 韧性标签 + WSN 集成）
├── 阶段二  train_gcn_regressor.py  GCN 静态评估器（图级韧性回归）
├── 阶段三  train_gcn_lstm.py       GCN-LSTM 级联失效预测器
├── 阶段四  run_nsga3_optimizer.py  NSGA-III 韧性增强器（三目标优化）
└── 阶段五  generate_final_report.py 常州园区镜像案例验证 + 论文图表
```

## 环境安装

```bash
# Python 3.13 + CUDA（RTX 4070 Ti 已验证）
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric networkx pandas numpy scikit-learn pymoo matplotlib scipy PyYAML

# 注意：torch-geometric 2.8 已内置 scatter 算子，勿装 torch-scatter/torch-sparse（会编译失败）
# 注意：Windows 沙箱下 pip 换装 torch 前先清理 site-packages 残留（~orch*/torch/functorch/torchgen）
```

## 运行

```bash
# 全量流水线（600 节点 × 3 拓扑 × 30 图，CUDA 加速）
python run_pipeline.py

# 快速冒烟（小规模验证）
python run_pipeline.py --quick

# 新算法调优（对比 GCN/GAT/GraphSAGE/GCN-GAT 集成，最优者替换主模型）
python train_models_compare.py --config config.yaml

# 单阶段执行
python data_factory.py --config config.yaml
python train_gcn_regressor.py --config config.yaml
python train_gcn_lstm.py --config config.yaml
python run_nsga3_optimizer.py --config config.yaml
python generate_final_report.py --config config.yaml
```

## 目录结构

```
resilience_agent/
├── config.yaml                     # 全局配置（园区/拓扑/模型/优化参数）
├── data_factory.py                 # 阶段一：数据工厂
├── train_gcn_regressor.py          # 阶段二：GCN 静态评估器
├── train_gcn_lstm.py               # 阶段三：GCN-LSTM 级联预测器
├── run_nsga3_optimizer.py          # 阶段四：NSGA-III 优化器
├── generate_final_report.py        # 阶段五：案例验证与报告
├── run_pipeline.py                 # 主控制器
├── src/
│   ├── topology_factory.py         # 异构节点编码 + 三种拓扑变体
│   ├── resilience_labels.py        # 韧性标签（鲁棒性/抗毁性/k连通/级联）
│   ├── wsn_data_loader.py          # 4 个 WSN 数据集集成
│   ├── gcn_regressor.py            # GCN 模型
│   ├── gcn_lstm.py                 # GCN-LSTM 模型
│   └── nsga3_optimizer.py          # NSGA-III 问题定义
├── data/raw/                       # 拓扑数据集 + WSN 原始 CSV
├── data/processed/                 # PyG 数据集 + 元数据 + 级联序列
├── models/                         # 模型 checkpoint
├── results/                        # 指标 CSV + 图表
└── outputs/                        # 论文图表 + methodology + LaTeX 表
```

## WSN 数据集集成

| 数据集 | 内容 | 用途 |
|--------|------|------|
| WsnData (1).csv | 传感器时空数据（X/Y/数据/电池/温度/故障） | 节点时空特征 + 故障标签 |
| WSN-DS.csv | 374k 行网络数据（距离 CH/能耗/攻击类型） | 节点性能特征 + 攻击标签 |
| WSN-DS2.csv | WSN-DS 变体（label 列） | 同上 |
| WSN_Latency_Categorical_Dataset.csv | 网络延迟分类（跳数/时延/缓冲区/信道利用率） | 网络质量特征 + 延迟风险 |

## 复现约束

- 随机种子锁定：所有模块 `seed=42`（numpy/torch 均设）
- 固定种子：数据生成 / 种群初始化 / 级联仿真
- 大规模图：PyG 内部稀疏存储，无需显式 GraphSage 采样（600 节点规模安全）
- 异常回滚：NSGA-III 非支配解过少时自动调高变异概率（0.1→0.3）重启
