#!/usr/bin/env python3
"""
run_pipeline.py - Agent 主控执行流（Main Controller）
按顺序执行五阶段流水线，支持产物存在性跳过与异常回滚。

用法:
    python run_pipeline.py                  # 全量执行（含跳过检查）
    python run_pipeline.py --quick          # 快速冒烟（小规模）
    python run_pipeline.py --force          # 强制重跑全部阶段
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils import setup_logger


STAGES = [
    # (名称, 命令, 产物检查列表)
    ("阶段一 数据工厂", "data_factory.py --config config.yaml",
     ["data/raw/topologies.pt", "data/processed/metadata.csv"]),
    ("阶段二 GCN评估器", "train_gcn_regressor.py --config config.yaml",
     ["models/gcn_resilience.pth"]),
    ("阶段三 GCN-LSTM", "train_gcn_lstm.py --config config.yaml",
     ["models/gcn_lstm_cascade.pth"]),
    ("阶段四 NSGA-III", "run_nsga3_optimizer.py --config config.yaml",
     ["results/tables/pareto_front.csv"]),
    ("阶段五 案例报告", "generate_final_report.py --config config.yaml",
     ["outputs/figures/fig_resilience_comparison.png",
      "outputs/paper_draft/methodology.txt"]),
]


def check_exists(paths):
    """检查产物是否已存在。"""
    return all(Path(p).exists() for p in paths)


def run_stage(name, cmd, quick=False, logger=None):
    """执行单个阶段。"""
    if quick:
        # 各阶段 CLI 均支持 --quick
        if "--quick" not in cmd:
            cmd += " --quick"
    logger.info(">>> 执行 %s: python %s", name, cmd)
    ret = subprocess.run([sys.executable] + cmd.split(), cwd=str(ROOT))
    if ret.returncode != 0:
        raise RuntimeError(f"{name} 执行失败 (exit={ret.returncode})")
    logger.info("<<< %s 完成", name)


def main():
    parser = argparse.ArgumentParser(description="韧性评估与增强流水线主控")
    parser.add_argument("--quick", action="store_true", help="快速冒烟")
    parser.add_argument("--force", action="store_true", help="强制重跑全部")
    args = parser.parse_args()

    logger = setup_logger("pipeline", log_file="logs/pipeline.log")
    logger.info("=" * 60)
    logger.info("启动流水线 (quick=%s, force=%s)", args.quick, args.force)
    logger.info("=" * 60)

    for name, cmd, artifacts in STAGES:
        try:
            if not args.force and check_exists(artifacts):
                logger.info("跳过 %s（产物已存在）", name)
                continue
            run_stage(name, cmd, quick=args.quick, logger=logger)
        except RuntimeError as e:
            logger.error("流水线中断: %s", e)
            # 异常回滚提示：若数据缺失则重新触发阶段一
            if not Path("data/raw/topologies.pt").exists():
                logger.warning("检测到数据缺失，自动重跑阶段一...")
                run_stage("阶段一 数据工厂", "data_factory.py --config config.yaml",
                          quick=args.quick, logger=logger)
            else:
                sys.exit(1)

    logger.info("=" * 60)
    logger.info("流水线全部完成！产物位于 /outputs 与 /results")
    logger.info("=" * 60)
    print("\n✅ Pipeline Finished Successfully. Outputs located in /outputs.")


if __name__ == "__main__":
    main()
