#!/usr/bin/env python3
"""
run_optimization_plan.py - 论文对标优化实验编排器（串行 GPU 队列）
按计划顺序执行 8 个实验阶段（P0.1 已单独完成）：

P0.2 多指标联合 → P0.3 排名一致性 → P1.1 韧性曲线 → P1.2 可解释性
→ P1.3 级联增强 → P2.1 对比学习 → P2.2 优化增强 → P2.3 不确定性

用法:
    python run_optimization_plan.py
"""
import subprocess
import sys
import time

PY = sys.executable
STEPS = [
    ("P0.2 多指标联合预测", ["train_multihead.py", "--epochs", "150"]),
    ("P0.3 排名一致性", ["rank_consistency.py"]),
    ("P1.1 韧性时间曲线", ["train_curve.py", "--epochs", "100"]),
    ("P1.2 可解释性归因", ["explain_widegat.py"]),
    ("P1.3 级联增强对比", ["cascade_recovery_analysis.py", "--n_graphs", "200"]),
    ("P2.1 InfoNCE 对比学习", ["train_contrastive.py", "--epochs", "60"]),
    ("P2.2 NSGA3+Louvain", ["run_nsga3_enhanced.py", "--n_gen", "60"]),
    ("P2.3 MC 不确定性", ["mc_uncertainty.py"]),
]


def main():
    total_t0 = time.time()
    for name, args in STEPS:
        t0 = time.time()
        print(f"\n{'='*60}\n▶ 阶段: {name} ({args[0]})\n{'='*60}", flush=True)
        r = subprocess.run([PY] + args)
        dt = time.time() - t0
        status = "✅ 完成" if r.returncode == 0 else f"❌ 失败 (exit {r.returncode})"
        print(f"[{name}] {status} | 耗时 {dt/60:.1f} 分钟", flush=True)
        if r.returncode != 0:
            print(f"⚠️ 跳过后续阶段（{name} 失败）")
    print(f"\n全部实验完成，总耗时 {(time.time()-total_t0)/3600:.1f} 小时")


if __name__ == "__main__":
    main()
