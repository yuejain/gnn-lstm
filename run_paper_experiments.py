#!/usr/bin/env python3
"""
run_paper_experiments.py - 论文补充实验编排器（串行执行）
按优先级依次运行，GPU 复用：
  1. run_width_ablation.py   (宽度消融 h128/256/512)
  2. run_seed_robustness.py  (3-seed 统计稳健性)
  3. run_nsga3_optimizer.py  (NSGA-III 韧性优化案例，用新 WideGAT 模型)

依赖：baseline 对比（train_models_compare.py）完成后运行本脚本。
用法:
    python run_paper_experiments.py
"""
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent
STEPS = [
    ("Baseline 对比", "train_models_compare.py", ["--epochs", "250"]),
    ("宽度消融", "run_width_ablation.py", ["--epochs", "150"]),
    ("3-seed 稳健性", "run_seed_robustness.py", ["--epochs", "150"]),
    ("NSGA-III 案例", "run_nsga3_optimizer.py", []),
]


def main():
    os.chdir(str(ROOT)) if (os := __import__("os")) else None
    total_t0 = time.time()
    for name, script, args in STEPS:
        t0 = time.time()
        print(f"\n{'='*60}\n▶ 阶段: {name} ({script})\n{'='*60}", flush=True)
        r = subprocess.run([PY, script] + args)
        dt = time.time() - t0
        status = "✅ 完成" if r.returncode == 0 else f"❌ 失败 (exit {r.returncode})"
        print(f"[{name}] {status} | 耗时 {dt/60:.1f} 分钟", flush=True)
        if r.returncode != 0:
            print(f"跳过后续阶段（{name} 失败）")
            break
    print(f"\n全部实验完成，总耗时 {(time.time()-total_t0)/3600:.1f} 小时")


if __name__ == "__main__":
    main()
