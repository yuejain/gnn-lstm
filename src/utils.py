"""
utils.py - 通用工具模块
功能: 随机种子锁定、设备选择、日志、文件 IO、路径管理
"""
import os
import sys
import random
import logging
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # torch 未安装/重装中时降级


def set_seed(seed: int = 42):
    """锁定所有随机源，确保实验可复现。torch 不可用时自动降级。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except (AttributeError, ImportError):
            pass  # torch 未完全就绪时降级（如重装期间）


def get_device(device: str = "auto", force_cuda: bool = False):
    """
    选择计算设备。
    - force_cuda=True: 强制 GPU，CUDA 不可用时抛错（训练一律使用，避免静默降级 CPU）
    - device="auto" + force_cuda=False: 优先 CUDA，无则 CPU
    - device="cuda"/"cpu": 显式指定
    """
    if torch is None:
        raise RuntimeError("torch 不可用，无法选择设备")
    if device in ("auto", "cuda") and (force_cuda or device == "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 不可用！训练强制 GPU 模式失败。请检查："
                "1) 是否安装 CUDA 版 torch (pip install torch --index-url https://download.pytorch.org/whl/cu128) "
                "2) 是否被其他进程占满显存 (nvidia-smi 查看)")
        return torch.device("cuda")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_config(config_path: str = "config.yaml"):
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    """确保目录存在。"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_project_root():
    """返回项目根目录（resilience_agent/）。"""
    return Path(__file__).resolve().parent.parent


def setup_logger(name: str = "resilience", log_file: str = None):
    """配置日志器，同时输出到控制台与文件。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_pickle(obj, path):
    """保存 pickle 文件。"""
    import pickle
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """加载 pickle 文件。"""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def save_csv(df, path):
    """保存 DataFrame 为 CSV。

    沙箱环境可能阻止覆盖已存在文件（os.replace/删除被拦截）→
    遇 PermissionError/OSError 时自动改用唯一文件名（_v2/_v3...），
    保证训练结果永不丢失。
    """
    import time
    ensure_dir(Path(path).parent)
    path = str(path)
    for attempt in range(6):
        try:
            df.to_csv(path, index=False)
            return
        except (PermissionError, OSError):
            time.sleep(2 + attempt * 2)
    # 覆盖冲突 → 改用唯一名
    base, ext = os.path.splitext(path)
    for v in range(2, 100):
        alt = f"{base}_v{v}{ext}"
        try:
            df.to_csv(alt, index=False)
            print(f"[save_csv] 原文件被占用，已保存到 {alt}")
            return
        except (PermissionError, OSError):
            continue
    # 最终兜底
    df.to_csv(path, index=False)


def timestamp_str():
    """返回时间戳字符串。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
