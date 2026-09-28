"""合成数据集生成器（带真实标签 + 真实簇数）。

设计目标（难度梯度，避免数据泄漏 / 天花板效应）：
- blobs / varied / anisotropic：凸结构，参数化方法（KMeans/GMM）可达高 ARI。
- moons / circles / noisy_circles：非凸结构，欧氏 KMeans 必败，
  密度 / 连通性方法（Spectral/DBSCAN/HDBSCAN）占优。
- no_structure：均匀分布噪声，所有方法 ARI≈0，内部指标（轮廓）应偏低，证明系统不会"无中生有"。
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

from ..core.types import Dataset


def _blobs(random_state: int) -> Dataset:
    X, y = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=random_state)
    return Dataset("blobs", X, y, 4, "easy", "标准球形凸簇")


def _varied(random_state: int) -> Dataset:
    X, y = make_blobs(
        n_samples=600,
        centers=4,
        cluster_std=[1.0, 2.5, 0.5, 1.8],
        random_state=random_state,
    )
    return Dataset("varied", X, y, 4, "easy", "不同方差的凸簇")


def _anisotropic(random_state: int) -> Dataset:
    X, y = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=random_state)
    # 各向异性拉伸：用固定变换旋转 + 拉长
    rng = np.random.default_rng(random_state)
    t = rng.standard_normal((2, 2))
    u, s, _ = np.linalg.svd(t)
    trans = u @ np.diag(s * 3.0) @ u.T  # (2,2) 矩阵，避免退化为向量
    X = X @ trans.T
    return Dataset("anisotropic", X, y, 4, "medium", "拉伸的各向异性凸簇（KMeans 受损）")


def _moons(random_state: int) -> Dataset:
    X, y = make_moons(n_samples=600, noise=0.06, random_state=random_state)
    return Dataset("moons", X, y, 2, "hard", "双月非凸（KMeans 必败）")


def _circles(random_state: int) -> Dataset:
    X, y = make_circles(n_samples=600, factor=0.5, noise=0.05, random_state=random_state)
    return Dataset("circles", X, y, 2, "hard", "同心圆非凸（KMeans 必败）")


def _noisy_circles(random_state: int) -> Dataset:
    X, y = make_circles(n_samples=500, factor=0.5, noise=0.08, random_state=random_state)
    rng = np.random.default_rng(random_state + 1)
    noise = rng.standard_normal((200, 2)) * 1.5
    X = np.vstack([X, noise])
    y = np.hstack([y, rng.integers(0, 2, size=200)])  # 噪声点无结构，标签随机
    return Dataset("noisy_circles", X, y, 2, "hard", "同心圆 + 均匀背景噪声")


def _no_structure(random_state: int) -> Dataset:
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-3, 3, size=(600, 2))
    y = rng.integers(0, 3, size=600)  # 纯噪声，标签随机，ARI 应≈0
    return Dataset("no_structure", X, y, 3, "trivial", "均匀分布噪声（无簇结构）")


# 注册表：name -> (callable, 默认说明)
SYNTHETIC_DATASETS: dict[str, object] = {
    "blobs": _blobs,
    "varied": _varied,
    "anisotropic": _anisotropic,
    "moons": _moons,
    "circles": _circles,
    "noisy_circles": _noisy_circles,
    "no_structure": _no_structure,
}


def list_datasets() -> list[str]:
    return list(SYNTHETIC_DATASETS.keys())


def get_dataset(name: str, random_state: int = 42) -> Dataset:
    if name not in SYNTHETIC_DATASETS:
        raise KeyError(f"未知数据集: {name}")
    return SYNTHETIC_DATASETS[name](random_state)  # type: ignore[call-arg]


def default_datasets(random_state: int = 42) -> list[Dataset]:
    return [get_dataset(n, random_state) for n in list_datasets()]
