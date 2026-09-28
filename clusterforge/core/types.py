"""ClusterForge 核心数据结构（dataclass）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    """一份带真实标签的合成 / 载入数据集。

    - y: 真实簇标签（int，连续 0..n_clusters-1），用于外部指标（ARI/NMI）。
    - n_clusters: 真实簇数，供参数化算法作为 oracle k（最佳情形）。
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    n_clusters: int
    difficulty: str = ""
    note: str = ""


@dataclass
class ClusterResult:
    """单次聚类产出。

    - labels: int 数组，-1 表示噪声（仅密度类算法）。
    - n_clusters: 实际发现的簇数（不含噪声）。
    - supports_noise: 该算法是否产出噪声标签。
    """

    name: str
    labels: np.ndarray
    n_clusters: int
    runtime_s: float
    supports_noise: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class EvalRow:
    """一条跨算法 × 数据集的评测记录（落盘 benchmark.json）。"""

    algorithm: str
    dataset: str
    n_clusters_found: int
    n_clusters_true: int
    ari: float
    nmi: float
    homogeneity: float
    completeness: float
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    runtime_s: float
    available: bool = True
    skipped: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "n_clusters_found": self.n_clusters_found,
            "n_clusters_true": self.n_clusters_true,
            "ari": round(float(self.ari), 6),
            "nmi": round(float(self.nmi), 6),
            "homogeneity": round(float(self.homogeneity), 6),
            "completeness": round(float(self.completeness), 6),
            "silhouette": round(float(self.silhouette), 6),
            "calinski_harabasz": round(float(self.calinski_harabasz), 6),
            "davies_bouldin": round(float(self.davies_bouldin), 6),
            "runtime_s": round(float(self.runtime_s), 6),
            "available": self.available,
            "skipped": self.skipped,
            "note": self.note,
        }
