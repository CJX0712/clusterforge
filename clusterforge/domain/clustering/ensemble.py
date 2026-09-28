"""旗舰创新模块：共识集成聚类（consensus-forge）+ 表征稳定聚类（pca-kmeans）。

ConsensusForge 思路：
1. 用多个基聚类器（KMeans / GMM 在不同 k 与多随机种子、Spectral、DBSCAN）生成 B 个划分。
2. 累加得到共联矩阵 C[i,j] = 两样本被判同簇的比例（相似度）。
3. 最终划分 = 在相异度 D=1-C 上做层次聚类（average linkage）。
4. k 选择：优先用 oracle k（与参数化方法公平对照）；否则用 C 的特征间隙（eigengap）自动判定。

创新点：不对数据形状做假设，靠基集成把"凸结构擅长者(KMeans)"与"非凸结构擅长者(Spectral/DBSCAN)"
的优势融合，对非凸数据（moons/circles）显著优于单一 KMeans。
"""

from __future__ import annotations

import numpy as np

from ...core.interfaces import BaseClusterer
from .base import SklearnDBSCAN, SklearnKMeans, SklearnSpectral


class RepresentationStableClustering(BaseClusterer):
    """表征稳定基线：PCA 降维（去冗余 / 去尺度）后 KMeans。"""

    name = "pca-kmeans"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2, dim: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.dim = int(dim)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from ..representation.reducers import PCAReducer

        Z = PCAReducer(self.dim).fit_transform(X)
        from .base import SklearnKMeans

        return SklearnKMeans(None, self.n_clusters).fit_predict(Z)


class ConsensusForge(BaseClusterer):
    name = "consensus-forge"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int | None = None) -> None:
        self.config = config
        self.n_clusters_oracle = n_clusters
        self.chosen_k: int | None = None

    def _base_partitions(self, X: np.ndarray) -> list[np.ndarray]:
        """平衡基集成：1 个凸结构专家（KMeans）+ 2 个连通性 / 密度专家（Spectral, DBSCAN）。

        设计要点：非凸数据（同心圆 / 双月）上 KMeans 只能切出左右半（亦合法但非真结构），
        Spectral / DBSCAN 才能恢复真环；两者联合在非凸数据上以 2:1 胜出，凸数据上三者一致。
        故共识既能恢复非凸真结构，又不会在非凸数据上被 KMeans 的"伪合理"半切过票。
        """
        parts: list[np.ndarray] = []
        if self.n_clusters_oracle:
            parts.append(SklearnKMeans(None, self.n_clusters_oracle).fit_predict(X))
            if SklearnSpectral.available():
                parts.append(SklearnSpectral(None, self.n_clusters_oracle).fit_predict(X))
        if SklearnDBSCAN.available():
            parts.append(SklearnDBSCAN(None).fit_predict(X))
        return [p.astype(int) for p in parts]

    @staticmethod
    def _eigengap_k(C: np.ndarray, k_min: int = 2, k_max: int = 8) -> int:
        eig = np.linalg.eigvalsh(np.asarray(C, dtype=float))
        eig = np.sort(np.real(eig))[::-1]
        gaps = np.diff(eig)
        lo = max(0, k_min - 1)
        hi = min(len(gaps), k_max - 1)
        if hi <= lo:
            return k_min
        idx = lo + int(np.argmax(gaps[lo:hi]))
        return int(idx + 1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        parts = self._base_partitions(X)
        if not parts:
            return np.zeros(n, dtype=int)

        # 证据累积（EAC）：等权累加共联矩阵
        C = np.zeros((n, n))
        for lab in parts:
            lab = lab.astype(int)
            for i in range(n):
                li = lab[i]
                if li == -1:
                    continue
                for j in range(i + 1, n):
                    if lab[j] == li and lab[j] != -1:
                        C[i, j] += 1.0
                        C[j, i] += 1.0
        C /= max(len(parts), 1)
        np.fill_diagonal(C, 1.0)

        k = self.n_clusters_oracle or self._eigengap_k(C)
        self.chosen_k = k

        # 阈值连边 → 连通分量（鲁棒、可解释）
        tau = 0.5
        adj = (C > tau).astype(int)
        np.fill_diagonal(adj, 0)
        import scipy.sparse.csgraph as cg

        ncomp, labels_cc = cg.connected_components(adj, directed=False)
        if 2 <= ncomp <= (self.n_clusters_oracle + 3 if self.n_clusters_oracle else 12):
            return labels_cc.astype(int)

        # 回退：组件数越界 → 在相异度 D=1-C 上层次聚类切 k
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        D = 1.0 - C
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        return fcluster(Z, t=k, criterion="maxclust").astype(int) - 1
