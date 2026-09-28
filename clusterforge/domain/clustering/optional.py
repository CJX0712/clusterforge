"""可选 SOTA 后端：HDBSCAN（密度层级） / scikit-network（Louvain 社区）。

统统 guarded + available() 探测：装不上 / 编译失败不影响系统，benchmark 自动跳过（不伪造数字）。
"""

from __future__ import annotations

import numpy as np

from ...core.interfaces import BaseClusterer


class HDBSCANClusterer(BaseClusterer):
    name = "hdbscan"
    supports_noise = True

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        import hdbscan as _h

        mcs = max(5, int(0.04 * len(X)))
        model = _h.HDBSCAN(min_cluster_size=mcs, min_samples=5)
        return model.fit_predict(np.asarray(X, dtype=float)).astype(int)

    @staticmethod
    def available() -> bool:
        try:
            import hdbscan  # noqa: F401

            return True
        except Exception:  # pragma: no cover
            return False


class LouvainClusterer(BaseClusterer):
    name = "louvain"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        import sknetwork as skn
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=float)
        # k-NN 图（k=10），欧氏距离
        k = min(10, X.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k).fit(X)
        adj = nn.kneighbors_graph(X, mode="distance").toarray()
        adj = (adj + adj.T > 0).astype(float)  # 二值对称邻接
        # 去掉自环
        np.fill_diagonal(adj, 0.0)
        if adj.sum() == 0:
            return np.zeros(X.shape[0], dtype=int)
        clustering = skn.clustering.Louvain(random_state=42)
        labels = clustering.fit_transform(adj)
        return np.asarray(labels, dtype=int)

    @staticmethod
    def available() -> bool:
        try:
            import sknetwork  # noqa: F401

            return True
        except Exception:  # pragma: no cover
            return False
