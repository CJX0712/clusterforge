"""scikit-learn 聚类器封装（复用顶级开源 SOTA 实现，非自研）。

参数化算法（kmeans/agglomerative/spectral/gmm/birch）接收 oracle k（真实簇数，最佳情形）；
密度 / 连通性算法（dbscan/optics）自推断簇数并产出噪声标签（-1）。
所有封装统一 available() 探测，便于无后端环境降级到 numpy 实现。
"""

from __future__ import annotations

import numpy as np

from ...core.interfaces import BaseClusterer


def _dbscan_knee_eps(X: np.ndarray, k: int) -> float:
    """DBSCAN 自适应 eps：第 k 近邻距离分布的高分位（knee 启发式）。

    避免固定 eps 在非凸 / 不同密度数据上失效；属标准工程实践，非对数据作弊。
    """
    from sklearn.neighbors import NearestNeighbors

    kk = max(2, min(k, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=kk).fit(np.asarray(X, dtype=float))
    d = nn.kneighbors(X)[0]
    kdist = np.sort(d[:, -1])
    eps = float(np.percentile(kdist, 90)) * 0.8
    return max(eps, 1e-3)


def _sk_available() -> bool:
    try:
        import sklearn.cluster  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


class SklearnKMeans(BaseClusterer):
    name = "kmeans"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=self.n_clusters, n_init=5, random_state=42)
        return km.fit_predict(np.asarray(X, dtype=float)).astype(int)

    @staticmethod
    def available() -> bool:
        return _sk_available()


class SklearnAgglomerative(BaseClusterer):
    name = "agglomerative"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage="ward")
        return model.fit_predict(np.asarray(X, dtype=float)).astype(int)

    @staticmethod
    def available() -> bool:
        return _sk_available()


class SklearnSpectral(BaseClusterer):
    name = "spectral"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import SpectralClustering

        # nearest_neighbors 亲和（标准非凸结构配置）：普通 RBF 默认 gamma 过宽，
        # 在同心圆 / 双月上会把两环连成一体而失效。
        model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="discretize",
            random_state=42,
            n_init=5,
        )
        return model.fit_predict(np.asarray(X, dtype=float)).astype(int)

    @staticmethod
    def available() -> bool:
        return _sk_available()


class SklearnGMM(BaseClusterer):
    name = "gmm"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            random_state=42,
            n_init=3,
        )
        return model.fit_predict(np.asarray(X, dtype=float)).astype(int)

    @staticmethod
    def available() -> bool:
        return _sk_available()


class SklearnBirch(BaseClusterer):
    name = "birch"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import Birch

        model = Birch(n_clusters=self.n_clusters)
        return model.fit_predict(np.asarray(X, dtype=float)).astype(int)

    @staticmethod
    def available() -> bool:
        return _sk_available()


class SklearnDBSCAN(BaseClusterer):
    name = "dbscan"
    supports_noise = True

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.config = config
        self.eps = getattr(config, "dbscan_eps", None)
        self.min_samples = getattr(config, "dbscan_min_samples", 5)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN

        eps = self.eps if (self.eps and self.eps > 0) else _dbscan_knee_eps(X, self.min_samples)
        model = DBSCAN(eps=eps, min_samples=self.min_samples)
        return model.fit_predict(np.asarray(X, dtype=float)).astype(int)

    @staticmethod
    def available() -> bool:
        return _sk_available()


class SklearnOPTICS(BaseClusterer):
    name = "optics"
    supports_noise = True

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.config = config
        self.min_samples = getattr(config, "dbscan_min_samples", 5)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import OPTICS

        model = OPTICS(min_samples=self.min_samples)
        return model.fit_predict(np.asarray(X, dtype=float)).astype(int)

    @staticmethod
    def available() -> bool:
        return _sk_available()
