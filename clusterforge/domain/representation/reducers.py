"""降维 / 表征学习器。

- PCA：scikit-learn 为主，纯 numpy Jacobi 实现作离线兜底。
- UMAP：可选 SOTA 后端（guard）。
- TSNE：scikit-learn 流形学习（可视化常用）。
"""

from __future__ import annotations

import numpy as np

from ...core.interfaces import BaseReducer


class PCAReducer(BaseReducer):
    name = "pca"

    def __init__(self, dim: int = 2) -> None:
        self.dim = int(dim)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        d = min(self.dim, np.asarray(X).shape[1])
        return PCA(n_components=d, random_state=42).fit_transform(np.asarray(X, dtype=float))

    @staticmethod
    def available() -> bool:
        return True


class NumpyPCAReducer(BaseReducer):
    name = "pca-numpy"

    def __init__(self, dim: int = 2) -> None:
        self.dim = int(dim)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from ..clustering.numpy_impl import pca_numpy

        Z, _, _, _ = pca_numpy(np.asarray(X, dtype=float), self.dim)
        return Z

    @staticmethod
    def available() -> bool:
        return True


class UMAPReducer(BaseReducer):
    name = "umap"

    def __init__(self, dim: int = 2) -> None:
        self.dim = int(dim)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        import umap

        d = min(self.dim, np.asarray(X).shape[1])
        return umap.UMAP(n_components=d, random_state=42).fit_transform(np.asarray(X, dtype=float))

    @staticmethod
    def available() -> bool:
        try:
            import umap  # noqa: F401

            return True
        except Exception:  # pragma: no cover
            return False


class TSNEReducer(BaseReducer):
    name = "tsne"

    def __init__(self, dim: int = 2) -> None:
        self.dim = int(dim)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE

        d = min(self.dim, np.asarray(X).shape[1])
        return TSNE(n_components=d, random_state=42).fit_transform(np.asarray(X, dtype=float))

    @staticmethod
    def available() -> bool:
        return True
