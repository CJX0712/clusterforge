"""纯 numpy 离线兜底实现（无 torch / 无重型依赖，零下载可跑）。

提供三类可被独立验证的算法引擎：
- KMeans（Lloyd + k-means++ 初始化）：不变量 = 惯性（inertia）单调非增。
- GMM（对角协方差 EM）：不变量 = 责任度每行和为 1，对数似然单调非减。
- Jacobi 特征分解 → PCA：不变量 = A·v ≈ λv，特征向量正交归一。
- 层次聚类（average linkage，借 scipy 距离，scipy 属核心依赖）。

这些实现同时作为「SOTA 后端（HDBSCAN/UMAP）不可用」时的兜底与数值参照。
"""

from __future__ import annotations

import numpy as np

from ...core.interfaces import BaseClusterer


# ===================== KMeans =====================
def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    i0 = int(rng.integers(n))
    centers[0] = X[i0]
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        probs = closest / (closest.sum() + 1e-12)
        i = int(rng.choice(n, p=probs))
        centers[c] = X[i]
        d = np.sum((X - centers[c]) ** 2, axis=1)
        closest = np.minimum(closest, d)
    return centers


def _lloyd(X: np.ndarray, centroids: np.ndarray, max_iter: int, tol: float):
    prev_inertia = None
    hist = []
    labels = None
    for _ in range(max_iter):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = float(dists[np.arange(X.shape[0]), labels].sum())
        hist.append(inertia)
        if prev_inertia is not None and (prev_inertia - inertia) < tol * abs(prev_inertia):
            break
        newc = np.empty_like(centroids)
        for c in range(centroids.shape[0]):
            mask = labels == c
            newc[c] = X[mask].mean(axis=0) if mask.sum() > 0 else centroids[c]
        centroids = newc
        prev_inertia = inertia
    return labels, inertia, hist


def kmeans_numpy(X, k, random_state=42, n_init=5, max_iter=300, tol=1e-4):
    rng = np.random.default_rng(random_state)
    best = None
    for _ in range(n_init):
        centroids = _kmeans_plus_plus(X, k, rng)
        labels, inertia, hist = _lloyd(X, centroids, max_iter, tol)
        if best is None or inertia < best[1]:
            best = (labels, inertia, centroids, hist)
    return best


class NumpyKMeans(BaseClusterer):
    name = "numpy-kmeans"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        labels, _, _, _ = kmeans_numpy(np.asarray(X, dtype=float), self.n_clusters, random_state=42)
        return labels.astype(int)


# ===================== GMM (对角协方差 EM) =====================
def _gmm_responsibilities(X, means, covs, weights):
    """返回责任度矩阵 resp (n,k)，每行和为 1（不变量）。"""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    k = means.shape[0]
    logp = np.zeros((n, k))
    for c in range(k):
        diff = X - means[c]
        prec = 1.0 / (covs[c] + 1e-9)
        logdet = np.sum(np.log(covs[c] + 1e-9))
        maha = np.sum(diff * diff * prec, axis=1)
        logp[:, c] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
    lw = logp + np.log(weights + 1e-12)
    m = lw.max(axis=1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(lw - m).sum(axis=1) + 1e-12)
    return np.exp(lw - logZ[:, None]), logZ


def gmm_em_numpy(X, k, random_state=42, n_iter=100, tol=1e-4):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    means = X[rng.choice(n, k, replace=False)]
    weights = np.full(k, 1.0 / k)
    covs = np.tile(X.var(axis=0) + 1e-6, (k, 1))
    log_resp = np.zeros((n, k))
    ll_hist = []
    for _ in range(n_iter):
        resp, logZ = _gmm_responsibilities(X, means, covs, weights)
        ll_hist.append(float(logZ.sum()))
        Nk = resp.sum(axis=0) + 1e-9
        weights = Nk / n
        means = (resp.T @ X) / Nk[:, None]
        for c in range(k):
            diff = X - means[c]
            covs[c] = np.maximum((resp[:, c, None] * diff * diff).sum(axis=0) / Nk[c], 1e-6)
        if len(ll_hist) > 1 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            break
    labels = np.argmax(log_resp, axis=1)
    return labels.astype(int), means, covs, weights, ll_hist


class NumpyGMM(BaseClusterer):
    name = "numpy-gmm"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        labels, _, _, _, _ = gmm_em_numpy(
            np.asarray(X, dtype=float), self.n_clusters, random_state=42
        )
        return labels


# ===================== Jacobi 特征分解 → PCA =====================
def jacobi_eig(A, max_iter=100, tol=1e-9):
    A = np.array(A, dtype=float).copy()
    n = A.shape[0]
    V = np.eye(n)
    for _ in range(max_iter):
        off = np.abs(np.triu(A, 1))
        p, q = np.unravel_index(np.argmax(off), off.shape)
        if abs(A[p, q]) < tol:
            break
        app, aqq, apq = A[p, p], A[q, q], A[p, q]
        phi = -0.5 * np.arctan2(2.0 * apq, app - aqq)
        c, s = np.cos(phi), np.sin(phi)
        A[p, p] = c * c * app - 2 * c * s * apq + s * s * aqq
        A[q, q] = s * s * app + 2 * c * s * apq + c * c * aqq
        A[p, q] = 0.0
        A[q, p] = 0.0
        for k in range(n):
            if k != p and k != q:
                akp, akq = A[k, p], A[k, q]
                A[k, p] = c * akp - s * akq
                A[p, k] = A[k, p]
                A[k, q] = s * akp + c * akq
                A[q, k] = A[k, q]
        Vp, Vq = V[:, p].copy(), V[:, q].copy()
        V[:, p] = c * Vp - s * Vq
        V[:, q] = s * Vp + c * Vq
    return np.diag(A), V


def pca_numpy(X, dim: int, random_state: int = 42):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    eigvals, eigvecs = jacobi_eig(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    dim = min(dim, X.shape[1])
    comps = eigvecs[:, :dim]
    Z = Xc @ comps
    return Z, eigvals, eigvecs, comps


class NumpyPCAKMeans(BaseClusterer):
    """表征稳定基线：先 PCA 降维（纯 numpy Jacobi）再 KMeans。"""

    name = "numpy-pca-kmeans"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2, dim: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.dim = int(dim)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        Z, _, _, _ = pca_numpy(np.asarray(X, dtype=float), self.dim)
        labels, _, _, _ = kmeans_numpy(Z, self.n_clusters, random_state=42)
        return labels.astype(int)


# ===================== 层次聚类（average linkage，scipy 距离） =====================
class NumpyAgglomerative(BaseClusterer):
    name = "numpy-agglomerative"
    supports_noise = False

    def __init__(self, config=None, n_clusters: int = 2) -> None:
        self.n_clusters = int(n_clusters)
        self.config = config

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        Z = linkage(pdist(np.asarray(X, dtype=float)), method="average")
        return fcluster(Z, t=self.n_clusters, criterion="maxclust").astype(int) - 1
