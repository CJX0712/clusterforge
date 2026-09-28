"""聚类评测指标：外部（有真实标签）+ 内部（无标签）。

外部指标复用 scikit-learn（别名导入避免与本地同名函数递归）。
内部指标以 scikit-learn 为主，纯 numpy 实现作离线兜底（scipy cdist 仅算距离）。
所有指标均带范围 / 不变量，供单测交叉验证。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import BackendUnavailable, DataError

# ---- sklearn 探测（别名导入，杜绝递归） ----
try:
    from sklearn.metrics import (
        adjusted_rand_score as _sk_ari,
    )
    from sklearn.metrics import (
        calinski_harabasz_score as _sk_ch,
    )
    from sklearn.metrics import (
        completeness_score as _sk_comp,
    )
    from sklearn.metrics import (
        davies_bouldin_score as _sk_db,
    )
    from sklearn.metrics import (
        homogeneity_score as _sk_hom,
    )
    from sklearn.metrics import (
        normalized_mutual_info_score as _sk_nmi,
    )
    from sklearn.metrics import (
        silhouette_score as _sk_silhouette,
    )

    _HAVE_SK = True
except Exception:  # pragma: no cover
    _HAVE_SK = False


def _drop_noise(y: np.ndarray, X: np.ndarray | None = None):
    y = np.asarray(y)
    mask = y != -1
    if X is None:
        return y[mask]
    return y[mask], X[mask]


def _align(y_true: np.ndarray, y_pred: np.ndarray):
    """外部指标对齐：仅保留两方都非噪声（-1）的样本，保证等长。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = (y_true != -1) & (y_pred != -1)
    return y_true[mask], y_pred[mask]


# ===================== 外部指标 =====================
def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not _HAVE_SK:
        raise BackendUnavailable("外部指标需要 scikit-learn")
    yt, yp = _align(y_true, y_pred)
    if yt.shape[0] < 2:
        return 0.0
    return float(_sk_ari(yt, yp))


def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not _HAVE_SK:
        raise BackendUnavailable("外部指标需要 scikit-learn")
    yt, yp = _align(y_true, y_pred)
    if yt.shape[0] < 2:
        return 0.0
    return float(_sk_nmi(yt, yp))


def homogeneity_completeness(y_true: np.ndarray, y_pred: np.ndarray):
    if not _HAVE_SK:
        raise BackendUnavailable("外部指标需要 scikit-learn")
    yt, yp = _align(y_true, y_pred)
    if yt.shape[0] < 2:
        return 0.0, 0.0
    return float(_sk_hom(yt, yp)), float(_sk_comp(yt, yp))


# ===================== 内部指标（sklearn 优先，numpy 兜底） =====================
def _ch_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = labels.astype(int)
    n = X.shape[0]
    uniq = np.unique(labels)
    k = uniq.size
    if k < 2 or k >= n:
        return 0.0
    mean = X.mean(axis=0)
    B = 0.0
    W = 0.0
    for c in uniq:
        Xc = X[labels == c]
        nc = Xc.shape[0]
        if nc == 0:
            continue
        diff = Xc.mean(axis=0) - mean
        B += nc * float(diff @ diff)
        Xc_c = Xc - Xc.mean(axis=0)
        W += float(np.trace(Xc_c @ Xc_c.T))
    return float((B / (k - 1)) / (W / (n - k) + 1e-12))


def _db_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    labels = labels.astype(int)
    uniq = np.unique(labels)
    if uniq.size < 2:
        return 0.0
    centroids = []
    scat = []
    for c in uniq:
        Xc = X[labels == c]
        if Xc.shape[0] == 0:
            continue
        cent = Xc.mean(axis=0)
        centroids.append(cent)
        scat.append(float(np.mean(np.linalg.norm(Xc - cent, axis=1))))
    centroids = np.array(centroids)
    scat = np.array(scat)
    kk = centroids.shape[0]
    if kk < 2:
        return 0.0
    R = 0.0
    for i in range(kk):
        maxr = -1.0
        for j in range(kk):
            if i == j:
                continue
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d == 0.0:
                d = 1e-12
            val = (scat[i] + scat[j]) / d
            if val > maxr:
                maxr = val
        R += maxr
    return float(R / kk)


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    # 统一去噪：噪声点（-1）不参与内部指标计算
    lmask = labels != -1
    yt2 = labels[lmask]
    Xt2 = X[lmask]
    k = int(np.unique(yt2).size)
    if k < 2 or k >= Xt2.shape[0]:
        return 0.0
    if _HAVE_SK:
        try:
            return float(_sk_silhouette(Xt2, yt2, metric="euclidean"))
        except Exception:
            pass
    # numpy 兜底
    from scipy.spatial.distance import cdist

    D = cdist(Xt2, Xt2)
    n = Xt2.shape[0]
    uniq = np.unique(yt2)
    s = np.zeros(n)
    for i in range(n):
        li = yt2[i]
        same = yt2 == li
        same[i] = False
        if same.sum() == 0:
            s[i] = 0.0
            continue
        a = float(D[i, same].mean())
        others = [uj for uj in uniq if uj != li]
        bs = [float(D[i, yt2 == uj].mean()) for uj in others if (yt2 == uj).sum() > 0]
        b = min(bs) if bs else 0.0
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    lmask = labels != -1
    Xt = np.asarray(X, dtype=float)[lmask]
    yt = labels[lmask]
    k = int(np.unique(yt).size)
    if k < 2 or k >= Xt.shape[0]:
        return 0.0
    if _HAVE_SK:
        try:
            return float(_sk_ch(Xt, yt))
        except Exception:
            pass
    return _ch_numpy(Xt, yt)


def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    lmask = labels != -1
    Xt = np.asarray(X, dtype=float)[lmask]
    yt = labels[lmask]
    k = int(np.unique(yt).size)
    if k < 2:
        return 0.0
    if _HAVE_SK:
        try:
            return float(_sk_db(Xt, yt))
        except Exception:
            pass
    return _db_numpy(Xt, yt)


def evaluate_external(y_true: np.ndarray, y_pred: np.ndarray):
    """返回 (ari, nmi, homogeneity, completeness)。"""
    ari = adjusted_rand_index(y_true, y_pred)
    nmi = normalized_mutual_info(y_true, y_pred)
    hom, comp = homogeneity_completeness(y_true, y_pred)
    return ari, nmi, hom, comp


def evaluate_internal(X: np.ndarray, labels: np.ndarray):
    """返回 (silhouette, calinski_harabasz, davies_bouldin)。"""
    return silhouette(X, labels), calinski_harabasz(X, labels), davies_bouldin(X, labels)


def check_labels(y: np.ndarray) -> None:
    """契约校验：标签须为 int 数组，噪声用 -1。"""
    if not np.issubdtype(np.asarray(y).dtype, np.integer):
        raise DataError("标签须为整数数组")
