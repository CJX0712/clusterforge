import numpy as np

from clusterforge.domain.clustering.numpy_impl import (
    NumpyGMM,
    NumpyKMeans,
    NumpyPCAKMeans,
    _gmm_responsibilities,
    gmm_em_numpy,
    jacobi_eig,
    kmeans_numpy,
    pca_numpy,
)


def _two_blobs(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0, 0.5, (150, 2)), rng.normal(8, 0.5, (150, 2))])


def test_kmeans_inertia_monotonic():
    X = _two_blobs()
    _, inertia, _, hist = kmeans_numpy(X, 2, random_state=1)
    hist = np.asarray(hist)
    assert np.all(np.diff(hist) <= 1e-6), hist  # 惯性单调非增（不变量）


def test_kmeans_separates():
    X = _two_blobs()
    labels, _, _, _ = kmeans_numpy(X, 2, random_state=1)
    assert len(np.unique(labels)) == 2
    # 每团应主要由单一簇覆盖
    for c in (0, 1):
        mask = X[:, 0] < 4
        frac = (labels[mask] == c).mean()
        assert max(frac, 1 - frac) > 0.9


def test_gmm_responsibilities_sum_one():
    X = _two_blobs()
    labels, means, covs, weights, _ = gmm_em_numpy(X, 2, random_state=1)
    resp, _ = _gmm_responsibilities(X, means, covs, weights)
    assert np.allclose(resp.sum(axis=1), 1.0, atol=1e-6)  # 不变量


def test_gmm_loglik_non_decreasing():
    X = _two_blobs()
    _, _, _, _, ll = gmm_em_numpy(X, 2, random_state=1, n_iter=50)
    assert np.all(np.diff(ll) >= -1e-6), ll  # 对数似然单调非减


def test_jacobi_eig_invariant():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (6, 6))
    A = A @ A.T + np.eye(6) * 0.1
    w, V = jacobi_eig(A)
    resid = np.max(np.abs(A @ V - V * w))
    assert resid < 1e-6, resid
    assert np.max(np.abs(V.T @ V - np.eye(6))) < 1e-6  # 正交归一


def test_pca_reconstruction():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (100, 5))
    Z, eig, vecs, comps = pca_numpy(X, 2)
    assert Z.shape == (100, 2)
    assert eig[0] >= eig[1] >= 0


def test_numpy_clusterers_run():
    X = _two_blobs()
    for cls in (NumpyKMeans, NumpyGMM, NumpyPCAKMeans):
        inst = cls(None, n_clusters=2)
        lab = inst.fit_predict(X)
        assert set(np.unique(lab)).issubset({0, 1})
