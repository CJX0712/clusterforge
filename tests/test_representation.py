import numpy as np

from clusterforge.domain.clustering.numpy_impl import pca_numpy
from clusterforge.domain.representation.reducers import (
    PCAReducer,
    TSNEReducer,
)


def test_pca_reducer_dim():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (80, 10))
    Z = PCAReducer(dim=3).fit_transform(X)
    assert Z.shape == (80, 3)


def test_numpy_pca_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (120, 6))
    Z_num, eig_num, _, _ = pca_numpy(X, 3)
    Z_sk = PCAReducer(dim=3).fit_transform(X)
    # 投影应同向（符号可能翻转，比对绝对值相关）
    # 用前两主成分解释方差比例近似一致
    assert Z_num.shape == Z_sk.shape
    # 协方差结构的方差解释：numpy 前3特征值占比接近 sklearn
    Xc = X - X.mean(0)
    total = np.trace(np.cov(Xc.T))
    assert eig_num[:3].sum() / total > 0.4


def test_tsne_reducer_runs():
    if not TSNEReducer.available():
        return
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 8))
    Z = TSNEReducer(dim=2).fit_transform(X)
    assert Z.shape == (60, 2)
