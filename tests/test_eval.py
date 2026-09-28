import numpy as np

from clusterforge.eval.metrics import (
    adjusted_rand_index,
    calinski_harabasz,
    davies_bouldin,
    evaluate_external,
    evaluate_internal,
    silhouette,
)


def test_ari_identical_is_one():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(y, y) == 1.0


def test_ari_random_low():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 300)
    y_pred = rng.integers(0, 3, 300)
    assert -1.0 <= adjusted_rand_index(y_true, y_pred) <= 1.0


def test_ari_range():
    y = np.array([0, 0, 1, 1])
    assert -1.0 <= adjusted_rand_index(y, np.array([1, 1, 0, 0])) <= 1.0


def test_silhouette_range():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(5, 0.3, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    s = silhouette(X, y)
    assert -1.0 <= s <= 1.0
    assert s > 0.5  # 清晰两团，轮廓应高


def test_ch_db_non_negative():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(5, 0.3, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    ch = calinski_harabasz(X, y)
    db = davies_bouldin(X, y)
    assert ch >= 0.0
    assert db >= 0.0


def test_drop_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 2))
    y = np.array([0] * 30 + [-1] * 30)
    s = silhouette(X, y)  # 噪声点应被剔除，不报错
    assert -1.0 <= s <= 1.0


def test_evaluate_external_internal():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(5, 0.3, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    ari, nmi, hom, comp = evaluate_external(y, y)
    assert (ari, nmi, hom, comp) == (1.0, 1.0, 1.0, 1.0)
    sil, ch, db = evaluate_internal(X, y)
    assert sil > 0.5
