import numpy as np

from clusterforge.data.synthetic import get_dataset
from clusterforge.domain.clustering.ensemble import ConsensusForge
from clusterforge.domain.clustering.registry import REGISTRY, available_entries


def _gauss2(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0, 0.5, (150, 2)), rng.normal(9, 0.5, (150, 2))])


def test_all_available_clusterers_run():
    X = _gauss2()
    for e in available_entries():
        inst = e["cls"](None, n_clusters=2) if e.get("accepts_k") else e["cls"](None)
        lab = inst.fit_predict(X)
        lab = np.asarray(lab).astype(int)
        assert lab.shape == (X.shape[0],)
        # 标签须为整数，噪声用 -1，簇标签非负（OPTICS 等可能过分割，属正常行为）
        assert np.issubdtype(lab.dtype, np.integer)
        assert lab.min() >= -1


def test_registry_has_flagship():
    names = {e["cls"].name for e in REGISTRY}
    assert "consensus-forge" in names
    assert any(e.get("flagship") for e in REGISTRY)


def test_consensus_runs_and_labels_valid():
    X = _gauss2()
    cf = ConsensusForge(None, n_clusters=2)
    lab = cf.fit_predict(X)
    lab = np.asarray(lab).astype(int)
    assert lab.shape == (X.shape[0],)
    assert lab.min() >= 0
    assert len(np.unique(lab)) >= 2


def test_consensus_handles_nonconvex():
    # 同心圆：KMeans 必败，共识集成应显著更优
    ds = get_dataset("circles", 42)
    from clusterforge.domain.clustering.base import SklearnKMeans

    km = SklearnKMeans(None, n_clusters=ds.n_clusters).fit_predict(ds.X)
    cf = ConsensusForge(None, n_clusters=ds.n_clusters).fit_predict(ds.X)
    from clusterforge.eval.metrics import adjusted_rand_index as ari

    assert ari(ds.y, cf) > 0.5
    assert ari(ds.y, km) < 0.3  # KMeans 在同心圆上接近随机
