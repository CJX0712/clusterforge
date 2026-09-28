import numpy as np

from clusterforge.core.config import Config
from clusterforge.core.types import ClusterResult, Dataset, EvalRow


def test_config_default():
    c = Config()
    assert c.random_state == 42
    assert c.n_jobs == 1


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ENV_CF_RANDOM_STATE", "7")
    monkeypatch.setenv("ENV_CF_USE_HDBSCAN", "false")
    monkeypatch.setenv("ENV_CF_BENCHMARK_SEEDS", "1,2,3")
    c = Config.from_env()
    assert c.random_state == 7
    assert c.use_hdbscan is False
    assert c.benchmark_seeds == (1, 2, 3)


def test_dataset_dataclass():
    ds = Dataset("x", np.zeros((3, 2)), np.array([0, 1, 0]), 2)
    assert ds.n_clusters == 2
    assert ds.X.shape == (3, 2)


def test_evalrow_as_dict():
    row = EvalRow("kmeans", "blobs", 4, 4, 1.0, 1.0, 1.0, 1.0, 0.5, 100.0, 0.3, 0.01)
    d = row.as_dict()
    assert d["algorithm"] == "kmeans"
    assert d["ari"] == 1.0
    assert "skipped" in d


def test_types_import_roundtrip():
    r = ClusterResult("kmeans", np.array([0, 1, 0]), 2, 0.1)
    assert r.n_clusters == 2
    assert r.supports_noise is False
