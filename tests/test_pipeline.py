from clusterforge.core.config import Config
from clusterforge.data.synthetic import get_dataset
from clusterforge.domain.clustering.registry import REGISTRY
from clusterforge.pipeline.pipeline import ClusterPipeline


def _entry(name):
    return next(e for e in REGISTRY if e["cls"].name == name)


def test_benchmark_runs_and_ranks():
    cfg = Config(benchmark_seeds=(42,))
    pipe = ClusterPipeline(cfg)
    rows = pipe.benchmark(seeds=(42,))
    summary = pipe.summarize(rows)
    assert summary["n_rows"] > 0
    assert summary["best_algorithm"] is not None
    assert all(r["n_datasets"] >= 1 for r in summary["ranking"])


def test_consensus_wins_on_circles():
    pipe = ClusterPipeline(Config())
    ds = get_dataset("circles", 42)
    r_km = pipe._eval_one(ds, _entry("kmeans"))
    r_cf = pipe._eval_one(ds, _entry("consensus-forge"))
    # 创新点：非凸数据上共识集成显著优于单一 KMeans
    assert r_cf.ari > r_km.ari + 0.3


def test_kmeans_wins_on_blobs():
    pipe = ClusterPipeline(Config())
    ds = get_dataset("blobs", 42)
    r_km = pipe._eval_one(ds, _entry("kmeans"))
    assert r_km.ari > 0.8  # 凸球形簇 KMeans 应近满分


def test_no_structure_low_ari():
    pipe = ClusterPipeline(Config())
    ds = get_dataset("no_structure", 42)
    r = pipe._eval_one(ds, _entry("kmeans"))
    # 均匀分布噪声无簇结构，ARI 应接近 0（系统不"无中生有"）
    assert r.ari < 0.2


def test_benchmark_reproducible():
    pipe = ClusterPipeline(Config(benchmark_seeds=(42,)))
    a = pipe.benchmark(seeds=(42,))
    b = pipe.benchmark(seeds=(42,))
    # 同 seed 应确定性一致
    assert [r.as_dict()["ari"] for r in a] == [r.as_dict()["ari"] for r in b]
