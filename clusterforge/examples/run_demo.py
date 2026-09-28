"""端到端演示：合成数据 → 跨算法基准 → 打印表 + 落盘 benchmark.json。

零下载可跑（核心依赖 numpy/scipy/scikit-learn；HDBSCAN/UMAP 不可用时自动跳过）。
"""

from __future__ import annotations

import json
import os
import time

from ..core.config import Config
from ..pipeline.pipeline import ClusterPipeline


def run(out_path: str = "benchmark.json") -> dict:
    cfg = Config.from_env()
    pipe = ClusterPipeline(cfg)
    t0 = time.perf_counter()
    rows = pipe.benchmark(seeds=cfg.benchmark_seeds)
    elapsed = time.perf_counter() - t0
    summary = pipe.summarize(rows)

    payload = {
        "system": "ClusterForge",
        "version": "0.1.0",
        "author": "晨星",
        "config": cfg.as_dict(),
        "generated_at_note": "确定性可复现基准（固定 random_state）",
        "elapsed_s": round(elapsed, 3),
        "summary": summary,
        "rows": [r.as_dict() for r in rows],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _print_table(payload: dict) -> None:
    rows = payload["rows"]
    hdr = ["algo", "dataset", "k_hat", "k_true", "ARI", "NMI", "sil", "CH", "DB", "sec"]
    print("  ".join(f"{h:<14}" for h in hdr))
    for r in rows:
        if r.get("skipped"):
            print(f"{r['algorithm']:<14}{r['dataset']:<14}{'--':<14}skip: {r.get('note', '')[:40]}")
            continue
        print(
            "  ".join(
                f"{str(r['algorithm']):<14}{r['dataset']:<14}"
                f"{r['n_clusters_found']:<14}{r['n_clusters_true']:<14}"
                f"{r['ari']:<14.3f}{r['nmi']:<14.3f}{r['silhouette']:<14.3f}"
                f"{r['calinski_harabasz']:<14.1f}{r['davies_bouldin']:<14.3f}{r['runtime_s']:<14.3f}"
            )
        )
    print("\n=== 聚合排名（按 mean ARI） ===")
    for i, r in enumerate(payload["summary"]["ranking"], 1):
        print(
            f"{i:>2}. {r['algorithm']:<16} meanARI={r['mean_ari']:.3f}  "
            f"minARI={r['min_ari']:.3f}  datasets={r['n_datasets']}  "
            f"mean_sec={r['mean_runtime_s']:.3f}"
        )
    print(f"\n最佳算法: {payload['summary']['best_algorithm']}")
    print(f"总耗时: {payload['elapsed_s']}s  评测条数: {payload['summary']['n_rows']}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "benchmark.json")
    payload = run(out)
    _print_table(payload)
    print(f"\n已落盘: {out}")
