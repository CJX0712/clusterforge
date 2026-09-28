"""ClusterForge 主流水线：run → evaluate → benchmark → summarize。

单向无环：cli → pipeline → {data, clustering, representation, eval} → core。
每个算法独立计时（per-request 口径），失败 / 后端不可用自动标记为 skipped（不伪造数字）。
"""

from __future__ import annotations

import time

import numpy as np

from ..core.config import Config
from ..core.types import Dataset, EvalRow
from ..data.synthetic import get_dataset, list_datasets
from ..domain.clustering.registry import available_entries
from ..eval.metrics import evaluate_external, evaluate_internal


class ClusterPipeline:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    # ---------- 单算法评测 ----------
    def _eval_one(self, dataset: Dataset, entry: dict) -> EvalRow:
        name = entry["cls"].name
        if not entry["cls"].available():
            return EvalRow(
                name,
                dataset.name,
                0,
                dataset.n_clusters,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                available=False,
                skipped=True,
                note="可选后端不可用，已自动跳过",
            )
        try:
            inst = (
                entry["cls"](self.config, n_clusters=dataset.n_clusters)
                if entry.get("accepts_k")
                else entry["cls"](self.config)
            )
            t0 = time.perf_counter()
            labels = inst.fit_predict(dataset.X)
            dt = time.perf_counter() - t0
        except Exception as ex:  # 单算法失败不影响整体
            return EvalRow(
                name,
                dataset.name,
                0,
                dataset.n_clusters,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                available=True,
                skipped=True,
                note=f"运行异常: {type(ex).__name__}: {ex}",
            )
        labels = np.asarray(labels).astype(int)
        n_found = int(np.unique(labels[labels != -1]).size)
        ari, nmi, hom, comp = evaluate_external(dataset.y, labels)
        sil, ch, db = evaluate_internal(dataset.X, labels)
        return EvalRow(
            name,
            dataset.name,
            n_found,
            dataset.n_clusters,
            ari,
            nmi,
            hom,
            comp,
            sil,
            ch,
            db,
            dt,
            available=True,
            skipped=False,
        )

    # ---------- 全基准 ----------
    def benchmark(self, seeds: tuple[int, ...] | None = None) -> list[EvalRow]:
        seeds = seeds if seeds is not None else (self.config.random_state,)
        entries = available_entries()
        rows: list[EvalRow] = []
        for seed in seeds:
            for dname in list_datasets():
                ds = get_dataset(dname, seed)
                for e in entries:
                    rows.append(self._eval_one(ds, e))
        return rows

    # ---------- 聚合摘要 ----------
    def summarize(self, rows: list[EvalRow]) -> dict:
        from collections import defaultdict

        by_algo = defaultdict(list)
        for r in rows:
            if not r.skipped and r.available:
                by_algo[r.algorithm].append(r)

        ranking = []
        for algo, rs in by_algo.items():
            aris = [x.ari for x in rs]
            ranking.append(
                {
                    "algorithm": algo,
                    "mean_ari": float(np.mean(aris)),
                    "min_ari": float(np.min(aris)),
                    "n_datasets": len(rs),
                    "mean_runtime_s": float(np.mean([x.runtime_s for x in rs])),
                }
            )
        ranking.sort(key=lambda d: d["mean_ari"], reverse=True)
        best = ranking[0]["algorithm"] if ranking else None
        return {
            "n_rows": len(rows),
            "n_algorithms": len(by_algo),
            "ranking": ranking,
            "best_algorithm": best,
        }
