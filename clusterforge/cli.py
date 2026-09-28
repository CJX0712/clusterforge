"""ClusterForge 命令行入口（argparse）。

子命令：
  benchmark            跨算法 × 数据集基准，落盘 benchmark.json
  run --dataset --algorithm   单算法单数据集评测
  datasets / algorithms        列出可用数据集 / 算法
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .core.config import Config
from .data.synthetic import get_dataset, list_datasets
from .domain.clustering.registry import REGISTRY, available_entries
from .pipeline.pipeline import ClusterPipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="clusterforge", description="ClusterForge · 聚类与表征学习系统"
    )
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("benchmark", help="运行全基准并落盘 JSON")
    b.add_argument("--out", default="benchmark.json")
    b.add_argument("--seeds", default="42,123,7")

    r = sub.add_parser("run", help="单次评测")
    r.add_argument("--dataset", default="blobs")
    r.add_argument("--algorithm", default="kmeans")

    sub.add_parser("datasets", help="列出数据集")
    sub.add_parser("algorithms", help="列出可用算法")

    args = p.parse_args(argv)
    cfg = Config.from_env()
    pipe = ClusterPipeline(cfg)

    if args.cmd == "benchmark":
        seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
        rows = pipe.benchmark(seeds=seeds)
        summary = pipe.summarize(rows)
        payload = {
            "system": "ClusterForge",
            "version": "0.1.0",
            "author": "晨星",
            "config": cfg.as_dict(),
            "summary": summary,
            "rows": [rw.as_dict() for rw in rows],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("=== ClusterForge Benchmark ===")
        for rk in summary["ranking"]:
            print(
                f"{rk['algorithm']:<16} meanARI={rk['mean_ari']:.3f} "
                f"minARI={rk['min_ari']:.3f} ds={rk['n_datasets']} sec={rk['mean_runtime_s']:.3f}"
            )
        print(f"best={summary['best_algorithm']} rows={summary['n_rows']}")
        print(f"written: {os.path.abspath(args.out)}")
        return 0

    if args.cmd == "run":
        ds = get_dataset(args.dataset, cfg.random_state)
        entry = next((e for e in REGISTRY if e["cls"].name == args.algorithm), None)
        if entry is None:
            print(f"unknown algorithm: {args.algorithm}", file=sys.stderr)
            return 1
        row = pipe._eval_one(ds, entry)
        print(json.dumps(row.as_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "datasets":
        for d in list_datasets():
            print(d)
        return 0

    if args.cmd == "algorithms":
        for e in available_entries():
            print(f"{e['cls'].name:<16} tier={e['tier']:<10} label={e['label']}")
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
