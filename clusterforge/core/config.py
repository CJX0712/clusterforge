"""ClusterForge 全局配置 + ENV_CF_* 环境变量覆盖。

约定：环境变量 `ENV_CF_<UPPER_KEY>` 优先于默认值。
例如 ENV_CF_RANDOM_STATE=7 覆盖 random_state。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    random_state: int = 42
    n_jobs: int = 1

    # 密度聚类参数（dbscan_eps=None 表示按数据自适应 knee-eps）
    dbscan_eps: float | None = None
    dbscan_min_samples: int = 5

    # 可选 SOTA 后端开关（不可用时自动降级，不报错）
    use_hdbscan: bool = True
    use_umap: bool = True
    use_scikit_network: bool = True

    # 基准多 seed 复现
    benchmark_seeds: Tuple[int, ...] = (42, 123, 7)

    # 表征降维目标维度（用于 PCA->KMeans 等链路）
    reduce_dim: int = 2

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        prefix = "ENV_CF_"
        for f in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            env_key = prefix + f.upper()
            if env_key in os.environ:
                raw = os.environ[env_key]
                cur = getattr(cfg, f)
                try:
                    if isinstance(cur, bool):
                        setattr(cfg, f, raw.lower() in ("1", "true", "yes", "on"))
                    elif isinstance(cur, int):
                        setattr(cfg, f, int(raw))
                    elif isinstance(cur, float):
                        setattr(cfg, f, float(raw))
                    elif isinstance(cur, tuple):
                        setattr(cfg, f, tuple(int(x) for x in raw.split(",") if x.strip()))
                    else:
                        setattr(cfg, f, raw)
                except ValueError:
                    # 解析失败则保留默认
                    pass
        return cfg

    def as_dict(self) -> dict:
        return {
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "use_hdbscan": self.use_hdbscan,
            "use_umap": self.use_umap,
            "use_scikit_network": self.use_scikit_network,
            "benchmark_seeds": list(self.benchmark_seeds),
            "reduce_dim": self.reduce_dim,
        }
