"""聚类算法注册表（契约驱动，可插拔）。

- tier=sklearn：复用 scikit-learn SOTA 实现。
- tier=numpy：纯 numpy 离线兜底 / 数值参照。
- tier=optional：可选 SOTA 后端（HDBSCAN / scikit-network），不可用时自动跳过。
- tier=flagship：共识集成（consensus-forge）。

accepts_k=True 的算法会收到真实簇数（oracle k，最佳情形）；density 类自推断。
"""

from __future__ import annotations

from .base import (
    SklearnAgglomerative,
    SklearnBirch,
    SklearnDBSCAN,
    SklearnGMM,
    SklearnKMeans,
    SklearnOPTICS,
    SklearnSpectral,
)
from .ensemble import (
    ConsensusForge,
    RepresentationStableClustering,
)
from .numpy_impl import (
    NumpyAgglomerative,
    NumpyGMM,
    NumpyKMeans,
    NumpyPCAKMeans,
)
from .optional import (
    HDBSCANClusterer,
    LouvainClusterer,
)

REGISTRY: list[dict] = [
    {"cls": SklearnKMeans, "accepts_k": True, "tier": "sklearn", "label": "KMeans"},
    {"cls": SklearnAgglomerative, "accepts_k": True, "tier": "sklearn", "label": "Agglomerative"},
    {"cls": SklearnSpectral, "accepts_k": True, "tier": "sklearn", "label": "Spectral"},
    {"cls": SklearnGMM, "accepts_k": True, "tier": "sklearn", "label": "GMM(full)"},
    {"cls": SklearnBirch, "accepts_k": True, "tier": "sklearn", "label": "Birch"},
    {
        "cls": SklearnDBSCAN,
        "accepts_k": False,
        "tier": "sklearn",
        "density": True,
        "label": "DBSCAN",
    },
    {
        "cls": SklearnOPTICS,
        "accepts_k": False,
        "tier": "sklearn",
        "density": True,
        "label": "OPTICS",
    },
    {"cls": NumpyKMeans, "accepts_k": True, "tier": "numpy", "label": "numpy-KMeans"},
    {"cls": NumpyGMM, "accepts_k": True, "tier": "numpy", "label": "numpy-GMM"},
    {"cls": NumpyPCAKMeans, "accepts_k": True, "tier": "numpy", "label": "numpy-PCA-KMeans"},
    {"cls": NumpyAgglomerative, "accepts_k": True, "tier": "numpy", "label": "numpy-Agglomerative"},
    {
        "cls": RepresentationStableClustering,
        "accepts_k": True,
        "tier": "sklearn",
        "label": "PCA-KMeans",
    },
    {
        "cls": HDBSCANClusterer,
        "accepts_k": False,
        "tier": "optional",
        "density": True,
        "label": "HDBSCAN",
    },
    {"cls": LouvainClusterer, "accepts_k": False, "tier": "optional", "label": "Louvain"},
    {
        "cls": ConsensusForge,
        "accepts_k": True,
        "tier": "flagship",
        "flagship": True,
        "label": "ConsensusForge",
    },
]


def available_entries() -> list[dict]:
    return [e for e in REGISTRY if e["cls"].available()]
