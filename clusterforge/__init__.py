"""ClusterForge — 世界顶级聚类与表征学习系统。

作者: 晨星 (CJX0712)
复用顶级开源: scikit-learn / scipy / numpy；可选 SOTA 后端
HDBSCAN / UMAP / scikit-network 自动探测降级。
含纯 numpy 离线兜底（KMeans / GMM-EM / Jacobi-PCA），零下载可跑。
"""

__version__ = "0.1.0"
__author__ = "晨星"

from .core.config import Config
from .core.types import ClusterResult, Dataset, EvalRow

__all__ = ["Config", "Dataset", "ClusterResult", "EvalRow", "__version__", "__author__"]
