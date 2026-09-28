"""ClusterForge 抽象接口（契约先行）。

所有聚类器 / 降维器 / 指标 / 数据生成器实现统一契约，保证跨模块公平评测与可插拔。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClusterer(ABC):
    """聚类器契约。

    实现方须提供：
    - name: 稳定唯一名（落盘 / 对照表用）
    - supports_noise: 是否产出 -1 噪声标签
    - available(): 后端是否可用（可选 SOTA 探测）
    - fit_predict(X) -> labels (int, -1=noise)
    """

    name: str = "base"
    supports_noise: bool = False

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def available() -> bool:
        return True


class BaseReducer(ABC):
    """降维 / 表征学习器契约。fit_transform(X) -> Z (float)。"""

    name: str = "base"

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def available() -> bool:
        return True


class BaseMetric(ABC):
    """评测指标契约。compute(...) -> float；range/higher_better 用于交叉验证。"""

    name: str = "base"
    higher_better: bool = True
    low: float = 0.0
    high: float = 1.0

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> float: ...


class BaseDatasetGenerator(ABC):
    """数据集生成器契约。生成带真实标签的数据。"""

    name: str = "base"

    @abstractmethod
    def generate(self, random_state: int): ...
