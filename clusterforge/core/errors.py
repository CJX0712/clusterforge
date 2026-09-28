"""ClusterForge 错误码体系（E100~E500）。

统一异常基类 + 领域细分，便于调用方按 code 做降级或告警。
"""

from __future__ import annotations


class ClusterForgeError(Exception):
    """所有 ClusterForge 异常的基类。"""

    code = "E000"
    message = "ClusterForge 内部错误"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(f"[{self.code}] {self.message}{(': ' + detail) if detail else ''}")


class DataError(ClusterForgeError):
    """数据层错误（E100~E199）。"""

    code = "E100"
    message = "数据不符合契约"


class ConfigError(ClusterForgeError):
    """配置层错误（E200~E299）。"""

    code = "E200"
    message = "配置非法"


class BackendUnavailable(ClusterForgeError):
    """可选后端不可用（E300~E399）。属预期降级，不应抛到用户。"""

    code = "E300"
    message = "可选后端不可用，已降级"


class NumericalError(ClusterForgeError):
    """数值不收敛 / 病态（E400~E499）。"""

    code = "E400"
    message = "数值计算异常"


class PipelineError(ClusterForgeError):
    """流水线编排错误（E500~E599）。"""

    code = "E500"
    message = "流水线编排错误"
