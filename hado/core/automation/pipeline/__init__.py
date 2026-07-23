from typing import TYPE_CHECKING
from hado.core.automation.pipeline.config import PipelineConfig
from hado.core.automation.pipeline.types import (
    AutostapleResult,
    BaseDesignResult,
    ConnectionOptimizationResult,
    MiteringResult,
    PipelineDiagnostics,
    ScaffoldRoutingResult,
    emit_runtime_message,
)

if TYPE_CHECKING:
    from hado.core.automation.pipeline.manager import HadoManager

__all__ = [
    "AutostapleResult",
    "BaseDesignResult",
    "ConnectionOptimizationResult",
    "HadoManager",
    "MiteringResult",
    "PipelineConfig",
    "PipelineDiagnostics",
    "ScaffoldRoutingResult",
    "emit_runtime_message",
]

def __getattr__(name):
    if name == "HadoManager":
        from hado.core.automation.pipeline.manager import HadoManager
        return HadoManager
    raise AttributeError(name)
