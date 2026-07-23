from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hado.core.automation.model.nucleotide_model import HadoNucleotideModel


@dataclass
class PipelineDiagnostics:
    stage_results: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    info_messages: list = field(default_factory=list)
    figures: dict = field(default_factory=dict)

    def record(self, stage: str, **values):
        self.stage_results[stage] = values

    def warn(self, message: str):
        self.warnings.append(message)

    def info(self, message: str):
        self.info_messages.append(message)

    def record_figure(self, stage: str, name: str, figure):
        self.figures.setdefault(stage, {})[name] = figure


def emit_runtime_message(
    message: str,
    *,
    diagnostics: PipelineDiagnostics | None = None,
    verbose: bool = False,
    warning: bool = False,
):
    """Record a runtime message and optionally render it in verbose mode."""
    if diagnostics is not None:
        if warning:
            diagnostics.warn(message)
        else:
            diagnostics.info(message)
    if verbose:
        print(message)


@dataclass(frozen=True)
class ScaffoldRoutingResult:
    edge_xsect_definitions: dict
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaseDesignResult:
    design: HadoNucleotideModel
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionOptimizationResult:
    optimal_connections: dict
    best_state: dict
    scaffold_helix_connections: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MiteringResult:
    final_positions: dict
    final_nts: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AutostapleResult:
    design: HadoNucleotideModel
    autobreak_cost: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
