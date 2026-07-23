"""Automation pipeline package."""

from hado.core.automation.model import HadoNucleotideModel, initialize_base_design
from hado.core.automation.pipeline import HadoManager, PipelineConfig, PipelineDiagnostics

__all__ = [
    "HadoManager",
    "HadoNucleotideModel",
    "PipelineConfig",
    "PipelineDiagnostics",
    "initialize_base_design",
]
