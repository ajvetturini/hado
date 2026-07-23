from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

"""
These are configurations used for various steps of the hado automation pipeline and are meant to just be a conveinent
way to manager variety of parameters used across algorithms in this pipeline. OVerall, the PipelineConfig is the 
main configuration object that is used to manage all of these parameters, but generally you will not need to really
interact with these unless you are doing specific algorithmic development / debugging.
"""


@dataclass(frozen=True)
class PipelineControlConfig:
    skip_mitering: bool = False
    skip_stapling: bool = False
    return_diagnostics: bool = False
    prepare_exports: bool = True


@dataclass(frozen=True)
class BaseDesignConfig:
    max_cross_section_size_override: bool = False


@dataclass(frozen=True)
class ConnectionOptimizationConfig:
    max_optimize_connections: int = 100
    max_rotation_iterations: int = 250
    animation_frequency: int = 10
    angle_step_size: int = 15
    max_scaf_path_iterations_hollowframe: int = 100
    min_hungarian_threshold: float = 0.2
    max_hungarian_retries: int = 25
    connections_only: bool = False


@dataclass(frozen=True)
class AutostapleConfig:
    override_staple_autobreak_limit: bool = False
    base_staple_length_to_simple_break: int = 500
    show_breakpoint_labels: bool = False


@dataclass(frozen=True)
class ScaffoldDecodingConfig:
    max_scaffold_decoding_iterations: int = 1000


@dataclass(frozen=True)
class VisualizationConfig:
    show_best_state_animation: bool = False
    miter_show_init_connections: bool = False
    miter_show_post_mitering_positions: bool = False
    show_miter_colors: bool = False
    cylinder_radius: float = 0.5
    cylinder_res: int = 10
    flip_init_connection_cylinders: bool = False
    flip_post_mitering_cylinders: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    control: PipelineControlConfig = field(default_factory=PipelineControlConfig)
    base_design: BaseDesignConfig = field(default_factory=BaseDesignConfig)
    connection_optimization: ConnectionOptimizationConfig = field(default_factory=ConnectionOptimizationConfig)
    autostaple: AutostapleConfig = field(default_factory=AutostapleConfig)
    scaffold_decoding: ScaffoldDecodingConfig = field(default_factory=ScaffoldDecodingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> "PipelineConfig":
        remaining = dict(kwargs)
        return cls(
            control=_build_config(PipelineControlConfig, remaining),
            base_design=_build_config(BaseDesignConfig, remaining),
            connection_optimization=_build_config(ConnectionOptimizationConfig, remaining),
            autostaple=_build_config(AutostapleConfig, remaining),
            scaffold_decoding=_build_config(ScaffoldDecodingConfig, remaining),
            visualization=_build_config(VisualizationConfig, remaining),
            extra=remaining,
        )

    def to_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for config in (
            self.control,
            self.base_design,
            self.connection_optimization,
            self.autostaple,
            self.scaffold_decoding,
            self.visualization,
        ):
            kwargs.update(asdict(config))
        kwargs.update(self.extra)
        return kwargs


def _build_config(config_type, kwargs: dict[str, Any]):
    field_names = config_type.__dataclass_fields__
    values = {name: kwargs.pop(name) for name in tuple(field_names) if name in kwargs}
    return config_type(**values)
