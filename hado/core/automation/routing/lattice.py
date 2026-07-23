from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class LatticeConfig:
    """
    Geometric parameterization of a lattice type used for DNA origami automation. Lattice definitions provide
    periodic scaffold and staple crossover offsets used throughout the hollowframe pipeline.
    """

    name: str
    grid_type: str
    axial_rise: float
    diameter: float
    bases_per_turn: float
    minor_groove_ang: float
    inter_helix_gap: float
    period: int
    init_offset: float
    scaffold_xovers: tuple
    staple_xovers: tuple
    forward_angle_to_index: tuple[tuple[float, int], ...]
    reverse_angle_to_index: tuple[tuple[float, int], ...]

    @property
    def helix_spacing(self) -> float:
        return self.diameter + self.inter_helix_gap

    @property
    def twist_per_base(self) -> float:
        turns_per_rotation = self.period / self.bases_per_turn
        return 360 / (self.period / turns_per_rotation)

    def as_model_attributes(self) -> dict[str, Any]:
        """Return legacy HadoNucleotideModel private attributes."""
        scaffold_xovers = [
            [tuple(crossover) for crossover in crossover_group]
            for crossover_group in self.scaffold_xovers
        ]
        staple_xovers = [tuple(crossover) for crossover in self.staple_xovers]

        return {
            "_grid_type": self.grid_type,
            "_axial_rise": self.axial_rise,
            "_diameter": self.diameter,
            "_bases_per_turn": self.bases_per_turn,
            "_minor_groove_ang": self.minor_groove_ang,
            "_inter_helix_gap": self.inter_helix_gap,
            "_period": self.period,
            "_helix_spacing": self.helix_spacing,
            "_twist_per_base": self.twist_per_base,
            "_init_offset": self.init_offset,
            "_scaf_xovers": scaffold_xovers,
            "_stap_xovers": staple_xovers,
        }

    def get_crossover_offsets(self, direction: bool, theta: float, tolerance: float):
        angle_to_index = self.forward_angle_to_index if direction else self.reverse_angle_to_index
        for angle, crossover_index in angle_to_index:
            if np.isclose(theta, angle, atol=tolerance):
                return self.scaffold_xovers[crossover_index], self.staple_xovers[crossover_index]
        raise ValueError(...)


_LATTICE_CONFIGS = {
    "dna_honeycomb": LatticeConfig(
        name="dna_honeycomb",
        grid_type="honeycomb",
        axial_rise=0.34,
        diameter=2.25,
        bases_per_turn=10.5,
        minor_groove_ang=150.0,
        inter_helix_gap=1.00,
        period=21,
        init_offset=0.0,
        scaffold_xovers=(
            ((1, 11), (2, 12)),
            ((8, 18), (9, 19)),
            ((4, 15), (5, 16)),
        ),
        staple_xovers=((6, 7), (13, 14), (20, 0)),
        forward_angle_to_index=((330.0, 0), (90.0, 1), (210.0, 2)),
        reverse_angle_to_index=((150.0, 0), (270.0, 1), (30.0, 2)),
    ),
    "dna_square": LatticeConfig(
        name="dna_square",
        grid_type="square",
        axial_rise=0.34,
        diameter=2.25,
        bases_per_turn=10.66,
        minor_groove_ang=150.0,
        inter_helix_gap=1.00,
        period=32,
        init_offset=180.0 + (360 / 10.66) / 2,  # ~196.8
        scaffold_xovers=(
            ((4, 26, 15), (5, 27, 16)),
            ((18, 28, 7), (19, 29, 8)),
            ((10, 20, 31), (11, 21, 0)),
            ((2, 12, 23), (3, 13, 24)),
        ),
        staple_xovers=((31, 0), (23, 24), (15, 16), (7, 8)),
        forward_angle_to_index=((0.0, 0), (270.0, 3), (180.0, 2), (90.0, 1)),
        reverse_angle_to_index=((180.0, 0), (90.0, 3), (0.0, 2), (270.0, 1)),
    ),
}

SUPPORTED_LATTICE_TYPES = tuple(_LATTICE_CONFIGS)


def get_lattice_config(lattice_type: str) -> LatticeConfig:
    """Return the lattice configuration matching a scaffold lattice type."""
    lattice_key = lattice_type.lower()
    try:
        return _LATTICE_CONFIGS[lattice_key]
    except KeyError as exc:
        supported = '", "'.join(SUPPORTED_LATTICE_TYPES)
        raise ValueError(
            f'Unsupported lattice_type specified in scaffold_args, valid options are "{supported}".'
        ) from exc
