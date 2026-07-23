from __future__ import annotations

import numpy as np

from hado.core.automation.routing.cross_sections import get_cross_section
from hado.core.automation.routing.lattice import get_lattice_config
from hado.core.automation.model.crossover_maps import define_xover_maps
from hado.core.utils import get_color_palette


def build_initial_model_state(model, max_cross_section_size_override: bool = False, diagnostics=None) -> dict:
    """Build the initial private state for a nucleotide-level model."""
    lattice_config = get_lattice_config(model.scaffold_args.lattice_type)
    state = lattice_config.as_model_attributes()
    largest_edge_length, largest_xsect = _get_model_extents(model, state["_axial_rise"], state["_period"])

    (
        scaffold_nucleotides, grid_positions, grid_directions, helix_to_bundle
    ) = _populate_edge_helices(model, state, largest_edge_length, largest_xsect, max_cross_section_size_override,
                               diagnostics)

    num_edges = len(model.edge_xsect_definitions)
    populated_helices = np.array([helix for edge in scaffold_nucleotides for helix in edge if np.any(helix)])
    state.update(
        {
            "_scaffold_nucleotides": populated_helices.copy(),
            "_staple_nucleotides": populated_helices.copy(),
            "_helix_to_bundle": np.array(helix_to_bundle),
            "_grid_locations": np.array(grid_positions),
            "_scaffold_dirs": np.array(grid_directions, dtype=np.bool_),
            "_staple_color_palette": get_color_palette(num_edges),
            "_staple_dirs": np.logical_not(np.array(grid_directions)),
            "_scaffold_crossovers": [],
            "_staple_crossovers": [],
            "_scaffold_start_point": [],
            "_staple_breaks": [],
            "_bundle_rotations": [],
            "_idx_edge_map": np.array(model.geometry.edges.copy()),
        }
    )

    state["_scaffold_xover_map"], state["_staple_xover_map"] = define_xover_maps(model, state,
                                                                                 num_edges, lattice_config)
    return state


def _get_model_extents(model, axial_rise: float, period: int) -> tuple[int, int]:
    largest_edge_length, largest_xsect = 0, 0
    for edge, xsect in model.edge_xsect_definitions.items():
        m_count, n_count = xsect["M"], xsect["N"]
        largest_xsect = max(int(largest_xsect), int(m_count + n_count))

        vi = model.geometry.get_vertex_position(edge[0])
        vj = model.geometry.get_vertex_position(edge[1])
        edge_length = np.linalg.norm(vj - vi)
        num_nucleotides = edge_length // axial_rise
        if num_nucleotides % period != 0:
            num_nucleotides += period - (num_nucleotides % period)
        num_nucleotides += 2 * period
        largest_edge_length = max(int(num_nucleotides), largest_edge_length)
    return largest_edge_length, largest_xsect


def _populate_edge_helices(model, state: dict, largest_edge_length: int, largest_xsect: int,
                           max_cross_section_size_override: bool, diagnostics):
    scaffold_nucleotides = []
    grid_positions = []
    grid_directions = []
    helix_to_bundle = []
    grid_reuse = {}

    for bundle_index, (edge, xsect) in enumerate(model.edge_xsect_definitions.items()):
        m_count, n_count = xsect["M"], xsect["N"]
        vi = model.geometry.get_vertex_position(edge[0])
        vj = model.geometry.get_vertex_position(edge[1])
        edge_length = np.linalg.norm(vj - vi)
        start_length_nts = int(edge_length // state["_axial_rise"])
        active_nts = np.ones(start_length_nts, dtype=np.bool_)

        offset = (largest_edge_length - start_length_nts) // 2
        start_location = offset
        final_location = start_location + start_length_nts

        if (m_count, n_count) not in grid_reuse:
            grid_reuse[(m_count, n_count)] = get_cross_section(
                m_count,
                n_count,
                state["_helix_spacing"],
                lattice_type=state["_grid_type"],
                grid_style=model.scaffold_args.grid_style,
                override=max_cross_section_size_override,
                diagnostics=diagnostics,
                custom_cross_section=model.scaffold_args.custom_cross_section,
                cross_section_generator=model.scaffold_args.cross_section_generator,
            )
        grids_53, grids_35 = grid_reuse[(m_count, n_count)]

        current_helix = 0
        init_array = np.zeros((largest_xsect, largest_edge_length), dtype=np.bool_)
        for helix_index in range(m_count):
            grid_positions.append(list(grids_53[helix_index]))
            init_array[current_helix, start_location:final_location] = active_nts
            grid_directions.append(True)
            helix_to_bundle.append(bundle_index)
            current_helix += 1

        for helix_index in range(n_count):
            grid_positions.append(list(grids_35[helix_index]))
            init_array[current_helix, start_location:final_location] = active_nts
            grid_directions.append(False)
            helix_to_bundle.append(bundle_index)
            current_helix += 1

        scaffold_nucleotides.append(init_array)

    return scaffold_nucleotides, grid_positions, grid_directions, helix_to_bundle
