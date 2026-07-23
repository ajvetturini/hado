from __future__ import annotations

import networkx as nx
import numpy as np


def fix_leading_zero_index(crossover_map, period):
    """Align wrapped crossover maps whose first index is zero."""
    if crossover_map[0] == 0:
        crossover_map = np.roll(crossover_map, -1)
        crossover_map[-1] = crossover_map[-2] + period
    return crossover_map


def define_xover_maps(model, state: dict | None = None, num_edges: int | None = None, lattice_config=None):
    """Define allowed scaffold and staple crossover indices."""
    data = _ModelData(model, state)
    scaffold_xover_map, staple_xover_map = {}, {}
    if len(data.grid_locations) != len(data.scaffold_dirs):
        raise RuntimeError("ERROR: incorrect grid specification")
    _, num_nts = data.scaffold_nucleotides.shape

    cur_min_idx = 0
    for bundle_index in range(num_edges if num_edges is not None else len(model.edge_xsect_definitions)):
        indices_i = [j for j, bundle in enumerate(data.helix_to_bundle) if bundle == bundle_index]
        grid_locs_i = data.grid_locations[indices_i]
        scaf_dirs_i = data.scaffold_dirs[indices_i]
        grid_graph = _create_grid_graph(grid_locs_i, data.helix_spacing)

        for helix_index in range(len(indices_i)):
            for neighbor_index in list(grid_graph.neighbors(helix_index)):
                global_helix = int(cur_min_idx + helix_index)
                global_neighbor = int(cur_min_idx + neighbor_index)

                if (global_helix, global_neighbor) in scaffold_xover_map:
                    continue

                scaf_map_both, stap_map_both = _check_grid_vectors(
                    data,
                    grid_locs_i,
                    scaf_dirs_i,
                    helix_index,
                    neighbor_index,
                    num_nts,
                    global_helix,
                    global_neighbor,
                    lattice_config,
                )
                scaf_jn, scaf_nj = scaf_map_both
                stap_jn, stap_nj = stap_map_both

                scaffold_xover_map[(global_helix, global_neighbor)] = scaf_jn
                scaffold_xover_map[(global_neighbor, global_helix)] = scaf_nj
                staple_xover_map[(global_helix, global_neighbor)] = stap_jn
                staple_xover_map[(global_neighbor, global_helix)] = stap_nj
        cur_min_idx += len(indices_i)
    return scaffold_xover_map, staple_xover_map


def get_nearest_scaffold_crossover_index(model, helix_from, helix_to, idx):
    """Find the nearest valid scaffold crossover index."""
    xover_ft = model._scaffold_xover_map[(int(helix_from), int(helix_to))]
    valid_indices = np.where(xover_ft != -1)[0]
    scaffold_dir = model._scaffold_dirs[helix_from]

    diffs = np.abs(valid_indices - idx)
    min_diff_idx = np.argmin(diffs)
    correct_nt_from_to = int(valid_indices[min_diff_idx])

    if valid_indices[(min_diff_idx + 1) % len(valid_indices)] - correct_nt_from_to == 1:
        correct_nt_to_from = int(correct_nt_from_to + 1)
    else:
        correct_nt_to_from = int(correct_nt_from_to - 1)

    if scaffold_dir:
        return min(correct_nt_from_to, correct_nt_to_from)
    return max(correct_nt_from_to, correct_nt_to_from)


def get_all_scaffold_crossover_options(model, helix_from, helix_to):
    """Return all valid scaffold crossover positions between two helices."""
    xover_ft = np.array(model._scaffold_xover_map[(int(helix_from), int(helix_to))])
    valid_indices = np.where(xover_ft != -1)[0]
    if model._scaffold_dirs[helix_from]:
        return valid_indices[::2]
    return valid_indices[1::2]


def get_all_valid_scaffold_crossover_options(model, helix_from, helix_to):
    """Return active scaffold crossover options as paired nucleotide indices."""

    def _is_valid(helix, nt):
        active_nts = np.where(model._scaffold_nucleotides[helix])[0]
        if nt not in active_nts:
            return False
        min_run_post_bundle_connection = model.staple_args.min_run_post_bundle_connection
        first_nt, last_nt = active_nts[0], active_nts[-1]
        if abs(first_nt - nt) < min_run_post_bundle_connection:
            return False
        if abs(last_nt - nt) < min_run_post_bundle_connection:
            return False
        return True

    positions1 = get_all_scaffold_crossover_options(model, helix_from, helix_to)
    positions2 = get_all_scaffold_crossover_options(model, helix_to, helix_from)
    h1toh2 = fix_leading_zero_index(positions1, model.get_period())
    h2toh1 = fix_leading_zero_index(positions2, model.get_period())

    valid_options = set()
    for i, j in zip(h1toh2, h2toh1):
        if _is_valid(helix_from, i) and _is_valid(helix_to, j):
            valid_options.add((i, j))
    return valid_options


def replace_scaffold_positions(model, h1, h2, oldnt1, oldnt2, newnt1, newnt2):
    """Update scaffold crossover positions in place."""
    min_old_nt, max_old_nt = min(oldnt1, oldnt2), max(oldnt1, oldnt2)
    min_new_nt, max_new_nt = min(newnt1, newnt2), max(newnt1, newnt2)
    match1 = (h1, min_old_nt, h2, -1)
    match2 = (h2, min_old_nt, h1, -1)
    match3 = (h1, max_old_nt, h2, -1)
    match4 = (h2, max_old_nt, h1, -1)
    found_min, found_max = False, False

    for i, crossover in enumerate(model._scaffold_crossovers):
        current_row = tuple(crossover)
        if current_row == match1:
            model._scaffold_crossovers[i] = np.array([h1, min_new_nt, h2, -1])
            found_min = True
        elif current_row == match2:
            model._scaffold_crossovers[i] = np.array([h2, min_new_nt, h1, -1])
            found_min = True
        elif current_row == match3:
            model._scaffold_crossovers[i] = np.array([h1, max_new_nt, h2, -1])
            found_max = True
        elif current_row == match4:
            model._scaffold_crossovers[i] = np.array([h2, max_new_nt, h1, -1])
            found_max = True

    if not found_min or not found_max:
        raise Exception("ERROR: The crossovers between h1 and h2 are not found in the scaffold crossovers")


def get_all_staple_crossover_options(model, helix_from, helix_to):
    """Return all valid staple crossover positions between two helices."""
    xover_ft = np.array(model._staple_xover_map[(int(helix_from), int(helix_to))])
    valid_indices = np.where(xover_ft != -1)[0]
    if model._staple_dirs[helix_from]:
        return valid_indices[::2]
    return valid_indices[1::2]


def _check_grid_vectors(data, grid_locs, grid_dirs, j, n, num_nts, global_j, global_n, lattice_config, tolerance=1e-5):
    blank_scaf_map1 = np.full(num_nts, -1, dtype=np.int32)
    blank_scaf_map2 = np.full(num_nts, -1, dtype=np.int32)
    blank_stap_map1 = np.full(num_nts, -1, dtype=np.int32)
    blank_stap_map2 = np.full(num_nts, -1, dtype=np.int32)
    grid_j, grid_n = grid_locs[j], grid_locs[n]
    jn = (grid_n - grid_j) / np.linalg.norm(grid_n - grid_j)
    theta = np.degrees(np.arctan2(jn[1], jn[0]))

    if theta < 0:
        theta += 360

    scaf_xo, stap_xo = lattice_config.get_crossover_offsets(grid_dirs[j], theta, tolerance)
    num_periods = num_nts // data.period
    for period_index in range(num_periods):
        base_period_offset = period_index * data.period
        for scaffold_offsets in scaf_xo:
            for offset in list(scaffold_offsets):
                idx = (offset + base_period_offset) % num_nts
                blank_scaf_map1[idx] = global_n
                blank_scaf_map2[idx] = global_j

        for offset in list(stap_xo):
            idx = (offset + base_period_offset) % num_nts
            blank_stap_map1[idx] = global_n
            blank_stap_map2[idx] = global_j

    return (blank_scaf_map1, blank_scaf_map2), (blank_stap_map1, blank_stap_map2)


# def _select_crossover_offsets(data, direction, theta, tolerance):
#     if data.grid_type == "honeycomb":
#         return _select_honeycomb_offsets(data, direction, theta, tolerance)
#     if data.grid_type == "square":
#         return _select_square_offsets(data, direction, theta, tolerance)
#     raise ValueError("ERROR: Invalid lattice type.")


# def _select_honeycomb_offsets(data, direction, theta, tolerance):
#     if direction:
#         angle_to_index = ((330.0, 0), (90.0, 1), (210.0, 2))
#     else:
#         angle_to_index = ((150.0, 0), (270.0, 1), (30.0, 2))
#     return _lookup_offsets(data, angle_to_index, theta, tolerance, "honeycomb")


# def _select_square_offsets(data, direction, theta, tolerance):
#     # This is a bit different due to initial offset presumed by square matrix
#     if direction:
#         angle_to_index = ((0.0, 0), (270.0, 3), (180.0, 2), (90.0, 1))
#     else:
#         angle_to_index = ((180.0, 0), (90.0, 3), (0.0, 2), (270.0, 1))
#     return _lookup_offsets(data, angle_to_index, theta, tolerance, "square")


# def _lookup_offsets(data, angle_to_index, theta, tolerance, lattice_name):
#     for angle, crossover_index in angle_to_index:
#         if np.isclose(theta, angle, atol=tolerance):
#             return data.scaf_xovers[crossover_index], data.stap_xovers[crossover_index]
#     raise ValueError(f"ERROR: Invalid helix direction / grid vector angle found for {lattice_name}.")


def _create_grid_graph(grid_locs, helix_spacing: float, tolerance=1e-5):
    graph = nx.Graph()
    num_nodes = grid_locs.shape[0]

    for i in range(num_nodes):
        graph.add_node(i, pos=tuple(grid_locs[i]))

    for i in range(num_nodes):
        current_pt = grid_locs[i]
        remaining_pts = grid_locs[i + 1 :]
        dists = np.linalg.norm(remaining_pts - current_pt, axis=1)
        match_indices = np.where(np.isclose(dists, helix_spacing, atol=tolerance))[0]
        for match_idx in match_indices:
            graph.add_edge(i, i + 1 + match_idx)

    return graph


class _ModelData:
    def __init__(self, model, state: dict | None = None):
        state = state or {}
        self.grid_locations = _state_or_model(state, model, "_grid_locations")
        self.scaffold_dirs = _state_or_model(state, model, "_scaffold_dirs")
        self.scaffold_nucleotides = _state_or_model(state, model, "_scaffold_nucleotides")
        self.helix_to_bundle = _state_or_model(state, model, "_helix_to_bundle")
        self.helix_spacing = _state_or_model(state, model, "_helix_spacing")
        self.grid_type = _state_or_model(state, model, "_grid_type")
        self.scaf_xovers = _state_or_model(state, model, "_scaf_xovers")
        self.stap_xovers = _state_or_model(state, model, "_stap_xovers")
        self.period = _state_or_model(state, model, "_period")


def _state_or_model(state: dict, model, attr: str):
    if attr in state:
        return state[attr]
    return getattr(model, attr)