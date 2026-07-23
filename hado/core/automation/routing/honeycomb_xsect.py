import numpy as np
import networkx as nx
from hado.core.automation.diagnostics.visualization import (
    build_cross_section_points_figure,
    build_honeycomb_grid_figure,
    build_honeycomb_symmetry_graph_figure,
)
from itertools import product, combinations
from typing import Tuple
from hado.core.utils import MAX_NODES_FOR_CROSS_SECTION_SEARCH

MAX_PATH_LIST_SIZE = 1000
DIAMETER_SEARCH_BRACKET_LOOKAHEAD = 3


def get_honeycomb_ring_diameter(evens: list, odds: list, helix_diameter: float = 0.0) -> float:
    """Return the outer diameter of a honeycomb cross-section."""
    points = np.array(evens + odds, dtype=float)
    if len(points) == 0:
        raise ValueError('ERROR: Cross-section must contain at least one helix.')
    if helix_diameter < 0:
        raise ValueError('ERROR: Helix diameter must be non-negative.')
    if len(points) == 1:
        return float(helix_diameter)

    diffs = points[:, None, :] - points[None, :, :]
    center_to_center_diameter = np.sqrt(np.sum(diffs ** 2, axis=-1)).max()
    return float(center_to_center_diameter + helix_diameter)


def _estimate_honeycomb_diameter_search_start(target_diameter: float,
                                               L: float,
                                               helix_diameter: float,
                                               min_total_helices: int,
                                               max_total_helices: int) -> int:
    """Return the first even helix count worth checking for a diameter search."""
    center_to_center_target = max(float(target_diameter) - float(helix_diameter), 0.0)
    if center_to_center_target <= 0:
        return min_total_helices

    required_graph_edges = max(1, int(np.ceil(center_to_center_target / L)))
    start_total = 2 * max(required_graph_edges - 1, 1)
    if start_total % 2 != 0:
        start_total += 1

    return min(max(start_total, min_total_helices), max_total_helices)


def select_honeycomb_ring_by_diameter(target_diameter: float,
                                      L: float,
                                      helix_diameter: float = 0.0,
                                      min_total_helices: int = 2,
                                      max_total_helices: int = MAX_NODES_FOR_CROSS_SECTION_SEARCH,
                                      override: bool = False,
                                      diagnostics=None,
                                      ) -> dict:
    """
    Select the even-M/N hollow honeycomb ring whose outer diameter is closest to the target diameter.

    The returned n_per_edge is always even because the standard hollowframe routing path can use even counts directly
    on each edge without invoking the odd-count cycle heuristic.
    """
    if target_diameter <= 0:
        raise ValueError('ERROR: Target honeycomb-ring diameter must be positive.')
    if L <= 0:
        raise ValueError('ERROR: Length between helices must be positive and larger than 0')
    if min_total_helices < 2:
        raise ValueError('ERROR: Minimum total helices must be at least 2.')
    if max_total_helices < min_total_helices:
        raise ValueError('ERROR: Maximum total helices must be at least the minimum total helices.')

    if not override:
        max_total_helices = min(max_total_helices, MAX_NODES_FOR_CROSS_SECTION_SEARCH - 1)

    if min_total_helices % 2 != 0:
        min_total_helices += 1
    if max_total_helices % 2 != 0:
        max_total_helices -= 1

    min_total_helices = _estimate_honeycomb_diameter_search_start(
        target_diameter,
        L,
        helix_diameter,
        min_total_helices,
        max_total_helices,
    )

    best = None
    bracket_lookahead_remaining = None
    for total_helices in range(min_total_helices, max_total_helices + 1, 2):
        M = N = total_helices // 2
        evens, odds, metadata = set_hollow_honeycomb(M, N, L, override=override, diagnostics=diagnostics)
        actual_diameter = get_honeycomb_ring_diameter(evens, odds, helix_diameter)
        candidate = {
            "M": M,
            "N": N,
            "n_per_edge": total_helices,
            "target_diameter": float(target_diameter),
            "actual_diameter": actual_diameter,
            "evens": evens,
            "odds": odds,
            "metadata": metadata,
        }

        if best is None:
            best = candidate
        else:
            best_error = abs(best["actual_diameter"] - target_diameter)
            candidate_error = abs(candidate["actual_diameter"] - target_diameter)
            if (
                candidate_error < best_error
                or (
                    np.isclose(candidate_error, best_error, rtol=1e-12, atol=1e-9)

                    # Place preference on the more symmetric cross-sections which
                    # can be simply checked using below
                    and candidate["n_per_edge"] > best["n_per_edge"]
                )
            ):
                best = candidate

        if bracket_lookahead_remaining is not None:
            if bracket_lookahead_remaining <= 0:
                break
            bracket_lookahead_remaining -= 1
        elif actual_diameter >= target_diameter:
            bracket_lookahead_remaining = DIAMETER_SEARCH_BRACKET_LOOKAHEAD - 1

    if best is None:
        raise ValueError('ERROR: No honeycomb-ring cross-section candidates were generated.')
    return best


def get_cross_section(M: int,
                      N: int,
                      L: float,
                      lattice_type: str = 'honeycomb',
                      grid_style: str ='hollow',
                      override: bool = False,
                      diagnostics=None,
                      ) -> Tuple[list, list]:
    """
    Uses a honeycomb grid to determine the cross-section for an initialized cadnano model. This feature (by default)
    uses the maximal enclose-cross section area as the selected cross-section design. Overall, this function leverages
    symmetry alongside depth-first-search to populate potential cross sections that contain M "even" (or 5' to 3'
    helices) alongside N "odd" (or 3' to 5' helices). For hollowframe (and in general) it should be best practice to
    keep M == N.

    Eventually, it may make sense to couple this function to `optimize_connections` for the periodic assembly of
    the hollowframe structure to actually inform the cross-section design for colloidal assembly of complex materials.

    :param M: Total integer number of helices that run 5' to 3' in a given helix bundle.
    :type M: int

    :param N: Total integer number of helices that run 3' to 5' in a given helix bundle.
    :type N: int

    :param L: The center-to-center distance between two neighboring helices. This value is the scaffold diameter added
        to the inter-helix spacing gap.
    :type L: float

    :param lattice_type: The type of lattice configuration (e.g., 'honeycomb').
        Defaults to 'honeycomb' and currently only `honeycomb` is supported.
    :type lattice_type: str, optional

    :param grid_style: The style of the internal grid lattice (e.g., 'hollow').
        Defaults to 'hollow' and currently only `hollow` is supported.
    :type grid_style: str, optional

    :return: A tuple containing three elements:

        * **design** (*GlobalModel*): Final design model containing scaffold, staple, crossover, break point details.
        * **sequence_strings** (*list*): The scaffold and staple sequences that will be written to a CSV file
        * **staple_colors** (*dict*): Currently unused, but the dictionary can color the output caDNAno staples
    :rtype: tuple
    """
    if grid_style.lower() not in ['hollow']:
        raise ValueError('ERROR: helix_bundle_style can only be `hollow` currently.')
    if abs(M - N) > 2:
        raise ValueError('ERROR: M and N must be within 2 helix of each other.')

    if grid_style == 'hollow':  # Default == hollow_frame, the below _set_hollow works for all M / N
        selected_even, selected_odd, _ = _set_hollow(M, N, L, lattice_type, override, diagnostics=diagnostics)

    else:
        raise NotImplementedError('Only hollow grid_style is currently implemented.')

    return selected_even, selected_odd


def convert_stacked_grids_to_planar(grids, honeycomb_mapping, precision=5, col_buffer=3, row_buffer=2):
    """ Converts the helix bundle cross-sections (which are all stacked on eachother) and properly allocates them in
    honeycomb grid positions for caDNAno export """
    max_column = 30
    cur_column, cur_row, local_max_height = 0, 0, 0  # Initial

    grid_data = {}
    for gct, g in enumerate(grids):
        temp = []
        temp2 = []
        for grid_member in g:
            gm = np.round(grid_member, decimals=precision)
            temp.append(honeycomb_mapping[tuple(gm)])
            temp2.append(gm)

        temp = np.array(temp)
        temp2 = np.array(temp2)
        min_c, max_c = np.min(temp[:, 0]), np.max(temp[:, 0])
        min_r, max_r = np.min(temp[:, 1]), np.max(temp[:, 1])

        width_needed = (max_c - min_c) + col_buffer  # Add buffer between bundles
        height_needed = (max_r - min_r) + row_buffer

        # Store all helices within the needed bounds and given the cur row / col
        # values
        if cur_column + width_needed <= max_column:
            lower_bound = (cur_column, cur_row)
            upper_bound = (cur_column + width_needed, cur_row + height_needed)
            cur_column += (width_needed + 1)
            if height_needed > local_max_height:
                local_max_height = height_needed

        else:
            cur_row += (local_max_height + row_buffer)
            local_max_height = height_needed
            lower_bound = (0, cur_row)
            upper_bound = (cur_column + width_needed, cur_row + height_needed)
            cur_column = width_needed + 1

        min_col, min_row = np.min(temp[:, 0]), np.min(temp[:, 1])

        if (lower_bound[0] % 2 == 0 and min_col % 2 != 0) or (lower_bound[0] % 2 != 0 and min_col % 2 == 0):
            lower_bound = (lower_bound[0] + 1, lower_bound[1])

        if (lower_bound[1] % 2 == 0 and min_row % 2 != 0) or (lower_bound[1] % 2 != 0 and min_row % 2 == 0):
            lower_bound = (lower_bound[0], lower_bound[1] + 1)

        bundle_mapping = {}
        for ct, gp in enumerate(temp):
            cadnano_row = lower_bound[0] + (gp[0] - min_col)
            cadnano_col = lower_bound[1] + (gp[1] - min_row)
            bundle_mapping[tuple(temp2[ct])] = (cadnano_row, cadnano_col)

        grid_data[gct] = bundle_mapping

    return grid_data


def get_honeycomb_mapping_cadnano(spacing_length, grids_to_fill, precision):
    """ Gets a mapping for a set of grid_positions for use in export to cadnano """
    L = spacing_length
    BIG_X = 512
    BIG_Y = 512
    a, b = _generate_honeycomb_grid(BIG_X, BIG_Y, L, visualize=False)

    temp = np.array(a + b)

    start_n = 10  # Just set this sufficiently high
    size_threshold = 15  # How large of a grid to section out
    while True:
        max_dist_away = start_n * L
        trimmed_grid = temp[np.linalg.norm(temp, axis=1) <= max_dist_away]

        all_unique_x = np.unique(trimmed_grid[:, 0])

        # Find which x has the largest number of points
        max_count, max_x = 0, []
        for x in all_unique_x:
            count = np.sum(trimmed_grid[:, 0] == x)
            if count == max_count:
                max_x.append(x)
            elif count > max_count:
                max_count = count
                max_x = [x]

        # For honeycomb, this should be at least 3 in length:
        if len(max_x) % 2 != 0 and len(max_x) > 1 and max_count >= size_threshold:
            break
        elif len(trimmed_grid) == len(temp):
            start_n = 2  # Set this smaller
            size_threshold = max_count
        else:
            start_n += 1

    middle_idx = int(len(max_x) / 2)
    first_col_pattern = trimmed_grid[trimmed_grid[:, 0] == max_x[middle_idx]]
    second_col_pattern = trimmed_grid[trimmed_grid[:, 0] == max_x[middle_idx + 1]]

    # If the max value in second_col_pattern[:, 1] > first_col_pattern[:, 1], flip:
    if np.max(second_col_pattern[:, 1]) > np.max(first_col_pattern[:, 1]):
        first_col_y = np.unique(second_col_pattern[:, 1])
        second_col_y = np.unique(first_col_pattern[:, 1])

    else:
        # Otherwise maintain:
        first_col_y = np.unique(first_col_pattern[:, 1])
        second_col_y = np.unique(second_col_pattern[:, 1])

    repeating_grid = []
    for ct, x in enumerate(np.unique(trimmed_grid[:, 0])):
        if ct % 2 == 0:
            col = [(x, y) for y in first_col_y]
        else:
            col = [(x, y) for y in second_col_y]
        repeating_grid.extend(col)

    # Below is a very large cadnano-like honeycomb grid:
    complete_grid = np.array(repeating_grid)
    honeycomb_mapping = _build_honeycomb_index_map(complete_grid, precision)
    correct_mapping = convert_stacked_grids_to_planar(grids_to_fill, honeycomb_mapping, precision)
    return correct_mapping

def _generate_honeycomb_grid(M, N, L, visualize=False, diagnostics=None):
    """
    Generates a honeycomb grid and finds the closest M "E" and N "O" vertices to the origin,
    in a clockwise order. E and O represent even-running or odd-running (i.e., 5' -> 3' or 3' -> 5', respectively)
    """
    hex_directions = [
        (0, 1), (-np.sqrt(3) / 2, 0.5), (-np.sqrt(3) / 2, -0.5),
        (0, -1), (np.sqrt(3) / 2, -0.5), (np.sqrt(3) / 2, 0.5)
    ]

    max_radius = 1

    while True:
        temp_e, temp_o = set(), set()
        check = set()
        num_hexagons = 3 * max_radius * (max_radius + 1) + 1

        if 3 * num_hexagons >= max(M, N):  # We need at least max(M, N) total vertices

            for q in range(-max_radius, max_radius + 1):
                for r in range(max(-max_radius, -q - max_radius), min(max_radius, -q + max_radius) + 1):
                    center_x = L * np.sqrt(3) * (r + q / 2)
                    center_y = L * (3 / 2) * q

                    for i, (dx, dy) in enumerate(hex_directions):
                        vertex_x = center_x + L * dx
                        vertex_y = center_y + L * dy

                        vertex = (round(vertex_x, 10), round(vertex_y, 10))

                        if vertex in check:
                            continue
                        if i % 2 == 0:
                            temp_e.add(vertex)
                        else:
                            temp_o.add(vertex)
                        check.add(vertex)

            temp_e = list(temp_e)
            temp_o = list(temp_o)

            def clockwise_angle(v):
                return (np.arctan2(v[1], v[0]) - np.pi / 2) % (2 * np.pi)

            temp_e = sorted(temp_e, key=clockwise_angle)
            temp_o = sorted(temp_o, key=clockwise_angle)

            temp_e = sorted(temp_e, key=lambda v: np.hypot(v[0], v[1]))
            temp_o = sorted(temp_o, key=lambda v: np.hypot(v[0], v[1]))

            if len(temp_e) >= M and len(temp_o) >= N:
                E_vertices = temp_e[:M]
                O_vertices = temp_o[:N]
                break

        max_radius += 1

    if visualize:
        fig = build_honeycomb_grid_figure(E_vertices, O_vertices)
        if diagnostics is not None:
            diagnostics.record_figure('base_design', 'honeycomb_grid', fig)

    return E_vertices, O_vertices


def _build_honeycomb_index_map(points, precision=5):
    points = np.round(points, decimals=precision)
    x = np.sort(np.unique(points[:, 0]))
    y = np.sort(np.unique(points[:, 1]))[::-1]  # Largest -> Smallest

    map_x = {val: i - 1 for i, val in enumerate(x)}

    map_y = {}
    cur_y, ct = -1, 0
    for val in y:
        map_y[val] = cur_y
        ct += 1
        if ct % 2 == 0:
            cur_y += 1

    xy_to_grid = {}
    for p in points:
        xy_to_grid[tuple(p)] = (map_x[p[0]], map_y[p[1]])
    return xy_to_grid


def _unique_swap_neg_pairs(pairs):
    """ Function that makes sure there are no duplicate cross-sections """
    seen = set()
    out = []
    for a, b in pairs:
        at = tuple(np.asarray(a).tolist())
        bt = tuple(np.asarray(b).tolist())

        neg_swapped = (tuple((-np.asarray(b)).tolist()), tuple((-np.asarray(a)).tolist()))

        # Choose canonical representation using lexicographic
        key = (at, bt)
        canonical = key if key <= neg_swapped else neg_swapped

        if canonical not in seen:
            seen.add(canonical)
            out.append((a, b))
    return out

def _get_honeycomb_symmetry_graph(M, N, L, visualize=False, diagnostics=None):
    # Generate sufficiently large grid
    graph = nx.Graph()
    e, o = _generate_honeycomb_grid(3*M, 3*N, L, False, diagnostics=diagnostics)
    e, o = np.array(e), np.array(o)

    # Sort by symmetry from 0-x
    e = e[e[:, 0] >= 0]
    o = o[o[:, 0] >= 0]
    e_tuples = [(float(p[0]), float(p[1])) for p in e]
    o_tuples = [(float(p[0]), float(p[1])) for p in o]
    adjacency_precision = 8
    o_lookup = {
        (round(point[0], adjacency_precision), round(point[1], adjacency_precision)): point
        for point in o_tuples
    }
    for node in e_tuples:
        graph.add_node(node, parity='even')
    for node in o_tuples:
        graph.add_node(node, parity='odd')

    neighbor_offsets = (
        (0.0, L),
        (0.0, -L),
        (np.sqrt(3) * L / 2, L / 2),
        (np.sqrt(3) * L / 2, -L / 2),
        (-np.sqrt(3) * L / 2, L / 2),
        (-np.sqrt(3) * L / 2, -L / 2),
    )
    for p in e_tuples:
        for dx, dy in neighbor_offsets:
            q_key = (round(p[0] + dx, adjacency_precision), round(p[1] + dy, adjacency_precision))
            q = o_lookup.get(q_key)
            if q is not None:
                graph.add_edge(p, q, weight=L)

    connect_from_e = e[(e[:, 1] > 0) & (e[:, 0] == 0)]
    connect_from_o = o[(o[:, 1] > 0) & (o[:, 0] == 0)]
    connect_to_e = e[(e[:, 1] < 0) & (e[:, 0] == 0)]
    connect_to_o = o[(o[:, 1] < 0) & (o[:, 0] == 0)]

    # We return all pairs of (connect_from_e + connect_from_o) to (connect_to_e + connect_to_o)
    cf = np.concat((connect_from_e, connect_from_o))
    ct = np.concat((connect_to_e, connect_to_o))
    pairs = list(product(cf, ct))

    # However, also make sure unique pairings:
    unique_pairs = _unique_swap_neg_pairs(pairs)

    for c in cf:
        graph.nodes[(float(c[0]), float(c[1]))]['start'] = True
    for c in ct:
        graph.nodes[(float(c[0]), float(c[1]))]['end'] = True

    if visualize:
        fig = build_honeycomb_symmetry_graph_figure(e, o, cf, ct)
        if diagnostics is not None:
            diagnostics.record_figure('base_design', 'honeycomb_symmetry_graph', fig)
    return graph, unique_pairs


def _find_path_optimized(graph, pairs, traversal_length, total_nodes, override):
    found_paths_set = set()
    seen_path_tuples = set()
    active_nodes = {n for n in graph.nodes if
                    (isinstance(n, np.ndarray) and n[0] != 0) or (not isinstance(n, np.ndarray) and n[0] != 0)}

    for p in pairs:
        p0 = (float(p[0][0]), float(p[0][1]))
        p1 = (float(p[1][0]), float(p[1][1]))
        tmp_p = (p0, p1)

        current_allowed = active_nodes | {p0, p1}
        temp_graph = graph.subgraph(current_allowed)

        if not nx.has_path(temp_graph, p0, p1):
            continue

        paths_gen = nx.all_simple_paths(temp_graph, p0, p1, cutoff=traversal_length - 1)

        for path in paths_gen:
            if len(path) == traversal_length:
                path_tuple = tuple((float(_p[0]), float(_p[1])) for _p in path)
                if path_tuple not in seen_path_tuples:
                    seen_path_tuples.add(path_tuple)
                    found_paths_set.add((tmp_p, path_tuple))

            if not override:
                if total_nodes >= MAX_NODES_FOR_CROSS_SECTION_SEARCH or \
                        len(found_paths_set) > MAX_PATH_LIST_SIZE:
                    break
        else:
            continue
        break
    return found_paths_set

def _find_path(graph, pairs, traversal_length, total_nodes, override):
    # Prior to sorting by area, we need to mirror the paths across their y-axis:
    found_paths = _find_path_optimized(graph, pairs, traversal_length, total_nodes, override)
    full_found_paths = []
    for carry in found_paths:
        prefix, path = carry
        base = list(path).copy()
        reversed = list(path)[::-1]
        trimmed = reversed[1:-1]  # Do not repeat
        for t in trimmed:
            base.append((-t[0], t[1]))
        full_found_paths.append((prefix, base))

    sorted_paths_area, sorted_areas, indices_stored = _sort_paths_by_area(full_found_paths)
    sorted_paths_sym, sorted_sym_scores = _sort_by_rotational_symmetry(full_found_paths, indices_stored)
    area_dict = {tuple(map(tuple, path)): area for path, area in zip(sorted_paths_area, sorted_areas)}
    sym_dict = {tuple(map(tuple, path)): sym for path, sym in zip(sorted_paths_sym, sorted_sym_scores)}

    combined_scores = {}
    scores_to_xsect = {}
    for i in indices_stored:
        _, path = full_found_paths[i]
        key = tuple(map(tuple, path))  # convert path to hashable type (tuple of tuples)
        combined_scores[key] = [area_dict[key], sym_dict[key]]
        scores_to_xsect[(area_dict[key], sym_dict[key])] = key

    combined_paths = list(combined_scores.keys())
    combined_scores_list = list(combined_scores.values())

    return combined_paths, combined_scores_list, scores_to_xsect

def _sort_paths_by_area(found_paths):
    """ Sorts all found paths by maximal enclosed area using shoelace formula """
    def _area(pts):
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        x.append(pts[0][0])  # CLose the loop
        y.append(pts[0][1])

        # Shoelace Formula
        area = 0.0
        for i in range(len(pts)):
          area += x[i] * y[i+1] - x[i+1] * y[i]
        return abs(area) * 0.5

    areas_and_paths = []
    indices_stored, found_areas = [], set()
    for ci, c in enumerate(found_paths):
        o = c[1]
        area = round(_area(o), 2)
        if area in found_areas: continue
        found_areas.add(area)
        indices_stored.append(ci)
        areas_and_paths.append((area, o))

    areas_and_paths.sort(key=lambda x: x[0], reverse=True)
    sorted_cycles = [c for A, c in areas_and_paths]
    sorted_areas = [A for A, c in areas_and_paths]
    return sorted_cycles, sorted_areas, indices_stored

def _rotational_symmetry_score(pts, max_order=15, tol=1e-3):
    """ Caclulates the k-fold rotational symmetry by rotating 2D points about their centroid and determining if there
    is overlap with the original, unrotated points.
    """
    pts = np.array(pts)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid

    best_order = 1
    for k in range(2, max_order+1):
        angle = 2*np.pi / k
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle),  np.cos(angle)]])
        pts_rot = pts_centered @ rot_matrix.T

        # Compute min distance from rotated points to original points
        dists = np.min(np.linalg.norm(pts_rot[:, None, :] - pts_centered[None, :, :], axis=2), axis=1)
        if np.max(dists) < tol:
            best_order = k
    return best_order

def _sort_by_rotational_symmetry(found_paths, indices_found):
    """ Sorts all found paths based on a rotation symmetry check """
    scored_paths = []
    for ci in indices_found:
        c = found_paths[ci]
        _, base = c

        # Base is already a  symmetric copy of the path across Y-axis
        base_score = _rotational_symmetry_score(base)
        scored_paths.append((base_score, base))

    scored_paths.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in scored_paths], [score for score, c in scored_paths]

def _find_paths_between_honeycomb_points(graph, pairs, N_even, total_nodes, override):
    while True:
        override = override
        sorted_paths, sorted_areas, score_to_xsect = _find_path(graph, pairs, N_even, total_nodes, override)
        if len(sorted_paths) > 0:
            return sorted_paths, sorted_areas, score_to_xsect
        elif N_even > total_nodes:
            return False, None, None
        else:
            N_even += 1

def _generate_honeycomb_compact(M, N, L, min_candidates=4):
    """ Used for generating the low-helix bundle (up to 12, non-inclusive) honeycomb grids.
    We do not need search / optimization here as there are very few combinations."""
    def _is_valid(pts):
        """ Checks all points in a subset are within L distance of at least one other point (and counts neighbors) """
        total_neighbors = 0
        graph = nx.Graph()
        graph.add_nodes_from(range(len(pts)))  # Used to ensure single connected component
        for i, p in enumerate(pts):
            neighbors = 0
            for j, q in enumerate(pts):
                if i == j:  # Skip self
                    continue
                if np.isclose(np.linalg.norm(p - q), L):
                    graph.add_edge(i, j)
                    neighbors += 1
            total_neighbors += neighbors

        for node in graph.nodes:
            if graph.degree[node] == 0:
                return 0, False  # Any unconnected nodes are invalid solutions
        if nx.number_connected_components(graph) != 1:
            return 0, False  # Scaffold must be able to reach any helix within a cross-section

        avg_neigh = total_neighbors / len(pts)
        return avg_neigh, True

    # The smallest cross-section is M=1 and N=1 (and scales up from there). Up until 12, there exists no "hollowframe"
    # design option as the helices must be placed more compactly (where at least one helix has 3 neighbors instead of 2)
    # NOTE: I do not handle a case where M=1 and N=0 (or vice-versa) as that will require more thoughtful stapling than
    #       the paradigms used in this work. For 1-helix designs, see vHelix (doi https://doi.org/10.1038/nature14586)
    e, o = _generate_honeycomb_grid(2 * M, 2 * N, L, False)
    e, o = np.array(e), np.array(o)  # Arrays of (X, Y) positions

    # Choose the M even and N o helices closest to the origin. Note that we want to maximize the number of neighbors
    # in this selection to ensure compactness
    e_sorted = e[np.argsort(np.sum(e ** 2, axis=1))]
    o_sorted = o[np.argsort(np.sum(o ** 2, axis=1))]
    e_candidates = e_sorted[: max(M, min_candidates)]
    o_candidates = o_sorted[: max(N, min_candidates)]

    potential_combinations = []
    scored_combinations = []
    scores_to_xsect = {}
    for e_sub in combinations(e_candidates, M):
        for o_sub in combinations(o_candidates, N):
            subset = np.array(list(e_sub) + list(o_sub))
            avg_neighbors, is_valid = _is_valid(subset)
            if is_valid:
                sorted_evens = _sort_ccw(e_sub)
                sorted_odds = _sort_ccw(o_sub)

                potential_combinations.append((subset, sorted_evens, sorted_odds))
                rot_score = _rotational_symmetry_score(subset)
                scored_combinations.append([avg_neighbors, rot_score])

                # Map scores to the cross-section
                scores_to_xsect[(avg_neighbors, rot_score)] = (subset, sorted_evens, sorted_odds)


    # By default, select highest rotational symmetry (then highest avg neighbors as tie breaker) for selected_evens odds
    # lexsort will sort by temp[:, 1] first then temp[:, 0] using the format below (-1 idx is for max value)
    temp = np.array(scored_combinations)
    optimal_index = np.lexsort((temp[:, 0], temp[:, 1]))[-1]
    selected_even, selected_odd = potential_combinations[optimal_index][1], potential_combinations[optimal_index][2]

    # Return the proper selected_evens, selected_odds, and the metadata:
    return selected_even, selected_odd, (optimal_index, potential_combinations, scored_combinations, scores_to_xsect)

def _get_nearest_points(candidates, current_points, k):
    # Remove points that are already in the current set
    current_set = set(current_points)
    pool = [p for p in candidates if p not in current_set]

    if not pool or k == 0:
        return []

    pool_arr = np.array(pool)
    curr_arr = np.array(current_points)

    # Calculate distance matrix (rows=pool points, cols=current points)
    diff = pool_arr[:, None, :] - curr_arr[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    min_dists = dists.min(axis=1)

    # Sort and select top k indices
    nearest_indices = np.argsort(min_dists)[:k]
    return [pool[i] for i in nearest_indices]

def set_hollow_honeycomb(M, N, L, override=False, diagnostics=None):
    if M <= 0 or N <= 0:
        raise ValueError('ERROR: M and N must be positive and larger than 0')
    if L <= 0:
        raise ValueError('ERROR: Length between helices must be positive and larger than 0')
    if abs(M - N) > 2:
        raise ValueError('ERROR: M and N must be within 2 helices of each other.')

    M, N = int(M), int(N)
    has_odd = False
    if M != N:
        has_odd = True
        total = 2 * min(M, N)  # Use smallest config for hollow frame design + add back difference later
    else:
        total = M + N  # If M and N are equal

    # For total < 12: handle manually
    if total < 12:
        return _generate_honeycomb_compact(M, N, L)

    # Otherwise, setup the graph:
    # NOTE: This uses an exhaustive DFS-style approach and likely will not scale to very large M / N
    #       but it does work for DNA origami sized constructs (e.g., ~30-40 HB wide cross-sections)
    even_odd = int(total // 2)
    _even_odd_plus1 = even_odd + 1  # Needed for finding proper path length
    graph, pairs = _get_honeycomb_symmetry_graph(even_odd, even_odd, L, diagnostics=diagnostics)
    xsects, all_scores, score_to_xsect = _find_paths_between_honeycomb_points(graph, pairs, _even_odd_plus1, total,
                                                                              override)

    # The default behaviour (i.e., if not provided as input) will be to select the max surface area design as this
    # will likely be rotationally-symmetric (which is just an integer value)
    temp = np.array(all_scores)
    optimal_index = np.lexsort((temp[:, 0], temp[:, 1]))[-1]
    # optimal_index = max(range(len(all_scores)), key=lambda i: all_scores[i][0])  # Max area
    all_cross_sections = []

    if has_odd:
        # Handle odd values of M / N when total > 12. Overall, this is rather unverified functionality as I am unsure
        # how useful this would actually be. The has_odd flag requires a watertight surface mesh input (e.g., ply
        # file) to even be applicable and hollowframe structures (focus of hado) is limited to the case of M == N.
        # I moreso added this for completeness.
        # Also: Note that I do not incorporate these extra helices into the "score" of the cross-section, i don't
        # find that necessary
        base_e, base_o = _generate_honeycomb_grid(2 * M, 2 * N, L, False, diagnostics=diagnostics)
        num_nearest_points_to_find = abs(M - N)
        for x in xsects:
            current_x = list(x)
            even, odd = _get_even_odd_parity(x, graph)

            if M > N:
                # Find the num_nearest_points_to_find from base_e to the set of points found in x.
                # Note that x belongs to base_e so we must ignore those points.
                new_points = _get_nearest_points(base_e, current_x, num_nearest_points_to_find)
                even.extend(new_points)

            elif M < N:
                # Here we need to add new points from base_o
                new_points = _get_nearest_points(base_o, current_x, num_nearest_points_to_find)
                odd.extend(new_points)
            else:
                raise Exception('ERROR: This should not raise as has_odd can only be true if M != N')

            current_x.extend(new_points)
            all_cross_sections.append((tuple(current_x), even, odd))

    else:
        # Otherwise, when M and N are equal (hollowframe structures) simply:
        for x in xsects:
            even, odd = _get_even_odd_parity(x, graph)
            all_cross_sections.append((x, even, odd))

    # Use the max_area_path index for the returned default values:
    sorted_evens, sorted_odds = all_cross_sections[optimal_index][1], all_cross_sections[optimal_index][2]

    # Return the selected values + the metadata which is used in the UI
    return sorted_evens, sorted_odds, (optimal_index, all_cross_sections, all_scores, score_to_xsect)

def _sort_ccw(points):
    """ Sorts a list of poitns in CW order from the top (i.e, 90-degree angle point)"""
    pts = np.array(points)
    xs = pts[:, 0]
    ys = pts[:, 1]

    angles = np.arctan2(xs, ys)  # Note: arctan2(y, x) but we want clockwise from top
    angles = (angles + 2 * np.pi) % (2 * np.pi)  # Normalize
    indices = np.argsort(angles)
    return [points[i] for i in indices]

def _get_even_odd_parity(xsect, graph):
    """ Splits a cross section in even / odd helix components based on the graph parity """
    evens, odds = [], []
    for i in xsect:
        if i in graph.nodes:
            parity = graph.nodes[i]['parity']
        else:
            check_node = (-i[0], i[1])  # Mirrored nodes not in graph, but parity maintained across mirror
            try:
                parity = graph.nodes[check_node]['parity']
            except KeyError:
                raise ValueError('ERROR: Node not found in graph during parity check.')
        if parity == 'even':
            evens.append(i)
        else:
            odds.append(i)

    # Sort evens and odds both CCW from top:
    sorted_evens = _sort_ccw(evens)
    sorted_odds = _sort_ccw(odds)
    return sorted_evens, sorted_odds

def _set_hollow(M, N, L, lattice_type, override, diagnostics=None):
    if lattice_type == 'honeycomb':
        return set_hollow_honeycomb(M, N, L, override, diagnostics=diagnostics)
    if lattice_type == 'square':
        from hado.core.automation.routing.square_xsect import set_hollow_square
        return set_hollow_square(M, N, L, override, diagnostics=diagnostics)
    raise Exception(f'ERROR: Invalid grid style: {lattice_type}')

if __name__ == '__main__':
    evens, odds, metadata = set_hollow_honeycomb(24, 24, 5.25)
    build_cross_section_points_figure(evens, odds)
