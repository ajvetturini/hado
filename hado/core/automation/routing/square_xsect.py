from __future__ import annotations

from itertools import product
from typing import Tuple

import networkx as nx
import numpy as np


def get_square_ring_diameter(evens: list, odds: list, helix_diameter: float = 0.0) -> float:
    """Return the outer diameter of a square-lattice cross-section."""
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


def get_cross_section(M: int,
                      N: int,
                      L: float,
                      lattice_type: str = 'square',
                      grid_style: str = 'hollow',
                      override: bool = False,
                      diagnostics=None,
                      ) -> Tuple[list, list]:
    """
    Build a hollow square-lattice cross-section with M forward and N reverse helices.

    Helix parity follows a square checkerboard so neighboring helices are always opposite direction. For M == N this
    returns a rectangular perimeter that is as close to square as possible. Small or odd-count requests fall back to
    the nearest connected square-lattice shape.
    """
    if lattice_type.lower() != 'square':
        raise ValueError(f'ERROR: Invalid grid style: {lattice_type}')
    if grid_style.lower() not in ['hollow']:
        raise ValueError('ERROR: helix_bundle_style can only be `hollow` currently.')

    selected_even, selected_odd, _ = set_hollow_square(M, N, L, override=override, diagnostics=diagnostics)
    return selected_even, selected_odd


def get_square_mapping_cadnano(spacing_length, grids_to_fill, precision, col_buffer=3, row_buffer=2):
    """Map stacked square-lattice bundle coordinates into a planar caDNAno square grid."""
    if spacing_length <= 0:
        raise ValueError('ERROR: Length between helices must be positive and larger than 0')

    max_column = 30
    cur_column, cur_row, local_max_height = 0, 0, 0
    grid_data = {}

    for bundle_index, grid in enumerate(grids_to_fill):
        local_points = _square_points_to_lattice_indices(grid, spacing_length, precision)
        min_col, min_row = np.min(local_points, axis=0)
        max_col, max_row = np.max(local_points, axis=0)

        width_needed = int(max_col - min_col + 1 + col_buffer)
        height_needed = int(max_row - min_row + 1 + row_buffer)

        if cur_column + width_needed <= max_column:
            lower_bound = (cur_column, cur_row)
            cur_column += width_needed + 1
            local_max_height = max(local_max_height, height_needed)
        else:
            cur_row += local_max_height + row_buffer
            local_max_height = height_needed
            lower_bound = (0, cur_row)
            cur_column = width_needed + 1

        if (lower_bound[0] + lower_bound[1] - min_col - min_row) % 2 != 0:
            lower_bound = (lower_bound[0] + 1, lower_bound[1])

        bundle_mapping = {}
        for physical_point, local_point in zip(grid, local_points):
            cadnano_col = lower_bound[0] + int(local_point[0] - min_col)
            cadnano_row = lower_bound[1] + int(local_point[1] - min_row)
            key = tuple(np.round(physical_point, decimals=precision))
            bundle_mapping[key] = (cadnano_col, cadnano_row)
        grid_data[bundle_index] = bundle_mapping

    return grid_data


def set_hollow_square(M: int, N: int, L: float, override: bool = False, diagnostics=None):
    if M <= 0 or N <= 0:
        raise ValueError('ERROR: M and N must be positive and larger than 0')
    if L <= 0:
        raise ValueError('ERROR: Length between helices must be positive and larger than 0')
    if abs(M - N) > 2:
        raise ValueError('ERROR: M and N must be within 2 helices of each other.')

    M, N = int(M), int(N)
    lattice_points = _select_square_lattice_points(M, N)
    evens, odds = _split_square_parity(lattice_points, L)

    if len(evens) != M or len(odds) != N:
        raise RuntimeError('ERROR: Square cross-section generator returned incorrect helix parity counts.')

    metadata = {
        'lattice_points': lattice_points,
        'num_helices': M + N,
    }
    return evens, odds, metadata


def _select_square_lattice_points(M: int, N: int) -> list[tuple[int, int]]:
    if M == N:
        return _select_equal_square_lattice_points(M + N)

    base_count = min(M, N)
    points = _select_equal_square_lattice_points(2 * base_count)
    desired_parity = 0 if M > N else 1
    points.extend(_nearest_adjacent_points(points, desired_parity, abs(M - N)))
    return points


def _square_points_to_lattice_indices(points, spacing_length: float, precision: int) -> np.ndarray:
    scaled = np.asarray(points, dtype=float) / spacing_length
    shifted = scaled - np.min(scaled, axis=0)
    rounded = np.rint(shifted).astype(int)
    if not np.allclose(shifted, rounded, atol=10 ** (-precision)):
        raise ValueError('ERROR: Square cross-section points are not aligned to the square lattice spacing.')
    return rounded


def _select_equal_square_lattice_points(total_helices: int) -> list[tuple[int, int]]:
    if total_helices == 2:
        return [(0, 0), (1, 0)]

    width, height = _rectangle_dimensions(total_helices)
    return _rectangle_perimeter_points(width, height)


def _rectangle_dimensions(total_helices: int) -> tuple[int, int]:
    if total_helices < 4 or total_helices % 2 != 0:
        raise ValueError('ERROR: Equal M/N square cross-sections require an even helix count of at least 2.')

    target_sum = total_helices // 2 + 2
    candidates = []
    for width in range(2, target_sum - 1):
        height = target_sum - width
        if height < 2:
            continue
        candidates.append((width * height, -abs(width - height), width, height))

    if not candidates:
        raise ValueError('ERROR: Unable to find square-lattice rectangle dimensions.')

    _, _, width, height = max(candidates)
    return width, height


def _rectangle_perimeter_points(width: int, height: int) -> list[tuple[int, int]]:
    points = []
    for x, y in product(range(width), range(height)):
        if x in (0, width - 1) or y in (0, height - 1):
            points.append((x, y))
    return points


def _nearest_adjacent_points(points: list[tuple[int, int]], desired_parity: int, count: int) -> list[tuple[int, int]]:
    selected = set(points)
    candidates = []
    min_x = min(x for x, _ in selected) - 1
    max_x = max(x for x, _ in selected) + 1
    min_y = min(y for _, y in selected) - 1
    max_y = max(y for _, y in selected) + 1

    for x, y in product(range(min_x, max_x + 1), range(min_y, max_y + 1)):
        point = (x, y)
        if point in selected or _square_parity(point) != desired_parity:
            continue
        if any(_manhattan_distance(point, existing) == 1 for existing in selected):
            candidates.append(point)

    centroid = np.array(points, dtype=float).mean(axis=0)
    candidates.sort(key=lambda p: (np.linalg.norm(np.array(p, dtype=float) - centroid), p[1], p[0]))
    extra = candidates[:count]
    if len(extra) != count:
        raise ValueError('ERROR: Unable to place requested odd square-lattice helix count.')

    graph = nx.Graph()
    all_points = points + extra
    graph.add_nodes_from(all_points)
    for i, point in enumerate(all_points):
        for other in all_points[i + 1:]:
            if _manhattan_distance(point, other) == 1:
                graph.add_edge(point, other)
    if not nx.is_connected(graph):
        raise ValueError('ERROR: Square cross-section must be connected.')

    return extra


def _split_square_parity(points: list[tuple[int, int]], L: float) -> tuple[list, list]:
    centered = _center_lattice_points(points, L)
    evens, odds = [], []
    for lattice_point, physical_point in zip(points, centered):
        if _square_parity(lattice_point) == 0:
            evens.append(physical_point)
        else:
            odds.append(physical_point)
    return _sort_clockwise_from_top(evens), _sort_clockwise_from_top(odds)


def _center_lattice_points(points: list[tuple[int, int]], L: float) -> list[tuple[float, float]]:
    pts = np.array(points, dtype=float)
    centroid = pts.mean(axis=0)
    centered = (pts - centroid) * L
    return [(float(point[0]), float(point[1])) for point in centered]


def _sort_clockwise_from_top(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = np.array(points, dtype=float)
    angles = np.arctan2(pts[:, 0], pts[:, 1])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    indices = np.argsort(angles)
    return [points[i] for i in indices]


def _square_parity(point: tuple[int, int]) -> int:
    return (point[0] + point[1]) % 2


def _manhattan_distance(point: tuple[int, int], other: tuple[int, int]) -> int:
    return abs(point[0] - other[0]) + abs(point[1] - other[1])
