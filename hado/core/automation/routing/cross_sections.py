from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from hado.core.automation.routing.honeycomb_xsect import set_hollow_honeycomb
from hado.core.automation.routing.square_xsect import set_hollow_square

CrossSectionBuilder = Callable[[int, int, float, bool, object], tuple[list, list, object]]

_CROSS_SECTION_BUILDERS: dict[str, CrossSectionBuilder] = {
    "honeycomb": set_hollow_honeycomb,
    "square": set_hollow_square,
}


def get_cross_section(M: int,
                      N: int,
                      L: float,
                      lattice_type: str = 'honeycomb',
                      grid_style: str = 'hollow',
                      override: bool = False,
                      diagnostics=None,
                      custom_cross_section=None,
                      cross_section_generator=None,
                      ) -> Tuple[list, list]:
    """Return forward and reverse helix coordinates for a registered lattice cross-section."""
    if grid_style.lower() not in ['hollow']:
        raise ValueError('ERROR: helix_bundle_style can only be hollow currently.')

    if custom_cross_section is not None or cross_section_generator is not None:
        return get_custom_cross_section(
            M,
            N,
            L,
            custom_cross_section=custom_cross_section,
            cross_section_generator=cross_section_generator,
        )

    if abs(M - N) > 2:
        raise ValueError('ERROR: M and N must be within 2 helix of each other.')

    lattice_key = lattice_type.lower()
    try:
        builder = _CROSS_SECTION_BUILDERS[lattice_key]
    except KeyError as exc:
        supported = '", "'.join(sorted(_CROSS_SECTION_BUILDERS))
        raise ValueError(
            f'Unsupported lattice cross-section "{lattice_type}". Supported options are "{supported}".'
        ) from exc

    selected_even, selected_odd, _ = builder(M, N, L, override, diagnostics=diagnostics)
    return selected_even, selected_odd


def get_custom_cross_section(M: int,
                             N: int,
                             L: float,
                             custom_cross_section=None,
                             cross_section_generator=None,
                             ) -> Tuple[list, list]:
    """Return validated user-provided cross-section points."""
    if custom_cross_section is not None and cross_section_generator is not None:
        raise ValueError('ERROR: Specify either custom_cross_section or cross_section_generator, not both.')

    if cross_section_generator is not None:
        if not callable(cross_section_generator):
            raise TypeError('ERROR: cross_section_generator must be callable.')
        custom_cross_section = cross_section_generator(M, N, L)

    evens, odds = _select_custom_cross_section(custom_cross_section, M, N)
    return validate_cross_section_points(evens, odds, M, N)


def validate_cross_section_points(evens, odds, M: int | None = None, N: int | None = None) -> Tuple[list, list]:
    """Validate and normalize forward/reverse helix coordinates."""
    evens_array = _coerce_points(evens, 'evens')
    odds_array = _coerce_points(odds, 'odds')

    if M is not None and len(evens_array) != int(M):
        raise ValueError(f'ERROR: Expected {M} even-running helices, found {len(evens_array)}.')
    if N is not None and len(odds_array) != int(N):
        raise ValueError(f'ERROR: Expected {N} odd-running helices, found {len(odds_array)}.')

    all_points = np.vstack((evens_array, odds_array))
    unique_points = np.unique(np.round(all_points, decimals=8), axis=0)
    if len(unique_points) != len(all_points):
        raise ValueError('ERROR: Custom cross-section contains duplicate helix coordinates.')

    return _array_to_points(evens_array), _array_to_points(odds_array)


def serialize_custom_cross_section(custom_cross_section):
    """Return a JSON-compatible custom cross-section if possible."""
    if custom_cross_section is None:
        return None
    evens, odds = _select_custom_cross_section(custom_cross_section, None, None)
    evens, odds = validate_cross_section_points(evens, odds)
    return {'evens': evens, 'odds': odds}


def _select_custom_cross_section(custom_cross_section, M, N):
    if custom_cross_section is None:
        raise ValueError('ERROR: custom_cross_section must be provided.')

    if isinstance(custom_cross_section, dict):
        if 'evens' in custom_cross_section and 'odds' in custom_cross_section:
            return custom_cross_section['evens'], custom_cross_section['odds']

        keys_to_try = []
        if M is not None and N is not None:
            keys_to_try.extend(((int(M), int(N)), f'{int(M)},{int(N)}', str((int(M), int(N)))))
        for key in keys_to_try:
            if key in custom_cross_section:
                return _select_custom_cross_section(custom_cross_section[key], M, N)
        raise ValueError('ERROR: Custom cross-section dictionary must contain evens/odds or an M,N entry.')

    if isinstance(custom_cross_section, (list, tuple)) and len(custom_cross_section) >= 2:
        return custom_cross_section[0], custom_cross_section[1]

    raise TypeError('ERROR: custom_cross_section must be (evens, odds), a dict with evens/odds, or an M,N mapping.')


def _coerce_points(points, label: str) -> np.ndarray:
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f'ERROR: Custom cross-section {label} must be a sequence of 2D coordinates.')
    if len(points_array) == 0:
        raise ValueError(f'ERROR: Custom cross-section {label} must contain at least one coordinate.')
    if not np.all(np.isfinite(points_array)):
        raise ValueError(f'ERROR: Custom cross-section {label} contains non-finite coordinates.')
    return points_array


def _array_to_points(points: np.ndarray) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in points]
