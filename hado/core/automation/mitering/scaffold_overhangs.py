from __future__ import annotations

import numpy as np

from hado.core.automation.autostaple.staple_end_connections import _calc_num_polyt_spacers
from hado.core.automation.model.nucleotide_model import HadoNucleotideModel


def add_scaffold_3p_overhangs(design: HadoNucleotideModel, final_positions: dict, staple_args, verbose: bool = False,
                              diagnostics=None) -> int:
    """Add scaffold-only overhangs at the 3' connection end to relieve torsion in backbone between helix bundle c
    connections. This is done separately of mitering step to ensure that these overhangs remain unpaired prior to
    staple assignment.
    """
    if not final_positions:
        return 0

    scaffold_nucleotides = design.get_scaffold_nucleotides()
    staple_nucleotides = design.get_staple_nucleotides()
    scaffold_crossovers = design.get_scaffold_crossovers()
    scaffold_directions = design.get_scaffold_directions()
    if scaffold_crossovers.size == 0:
        return 0

    updated_crossovers = np.array(scaffold_crossovers, copy=True)
    helix_to_bundle = design.get_helix_to_bundle()
    total_added = 0

    # Find crossover of (helix1, nt1) -> (helix2, nt2) so that it can be extended at 3' end (sender)
    for idx, row in enumerate(updated_crossovers):
        sender, sender_nt, receiver, receiver_nt = (int(x) for x in row)
        if receiver_nt == -1 or helix_to_bundle[sender] == helix_to_bundle[receiver]:
            # Ignore internal crossovers as only looking between connected helices
            continue

        shared_vertex = design.get_shared_vertex(
            int(helix_to_bundle[sender]),
            int(helix_to_bundle[receiver]),
        )
        if shared_vertex is None:
            raise ValueError("ERROR: External scaffold crossover helices do not share a vertex.")

        spacer_count = _calc_num_polyt_spacers(sender, receiver, shared_vertex, final_positions, staple_args,
                                               verbose, diagnostics)
        new_sender_nt, added = _extend_3p_scaffold_end(scaffold_nucleotides, staple_nucleotides, scaffold_directions,
                                                       sender, sender_nt, spacer_count)
        updated_crossovers[idx, 1] = new_sender_nt
        total_added += added

    design.set_scaffold_nucleotides(scaffold_nucleotides)
    design.set_scaffold_crossovers(updated_crossovers)
    _verify_scaffold_overhangs_are_unpaired(design)
    return total_added


def _extend_3p_scaffold_end(scaffold_nucleotides: np.ndarray, staple_nucleotides: np.ndarray, scaffold_dirs: np.ndarray,
                            helix: int, three_prime_nt: int, spacer_count: int) -> tuple[int, int]:
    if spacer_count <= 0:
        return three_prime_nt, 0

    helix_scaffold_nts = scaffold_nucleotides[helix]
    active_nts = np.where(helix_scaffold_nts)[0]
    if len(active_nts) == 0:
        raise ValueError(f"ERROR: Cannot add scaffold overhang on empty helix {helix}.")

    forward = bool(scaffold_dirs[helix])
    current_three_prime = int(active_nts[-1] if forward else active_nts[0])
    if int(three_prime_nt) != current_three_prime:
        raise ValueError(
            f"ERROR: Scaffold crossover on helix {helix} is not located at the current 3' end."
        )

    if forward:
        new_three_prime = min(three_prime_nt + spacer_count, len(helix_scaffold_nts) - 1)
        start, stop = three_prime_nt, new_three_prime
    else:
        new_three_prime = max(three_prime_nt - spacer_count, 0)
        start, stop = new_three_prime, three_prime_nt

    added_positions = np.array(
        [nt for nt in range(start, stop + 1) if not helix_scaffold_nts[nt]],
        dtype=np.int64,
    )
    if added_positions.size and np.any(staple_nucleotides[helix, added_positions]):
        raise ValueError("ERROR: Scaffold ssDNA overhang positions are already occupied by staples.")

    helix_scaffold_nts[start:stop + 1] = True
    _verify_contiguous_helix(scaffold_nucleotides[helix], helix)
    return int(new_three_prime), int(added_positions.size)


def _verify_scaffold_overhangs_are_unpaired(design: HadoNucleotideModel):
    scaffold_nucleotides = design.get_scaffold_nucleotides().astype(bool)
    staple_nucleotides = design.get_staple_nucleotides().astype(bool)
    scaffold_only = scaffold_nucleotides & ~staple_nucleotides
    if np.any(staple_nucleotides[scaffold_only]):
        raise ValueError("ERROR: Scaffold overhang verification failed; overhangs must be staple-unpaired.")


def _verify_contiguous_helix(nts: np.ndarray, helix: int):
    active_nts = np.where(nts)[0]
    if len(active_nts) == 0:
        raise ValueError(f"ERROR: Scaffold helix {helix} has no active nucleotides.")
    expected = active_nts[-1] - active_nts[0] + 1
    if len(active_nts) != expected:
        raise ValueError(f"ERROR: Scaffold overhang created a non-contiguous helix {helix}.")
