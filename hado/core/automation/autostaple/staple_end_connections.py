from __future__ import annotations

import numpy as np

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.automation.pipeline.types import emit_runtime_message


def _get_end_nt(helix, cur_vert, current_nt_vals, design: HadoNucleotideModel,):
    helix_to_bundle = design.get_helix_to_bundle()
    idx_edge_map = design.get_idx_edge_map()

    bundle = helix_to_bundle[helix]
    edge = idx_edge_map[bundle]

    all_nts = np.where(current_nt_vals[helix])[0]
    if edge[0] == cur_vert:
        return min(all_nts)
    else:
        return max(all_nts)

def _add_staple_map_crossovers(staple_map, design: HadoNucleotideModel, final_positions: dict, staple_args, verbose, diagnostics=None):
    """ Connects the 3' and 5' ends of staples using the final_positions from the optimal configuration """
    helix_to_bundle = design.get_helix_to_bundle()
    staple_nucleotides = design.get_staple_nucleotides()

    staple_end_xovers = []

    for v, connections in staple_map.items():
        for c in connections:
            h1, h2 = c

            # The order below is FLIPPED becuase the function traditionally works for SCAFFOLD (which runs opposite
            # direction of staples)
            senders_at_h1, _ = design.get_receiver_indices(helix_to_bundle[h1], v)
            senders_at_h2, _ = design.get_receiver_indices(helix_to_bundle[h2], v)
            receivers_at_h1, _ = design.get_sender_indices(helix_to_bundle[h1], v)
            receivers_at_h2, _ = design.get_sender_indices(helix_to_bundle[h2], v)

            if h1 in senders_at_h1 and h2 in receivers_at_h2:
                sender = h1
                receiver = h2
            elif h2 in senders_at_h2 and h1 in receivers_at_h1:
                sender = h2
                receiver = h1
            else:
                raise ValueError('ERROR: Can not connect staple ends.')

            sender_nt = _get_end_nt(sender, v, staple_nucleotides, design)
            receiver_nt = _get_end_nt(receiver, v, staple_nucleotides, design)

            # Calculate # of Poly-T spacers to add to the staple to minimize blunt end stacking and update sender_nt
            if final_positions != {}:
                num_spacers = _calc_num_polyt_spacers(sender, receiver, v, final_positions, staple_args, verbose, diagnostics)
                new_sender_nt, staple_nucleotides = _update_nts_for_spacer(num_spacers, sender, sender_nt,
                                                                           staple_nucleotides, design)

                staple_end_xovers.append([sender, new_sender_nt, receiver, receiver_nt])
            else:
                staple_end_xovers.append([sender, sender_nt, receiver, receiver_nt])

    return staple_end_xovers, staple_nucleotides

def _update_nts_for_spacer(num, sender_idx, sender_nt, all_nts, design: HadoNucleotideModel):
    """ Updates the sender nucleotide position and ensures the strand remains a single contiguous chain """
    staple_dirs = design.get_staple_directions()
    dir_to_add_in = staple_dirs[sender_idx]
    sender_nts = all_nts[sender_idx]
    max_len = len(sender_nts)

    if dir_to_add_in:
        new_sender_nt = min(sender_nt + num, max_len - 1)
    else:
        new_sender_nt = max(sender_nt - num, 0)

    v1, v2 = min(sender_nt, new_sender_nt), max(sender_nt, new_sender_nt)
    sender_nts[v1:v2 + 1] = True


    true_indices = np.where(sender_nts)[0]

    if len(true_indices) > 0:
        # If contiguous, the distance between the first and last True
        # must match the total count of True values.
        first_true = true_indices[0]
        last_true = true_indices[-1]
        actual_count = len(true_indices)
        expected_count = last_true - first_true + 1

        if actual_count != expected_count:
            raise ValueError(
                f"ERror updating nts on Helix {sender_idx}. The update created a gap or merged two "
                f"separate segments."
            )

    all_nts[sender_idx] = sender_nts
    return new_sender_nt, all_nts

def _calc_num_polyt_spacers(sender_nt, receiver_nt, vertex, positions, staple_args, verbose, diagnostics=None):
    """ Calculates the total # of poly-T spacers that will be added-on to the sender nucleotide. Overall, this procedure
    comes from the fact that a phosphate-phosphate distance of ~0.55 nm is B-Form DNA (which is traditionally presumed
    in DNA origami). A smaller distance (literature used 0.42, see below citations) is considered to reduce tension
    in the connection.

    Citations:
    1) Jun, Hyungmin et al. Science Advances 5, no. 1 (2019): eaav0655. https://doi.org/10.1126/sciadv.aav0655.
    2) Jun, Hyungmin et al. ACS Nano 13, no. 2 (2019): 2083–93. https://doi.org/10.1021/acsnano.8b08671.
    """
    def _get_pos(nt):
        for vp in positions[nt]:
            vertex_of_vp, position_at_v = vp
            if vertex_of_vp == vertex:
                return position_at_v
        raise Exception('ERROR: Connection not found at vertex for assigning poly t spacing.')

    sender_xyz = _get_pos(sender_nt)
    receiver_xyz = _get_pos(receiver_nt)
    dist = np.linalg.norm(sender_xyz - receiver_xyz)
    total_spacer_count = int(dist / staple_args.polyt_bulge_dist)
    if total_spacer_count > staple_args.max_staple_spacer_length:
        total_spacer_count = staple_args.max_staple_spacer_length
        emit_runtime_message(
            f'WARNING: Total poly-t spacer is very large ({total_spacer_count}), falling back to max threshold.',
            diagnostics=diagnostics,
            verbose=verbose,
            warning=True,
        )
    return total_spacer_count
