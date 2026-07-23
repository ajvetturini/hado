from __future__ import annotations

from collections import Counter

import numpy as np

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel


def _make_staple_ends_flush(design: HadoNucleotideModel, cur_staple_nts: np.ndarray, staple_args):
    """ Creates a new staples_nts array that makes the 3' and 5' ends of staples at pendant nodes flush """
    staple_nts = cur_staple_nts.copy()
    final_staples = []
    idx_edge_map = design.get_idx_edge_map()
    helix_to_bundle = design.get_helix_to_bundle()
    for bundle_id, edge in enumerate(idx_edge_map):
        helices_in_bundle = np.where(helix_to_bundle == bundle_id)[0]
        if len(helices_in_bundle) == 0:
            continue

        free_min, free_max = design.check_free_ends(helices_in_bundle[0])
        nts_subset = staple_nts[helices_in_bundle]
        if free_min:
            first_true = np.argmax(nts_subset, axis=1)
            has_true = np.any(nts_subset, axis=1)
            first_true = np.where(has_true, first_true, -1)

            t0_candidates = first_true[first_true >= 0]
            if t0_candidates.size == 0:
                continue
            t0 = t0_candidates.min()
            t0 = max(0, t0 - staple_args.flush_distance)

            for i, ft in enumerate(first_true):
                if ft >= 0 and t0 <= ft:
                    nts_subset[i, t0:ft + 1] = True
            final_staples.extend(nts_subset)

        elif free_max:
            # Repeat above but for the largest column index (i.e., longest helix in bundle length) wwith a True value
            reversed_indices = nts_subset.shape[1] - 1 - np.argmax(nts_subset[:, ::-1], axis=1)
            has_true = np.any(nts_subset, axis=1)
            last_true = np.where(has_true, reversed_indices, -1)
            t1_candidates = last_true[last_true >= 0]
            if t1_candidates.size == 0:
                continue
            t1 = t1_candidates.max()
            t1 = min(nts_subset.shape[1] - 1, t1 + staple_args.flush_distance)
            for i, lt in enumerate(last_true):
                if 0 <= lt <= t1:
                    nts_subset[i, lt:t1 + 1] = True
            final_staples.extend(nts_subset)
        else:
            # If the above are both False, then there is no modifications needed
            final_staples.extend(nts_subset)
            continue

    return final_staples

def _add_blunt_ends_to_degree_1_short_staples(short_staples, all_nts, design: HadoNucleotideModel, staple_args):
    """ Simply extends the nucleotides to reach the staple_args min length / prevent blunt end stacking thru a
    poly-t spacer
    """
    helices_in_short_staples = []
    for short_staple in short_staples:
        helices_in_short_staples.append(list({x[0] for x in short_staple}))
    flat_short_staples = [i for flat in helices_in_short_staples for i in flat]
    extension_dict = {}
    min_staple_length = staple_args.min_length_after_break
    blunt_length = staple_args.default_blunt_end_length
    for i in short_staples:
        unique_helices = {x[0] for x in i}
        cur_len = len(i)
        min_nts_needed = min_staple_length - cur_len - (len(unique_helices) * blunt_length)
        min_nts_per_helix = np.ceil(min_nts_needed / len(unique_helices))
        for u in unique_helices:
            extension_dict[u] = int(min_nts_per_helix + blunt_length)

    idx_edge_map = design.get_idx_edge_map()
    helix_to_bundle = design.get_helix_to_bundle()

    def _has_degree_1_end(h):
        temp = np.array(idx_edge_map).flatten()
        counts = Counter(temp)
        node = helix_to_bundle[h]
        v1, v2 = int(idx_edge_map[node][0]), int(idx_edge_map[node][1])
        if counts[v1] == 1:
            return True, 'min'
        elif counts[v2] == 1:
            return True, 'max'
        else:
            return False, None

    def _in_short_staple(h):
        if h in flat_short_staples:
            return extension_dict[h]

        else:
            return staple_args.default_blunt_end_length

    def _extend_nts(nts, end, length):
        actual_nts = np.where(nts)[0]
        if end == 'min':
            first_nt = min(actual_nts)
            new_first = max(0, first_nt - length)
            nts[new_first:first_nt] = True
            return nts

        elif end == 'max':
            last_nt = max(actual_nts)
            new_last = min(len(nts) - 1, last_nt + length)
            nts[last_nt:new_last+1] = True
            return nts

        else:
            raise ValueError('This should not raise, end should only be max or min.')

    final_staple_nts = []
    for helix_num, helix_nts in enumerate(all_nts):
        has_deg_1, which_end = _has_degree_1_end(helix_num)
        if has_deg_1:
            # Once we see there is a degree 1 node, check if there exists a short staple or not:
            extension_length = _in_short_staple(helix_num)
            extended_nts = _extend_nts(helix_nts, which_end, extension_length)
            final_staple_nts.append(extended_nts)
        else:
            # If not a degree 1 node, we do not need to change naythign:
            final_staple_nts.append(helix_nts)
    return np.array(final_staple_nts)
