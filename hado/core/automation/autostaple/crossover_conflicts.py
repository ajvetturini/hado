from __future__ import annotations

import math

import numpy as np

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel


def _verify_internal_bundle_xovers(xovers, nts, global_to_local, local_to_global, nearest_neighbors, cur_edge,
                                   scaffold_xovers, og_staple_xover_positions, design):
    """ Verifies that the nts included in the xovers are set True in nts (raises error if issue) """
    nearest_neighbors = set(nearest_neighbors)
    prevent_reuse = set()

    while True:
        crossovers_between = set()
        for x in xovers:
            h1, nt, h2, _ = x
            h1, h2 = global_to_local[h1], global_to_local[h2]
            nts_h1, nts_h2 = nts[h1], nts[h2]
            active_nts_h1, active_nts_h2 = np.where(nts_h1)[0], np.where(nts_h2)[0]
            min_h1, max_h1 = min(active_nts_h1), max(active_nts_h1)
            min_h2, max_h2 = min(active_nts_h2), max(active_nts_h2)

            if not ((nt > min_h1 and nt > min_h2) and (nt < max_h1 and nt < max_h2)):
                raise RuntimeError("ERROR: crossovers applied at end.")
            crossovers_between.add((h1, h2))  # Add local indices for potential_staple_xovers

        unresolved_neighbors = set(nearest_neighbors)
        for xo in crossovers_between:
            unresolved_neighbors.discard(xo)

        scaffold_xovers = _get_scaffold_xovers(local_to_global, design)
        for xo in scaffold_xovers:
            h1, _, h2, _ = xo
            unresolved_neighbors.discard((h1, h2))
            unresolved_neighbors.discard((h2, h1))

        if len(unresolved_neighbors) == 0:
            return xovers

        # First, clean up nearest_neighbors to only be (vi, vj) instead of (vi, vj) and (vj, vi)
        # Reminder: nearest_neighbors are the neighbor helices in a bundle that currently are not "glued together"
        #           thru scaffold / staple crossovers. This sometimes comes up on "shorter" edge designs (e.g., 40-50
        #           basepairs AFTER mitering takes place!)
        # Overall, this logic looks to see if a crossover can just be slightly moved up / down.
        compressed = {tuple(sorted(pair)) for pair in unresolved_neighbors}
        made_progress = False

        for c in compressed:
            # Look in og_staple_xover_positions to see if there is a crossover point
            is_position, data = _check_og_staple_map_positions(c, og_staple_xover_positions)
            if not is_position:
                continue

            added, xovers = _try_add_recovered_staple_xover(c, xovers, data, design, local_to_global)
            if added:
                made_progress = True
                continue

            # Check if we can safely move that crossover elsewhere (and if so, do so and modify design in place)
            can_be_moved, data2 = _check_scaffold_xovers(c, data, scaffold_xovers, design, local_to_global,
                                                         prevent_reuse)
            if not can_be_moved:
                continue

            moved, xovers = _move_scaffold_crossover(c, xovers, data2, design, local_to_global, prevent_reuse)
            if moved:
                made_progress = True

        if made_progress:
            continue

        scaffold_nts = design.get_scaffold_nucleotides()
        temp = set()
        min_sum = math.inf
        global_helix_min_nts = None

        for i in unresolved_neighbors:
            a, j = min(i), max(i)
            global_pair = (local_to_global[a], local_to_global[j])

            if global_pair not in temp:
                temp.add(global_pair)
                sum_a = np.sum(nts[a])
                sum_j = np.sum(nts[j])

                if sum_a < min_sum:
                    min_sum = sum_a
                    global_helix_min_nts = local_to_global[a]
                if sum_j < min_sum:
                    min_sum = sum_j
                    global_helix_min_nts = local_to_global[j]

        num_scaf_nts_post_mitering = np.sum(scaffold_nts[global_helix_min_nts])
        edge_between = tuple(design.geometry.edges[cur_edge])
        raise Exception(f'ERROR: Helix {global_helix_min_nts} on edge (vi, vj) = {edge_between} is too short '
                        f'post-mitering ({num_scaf_nts_post_mitering} nts) and will lead to un-winding. Try '
                        f'lengthening that edge, reducing the mitering threshold, or reducing the values of the'
                        f'various min `StapleArgs` values (these should be between 3 and 5).')


def _try_add_recovered_staple_xover(helix_pair, current_staple_xovers, crossover_positions, design: HadoNucleotideModel,
                                    local_to_global):
    """Adds a blocked crossover directly once a prior scaffold move has freed the site."""
    min_run_between = design.staple_args.min_dist_between_xovers
    staple_dirs = design.get_staple_directions()
    final_h1, final_h2 = helix_pair
    final_h1, final_h2 = local_to_global[final_h1], local_to_global[final_h2]
    scaffold_xovers = _get_scaffold_xovers(local_to_global, design)

    def _is_far_from_scaffold(helix, nt):
        for h1, nt1, h2, _ in scaffold_xovers:
            if h1 == helix and abs(nt - nt1) <= min_run_between:
                return False
            if h2 == helix and abs(nt - nt1) <= min_run_between:
                return False
        return True

    def _is_far_from_staples(helix, nt):
        for stap_xo in current_staple_xovers:
            a, b, c, _ = stap_xo
            if (a == helix or c == helix) and abs(nt - b) < min_run_between:
                return False
        return True

    for final_nt11, final_nt12 in sorted(crossover_positions):
        if staple_dirs[final_h1]:
            new_stap_xo1 = [final_h1, final_nt11, final_h2, -1]
            new_stap_xo2 = [final_h2, final_nt12, final_h1, -1]
        else:
            new_stap_xo1 = [final_h2, final_nt11, final_h1, -1]
            new_stap_xo2 = [final_h1, final_nt12, final_h2, -1]

        if not _is_far_from_scaffold(new_stap_xo1[0], new_stap_xo1[1]):
            continue
        if not _is_far_from_scaffold(new_stap_xo2[0], new_stap_xo2[1]):
            continue
        if not _is_far_from_staples(new_stap_xo1[0], new_stap_xo1[1]):
            continue
        if not _is_far_from_staples(new_stap_xo2[0], new_stap_xo2[1]):
            continue

        return True, np.vstack([current_staple_xovers, new_stap_xo1, new_stap_xo2])

    return False, current_staple_xovers


def _move_scaffold_crossover(helix_pair, current_staple_xovers, new_location_data, design: HadoNucleotideModel,
                             local_to_global, prevent_reuse):
    """ Determines a move point for a scaffold crossover to fit in a staple crossover """
    min_run_between = design.staple_args.min_dist_between_xovers

    def _is_valid(_h1, _h2, check_val):
        for stap_xo in current_staple_xovers:
            a, b, c, d = stap_xo
            if (a == _h1 or c == _h1) and (a == _h2 or c == _h2):
                if abs(check_val - b) < min_run_between:
                    return False
        return True

    final_h1, final_h2 = helix_pair
    final_h1, final_h2 = local_to_global[final_h1], local_to_global[final_h2]
    staple_dirs = design.get_staple_directions()

    for swap_pair, carry in new_location_data.items():
        new_helix_pair_staple_xover, scaffold_xover_to_move, new_scaffold_locations = carry
        final_nt11, final_nt12 = new_helix_pair_staple_xover
        h1, h2, (nt1, nt2) = scaffold_xover_to_move

        for new_xo in new_scaffold_locations:
            mid_xo = (new_xo[0] + new_xo[1]) / 2
            if _is_valid(h1, h2, mid_xo):
                design.replace_scaffold_positions(h1, h2, nt1, nt2, new_xo[0], new_xo[1])

                if staple_dirs[final_h1]:
                    new_stap_xo1 = [final_h1, final_nt11, final_h2, -1]
                    new_stap_xo2 = [final_h2, final_nt12, final_h1, -1]
                else:
                    new_stap_xo1 = [final_h2, final_nt11, final_h1, -1]
                    new_stap_xo2 = [final_h1, final_nt12, final_h2, -1]

                prevent_reuse.add((h1, h2))
                return True, np.vstack([current_staple_xovers, new_stap_xo1, new_stap_xo2])

    return False, current_staple_xovers


def _check_scaffold_xovers(helix_pair, crossover_positions, scaffold_xovers, design, local_to_global, prevent_reuse):
    """ Finds the internal crossovers containing the helices of helix_pair and determines if they can be moved based
    on hte avilable crossover positions for the helix_pair.
    """
    h1, h2 = helix_pair
    h1, h2 = local_to_global[h1], local_to_global[h2]
    check_these = set()
    # Get crossovers on either h1 / h2
    for sx in scaffold_xovers:
        hh1, nt1, hh2, nt2 = sx
        if hh1 in [h1, h2] or hh2 in [h1, h2]:
            check_these.add((hh1, nt1, hh2, nt2))

    # Format properly into (h1, h2, (nt1, nt2)) where nt1 < nt2 by 1
    filtered = set()
    for c in check_these:
        hh1, nt1, hh2, nt2 = c
        hh1, hh2 = min(hh1, hh2), max(hh1, hh2)

        if (hh1, hh2) in prevent_reuse:
            continue
        if (hh2, nt1+1, hh1, nt2) in check_these:
            filtered.add((hh1, hh2, (nt1, nt1+1)))
        elif (hh2, nt1-1, hh1, nt2) in check_these:
            filtered.add((hh1, hh2, (nt1-1, nt1)))

    data = {}
    for c in crossover_positions:
        for f in filtered:
            hh1, hh2, (nt1,  nt2) = f
            all_scaf_locations = design.get_all_valid_scaffold_crossover_options(hh1, hh2)
            filtered_potential_new_scaf_locs = set(i for i in all_scaf_locations if i != (nt1, nt2))
            ranked_list = _rank_by_middle_most(filtered_potential_new_scaf_locs, design, hh1, hh2)
            data[(hh1, hh2)] = (c, f, ranked_list)

    return len(data) > 0, data


def _rank_by_middle_most(filtered_scaf_locations, design: HadoNucleotideModel, h1, h2):
    """ Returns a filtered list of scaffold crossover locations based on proximity to "middle" of scaffold to
    replace position.
    """
    scaf_nts = design.get_scaffold_nucleotides()
    scored = []
    minh1, maxh1 = min(np.where(scaf_nts[h1])[0]), max(np.where(scaf_nts[h1])[0])
    minh2, maxh2 = min(np.where(scaf_nts[h2])[0]), max(np.where(scaf_nts[h2])[0])
    midh1, midh2 = (maxh1 + minh1) / 2, (minh2 + maxh2) / 2
    for f in filtered_scaf_locations:
        middle = (f[0] + f[1]) / 2
        score = ((midh1 - middle)**2 + (midh2 - middle)**2) / 2
        scored.append((f, score))

    return [f for f, _ in sorted(scored, key=lambda x: x[1])]


def _check_og_staple_map_positions(helix_pair, staple_xovers):
    """ Checks what positions (if there are any) a staple crossover can be placed between a helix pair """
    h1, h2 = helix_pair
    h1nts, h2nts = staple_xovers[h1], staple_xovers[h2]

    h1toh2 = np.where(h1nts == h2)[0]
    h2toh1 = np.where(h2nts == h1)[0]
    check_these_pairs = _positions_in_both(h1toh2, h2toh1)

    return len(check_these_pairs) > 0, check_these_pairs


def _positions_in_both(xover1, xover2):
    """ Finds that pairs of (nt1, nt2) that lead to a crossover that are present in both xover1 and xover 2 arrays """
    valid_pairs = set()
    for x in xover1:
        xp1 = x + 1
        if xp1 in xover1 and x in xover2 and xp1 in xover2:
            valid_pairs.add((x, xp1))
    return valid_pairs


def _remove_xovers_near_scaffold_xovers(potential_staple_xovers, scaffold_xovers, global_to_local, staple_args):
    """ Ensures that staple and scaffold crossovers are not placed too closely together using staple_args """
    min_dist_between_xovers = staple_args.min_dist_between_xovers
    invalid_scaffold_breaks = set()

    for s in scaffold_xovers:
        local_h1, local_h2 = global_to_local[s[0]], global_to_local[s[2]]
        local_h1, local_h2 = min(local_h1, local_h2), max(local_h1, local_h2)
        nt_to_check = s[1]

        for helix in [local_h1, local_h2]:
            for offset in range(-min_dist_between_xovers, min_dist_between_xovers + 1):
                nt_index = nt_to_check + offset
                if 0 <= nt_index < potential_staple_xovers.shape[1]:
                    if potential_staple_xovers[helix][nt_index] != -2:
                        potential_staple_xovers[helix][nt_index] = -3
                    invalid_scaffold_breaks.add((helix, nt_index))

    return potential_staple_xovers


def _get_scaffold_xovers(global_ids, design: HadoNucleotideModel, internal_only: bool = True):
    """ Returns the scaffold crossovers for the given global helix indices """
    scaffold_xovers = set()
    cur_scaffold_xovers = design.get_scaffold_crossovers()
    for local_id, global_id in global_ids.items():
        xovers_in_global_helix = np.where(
            (cur_scaffold_xovers[:, 0] == global_id) |
            (cur_scaffold_xovers[:, 2] == global_id)
        )[0]
        xovers = cur_scaffold_xovers[xovers_in_global_helix]

        # We ignore the "end" crossovers by looking at structure of the above list:
        for x in xovers:
            if internal_only:
                if x[-1] == -1:
                    scaffold_xovers.add((int(x[0]), int(x[1]), int(x[2]), int(x[3])))
            else:
                scaffold_xovers.add((int(x[0]), int(x[1]), int(x[2]), int(x[3])))
    return list(scaffold_xovers)
