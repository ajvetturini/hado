from __future__ import annotations

import networkx as nx
import numpy as np

from hado.core.automation.autostaple.crossover_conflicts import (
    _get_scaffold_xovers,
    _remove_xovers_near_scaffold_xovers,
    _verify_internal_bundle_xovers,
)
from hado.core.automation.model.nucleotide_model import HadoNucleotideModel, _fix_leading_zero_index


def _get_staple_crossover_map(hb_idx, grid_positions, staple_nts, staple_dirs, local_scaf_nts,
                              design: HadoNucleotideModel, staple_args):
    """ Creates 2D array mapping indices of the cadnano grid to where nucleotiides are currently active and where
    nearby crossover points are (to prevent placing crossovers too close to each other).
    """
    potential_staple_xovers = np.where(staple_nts, -1, -2)
    scaf_nts = np.where(local_scaf_nts, -1, -2)
    min_run_from_binding_edge = staple_args.min_run_post_bundle_connection
    min_run_post_xover = staple_args.min_run_post_xover
    check_distance = min_run_from_binding_edge + min_run_post_xover

    for i, (r, rscaf) in enumerate(zip(potential_staple_xovers, scaf_nts)):
        m1_indices = np.where(r == -1)[0]
        m2_indices = np.where(rscaf == -1)[0]
        min_val1, max_val1 = m1_indices[0], m1_indices[-1]
        min_val2, max_val2 = m2_indices[0], m2_indices[-1]
        min_val = max(min_val1, min_val2)  # max(min value) is first point of dsDNA (where we want to measure from)
        max_val = min(max_val1, max_val2)  # min(max value) is last point of dsDNA (where we want to measure to)

        third_period = design.get_period() // 3
        if min_val + third_period > max_val:
            raise Exception('ERROR: Not enough nucleotides to place staple crossovers with the given '
                            'distance between them.')
        r[min_val1:min_val + check_distance + 1] = -3
        r[max_val - check_distance:max_val1 + 1] = -3

    num_helices = np.arange(len(grid_positions))

    nearest_neighbors = set()
    allowable_dist = design.get_spacing_distance()
    for i in num_helices:
        other_helix_indices = np.where(num_helices != i)[0]

        dist = np.round(np.linalg.norm(grid_positions[other_helix_indices] - grid_positions[i], axis=1), 3)
        neighbor_tuples = [other_helix_indices[j] for j in np.where(dist <= allowable_dist)[0]]
        for n in neighbor_tuples:
            if staple_dirs[i] != staple_dirs[n]:  # Only consider anti-parallel neighbors
                nearest_neighbors.add((int(i), int(n)))

    helix_to_bundle = design.get_helix_to_bundle()
    indices = np.where(helix_to_bundle == hb_idx)[0]
    checked = set()
    for pair in nearest_neighbors:
        if pair in checked: continue

        h1, h2 = pair
        h1toh2 = design.get_all_staple_crossover_options(indices[h1], indices[h2])
        h2toh1 = design.get_all_staple_crossover_options(indices[h2], indices[h1])
        if len(h1toh2) != len(h2toh1):
            raise RuntimeError('ERROR: Staple crossover options are not same length.')

        # Need to carefully handle 0 index used by cadnano grid style because below logic assumes index 0 in h1toh2 and
        # index 0 in h2toh1 are within +- 1 of each other, but this case of 0 throws off the logic:
        h1toh2 = _fix_leading_zero_index(h1toh2, design.get_period())
        h2toh1 = _fix_leading_zero_index(h2toh1, design.get_period())

        h1_array = potential_staple_xovers[h1].copy()
        h2_array = potential_staple_xovers[h2].copy()

        for i, j in zip(h1toh2, h2toh1):
            if i >= len(h1_array) or j >= len(h2_array): continue
            if abs(i - j) != 1:
                raise RuntimeError('ERROR: Crossover maps not properly aligned')

            # Check or instead of and because i / j are within +- 1 of eachother
            check1 = h1_array[i] == -1 or h1_array[j] == -1
            check2 = h2_array[i] == -1 or h2_array[j] == -1

            if check1 and check2:
                h1_array[i] = h2
                h1_array[j] = h2
                h2_array[i] = h1
                h2_array[j] = h1

        potential_staple_xovers[h1] = h1_array
        potential_staple_xovers[h2] = h2_array
        checked.add((h1, h2))
        checked.add((h2, h1))

    return potential_staple_xovers, nearest_neighbors

def _staple_helix_bundle(only_add, hb_idx, grid_positions, staple_nts, staple_dirs, local_scaf_nts,
                         design: HadoNucleotideModel, staple_args):
    """ Staples the neighbor-helices together in a given helix bundle """
    potential_staple_xovers, nn = _get_staple_crossover_map(hb_idx, grid_positions, staple_nts, staple_dirs,
                                                            local_scaf_nts, design, staple_args)
    if staple_nts.shape != potential_staple_xovers.shape:
        raise RuntimeError('ERROR: staple_nts and potential_staple_xovers must be the same shape.')

    local_ids = np.arange(len(staple_dirs))
    local_to_global, global_to_local = _convert_local_to_global(hb_idx, local_ids, design)
    scaffold_xovers = _get_scaffold_xovers(local_to_global, design)
    staple_xover_positions = potential_staple_xovers.copy()
    if len(scaffold_xovers) > 0:
        potential_staple_xovers = _remove_xovers_near_scaffold_xovers(potential_staple_xovers, scaffold_xovers,
                                                                      global_to_local, staple_args)
    graph = _convert_array_to_graph(potential_staple_xovers)
    _add_all_staple_crossovers(graph, potential_staple_xovers)

    # Recombine any crossovers that are too closely placed. HOWEVER, this should NOT execute for any hollowframe
    # designs if the recommended default StapleArgs are used due to the design paradigms of hollowframes.
    if not only_add:
        _recombine_crossovers_using_staple_args(graph, local_ids, staple_args)
        bundle_xovers = _convert_graph_to_xovers_list(graph, staple_dirs, local_to_global)
        start_break = None
    else:
        start_break = _get_start_position(potential_staple_xovers, scaffold_xovers, staple_args)
        start_break = (local_to_global[start_break[0]], start_break[1])
        bundle_xovers = _convert_graph_to_xovers_list(graph, staple_dirs, local_to_global)

    bundle_xovers = _verify_internal_bundle_xovers(bundle_xovers, staple_nts, global_to_local, local_to_global, nn,
                                                   hb_idx, scaffold_xovers, staple_xover_positions, design)
    new_graph = nx.Graph()
    for u, v in graph.edges():
        N_u_old, M_u = u
        N_v_old, M_v = v

        N_u_new = int(local_to_global.get(N_u_old, N_u_old))
        N_v_new = int(local_to_global.get(N_v_old, N_v_old))

        u_new = (N_u_new, M_u)
        v_new = (N_v_new, M_v)
        new_graph.add_edge(u_new, v_new)

    graph = new_graph
    if not only_add:
        _apply_verified_staple_xovers_to_graph(graph, bundle_xovers)
    return bundle_xovers, start_break, graph


def _apply_verified_staple_xovers_to_graph(graph, xovers):
    """Keep the autobreak graph in sync with recovered staple crossovers."""
    crossover_positions = {}
    for h1, nt, h2, maybe_nt in xovers:
        if int(maybe_nt) != -1:
            continue
        key = tuple(sorted((int(h1), int(h2))))
        crossover_positions.setdefault(key, set()).add(int(nt))

    for (h1, h2), positions in crossover_positions.items():
        sorted_positions = sorted(positions)
        if len(sorted_positions) % 2 != 0:
            raise RuntimeError('ERROR: Expected paired staple crossover positions.')

        for nt1, nt2 in zip(sorted_positions[0::2], sorted_positions[1::2]):
            if abs(nt2 - nt1) != 1:
                raise RuntimeError('ERROR: Expected consecutive nucleotides staple crossover.')

            edge1 = ((h1, nt1), (h1, nt2))
            edge2 = ((h2, nt1), (h2, nt2))
            if graph.has_edge(*edge1):
                graph.remove_edge(*edge1)
            if graph.has_edge(*edge2):
                graph.remove_edge(*edge2)

            graph.add_edge((h1, nt1), (h2, nt1))
            graph.add_edge((h1, nt2), (h2, nt2))


def _get_start_position(potential_staple_xovers, scaffold_xovers, staple_args):
    """ Finds a (helix, nt) that is a quality start position to break the scaffold circularity """
    min_dist_between_xovers = staple_args.min_dist_between_xovers
    start_point = None
    scaf_xovers = np.array(scaffold_xovers)
    for row_ct, row in enumerate(potential_staple_xovers):
        nts = np.where(row != -2)[0]
        found = False
        for nt in nts:
            valid = True
            for i in range(nt - min_dist_between_xovers, nt + min_dist_between_xovers + 1):
                if row[i] != -1:
                    valid = False
                    break

            if valid:
                start_point = (row_ct, nt)

                # Verify not near scaffold xovers:
                if len(scaf_xovers) > 0:
                    scaf_xovers_in_row_ct = scaf_xovers[(scaf_xovers[:, 0] == row_ct) | (scaf_xovers[:, 2] == row_ct)]
                    for s in scaf_xovers_in_row_ct:
                        if s[-1] != -1:
                            continue
                        check_nt = s[1]

                        # Check if the nt of start_point is within +- min_dist_between_xovers of check_nt and if not,
                        # found = True (else found = False and break)
                        if not (check_nt - min_dist_between_xovers <= nt <= check_nt + min_dist_between_xovers):
                            found = True
                        else:
                            found = False
                            break

                    if not found:
                        continue  # If we broke out of the inner loop, we need to check the next potential start point
                    found = True
                    break
                else:
                    found = True
                    break
        if found:
            break

    if start_point is None:
        raise RuntimeError('ERROR: Unable to find start point')
    return start_point

def _convert_graph_to_xovers_list(graph, staple_dirs, local_to_global):
    """Converts the graph representation of staple crossovers into a list of crossover tuples."""
    crossover_positions = {}
    for u, v in graph.edges:
        helix_i, nt_i = u
        helix_j, nt_j = v
        if helix_i == helix_j:
            continue
        if nt_i != nt_j:
            raise RuntimeError('ERROR: Expected staple crossover nucleotides to share an index.')

        key = tuple(sorted((helix_i, helix_j)))
        crossover_positions.setdefault(key, set()).add(int(nt_i))

    cleaned_xovers = []
    for (helix_i, helix_j), positions in sorted(crossover_positions.items()):
        sorted_positions = sorted(positions)
        if len(sorted_positions) % 2 != 0:
            raise RuntimeError('ERROR: Expected paired staple crossover positions.')

        if staple_dirs[helix_i] == staple_dirs[helix_j]:
            raise RuntimeError('ERROR: Staple crossovers should connect anti-parallel helices.')

        if staple_dirs[helix_i]:
            forward_helix, reverse_helix = helix_i, helix_j
        else:
            forward_helix, reverse_helix = helix_j, helix_i

        for nt1, nt2 in zip(sorted_positions[0::2], sorted_positions[1::2]):
            if abs(nt2 - nt1) != 1:
                raise RuntimeError('ERROR: Expected consecutive nucleotides staple crossover.')

            cleaned_xovers.append([local_to_global[forward_helix], nt1, local_to_global[reverse_helix], -1])
            cleaned_xovers.append([local_to_global[reverse_helix], nt2, local_to_global[forward_helix], -1])

    return np.array(cleaned_xovers)

def _recombine_crossovers_using_staple_args(graph, local_ids, staple_args):
    """ Uses the min run distance in staple_args to recombine staple crossovers while ensuring that helices within
    a bundle that are neighbors are still connected through some combination of staple / scaffold crossovers.
    """
    min_run_dist = staple_args.min_run_post_xover

    for i in local_ids:
        i_count = {}
        i_xovers_to_fix = set()

        edges_in_helix = [edge for edge in graph.edges if (edge[0][0] == i and edge[1][0] != i) or
                          (edge[0][0] != i and edge[1][0] == i)]

        # Grab unique values in edges_in_helix that are not i:
        unique_neighbors = set()
        for edge in edges_in_helix:
            if edge[0][0] == i:
                unique_neighbors.add(edge[1][0])
            else:
                unique_neighbors.add(edge[0][0])

        # Reformat into "pairs of two" which represent holliday junctions
        pairs_of_two = []
        for e in range(0, len(edges_in_helix), 2):
            p1, p2 = edges_in_helix[e], edges_in_helix[e + 1]
            if abs(p1[0][1] - p2[0][1]) != 1:
                raise RuntimeError('ERROR: Expected consecutive nucleotides staple crossover.')
            pairs_of_two.append((p1[0][0], p1[1][0], (p1[0][1], p2[0][1])))  # (helix_i, helix_j, (nt_i, nt_j))

        # Because pairs_of_two are consecutive, we compare each value to it's two neighbors using min_run_dist
        for j in range(len(pairs_of_two)):
            cur_pair = pairs_of_two[j]
            helix_i, helix_j, (nt_i, nt_j) = cur_pair

            prev_pair = pairs_of_two[j - 1] if j > 0 else None
            next_pair = pairs_of_two[j + 1] if j < len(pairs_of_two) - 1 else None

            # If the current pair is within min_run_dist of the previous or next pair, store:
            # Note we check nt_i to pos 1 and nt_j to pos 0 due to the directionality of the staple run
            if prev_pair and abs(nt_i - prev_pair[2][1]) < min_run_dist:
                pair = tuple(sorted([cur_pair, prev_pair]))
                i_xovers_to_fix.add(pair)

            elif next_pair and abs(nt_j - next_pair[2][0]) < min_run_dist:
                pair = tuple(sorted([cur_pair, next_pair]))
                i_xovers_to_fix.add(pair)

        for pair in pairs_of_two:
            h1, h2, _ = pair
            h1, h2 = min(h1, h2), max(h1, h2)
            if (h1, h2) not in i_count:
                i_count[(h1, h2)] = 1
            else:
                i_count[(h1, h2)] += 1

        # Finally, we can fix the values in i_xovers_to_fix using the above i_count to ensure that the helix
        # connections (via the staples) remain similarly valued:
        already_removed = set()
        for fix in i_xovers_to_fix:
            p1, p2 = fix
            neighbors1, neighbors2 = (min(p1[0], p1[1]), max(p1[0], p1[1])), (min(p2[0], p2[1]), max(p2[0], p2[1]))

            # Now use the i_count to determine which crossover to replace:
            test_edge = ((p1[0], p1[2][0]), (p1[1], p1[2][0]))
            if i_count[neighbors1] >= i_count[neighbors2] and test_edge not in already_removed:
                edge1_to_remove = ((p1[0], p1[2][0]), (p1[1], p1[2][0]))
                edge2_to_remove = ((p1[0], p1[2][1]), (p1[1], p1[2][1]))
                repair_edge1 = ((p1[0], p1[2][0]), (p1[0], p1[2][1]))
                repair_edge2 = ((p1[1], p1[2][0]), (p1[1], p1[2][1]))
                i_count[neighbors1] -= 1
            else:
                edge1_to_remove = ((p2[0], p2[2][0]), (p2[1], p2[2][0]))
                edge2_to_remove = ((p2[0], p2[2][1]), (p2[1], p2[2][1]))
                repair_edge1 = ((p2[0], p2[2][0]), (p2[0], p2[2][1]))
                repair_edge2 = ((p2[1], p2[2][0]), (p2[1], p2[2][1]))
                i_count[neighbors2] -= 1

            # Remove and add edges as defined in if / else above:
            if edge1_to_remove in already_removed:
                continue
            graph.remove_edge(*edge1_to_remove)
            graph.remove_edge(*edge2_to_remove)
            graph.add_edge(*repair_edge1)
            graph.add_edge(*repair_edge2)
            already_removed.add(edge1_to_remove)
            already_removed.add(edge2_to_remove)

def _convert_array_to_graph(potential_staple_xovers):
    """ Converts the potential staple crossovers array into a graph representation using networkx """
    g = nx.Graph()
    num_helices, num_nts = potential_staple_xovers.shape
    for helix_idx in range(num_helices):
        helix = potential_staple_xovers[helix_idx]
        nts_for_nodes = np.where(helix != -2)[0]

        for i in range(len(nts_for_nodes) - 1):
            pos1 = (int(helix_idx), int(nts_for_nodes[i]))
            pos2 = (int(helix_idx), int(nts_for_nodes[i + 1]))
            g.add_edge(pos1, pos2)

    # There should be 2*helices number of degree 1 nodes when initializing the graph as each helix is initially a
    # straight staple run with a 3' and 5' end
    num_degree_1_nodes = [n for n, d in g.degree() if d == 1]
    if len(num_degree_1_nodes) != 2 * num_helices:
        raise RuntimeError('ERROR: Unexpected number of degree 1 nodes in graph.')
    return g

def _remove_isolated_crossover_positions(candidates):
    """ Scaffold / stpale crossovers are not allowed to be within a set range of each other, and sometimes this
    procedure results in isolated crossover positions (i.e., a crossover from h1 -> h2 at nt1 but not at nt2
    """
    candidates_set = set(candidates)
    return [val for val in candidates if (val - 1 in candidates_set or val + 1 in candidates_set)]

def _apply_all_xovers(graph, xover_dict, helix_staple_crossovers_added):
    """ Loops over the xover_indices keys until no more valid crossovers can be applied, then stores the keys
    in the helix_staple_crossovers_added set to ensure that we do not repeat the same pairs of helices.
    """
    last_applied = []
    each_pair_cur_idx = {key: 0 for key in xover_dict.keys()}

    while True:
        progress = False

        for key in xover_dict.keys():
            if key in helix_staple_crossovers_added:
                continue

            helix_i, helix_j = key
            candidates = xover_dict[key]
            clean_candidates = _remove_isolated_crossover_positions(candidates)
            idx = each_pair_cur_idx[key]

            while idx + 1 < len(clean_candidates):
                i_pos = int(clean_candidates[idx])
                j_pos = int(clean_candidates[idx + 1])

                node_i_ipos = (helix_i, i_pos)
                node_i_jpos = (helix_i, j_pos)
                node_j_ipos = (helix_j, i_pos)
                node_j_jpos = (helix_j, j_pos)
                edges_to_remove = [
                    (node_i_ipos, node_i_jpos),
                    (node_j_ipos, node_j_jpos)
                ]

                if not all(graph.has_edge(u, v) for u, v in edges_to_remove):
                    if (idx + 3) <= len(clean_candidates):
                        idx += 2
                        continue
                    else:
                        break

                graph.remove_edge(node_i_ipos, node_i_jpos)
                graph.remove_edge(node_j_ipos, node_j_jpos)

                graph.add_edge(node_i_ipos, node_j_ipos)
                graph.add_edge(node_i_jpos, node_j_jpos)

                last_applied.append((i_pos, j_pos))
                each_pair_cur_idx[key] = idx + 2  # advance to next pair for this key
                progress = True
                break

        if not progress:
            break

def _add_all_staple_crossovers(graph, potential_staple_xovers):
    """ Similar to the caDNAno auto-staple functionality, this function will add in all staple crossovers to the graph
    which will then be used to optimize the staple design after the fact.
    """
    helix_staple_crossovers_added = set()  # Do not repeat the pairs of helices in below for loop

    ordered_indices = _order_by_nearest_neighbors(potential_staple_xovers)
    for i in ordered_indices:
        helix_i = potential_staple_xovers[i]
        nearest_neighbors = np.unique(helix_i[np.where((helix_i != -2) & (helix_i != -1) & (helix_i != -3))[0]])
        xover_indices = {}
        for j in nearest_neighbors:
            helix_j = potential_staple_xovers[j]
            j_in_i = np.where(helix_i == j)[0]
            i_in_j = np.where(helix_j == i)[0]
            crossovers_both_listed = j_in_i[np.isin(j_in_i, i_in_j)]  # Only store the values found in both
            xover_indices[(i, int(j))] = crossovers_both_listed

        # Overall, we are going to want to alternate between the xover_indices keys to ensure that we incorporate
        # the min_run_post_xover condition, as if we just apply all over xover_indices[first_key] then we might
        # never add any crossovers to the xover_indices[second_key] and vice versa.
        _apply_all_xovers(graph, xover_indices, helix_staple_crossovers_added)

        for key in xover_indices.keys():
            helix_staple_crossovers_added.add(key)
            helix_staple_crossovers_added.add((key[1], key[0]))

def _order_by_nearest_neighbors(potential_staple_xovers):
    """ orders the indices of the stpale_xovers array by the fewest number of nearest neighbors first """
    nearest_count = {}
    for i in range(len(potential_staple_xovers)):
        helix = potential_staple_xovers[i]
        nearest_neighbors = np.unique(helix[np.where((helix != -2) & (helix != -1) & (helix != -3))[0]])
        nearest_count[i] = len(nearest_neighbors)

    ordered_indices = sorted(nearest_count, key=nearest_count.get)
    return ordered_indices

def _convert_local_to_global(hb_idx, local_ids, design: HadoNucleotideModel):
    """ Converts local indices to global indices based on the helix bundle index """
    local_to_global = {}
    global_to_local = {}
    helix_to_bundle = design.get_helix_to_bundle()
    potential_global_ids = np.where(helix_to_bundle == hb_idx)[0]
    for local_id in local_ids:
        local_to_global[local_id] = potential_global_ids[local_id]
        global_to_local[potential_global_ids[local_id]] = local_id
    return local_to_global, global_to_local