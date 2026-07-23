from __future__ import annotations

import math
from collections import defaultdict, deque

import networkx as nx
import numpy as np


def simple_autobreak(graph, design, staple_args, largest_staple_length_for_dp: int = 500):
    """Performs a simple autobreak procedure using dynamic programming to select the globally optimal break
    points based on the StapleArgs parameters. The DP approach is used once staples are all beneath the
    largest_staple_length_for_dp threshold for computational efficiency.

    Overall, this algorithm can get quite expensive for very large staple sets (e.g., larger than M13/P8634 DNA
    origami scaffolds).
    """
    staple_breaks = []
    staples_to_repair = []
    min_length = staple_args.min_length_after_break
    max_length = staple_args.max_length_after_break
    staple_dirs = design.get_staple_directions()
    total_cost = 0

    sorted_staples, _extra_node_breaks = _collect_staples_from_graph(
        graph, largest_staple_length_for_dp, staple_dirs, staple_breaks
    )

    target = staple_args.target_staple_length
    for staple in sorted_staples:
        deg_1_nodes = any(graph.degree(n) == 1 for n in staple)
        circular = not deg_1_nodes
        if len(staple) < min_length:
            if deg_1_nodes:
                staples_to_repair.append(staple)
            else:
                total_cost += 1e6

        elif len(staple) > max_length:
            all_breaks, final_cost = _breakup_long_staple(
                target, staple, graph, staple_dirs, staple_args,
            )
            total_cost += final_cost
            if all_breaks is not None:
                staple_breaks.extend(all_breaks)
            continue

        elif min_length <= len(staple) <= max_length and not circular:
            continue

        else:
            staple_break = _break_single_staple(staple, graph, staple_dirs)
            total_cost += (len(staple) - target) ** 2
            if staple_break is not None:
                staple_breaks.append(staple_break)
            continue

    return staple_breaks, staples_to_repair, total_cost


def _collect_staples_from_graph(graph, staple_len_to_break, staple_dirs, staple_breaks):
    graph = graph.copy()
    og_sorted_staples = sorted(nx.connected_components(graph), key=len)
    new_staples = []
    staple_indices_to_remove = []
    all_nbs = []
    for si, s in enumerate(og_sorted_staples):
        if len(s) > staple_len_to_break:
            init_breaks, initial_broken_staples, node_breaks = _break_initial_very_long_staples(
                s, staple_len_to_break, graph, staple_dirs
            )

            if len(init_breaks) == 0:
                raise Exception('ERROR: Unable to break up very long staples')

            staple_indices_to_remove.append(si)
            new_staples.extend(initial_broken_staples)
            staple_breaks.extend(init_breaks)
            all_nbs.extend(node_breaks)

    sorted_staples = [
        staple for i, staple in enumerate(og_sorted_staples)
        if i not in staple_indices_to_remove
    ]

    for i in new_staples:
        potential_break_points = _remove_invalid_staple_break_points(i, graph)
        ordered_staple, _ = _order_staple_nodes(i, graph, potential_break_points)
        if len(ordered_staple) == len(i):
            sorted_staples.append(set(i))
        else:
            raise ValueError('ERROR: Missing nucleotides in initial breakup of staples, likely an error in graph '
                             'creation')
    for i in staple_breaks:
        graph.remove_edge((i[0], i[1]), (i[0], i[2]))
    return sorted_staples, all_nbs

def _break_single_staple(staple, graph, staple_dirs):
    """ Finds a single break point in the staple that is not near any xovers """
    potential_break_points = _remove_invalid_staple_break_points(staple, graph)

    if len(potential_break_points) == 0:
        return None

    # Next, with potential_break_points found we look to select one N_point in the staple:
    subgraph = graph.subgraph(staple).copy()
    candidate_breaks = _select_start_break(subgraph, potential_break_points)
    if candidate_breaks:
        b = candidate_breaks[0]
        break_pt = (b[0][0], b[0][1], b[1][1])
    else:
        # I don't really think this else can ever trigger due to previous conditions, but just in case:
        b = list(potential_break_points)[0]
        if staple_dirs[b[0]]:
            break_pt = (b[0], b[1], b[1] - 1)
        else:
            break_pt = (b[0], b[1], b[1] + 1)

    return break_pt

def _breakup_long_staple(target, staple, graph, staple_dirs, staple_args,):
    """
    Breaks a long staple into smaller segments that are strictly within the min/max length bounds
    using a dynamic programming approach with soft constraints.

    Overall, we calculate the max number of breaks a staple needs based on the target length and then procedurally
    calculate the staple lengths for all break combinations up to that max number of breaks. Then, we backtrack thru
    the filled in cost matrix to find the break points that lead to a set of staple segments that are as close to the
    target length as possible
    """
    graph = graph.copy()
    min_length = staple_args.min_length_after_break
    max_length = staple_args.max_length_after_break
    potential_break_points = _remove_invalid_staple_break_points(staple, graph)

    if len(potential_break_points) == 0:
        return None, np.inf

    cur_staple, ib = _order_staple_nodes(staple, graph, potential_break_points)
    if cur_staple is None:
        return None, np.inf

    final_breaks = [] if ib is None else [ib[1]]

    # Determine a reasonable range for the number of breaks
    min_breaks_needed = math.ceil(len(cur_staple) / max_length) - 1
    max_breaks_needed = math.floor(len(cur_staple) / min_length) - 1

    if max_breaks_needed < 0 or min_breaks_needed > max_breaks_needed:
        return None, np.inf

    max_k = max_breaks_needed
    break_indices = sorted([cur_staple.index(p) for p in potential_break_points])
    locations = [0] + break_indices if break_indices[0] != 0 else break_indices
    num_locations = len(locations)
    staple_end_index = len(cur_staple)

    dp_cost = [[math.inf] * num_locations for _ in range(max_k + 1)]
    dp_path = [[-1] * num_locations for _ in range(max_k + 1)]

    def _cost(start, end):
        seg_len = end - start
        penalty = 0
        if seg_len < min_length:
            # penalty += 1e6
            return math.inf  # Absolutely restrict staple size to a minimum bound

        elif seg_len > max_length:
            penalty += 1e6  # Smaller penalty for longer staples as too small staples are more expensive typically

        return (seg_len - target) ** 2 + penalty

    for i in range(1, num_locations):
        dp_cost[1][i] = _cost(locations[0], locations[i])

    # Calculate length of resulting staples based on what break points are selected so we can then select the best N
    # breaks leading to minimal MSE to a target
    for k in range(2, max_k + 1):
        for i in range(k, num_locations):
            for j in range(k - 1, i):
                cost = _cost(locations[j], locations[i])
                if dp_cost[k - 1][j] == math.inf:
                    continue

                new_total_cost = dp_cost[k - 1][j] + cost
                if new_total_cost < dp_cost[k][i]:
                    dp_cost[k][i] = new_total_cost
                    dp_path[k][i] = j

    # Find the best solution by checking the final segment and determine proper # of breaks
    min_final_cost = math.inf
    last_break_index = -1
    best_k = -1

    for k in range(min_breaks_needed, max_k + 1):
        for i in range(k, num_locations):
            if dp_cost[k][i] == math.inf:
                continue

            final_segment_cost = _cost(locations[i], staple_end_index)
            if final_segment_cost == math.inf:
                continue

            current_total_cost = dp_cost[k][i] + final_segment_cost
            if current_total_cost < min_final_cost:
                min_final_cost = current_total_cost
                last_break_index = i
                best_k = k

    if last_break_index == -1:
        # No partition exists that satisfies the min/max length constraints.
        return None, np.inf

    current_index = last_break_index
    optimal_breaks_indices = []
    for k in range(best_k, 0, -1):
        if current_index == -1:
            raise RuntimeError("ERROR: Long staple auto-break feature back-tracking failed.")
        optimal_breaks_indices.append(locations[current_index])
        current_index = dp_path[k][current_index]

    optimal_breaks_indices.reverse()
    all_breaks = [cur_staple[i] for i in optimal_breaks_indices]
    final_breaks += all_breaks

    best_breaks_structured = _structure_breaks(final_breaks, staple_dirs)
    return best_breaks_structured, min_final_cost

def _break_initial_very_long_staples(staple, staple_len_to_break, graph, staple_dirs):
    """ Breaks up initial very long staples into more manageable chunks for the DP approach to _breakup_long_staple
    This function is used to avoid expensive calculations in DP (here we just pick a middle-most point to get staples
    beneath a length of staple_len_to_break.
    """
    potential_break_points = _remove_invalid_staple_break_points(staple, graph)
    if len(potential_break_points) == 0:
        return [], [], []

    # ordered, initial_break = _order_staple_nodes(staple, graph, potential_break_points)
    cur_staple, ib = _order_staple_nodes(staple, graph, potential_break_points)

    staples_to_break = deque([cur_staple])
    finished_staples = []
    break_nodes = [] if ib is None else [ib[1]]

    update = True
    midpoint = np.inf
    cur_staple_to_break = None

    while staples_to_break or cur_staple_to_break is not None:
        if update:
            cur_staple_to_break = staples_to_break.popleft()
            midpoint = len(cur_staple_to_break) // 2

            potential_break_points = _remove_invalid_staple_break_points(cur_staple_to_break, graph)
            if len(potential_break_points) == 0:
                return None, np.inf

        if midpoint >= len(cur_staple_to_break):
            raise RuntimeError("Failed to find valid break point in staple.")

        node_at_midpoint = cur_staple_to_break[midpoint]

        if node_at_midpoint in potential_break_points:
            break_nodes.append(node_at_midpoint)

            prefix = cur_staple_to_break[:midpoint]
            suffix = cur_staple_to_break[midpoint:]

            if len(prefix) <= staple_len_to_break:
                finished_staples.append(prefix)
            else:
                staples_to_break.append(prefix)

            if len(suffix) <= staple_len_to_break:
                finished_staples.append(suffix)
            else:
                staples_to_break.append(suffix)

            cur_staple_to_break = None
            update = True
        else:
            midpoint += 1
            update = False

    best_breaks_structured = _structure_breaks(break_nodes, staple_dirs)
    return best_breaks_structured, finished_staples, break_nodes

def _structure_breaks(best_breaks, staple_dirs):
    """ Structures the best_breaks array to be compatible with design export """
    best_breaks_structured = []
    for b in best_breaks:
        if isinstance(b[0], tuple):
            # Case that handled when an initial break is added
            best_breaks_structured.append((b[0][0], b[0][1], b[1][1]))

        else:
            # Need to modify based on staple direction:
            if staple_dirs[b[0]]:
                best_breaks_structured.append((b[0], b[1], b[1] - 1))
            else:
                best_breaks_structured.append((b[0], b[1], b[1] + 1))
    return best_breaks_structured

def _select_start_break(subgraph, potential_break_points):
    """ Used in the break_long_staple to select an initial staple_break location as well as the break_short_staple """
    # First we sort the potential_break_points by looking for the "middle-most" value along the longest helix (i.e.,
    # helix with most potential_break_points):
    helix_dict = defaultdict(list)
    for h, nt in potential_break_points:
        helix_dict[h].append(nt)
    longest_helix = max(helix_dict.items(), key=lambda x: len(x[1]))[0]
    nt_list = sorted(helix_dict[longest_helix])
    midpoint = nt_list[len(nt_list) // 2]
    sorted_nts = sorted(nt_list, key=lambda x: abs(x - midpoint))

    sorted_break_points = [(longest_helix, nt) for nt in sorted_nts]

    # Begin looping for a candidate edge to break. Returning None is fine as previous function handles that case
    candidate_edges = []
    for u in sorted_break_points:
        for v in subgraph.neighbors(u):
            if v in potential_break_points and subgraph.has_edge(u, v):
                candidate_edges.append((u, v))
                break

    if len(candidate_edges) == 0:
        candidate_edges = None
    return candidate_edges

def _order_staple_nodes(staple, graph, potential_break_points):
    """ Orders a set of nodes from 5' to 3' from the graph """
    G_staple = graph.subgraph(staple).copy()
    initial_break = None

    if nx.is_connected(G_staple) and all(deg == 2 for _, deg in G_staple.degree()):
        candidate_edges = _select_start_break(G_staple, potential_break_points)

        # If there are multiple candidate_edges just use the first
        if candidate_edges:
            for candidate in candidate_edges:
                temp = G_staple.copy()
                temp.remove_edge(*candidate)
                try:
                    start_node = next(n for n, d in temp.degree() if d == 1)
                except StopIteration:
                    continue
                ordered = list(nx.dfs_preorder_nodes(temp, source=start_node))
                return ordered, candidate
        else:
            # If candidate_edges does not have any valid break points, just return none. This is likely a just-too-long
            # strand (e.g., ~66 nts)
            return None, None

    # If the above fails (i.e., there exists a 3' and 5' end already) then:
    try:
        temp = [n for n, d in G_staple.degree() if d == 1]
        temp2 = [n for n, d in G_staple.degree() if d == 0]
        if len(temp) > 2:
            raise ValueError(f'ERROR: Too many degree-1 nodes: {temp}, staple / graph is ill-defined.')
        if len(temp2) > 0:
            raise ValueError(f'ERROR: Found degree-0 nodes: {temp2}, staple / graph is ill-defined.')
        start_node = next(n for n, d in G_staple.degree() if d == 1)
        ordered = list(nx.dfs_preorder_nodes(G_staple, source=start_node))
    except StopIteration:
        raise ValueError("ERROR: Could not find a degree-1 node, staple is not a valid path / cycle")

    return ordered, initial_break

def _remove_invalid_staple_break_points(staple, graph):
    """ Removes potential break points in the staple that are too close to any staple / scaffold xovers """
    potential_break_points = set()
    for p in staple:
        if graph.nodes[p]['valid_breakpoint']:
            potential_break_points.add(p)
    return potential_break_points

