from __future__ import annotations

import numpy as np


def get_ordered_scaffold_nts(model):
    if len(model._scaffold_start_point) != 2:
        raise ValueError("ERROR: Start point must first be identified.")

    start_helix, start_nt = model._scaffold_start_point
    crossovers = model.get_scaffold_crossovers()

    nts_in_order = []
    cur_dir, cur_nt, cur_helix = model._scaffold_dirs[start_helix], start_nt, start_helix
    max_nts = np.sum(model._scaffold_nucleotides)
    count = 0
    while count <= max_nts:
        nts_in_order.append((cur_helix, cur_nt))
        cur_nt = cur_nt + 1 if cur_dir else cur_nt - 1
        count += 1

        crossovers_on_cur_helix = crossovers[crossovers[:, 0] == cur_helix]
        for crossover in crossovers_on_cur_helix:
            _, nt_xover, next_helix, maybe_next_nt = crossover
            if nt_xover == cur_nt:
                nts_in_order.append((cur_helix, cur_nt))
                count += 1
                cur_helix = next_helix
                cur_dir = model._scaffold_dirs[cur_helix]
                if maybe_next_nt != -1:
                    cur_nt = maybe_next_nt

        if count == max_nts:
            if cur_nt != nts_in_order[0][1]:
                raise RuntimeError("ERROR: Scaffold routing incomplete.")
            break

    return nts_in_order


def get_ordered_staple_nts(model):
    """Collect all staples in 5-prime to 3-prime order from model-level connectivity."""
    active_nts = _get_active_staple_nts(model)
    successors = _build_staple_successor_map(model, active_nts)
    _remove_staple_break_edges(model, successors)
    _apply_staple_crossovers(model, successors)
    return _walk_staple_successor_paths(active_nts, successors)


def _get_active_staple_nts(model):
    return {
        (int(helix), int(nt_idx))
        for helix, nt_idx in zip(*np.where(model._staple_nucleotides))
    }


def _build_staple_successor_map(model, active_nts):
    successors = {}
    for helix in range(model._staple_nucleotides.shape[0]):
        for nt_idx in np.flatnonzero(model._staple_nucleotides[helix]):
            nt_idx = int(nt_idx)
            next_nt = nt_idx + 1 if model._staple_dirs[helix] else nt_idx - 1
            current = (int(helix), nt_idx)
            next_position = (int(helix), int(next_nt))
            if next_position in active_nts:
                successors[current] = next_position
    return successors


def _remove_staple_break_edges(model, successors):
    for brk in model._staple_breaks:
        helix, nt1, nt2 = int(brk[0]), int(brk[1]), int(brk[2])
        if nt2 == -1:
            continue

        nt1, nt2 = min(nt1, nt2), max(nt1, nt2)
        if model._staple_dirs[helix]:
            three_prime, five_prime = (helix, nt1), (helix, nt2)
        else:
            three_prime, five_prime = (helix, nt2), (helix, nt1)

        if successors.get(three_prime) == five_prime:
            del successors[three_prime]


def _apply_staple_crossovers(model, successors):
    for crossover in model.get_staple_crossovers():
        sender_helix = int(crossover[0])
        three_prime = int(crossover[1])
        receiver_helix = int(crossover[2])
        five_prime = three_prime if int(crossover[3]) == -1 else int(crossover[3])

        sender = (sender_helix, three_prime)
        receiver = (receiver_helix, five_prime)
        _remove_predecessor_edge(successors, receiver)
        successors[sender] = receiver


def _remove_predecessor_edge(successors, receiver):
    for current, next_position in list(successors.items()):
        if next_position == receiver:
            del successors[current]
            return


def _walk_staple_successor_paths(active_nts, successors):
    predecessors = set(successors.values())
    starts = sorted(active_nts - predecessors)
    all_staples = []
    seen = set()

    for start in starts:
        if start not in seen:
            all_staples.append(_walk_staple_successors(start, active_nts, successors, seen))

    for start in sorted(active_nts - seen):
        if start not in seen:
            all_staples.append(_walk_staple_successors(start, active_nts, successors, seen))

    return all_staples


def _walk_staple_successors(start, active_nts, successors, seen):
    current = start
    path = []
    while current in active_nts and current not in seen:
        path.append(current)
        seen.add(current)
        next_position = successors.get(current)
        if next_position is None or next_position == start:
            break
        current = next_position
    return path
