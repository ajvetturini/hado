from __future__ import annotations

import numpy as np

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.automation.diagnostics.visualization import build_breakpoint_heatmap


def _select_scaffold_start_break_point(graph, breaks, staple_args):
    """ Determines which nucleotide position is suitable for the scaffold circularity break point """
    run_length = staple_args.min_run_post_xover

    for b in breaks:
        helix, nt = b[0], b[1]
        for i in range(nt-run_length, nt+run_length+1):
            if (helix, i) in graph.nodes:
                graph.nodes[(helix, i)]['valid_breakpoint'] = False

    valid_breaks = [
        n for n, data in graph.nodes(data=True)
        if data.get('valid_breakpoint', False) is True
    ]

    if len(valid_breaks) == 0:
        raise ValueError('ERROR: Unable to find valid break point for scaffold')

    return valid_breaks[0]

def _convert_breaklist_using_lg_map(break_list, local_to_global):
    """ Corrects helix indices using local -> global map """
    converted = []
    for b in break_list:
        h, n1, n2 = b
        converted.append((local_to_global[h], n1, n2))
    return converted

def _convert_polyt_staples_using_lg_map(polyt_list, local_to_global):
    """ Converts the candidate polyt sequences using local -> global map """
    converted = []
    for pt in polyt_list:
        converted_strand = []
        for helix_nt_pair in pt:
            h, n1 = helix_nt_pair
            converted_strand.append((local_to_global[h], n1))
        converted.append(converted_strand)
    return converted

def _label_nodes_for_breaks(graph, all_scaffold_xovers, all_staple_xovers, design: HadoNucleotideModel,
                            new_staple_nts, staple_args, show_breakpoint_labels, diagnostics=None):
    """ labels nodes as either able to be removed (or not) based on the staple distance args for autobreak function """
    staple_xovers = np.array(all_staple_xovers)
    scaffold_xovers = np.array(all_scaffold_xovers)
    scaffold_nts = design.get_scaffold_nucleotides()
    min_run_post_xover = staple_args.min_run_post_xover
    min_run_post_bundle_connection = staple_args.min_run_post_bundle_connection

    def _process_bounds(xovers_array, helix_h, nt_n):
        """ Checks relation of a nucleotide position and determines if a crossover can be placed based on args """
        crossovers_in_helix = np.where(np.any(xovers_array[:, [0, 2]] == helix_h, axis=1))[0]
        for c in crossovers_in_helix:
            xover_c = xovers_array[c]
            if xover_c[-1] == -1:
                crossover_nt = xover_c[1]
            else:
                crossover_nt = xover_c[1] if xover_c[0] == helix_h else xover_c[3]

            # Must check min_run_post_xover+1 because I am placing 4-way junctions in my staple crossovers (i.e.,
            # helix i and helix j may be connected via crossovers at (nt_n, nt_n+1) and so we dont want to
            # break one half of the 4-way junction
            check_on = abs(nt_n - crossover_nt)
            if check_on != 0 and check_on < min_run_post_xover:
                return True
        return False

    def _bounded_by_crossovers(h, nt):
        """ Check if a (helix, nucleotide) is within the staple_args range and thus can NOT be set as a breakpoint """
        scaf_nts = np.where(scaffold_nts[h])[0]
        stap_nts = np.where(new_staple_nts[h])[0]
        scaf_start, scaf_stop = min(scaf_nts), max(scaf_nts)
        stap_start, stap_stop = min(stap_nts), max(stap_nts)

        # Be extra careful checking start / end points of helices. We do not want a break point in the ssDNA overhang
        # as that will not fold properly. Since all helix ends (start / stop) connect two bundles, we check against
        # min_run_post_bundle_connection instead of min_run_post_xover
        ds_start = max(scaf_start, stap_start)
        ds_stop = min(scaf_stop, stap_stop)
        if nt < (ds_start + min_run_post_bundle_connection) or nt > (ds_stop - min_run_post_bundle_connection + 1):
            return True

        # If the nt is not located at the start / end of a helix run, we then need to check its relation to any previous
        # crossovers:
        invalid_staples = _process_bounds(staple_xovers, h, nt)
        invalid_scaffold = _process_bounds(scaffold_xovers, h, nt)

        if invalid_staples or invalid_scaffold:
            return True

        # Otherwise return False signalling "not bounded, eligible for break point"
        return False

    for n in graph.nodes:
        helix, nt_position = n
        graph.nodes[n]['valid_breakpoint'] = not _bounded_by_crossovers(helix, nt_position)

    _invalidate_free_end_runs_before_first_crossover(graph, design, new_staple_nts)

    viz_data = []
    for n in graph.nodes:
        helix, nt_position = n
        viz_data.append({
            'helix': helix,
            'nt_position': nt_position,
            'valid_breakpoint': graph.nodes[n]['valid_breakpoint'],
        })

    if show_breakpoint_labels:
        fig = build_breakpoint_heatmap(viz_data)
        if diagnostics is not None:
            diagnostics.record_figure('autostaple', 'breakpoint_heatmap', fig)

    return graph


def _invalidate_free_end_runs_before_first_crossover(graph, design: HadoNucleotideModel, new_staple_nts):
    """Prevent free-end overhangs from becoming their own short staples."""
    if not hasattr(design, 'check_free_ends'):
        return graph

    def _mark_from_endpoint(start):
        if start not in graph:
            return

        previous = None
        current = start
        while True:
            graph.nodes[current]['valid_breakpoint'] = False
            next_nodes = [n for n in graph.neighbors(current) if n != previous]
            if not next_nodes:
                break

            next_node = next_nodes[0]
            if next_node[0] != current[0]:
                break

            previous, current = current, next_node

    for helix, nts in enumerate(new_staple_nts):
        active_nts = np.where(nts)[0]
        if len(active_nts) == 0:
            continue

        free_min, free_max = design.check_free_ends(helix)
        if free_min:
            _mark_from_endpoint((helix, int(active_nts[0])))
        if free_max:
            _mark_from_endpoint((helix, int(active_nts[-1])))

    return graph
