import networkx as nx
import numpy as np
from copy import deepcopy

from hado.core.automation.autostaple.autobreak import simple_autobreak
from hado.core.automation.autostaple.breakpoint_labels import (
    _convert_breaklist_using_lg_map,
    _convert_polyt_staples_using_lg_map,
    _label_nodes_for_breaks,
    _select_scaffold_start_break_point,
)
from hado.core.automation.autostaple.crossover_conflicts import _get_scaffold_xovers
from hado.core.automation.autostaple.staple_bundle_graph import _staple_helix_bundle
from hado.core.automation.autostaple.staple_end_connections import _add_staple_map_crossovers
from hado.core.automation.autostaple.staple_extensions import (
    _add_blunt_ends_to_degree_1_short_staples,
    _make_staple_ends_flush,
)
from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.automation.routing.lattice import get_lattice_config
from hado.core.utils import MAX_NUM_NTS_FOR_HADO_AUTOBREAK

def autostaple_hollowframe(design: HadoNucleotideModel,
                           staple_map: dict,
                           final_positions: dict,
                           **kwargs
                           ) -> tuple[HadoNucleotideModel, float, dict]:
    """
    Autostapling for hollowframe nanostructures starts by adding in all staple crossovers. This is done because in
    hollowframe nanostructures each helix has 2 neighbors only (so there will not be a case of very short staple runs
    before re-crossing over as in the case when helices have 3 neighbors). We do not add crossovers if there is a
    nearby scaffold crossover as that is kinetically unfavorable.

    After adding all staple crossovers, however, we then need to define break points so that all staples are between
    20 and 60 nucleotides in length (so they can be ordered from IDT / ssDNA supplier). Dynamic Programming is used
    here define optimal staple sequences that minimizes the MSE from staple lengths to a set target staple length
    (e.g., 42 nucleotides).

    :param design: The HadoNucleotideModel that has had it's scaffold routed, grid positions defined, and mitering
        completed.
    :type design: HadoNucleotideModel

    :param staple_map: A mapping of helices to their connection (i.e., {helix_i: helix_j} where helix_i and helix_j
        belong to separate helix bundles).
    :type staple_map: dict

    :param final_positions: A mapping of helices 3' and 5' ends to their (designed) final Euclidean X Y Z position.
        This is used to determine the number of ssDNA overhangs.
    :type final_positions: dict

    :return: A revised HadoNucleotideModel state:
        * **design** (*HadoNucleotideModel*): Final design post-stapling that will then be used for exporting sequences.
    :rtype: HadoNucleotideModel
    """
    staple_args = design.staple_args
    lattice_type = design.scaffold_args.lattice_type
    staple_xovers = []
    verbose = kwargs.get('verbose', False)
    very_long_staple_easy_break = int(kwargs.get('base_staple_length_to_simple_break', 500))

    lattice_config = get_lattice_config(lattice_type)
    if lattice_config.period <= 0:
        raise ValueError('ERROR: Autostapling requires a periodic lattice configuration.')

    end_staple_xovers, new_staple_nts = _add_staple_map_crossovers(
        staple_map,
        design,
        final_positions,
        staple_args,
        verbose,
        kwargs.get('diagnostics'),
    )
    helix_to_bundle = design.get_helix_to_bundle()
    grid_locations = design.get_helix_bundle_grid_locations()
    staple_dirs = design.get_staple_directions()

    number_of_nucleotides = np.sum(design.get_scaffold_nucleotides())
    override = kwargs.get('override_staple_autobreak_limit', False)
    show_breakpoint_labels = kwargs.get('show_breakpoint_labels', False)
    if override:
        only_add = staple_args.only_add
    elif number_of_nucleotides > MAX_NUM_NTS_FOR_HADO_AUTOBREAK:
        # If a design is too large and the override is not set then do NOT use this feature as it is expensive
        # (the UI gives a message to the user in this case)
        only_add = True
    else:
        only_add = staple_args.only_add

    new_staple_nts = np.array(new_staple_nts)
    if not only_add and staple_args.make_flush:
        new_staple_nts = np.array(_make_staple_ends_flush(design, new_staple_nts, staple_args))

    bundle_indices = np.unique(helix_to_bundle)
    global_graph = nx.Graph()

    scaf_nts = design.get_scaffold_nucleotides()
    for i in bundle_indices:
        helix_indices = np.where(helix_to_bundle == i)[0]
        if len(helix_indices) == 1:
            # If only one helix, skip it as there are no neighbors to staple to
            continue

        grids = grid_locations[helix_indices]
        staple_nts = new_staple_nts[helix_indices]
        staple_dirs_i = staple_dirs[helix_indices]
        local_scaf_nts = scaf_nts[helix_indices]
        bundle_staple_xovers, _, graph = _staple_helix_bundle(
            only_add, i, grids, staple_nts, staple_dirs_i, local_scaf_nts, design, staple_args
        )
        staple_xovers.extend(bundle_staple_xovers)
        global_graph = nx.compose(global_graph, graph)

    # After adding crossovers, we add in the edge crossovers then perform the autobreaking:
    for ex in end_staple_xovers:
        h1, n1, h2, n2 = int(ex[0]), int(ex[1]), int(ex[2]), int(ex[3])
        global_graph.add_edge((h1, n1), (h2, n2))

    global_local_map = {i: i for i in range(len(helix_to_bundle))}
    all_scaffold_xovers = _get_scaffold_xovers(global_local_map, design, False)
    all_staple_xovers = staple_xovers + end_staple_xovers

    total_cost = 0.
    graph_before_breaks = _label_nodes_for_breaks(global_graph, all_scaffold_xovers, all_staple_xovers,
                                                  design, new_staple_nts, staple_args, show_breakpoint_labels,
                                                  kwargs.get('diagnostics'))
    graph_to_return = deepcopy(graph_before_breaks)
    if not only_add:
        # Label nodes that can be used as break points (i.e., not too close to staple / scaffold crossovers)
        break_list, poly_t_candidates, total_cost = simple_autobreak(graph_before_breaks, design, staple_args,
                                                                     very_long_staple_easy_break)

        break_list = _convert_breaklist_using_lg_map(break_list, global_local_map)
        poly_t_candidates = _convert_polyt_staples_using_lg_map(poly_t_candidates, global_local_map)

        if not staple_args.make_flush:
            new_staple_nts = _add_blunt_ends_to_degree_1_short_staples(poly_t_candidates, new_staple_nts,
                                                                       design, staple_args)

        scaffold_break_point = _select_scaffold_start_break_point(graph_before_breaks, break_list, staple_args)
    else:
        break_list = []
        scaffold_break_point = []

    design.set_staple_crossovers(np.array(all_staple_xovers))
    design.set_staple_breaks(np.array(break_list))
    design.set_staple_nucleotides(np.array(new_staple_nts))
    design.set_scaffold_start_point(np.array(scaffold_break_point))
    return design, total_cost, {'constrained_breakpoints': graph_to_return}
