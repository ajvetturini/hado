from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.automation.connections.connect_bundles import decompose_design_into_bundles, get_rotated_positions
from hado.core.automation.pipeline.types import emit_runtime_message
from hado.core.automation.diagnostics.visualization import build_mitering_figure

import numpy as np
from copy import deepcopy
from typing import Tuple

def miter_design(design: HadoNucleotideModel,
                 best_state: dict,
                 optimal_connections: dict,
                 miter_target_distance: float = 2.0,
                 **kwargs) -> Tuple[HadoNucleotideModel, np.ndarray]:
    """ This function will modify the design.scaffold_nucleotides and design.staple_nucleotides in an effort
    to miter the beam-like bundles edges to fit snug in 3D space. This function also ensure that the geometry of the
    DNA is respected.

    :param design: The HadoNucleotideModel that has had it's scaffold routed, grid positions defined, and connections paired.
    :type design: HadoNucleotideModel

    :param best_state: A mapping of helices 3' and 5' ends to their (designed) final Euclidean X Y Z position. This
        is used to determine the number of ssDNA overhangs inside of `autostaple_hollowframe`.
    :type best_state: dict

    :param optimal_connections: A mapping of 3' indices to 5' indices that are paired in the best_state.
    :type optimal_connections: dict

    :param miter_target_distance: Target euclidean distance between the 3' end of helix i to the 5' end of helix j
        as located in 3D space. Default is 2.0 nanometers (nm).
    :type miter_target_distance: float, optional

    :return: A tuple containing two elements:

        * **design** (*HadoNucleotideModel*): Final state representation of the HadoNucleotideModel prior to stapling.
        * **new_positions_for_plotting** (*dict*): Final positions of the best_state dict that was passed in after
            mitering has been applied.
    :rtype: tuple
    """
    # First collect the true grid 3D positions that are located at the MIDPOINT of the helices
    base_positions_and_axes = decompose_design_into_bundles(design)
    rotated_positions = get_rotated_positions(design, base_positions_and_axes, best_state)
    connections_dict = _convert_connections(optimal_connections)
    diagnostics = kwargs.get('diagnostics')

    if kwargs.get('miter_show_init_connections', False):
        fig = _plot_design(design, rotated_positions, optimal_connections, show_colors=kwargs.get('show_miter_colors', False),
                           cylinder_radius=kwargs.get('cylinder_radius', 0.5), cylinder_res=kwargs.get('cylinder_res', 10),
                           visual_override=kwargs.get('flip_init_connection_cylinders', False))
        if diagnostics is not None:
            diagnostics.record_figure('mitering', 'initial_connections', fig)

    helix_to_bundle = design.get_helix_to_bundle()
    idx_edge_map = design.get_idx_edge_map()
    scaffold_nts = design.get_scaffold_nucleotides()

    num_helices = len(helix_to_bundle)
    new_scaffold_nts = {i: deepcopy(scaffold_nts[i]) for i in range(num_helices)}
    middle_nts_per_helix = _get_middle_nts(new_scaffold_nts)
    new_positions_for_plotting = {i: [] for i in range(num_helices)}
    verbose = kwargs.get('verbose', False)

    # We need to miter the helix bundle 'beams' at each vertex in the design;
    unique_verts = np.unique(idx_edge_map)
    final_thresholds_met = {}
    for v in unique_verts:
        bundles_at_vertex = list(design.get_bundles_at_vertex(v))
        if len(bundles_at_vertex) == 1:
            # If only one helix bundle at a vertex, then there is no need for mitering
            # NOTE: This shouldn't raise becasue N=1 should cause an error during edge definition (the hado package
            #       is not intended for N=1 designs, see vHelix for this)
            continue

        # Otherwise, we grab all helices belonging to bundles located at this vertex:
        helices_at_v = np.where(np.isin(helix_to_bundle, bundles_at_vertex))[0]
        already_mitered_at_v = set()

        # Now loop over the helices located at this vertex and start mitering:
        for helix in helices_at_v:
            if helix in already_mitered_at_v:
                continue

            connected_helix = connections_dict[v][helix]

            # First, we trim the active nucleotide in half as the current rotated_positions are located at the mid-point
            nts_helix = _trim_half_nts(
                new_scaffold_nts[helix], helix, v, design, middle_nts_per_helix
            )
            nts_connected_helix = _trim_half_nts(
                new_scaffold_nts[connected_helix], connected_helix, v, design, middle_nts_per_helix
            )

            # Perform Mitering
            nts_mitered_helix, p_helix, nts_miters_connected, p_connected, final_dist = _miter_helices(
                nts_helix, helix, nts_connected_helix, connected_helix, v, miter_target_distance, rotated_positions,
                design,
                verbose,
                diagnostics,
            )

            # Update the scaffold nt positions:
            new_scaffold_nts[helix] = nts_mitered_helix[0]
            new_scaffold_nts[connected_helix] = nts_miters_connected[0]
            new_positions_for_plotting[helix].append((v, p_helix))
            new_positions_for_plotting[connected_helix].append((v, p_connected))
            final_thresholds_met[(helix, connected_helix)] = final_dist

            # Add to set to ensure no repeat-mitering
            already_mitered_at_v.add(helix)
            already_mitered_at_v.add(connected_helix)


    if kwargs.get('miter_show_post_mitering_positions', False):
        for v in unique_verts:
            bundles_at_vertex = list(design.get_bundles_at_vertex(v))
            if len(bundles_at_vertex) == 1:
                continue
            mitered_positions = _collect_new_positions(new_positions_for_plotting, v)
            fig = _plot_design(design, mitered_positions, optimal_connections[v],
                               show_colors=kwargs.get('show_miter_colors', False),
                               cylinder_radius=kwargs.get('cylinder_radius', 0.5),
                               cylinder_res=kwargs.get('cylinder_res', 10),
                               visual_override=kwargs.get('flip_post_mitering_cylinders', False)
                               )
            if diagnostics is not None:
                diagnostics.record_figure('mitering', f'post_mitering_vertex_{v}', fig)

    final_nts, new_ends_dict = _collect_final_nts_array(new_scaffold_nts)

    return new_positions_for_plotting, final_nts

def _miter_helices(nts_i, helix_i, nts_j, helix_j, shared_vertex, miter_target_distance, cur_positions,
                   design: HadoNucleotideModel, verbose: bool = False, diagnostics=None):
    """ Strictly uses Euclidean geometry to miter the edges for a snug fit """
    pi = cur_positions[helix_i].copy()
    pj = cur_positions[helix_j].copy()

    target_threshold = miter_target_distance
    axial_rise = design.get_axial_rise()

    initial_dist = _dist(pi, pj)
    if abs(initial_dist - target_threshold) <= axial_rise:
        return nts_i, pi, nts_j, pj, initial_dist

    elif initial_dist < target_threshold:
        # Remove the initial distance is lower than the lowest threshold, we must continue to remove nucleotides
        return _subtract_nucleotides_to_miter(
            nts_i, pi, helix_i, nts_j, pj, helix_j, shared_vertex, target_threshold, design, verbose, diagnostics
        )

    else:
        # Otherwise, the initial dist is greater than the upper threshold thus we are adding nucleotides until
        # we get beneath this range:
        return _add_nucleotides_to_miter(
            nts_i, pi, helix_i, nts_j, pj, helix_j, shared_vertex, target_threshold, design, verbose, diagnostics
        )

def _add_nucleotides_to_miter(nts_i, pi, helix_i, nts_j, pj, helix_j, shared_vertex, target_threshold,
                              design: HadoNucleotideModel, verbose, diagnostics=None):
    """ Helper to _miter_helices by adding edges until the threshold is met """
    # First, we need to define the vectors along which we will be "adding" nucleotides in:
    dv_i = _get_direction_vector_for_mitering(helix_i, design, shared_vertex, True)
    dv_j = _get_direction_vector_for_mitering(helix_j, design, shared_vertex, True)

    # Next, we will begin adding-back nucleotides to the helix until the threshold is met:
    cur_dist = _dist(pi, pj)
    axial_rise = design.get_axial_rise()

    # We want to get as close to the min distance between as possible which is why we check the lower threshold
    while cur_dist > target_threshold:
        # First we try and move both pi and pj along dv_i / dv_j respectively by the axial rise
        prev_dist = cur_dist
        pi_simul = pi + dv_i * axial_rise
        pj_simul = pj + dv_j * axial_rise
        dist_simul = _dist(pi_simul, pj_simul)

        if dist_simul < prev_dist:
            # moving both helices works, use this
            pi, pj = pi_simul, pj_simul
            nts_i = _add_nt(nts_i)
            nts_j = _add_nt(nts_j)
            cur_dist = dist_simul
        else:
            # If moving both does not decrease distance, we attempt to move ONE of pi / pj:
            pi_i_only = pi + dv_i * axial_rise
            dist_i_only = _dist(pi_i_only, pj)

            pj_j_only = pj + dv_j * axial_rise
            dist_j_only = _dist(pi, pj_j_only)

            # Identify which of the single moves are actual improvements
            possible_moves = []
            if dist_i_only < prev_dist:
                possible_moves.append({'helix': 'i', 'dist': dist_i_only, 'pos': pi_i_only})
            if dist_j_only < prev_dist:
                possible_moves.append({'helix': 'j', 'dist': dist_j_only, 'pos': pj_j_only})

            if not possible_moves:
                # BREAK if not possible moves lead to a better solution
                emit_runtime_message(
                    f'WARNING: Miter threshold not possible to be met for helix {helix_i} -> {helix_j}',
                    diagnostics=diagnostics,
                    verbose=verbose,
                    warning=True,
                )
                break

            # Otherwise, we select the best move based on whichever of nts_i / nts_j is SHORTER (we want longer beams)
            if len(possible_moves) == 1:
                best_move = possible_moves[0]
            else:
                # Both single moves are improvements, so we choose based on helix length and extend the shorter helix
                if len(np.where(nts_i[0])[0]) <= len(np.where(nts_j[0])[0]):
                    best_move = next((m for m in possible_moves if m['helix'] == 'i'))
                else:
                    best_move = next((m for m in possible_moves if m['helix'] == 'j'))

            if best_move['helix'] == 'i':
                pi = best_move['pos']
                nts_i = _add_nt(nts_i)
            else:
                pj = best_move['pos']
                nts_j = _add_nt(nts_j)

            cur_dist = best_move['dist']

        # Check if all nts have been filled in (this should raise an exception that the miter_threshold is set too low)
        period = design.get_period()
        break_loop_i = _verify_nts(nts_i, period)
        break_loop_j = _verify_nts(nts_j, period)
        if break_loop_i or break_loop_j:
            break

    return nts_i, pi, nts_j, pj, cur_dist

def _subtract_nucleotides_to_miter(nts_i, pi, helix_i, nts_j, pj, helix_j, shared_vertex, target_threshold,
                                   design: HadoNucleotideModel, verbose, diagnostics=None):
    """ Helper to _miter_helices by adding edges until the threshold is met """
    # First, we need to define the vectors along which we will be "adding" nucleotides in:
    dv_i = _get_direction_vector_for_mitering(helix_i, design, shared_vertex, False)
    dv_j = _get_direction_vector_for_mitering(helix_j, design, shared_vertex, False)
    cur_dist = _dist(pi, pj)
    axial_rise = design.get_axial_rise()

    while cur_dist < target_threshold:  # Use lower bound as check
        # First we try to move pi and pj along dv_i / dv_j respectively by the axial rise
        prev_dist = cur_dist
        pi_simul = pi + dv_i * axial_rise
        pj_simul = pj + dv_j * axial_rise
        dist_simul = _dist(pi_simul, pj_simul)

        if dist_simul > prev_dist:
            # moving both helices away works
            pi, pj = pi_simul, pj_simul
            nts_i = _subtract_nt(nts_i)
            nts_j = _subtract_nt(nts_j)
            cur_dist = dist_simul
        else:
            # Otherwise, try moving one at a time:
            pi_i_only = pi + dv_i * axial_rise
            dist_i_only = _dist(pi_i_only, pj)

            pj_j_only = pj + dv_j * axial_rise
            dist_j_only = _dist(pi, pj_j_only)

            # Identify which of the single moves are actual improvements (distance increases)
            possible_moves = []
            if dist_i_only > prev_dist:
                possible_moves.append({'helix': 'i', 'dist': dist_i_only, 'pos': pi_i_only})
            if dist_j_only > prev_dist:
                possible_moves.append({'helix': 'j', 'dist': dist_j_only, 'pos': pj_j_only})

            if not possible_moves:
                # Otherwise, if not the first pass, then we actually found the minimal distance between the points in the
                # previous step addition, so end:
                emit_runtime_message(
                    f'WARNING: Miter threshold not possible to be met for helix {helix_i} -> {helix_j}',
                    diagnostics=diagnostics,
                    verbose=verbose,
                    warning=True,
                )
                break

            # If there are possible moves, we seelct based on whichever is the LONGER helix to remove from
            if len(possible_moves) == 1:
                best_move = possible_moves[0]
            else:
                if len(np.where(nts_i[0])[0]) >= len(np.where(nts_j[0])[0]):
                    best_move = next((m for m in possible_moves if m['helix'] == 'i'))
                else:
                    best_move = next((m for m in possible_moves if m['helix'] == 'j'))
            if best_move['helix'] == 'i':
                pi = best_move['pos']
                nts_i = _subtract_nt(nts_i)
            else:  # helix == 'j'
                pj = best_move['pos']
                nts_j = _subtract_nt(nts_j)
            cur_dist = best_move['dist']

        # Check if all nts have been filled in (this should raise an exception that the miter_threshold is set too low)
        period = design.get_period()
        break_loop_i = _verify_nts(nts_i, period)
        break_loop_j = _verify_nts(nts_j, period)
        if break_loop_i or break_loop_j:
            break

    return nts_i, pi, nts_j, pj, cur_dist

def _add_nt(nts):
    """ Sets a False -> True flag in the proper direction """
    nts, add_dir, cur_idx = nts

    # Add nucleotide in the add_dir direction, then iterate cur_idx appropriately:
    nts[cur_idx] = True
    if add_dir:
        cur_idx -= 1
    else:
        cur_idx += 1

    return nts, add_dir, cur_idx

def _subtract_nt(nts):
    """ Sets a False -> True flag in the proper direction """
    nts, add_dir, cur_idx = nts

    # Remove nucleotide by first iterate cur_idx appropriately then setting to False
    if add_dir:
        cur_idx += 1
    else:
        cur_idx -= 1
    nts[cur_idx] = False

    return nts, add_dir, cur_idx

def _verify_nts(nts, period):
    """ Makes sure we are not exceeding bounds """
    nts, _, cur_idx = nts  # unpack
    half_period = period // 2
    if cur_idx == half_period or cur_idx == len(nts) - half_period:
        return True
    return False

def _get_direction_vector_for_mitering(helix, design: HadoNucleotideModel, shared_vertex, add_nucleotides: bool = True):
    """ Defines the direction vector to step the nucleotide position in to effectively miter a design thru addition """
    helix_to_bundle = design.get_helix_to_bundle()
    idx_edge_map = design.get_idx_edge_map()

    helix_bundle = helix_to_bundle[helix]
    idx_of_verts = idx_edge_map[helix_bundle]
    pi = design.get_point(idx_of_verts[0])
    pj = design.get_point(idx_of_verts[1])

    if add_nucleotides:
        # Get the vector pointing TOWARDS shared_vertex (which is one of pi / pj):
        if shared_vertex == idx_of_verts[0]:
            direction = pi - pj
        elif shared_vertex == idx_of_verts[1]:
            direction = pj - pi
        else:
            raise ValueError('ERROR: shared_Vertex is not part of the proper edges.')
    else:
        # Get the vector pointing AWAY shared_vertex (which is one of pi / pj):
        if shared_vertex == idx_of_verts[0]:
            direction = pj - pi
        elif shared_vertex == idx_of_verts[1]:
            direction = pi - pj
        else:
            raise ValueError('ERROR: shared_Vertex is not part of the proper edges.')

    norm = np.linalg.norm(direction)
    return direction / norm

def _plot_design(design: HadoNucleotideModel, rotated_positions: np.ndarray | dict, connections: np.ndarray | dict,
                 show_colors: bool = False, cylinder_radius: float = 0.5, cylinder_res: int = 10,
                 visual_override: bool = False):
    return build_mitering_figure(
        design,
        rotated_positions,
        connections,
        show_colors=show_colors,
        cylinder_radius=cylinder_radius,
        cylinder_res=cylinder_res,
        visual_override=visual_override,
    )

def _collect_new_positions(new_positions_for_plotting, v):
    """ Collects the proper helices to show """
    mitered_positions_at_v = {}
    for helix, new_vals in new_positions_for_plotting.items():
        for new_positions in new_vals:
            vertex_of_new_positions, new_position = new_positions
            if vertex_of_new_positions == v:
                mitered_positions_at_v[helix] = new_position
                break
    return mitered_positions_at_v

def _convert_connections(optimal_connections):
    """ Converts the optimal_connections to a dict of helix_i: helix_j for quick lookup """
    converted = {}
    for v, connections_at_vertex_v in optimal_connections.items():
        converted_at_v = {}
        for connection in connections_at_vertex_v:
            hi, hj = connection
            hi, hj = int(hi), int(hj)
            if hi in converted_at_v or hj in converted_at_v:
                raise Exception('ERROR: Duplicate connection found at mitering stage.')
            converted_at_v[hi] = hj
            converted_at_v[hj] = hi
        converted[v] = converted_at_v
    return converted

def _trim_half_nts(cur_nts, helix, shared_vertex, design: HadoNucleotideModel, middle_nts_per_helix):
    """ Trims the nucleotides active in the current design in HALF to support mitering algorithm to start. """
    middle_nt = deepcopy(middle_nts_per_helix[helix])
    idx_edge_map = design.get_idx_edge_map()
    helix_to_bundle = design.get_helix_to_bundle()
    helix_node = helix_to_bundle[helix]

    if idx_edge_map[helix_node][0] == shared_vertex:
        cur_nts[:middle_nt+1] = False
        return cur_nts, True, middle_nt

    elif idx_edge_map[helix_node][1] == shared_vertex:
        cur_nts[middle_nt:] = False
        return cur_nts, False, middle_nt

    else:
        raise Exception('ERROR: Invalid shared_vertex passed in to mitering procedure.')

def _dist(pi: np.ndarray, pj: np.ndarray):
    return np.linalg.norm(pj - pi)

def _get_middle_nts(new_nts):
    """ Calculates the middle active nt per helix """
    middle_nts = {}
    for k, v in new_nts.items():
        active_nts = np.where(v)[0]
        middle_nt = (min(active_nts) + max(active_nts)) // 2
        middle_nts[k] = middle_nt
    return middle_nts

def _collect_final_nts_array(new_scaffold_nts):
    all_nts = []
    new_ends = {}
    for helix in new_scaffold_nts.keys():
        nts = new_scaffold_nts[helix]
        active_nts = np.where(nts)[0]
        try:
            new_ends[helix] = (min(active_nts), max(active_nts))
        except ValueError:
            raise ValueError('ERROR: Design post-mitering has helix with no active nucleotides. This is likely due to'
                             'an overly large cross-section on too short of a edge. It could also be caused by an'
                             'overly sharp angle.')
        all_nts.append(nts)
    return np.array(all_nts), new_ends
