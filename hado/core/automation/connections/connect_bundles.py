import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import random
from collections import defaultdict
from typing import Tuple
from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.automation.diagnostics.visualization import build_bundle_positions_figure

def optimize_connections(design: HadoNucleotideModel,
                         **kwargs
                         ) -> Tuple[dict, dict, list]:
    """
    This function takes in a HadoNucleotideModel that has had the helix bundle cross-sections defined. THis function
    rotates each of the helix bundles (aligned along each edge of the geometry) to find a series of helix connections
    that lend to an "un-knotted" scaffold routing. This minimizes the total distance between the connected helices while
    rotating bundles in discrete steps (as this is just a rough approximation prior to mitering).

    :param design: The HadoNucleotideModel that has had it's scaffold routed and grid_positions defined.
    :type design: HadoNucleotideModel

    :return: A tuple containing three elements:

        * **design** (*HadoNucleotideModel*): Final design post-stapling that will then be used for exporting sequences.
        * **staple_map** (*dict*): A mapping of helices to their connection (i.e., {helix_i: helix_j} where helix_i and
            helix_j belong to separate helix bundles). This is sued in `autostaple_hollowframe`.
        * **best_state** (*dict*): A mapping of helices 3' and 5' ends to their (designed) final Euclidean X Y Z
            position. This is used to determine the number of ssDNA overhangs inside of `autostaple_hollowframe`.
    :rtype: tuple
    """
    all_flats, scaffold_helix_connections, optimal_connections = set(), None, None
    max_check = kwargs.get('max_optimize_connections', 100)
    if max_check <= 0:
        raise ValueError('ERROR: Can not set max_optimize_connections to be less than 0')
    iteration = 0
    best_state, intermediate_states = find_best_initial_state(design, **kwargs)

    # A small hacky thing for short cross-sections (less than 6) is to rotate state by 180 degrees as although
    # this will be "longer" in terms of total distance, it actually is more ammenable to mitering procedure
    if any(i < 6 for i in design.geometry.n_per_edge):
        best_state = {k: (v + 180) % 360 for k, v in best_state.items()}

    if kwargs.get('show_best_state_animation', False):
        _animate_rotations(intermediate_states, design, kwargs.get('diagnostics'))

    # There is a loop here because in high connectivity, low-N-per-edge configurations
    # then we need to carefully atreat the connectivity to ensure continuous scaffold
    while iteration < max_check:
        optimal_connections, flat_optimal_conditions = get_optimal_connections(design, best_state, all_flats)

        if kwargs.get('connections_only', False):
            return optimal_connections, best_state, []

        scaffold_helix_connections = connect_hollowframe(flat_optimal_conditions, design, **kwargs)
        if scaffold_helix_connections is not None:
            break
        iteration += 1
        all_flats.add(tuple(flat_optimal_conditions))

    if scaffold_helix_connections is None:
        raise RuntimeError('ERROR: Unable to find proper state.')
    design.set_bundle_rotations(best_state)
    return optimal_connections, best_state, scaffold_helix_connections

def get_optimal_connections(design: HadoNucleotideModel, best_state, constrained_connections: set = None):
    """ Uses the Hungarian matching algorithm to find optimal set of rotations between helix bundles """
    base_positions_and_axes = decompose_design_into_bundles(design)
    rotated_positions = get_rotated_positions(design, base_positions_and_axes, best_state)
    helix_to_bundle = design.get_helix_to_bundle()

    unique_verts = np.arange(len(design.geometry.vertices))
    staple_map_nearest_helices = {}
    for v in unique_verts:
        bundles_at_vertex = list(design.get_bundles_at_vertex(v))
        if len(bundles_at_vertex) == 1:
            # If only 1 helix bundle at a given vertex then we can skip mitering procedure
            continue

        # First get all helices at current bundle and split into groups based on scaffold_direction to split into
        # two groups for the linear assignment problem:
        positions = {}
        for b in bundles_at_vertex:
            indices = np.where(helix_to_bundle == b)[0]
            bundle_points = rotated_positions[indices]
            global_senders, local_senders = design.get_sender_indices(b, v)
            global_receivers, local_receivers = design.get_receiver_indices(b, v)
            positions[int(b)] = (bundle_points[local_senders], bundle_points[local_receivers],
                                 global_senders, global_receivers)

        connected_edges, _ = _solve_via_hungarian(positions, design, constrained_connections)
        staple_map_nearest_helices[int(v)] = connected_edges

    connections_per_vertex = [list(v) for v in staple_map_nearest_helices.values()]
    all_connections = [item for sublist in connections_per_vertex for item in sublist]
    return staple_map_nearest_helices, all_connections


def find_best_initial_state(design: HadoNucleotideModel, **kwargs):
    """ Finds the best initial state for hollow-frame like designs by ensuring that continuous segments along
    a helix cross-section traverse to the neighbor-helices at all vertices prior to the scaffold assignment.

    This uses a rather simple, greedy search that can be controlled in length by kwargs.max_rotation_iterations.
    Overall, this iterations value does not need to be set too high as there are only a few discrete rotations checked
    since the honeycomb cross-section have 60-degree rotational symmetry.
    """
    base_positions_and_axes = decompose_design_into_bundles(design)
    unique_verts = np.arange(len(design.geometry.vertices))
    global_helix_maps = _map_helices_in_order(design)
    helix_to_bundle = design.get_helix_to_bundle()

    def _score_constraints(helix_connection_maps, connections):
        """ Assigns the connections and scores the contiguous-ness of the design rotation state"""
        for i in connections:
            h1, h2 = i
            h1b, h2b = helix_to_bundle[h1], helix_to_bundle[h2]
            helix_connection_maps[h1b][h1] = h2b
            helix_connection_maps[h2b][h2] = h1b

        use_small_constraint = True
        for helices_in_bundle in helix_connection_maps.values():
            if len(helices_in_bundle) >= 8:
                use_small_constraint = False  # Use the contiguous check for larger cross-sections

        constraint_score = 0.
        if use_small_constraint:
            # For small cross-section (e.g., 2) we want to enforce that each bundle traverses to two unique bundles
            for k, v in helix_connection_maps.items():
                if all(i == min(v.values()) for k, i in v.items()):
                    constraint_score += 1000.
        else:
            # This penalizes any non-contiguous chains for larger cross-sections. For example, if we have 18 helices
            # and are listing which helix bundle (i.e., edge id) they travel next to, we'd want something like
            # 1 1 1 1 1 2 2 2 2 2 2 2 2 2 1 1 1 1 because that gives "easiest" solve to processing scaffold positions
            # as compared to something like 1 1 1 2 2 2 2 1 1 1 1 1 2 2 2 2 1 1
            for bundle_id, helix_map in helix_connection_maps.items():
                values_in_order = list(helix_map.values())
                cyclic_values = list(values_in_order) + [values_in_order[0]]
                value_blocks = defaultdict(int)
                prev_val = cyclic_values[0]
                value_blocks[prev_val] += 1
                for val in cyclic_values[1:]:
                    if val != prev_val:
                        value_blocks[val] += 1
                    prev_val = val

                for val, n_blocks in value_blocks.items():
                    threshold = 2 if val == cyclic_values[0] else 1
                    if n_blocks > threshold:
                        constraint_score += 1000.
        return constraint_score

    def _score(state):
        """ Scores a state configuration by trying to form continuous-like rings that minimize total distance """
        rotated_positions = get_rotated_positions(design, base_positions_and_axes, state)
        distance_score = 0.
        constraint_score = 0.
        for v in unique_verts:
            bundles_at_vertex = list(design.get_bundles_at_vertex(v))
            if len(bundles_at_vertex) == 1:
                # If only 1 helix bundle at a given vertex then we can skip mitering procedure
                continue

            positions = {}
            submaps = {i: j for i, j in global_helix_maps.items() if i in bundles_at_vertex}
            for b in bundles_at_vertex:
                indices = np.where(helix_to_bundle == b)[0]
                bundle_points = rotated_positions[indices]
                global_senders, local_senders = design.get_sender_indices(b, v)
                global_receivers, local_receivers = design.get_receiver_indices(b, v)
                positions[int(b)] = (bundle_points[local_senders], bundle_points[local_receivers],
                                     global_senders, global_receivers)

            connected_edges, distances_found = _solve_via_hungarian(positions, design)
            distance_score += np.sum(distances_found)
            constraint_score += _score_constraints(submaps, connected_edges)

        return distance_score + constraint_score, np.isclose(constraint_score, 0.0)

    random.seed(kwargs.get('random_seed', 8))
    max_iterations = kwargs.get('max_rotation_iterations', 250)
    store_frequency = kwargs.get('animation_frequency', 10)

    # First,using the Hungarian matching algorithm to find optimal scaffold connections for a hollowframe design:
    all_bundle_ids = np.arange(len(design.geometry.edges))
    angle_choices = np.arange(0, 360, kwargs.get('angle_step_size', 15))  # Options for rotation
    cur_state = {bundle_id: random.choice(angle_choices) for bundle_id in all_bundle_ids}
    cur_score, cur_is_valid = _score(cur_state)

    best_state = cur_state
    best_score = cur_score
    best_valid = cur_is_valid

    # Greedy walk as this is just a rough estimate
    animation_states = []
    for iteration in range(max_iterations):
        if iteration % store_frequency == 0:
            animation_states.append(best_state)

        bundle_id = random.choice(all_bundle_ids)
        current_angle = cur_state[bundle_id]

        new_angles = [a for a in angle_choices if a != current_angle]
        new_angle = random.choice(new_angles)

        new_state = dict(cur_state)
        new_state[bundle_id] = new_angle
        new_score, new_valid = _score(new_state)

        if best_valid:
            if new_valid and new_score < best_score:
                best_state = new_state
                best_score = new_score
                best_valid = True
                cur_state = new_state
        else:
            if new_score < best_score:
                best_state = new_state
                best_score = new_score
                best_valid = new_valid
                cur_state = new_state

    return best_state, animation_states

def connect_hollowframe(optimal_connections, design: HadoNucleotideModel, **kwargs):
    """ Connects the scaffold for a hollowframe nanostructure using a graph-based approach conceptually similar to
    Huang, CM., Kucinic, A., Johnson, J.A. et al. Integrated computer-aided engineering and design for DNA assemblies.
    Nat. Mater. 20, 1264–1271 (2021). https://doi.org/10.1038/s41563-021-00978-5

    The key differences in the algorithm here is that that I pre-define the optimal connections (instead of using
    user-defined inputs) as the hollowframe paradigm is built on using all connections for a mechanically stiff
    junction between helix bundles that is conducive to colloidal assembly. Furthermore, there are some minor structural
    nuances with how the internal crossovers are properly allocated in an efficient manner.
    """
    helix_to_bundle = design.get_helix_to_bundle()
    _, cnts = np.unique(helix_to_bundle, return_counts=True)

    helix_order = _preprocess_helices(design, optimal_connections)
    graph, initial_n_cycles = _initialize_graph_from_design_and_optimal_connections(
        optimal_connections, design, helix_order
    )
    current_N = initial_n_cycles
    max_iterations = kwargs.get('max_scaf_path_iterations_hollowframe', 100)
    iteration_count = 0
    mst_connections_made = []
    internal_xover_counter = {i: 0 for i in range(len(np.unique(helix_to_bundle)))}
    while current_N > 1:
        adjacency_graph = _create_adjacency_graph(graph, design)

        if not adjacency_graph.edges:
            # Retrun None as "invalid solution" to re-try with different (but less-) optimal_connections
            return None

        # Identify ideal edges from adjacency graph to combine cycles into longer cycle:
        mst = nx.minimum_spanning_tree(adjacency_graph, weight="weight")

        for n1, n2 in mst.edges():
            connection_made = _merge_cycles(graph, n1, n2, design, adjacency_graph, internal_xover_counter)
            internal_xover_counter[helix_to_bundle[connection_made[0]]] += 1
            mst_connections_made.append((min(connection_made), max(connection_made)))

        # Break once single scaffold (i.e., N=1) is found:
        current_N -= (len(mst.edges()))
        iteration_count += 1

        if iteration_count > max_iterations:
            return None

    final_connections = _decode_graph(graph, mst_connections_made)
    return final_connections


def decompose_design_into_bundles(design: HadoNucleotideModel):
    """ Considers an initial HadoNucleotideModel structure and adds local coordinate systems to the individual helix bundle
    groups. These coordinate systems + edge decompositions are used to optimize connections between helix bundle groups.
    """
    idx_map = design.get_idx_edge_map()
    helix_to_bundle = design.get_helix_to_bundle()
    grid_locations = design.get_helix_bundle_grid_locations()
    N = grid_locations.shape[0]
    transformed_grid = []
    local_axes_info = []
    for i, hb in enumerate(idx_map):
        v1, v2 = hb  # Unpack vertex pointers forming this edge
        p1, p2 = design.get_point(v1), design.get_point(v2)
        init_coordinate_system = np.array([0, 0, -1])  # -Z is pointing up in this example
        target_dir = p2 - p1
        target_dir /= np.linalg.norm(target_dir)

        # Compute rotation to align init_dir with target_dir
        rot_axis = np.cross(init_coordinate_system, target_dir)
        if np.allclose(rot_axis, 0):
            # Vectors are parallel (either same or opposite)
            if np.dot(init_coordinate_system, target_dir) < 0:
                # 180 degree rotation around any axis perpendicular to init_dir
                rot = R.from_rotvec(np.pi * np.array([1, 0, 0]))
            else:
                rot = R.identity()
        else:
            rot_angle = np.arccos(np.clip(np.dot(init_coordinate_system, target_dir), -1.0, 1.0))
            rot_axis /= np.linalg.norm(rot_axis)
            rot = R.from_rotvec(rot_angle * rot_axis)

        grid_members_in_hb = []
        temp = np.where(helix_to_bundle == i)[0]
        for _i in temp:
            grid_members_in_hb.append(grid_locations[_i])
        grid_locations_3D = np.hstack([grid_members_in_hb, np.zeros((len(temp), 1))])
        rotated_grid = rot.apply(grid_locations_3D)

        edge_center = (p1 + p2) / 2  # Translate to center
        transformed_positions = rotated_grid + edge_center

        # Local coordinate frame
        local_axes = [
            rot.apply(np.array([1, 0, 0])),
            rot.apply(np.array([0, 1, 0])),
            rot.apply(np.array([0, 0, 1])),
        ]
        transformed_grid.append(transformed_positions)
        temp1 = [edge_center]
        temp2 = [edge_center + local_axes[i] for i in range(3)]  # Translate axes by + X / Y / Z position
        local_axes_info.append(temp1 + temp2)
    return transformed_grid, local_axes_info

def get_rotated_positions(design: HadoNucleotideModel, base_grid_locations, state):
    """ Rotates all grids in a design about their centroid """
    helix_to_bundle = design.get_helix_to_bundle()
    rotated_positions = np.zeros((len(helix_to_bundle), 3))
    base_locations, base_axes = base_grid_locations
    for i, (bundle_id, angle) in enumerate(state.items()):
        base_pos = base_locations[bundle_id]
        base_local_axis = base_axes[bundle_id]
        updated_bundle = _rotate_bundle(base_pos, base_local_axis, angle)
        helices_in_bundle = np.where(helix_to_bundle == bundle_id)[0]
        for h, hi in enumerate(helices_in_bundle):
            rotated_positions[hi] = updated_bundle[h]
    return rotated_positions

def _decode_graph(graph, mst_connections_made):
    """ Decodes the final graph post-connections to be converted into nucleotide-level data via DFS """
    start_node = next(iter(graph.nodes))
    neighbors = list(graph.neighbors(start_node))
    if len(neighbors) != 2:
        raise RuntimeError('ERROR: Node should only have 2 neighbors in the final graph prior to decoding...')
    start_graph_node = graph.nodes[start_node]

    potential_2 = graph.nodes[neighbors[1]]
    if start_graph_node['scaf_dir']:
        first_connection_to_perform = neighbors[1] if potential_2['scaf_dir'] else neighbors[0]
    else:
        first_connection_to_perform = neighbors[1] if not potential_2['scaf_dir'] else neighbors[0]

    visited_edges = set()
    current_node = start_node
    prev_node = None
    final_connections = []

    while True:
        neighbors = list(graph.neighbors(current_node))

        next_node = None
        for nbr in neighbors:
            if nbr != prev_node:
                edge = (min(current_node, nbr), max(current_node, nbr))
                if prev_node is None:
                    # First connection needs to use the first_connection_to_perform
                    if nbr != first_connection_to_perform:
                        continue
                    else:
                        next_node = nbr
                        visited_edges.add(edge)
                        n0 = graph.nodes[current_node]
                        n1 = graph.nodes[next_node]
                        if n0['helix'] == n1['helix']:
                            pass
                        elif (min(n0['helix'], n1['helix']), max(n0['helix'], n1['helix'])) in mst_connections_made:
                            final_connections.append((int(n0['helix']), int(n1['helix']), 'INTERNAL_MIDDLE'))
                        elif n0['internal'] and n1['internal']:
                            final_connections.append((int(n0['helix']), int(n1['helix']), 'INTERNAL'))
                        elif n0['bundle'] == n1['bundle']:
                            final_connections.append((int(n0['helix']), int(n1['helix']), 'INTERNAL'))
                        elif n0['helix'] != n1['helix']:
                            final_connections.append((int(n0['helix']), int(n1['helix'])))
                        break


                elif edge not in visited_edges:
                    next_node = nbr
                    visited_edges.add(edge)
                    n0 = graph.nodes[current_node]
                    n1 = graph.nodes[next_node]
                    if n0['helix'] == n1['helix']:
                        pass
                    elif (min(n0['helix'], n1['helix']), max(n0['helix'], n1['helix'])) in mst_connections_made:
                        final_connections.append((int(n0['helix']), int(n1['helix']), 'INTERNAL_MIDDLE'))
                    elif n0['internal'] and n1['internal']:
                        final_connections.append((int(n0['helix']), int(n1['helix']), 'INTERNAL'))
                    elif n0['bundle'] == n1['bundle']:
                        final_connections.append((int(n0['helix']), int(n1['helix']), 'INTERNAL'))
                    elif n0['helix'] != n1['helix']:
                        final_connections.append((int(n0['helix']), int(n1['helix'])))
                    break

        if next_node is None:
            break

        prev_node, current_node = current_node, next_node

        if current_node == start_node:
            break

    return final_connections

def _get_helix_edges(graph, helix_id):
    """ Helper to find all current edges in the graph that represent continuous segments of a specific helix. An
    edge represents a segment of a helix if both of its nodes belong to that helix.
    """
    valid_edges = []
    for u, v in graph.edges():
        if graph.nodes[u].get('helix') == helix_id and graph.nodes[v].get('helix') == helix_id:
            valid_edges.append((u, v))
    return valid_edges

def _merge_cycles(graph, n1, n2, design, adjacency_graph, internal_xover_counter):
    """ Merges the top-level graph by breaking valid helix edges and crossing them over. """
    helix_to_bundle = design.get_helix_to_bundle()
    scaffold_dirs = design.get_scaffold_directions()
    potential_connections = adjacency_graph.edges[(n1, n2)]['potential_connections']

    # Sort potential_connections based on internal_xover_counter
    pc_scored = {i: internal_xover_counter[helix_to_bundle[i[0]]] for i in potential_connections}
    pc_sorted = [k for k, _ in sorted(pc_scored.items(), key=lambda kv: (kv[1], kv[0]))]

    connection = pc_sorted[0]  # just use the lowest number of connections
    h1, h2 = connection

    # Find valid edges to break instead of counting nodes
    edges_h1 = _get_helix_edges(graph, h1)
    edges_h2 = _get_helix_edges(graph, h2)

    if len(edges_h1) == 0 or len(edges_h2) == 0:
        raise RuntimeError(f"ERROR: Cannot merge, no valid internal edges found for helices {h1} or {h2}.")

    # Select the first available edge to break, maybe in future we want to carefully pick?
    edge_to_break_h1 = edges_h1[0]
    edge_to_break_h2 = edges_h2[0]

    bundle_h1 = helix_to_bundle[h1]
    bundle_h2 = helix_to_bundle[h2]
    if bundle_h1 != bundle_h2:
        raise ValueError('ERROR: Can not merge helices that do not belong to same helix bundle.')

    scaf_dir_h1 = scaffold_dirs[h1]
    scaf_dir_h2 = scaffold_dirs[h2]
    if scaf_dir_h1 == scaf_dir_h2:
        raise ValueError('ERROR: Can not merge helices with same scaffold direction.')

    graph.remove_edge(*edge_to_break_h1)
    graph.remove_edge(*edge_to_break_h2)

    cur_num_nodes = len(graph.nodes)

    # Add 4 new nodes to represent the crossover
    graph.add_node(cur_num_nodes, helix=h1, bundle=bundle_h1, scaf_dir=scaf_dir_h1, internal=True,
                   vertex=None, is_unique=None, is_connected=None)
    graph.add_node(cur_num_nodes + 1, helix=h1, bundle=bundle_h1, scaf_dir=scaf_dir_h1, internal=True,
                   vertex=None, is_unique=None, is_connected=None)

    graph.add_node(cur_num_nodes + 2, helix=h2, bundle=bundle_h2, scaf_dir=scaf_dir_h2, internal=True,
                   vertex=None, is_unique=None, is_connected=None)
    graph.add_node(cur_num_nodes + 3, helix=h2, bundle=bundle_h2, scaf_dir=scaf_dir_h2, internal=True,
                   vertex=None, is_unique=None, is_connected=None)

    # Reconnect the broken segments to the new crossover nodes
    graph.add_edge(edge_to_break_h1[0], cur_num_nodes)
    graph.add_edge(edge_to_break_h1[1], cur_num_nodes + 1)

    graph.add_edge(edge_to_break_h2[0], cur_num_nodes + 2)
    graph.add_edge(edge_to_break_h2[1], cur_num_nodes + 3)

    # Form the actual crossover connections between h1 and h2
    graph.add_edge(cur_num_nodes, cur_num_nodes + 2)
    graph.add_edge(cur_num_nodes + 1, cur_num_nodes + 3)

    if not all(deg == 2 for _, deg in graph.degree()):
        raise RuntimeError("ERROR: Each node should have exactly 2 edges after graph construction.")

    return connection

def _create_adjacency_graph(graph, design: HadoNucleotideModel):
    """ Creates an adjacency matrix where each node is a specific component (cycle) of the graph and edges dictate
    which cycles can be 'combined' into a larger cycle (thereby reducing curN cycles in the top-level graph)
    """
    components = list(nx.connected_components(graph))
    n = len(components)
    cycles = [graph.subgraph(c).copy() for c in components]

    # Now create a new graph where the nodes will be the cycles and the edges are going to be a definition of
    # which cycles can be combined into longer cycles
    adjacency_graph = nx.Graph()
    for i in range(n):
        adjacency_graph.add_node(i, cycle_nodes=list(cycles[i].nodes(data=True)))

    for i in range(n):
        for j in range(i + 1, n):
            cycle_i = adjacency_graph.nodes[i]["cycle_nodes"]
            cycle_j = adjacency_graph.nodes[j]["cycle_nodes"]

            share_potential_connection, potential_connections = _share_potential_connection(cycle_i, cycle_j, design)
            if share_potential_connection:
                adjacency_graph.add_edge(i, j, weight=len(potential_connections),
                                         potential_connections=potential_connections)
    return adjacency_graph

def _share_potential_connection(cycle_i, cycle_j, design: HadoNucleotideModel):
    """ Function that determines if two cycles can be joined in the adjacency graph and ensures that a continuous
    scaffold is maintained. Overall, this function is finding which helices can have an internal crosover placed
    between them to merge two cycles together
    """
    # Extract bundle -> helices map for each cycle
    bundle_to_helices_i = {}
    for _, data in cycle_i:
        bundle_to_helices_i.setdefault(int(data["bundle"]), set()).add(int(data["helix"]))

    bundle_to_helices_j = {}
    for _, data in cycle_j:
        bundle_to_helices_j.setdefault(int(data["bundle"]), set()).add(int(data["helix"]))

    # Find shared bundles
    shared_bundles = set(bundle_to_helices_i.keys()) & set(bundle_to_helices_j.keys())
    potential_neighbors = []

    for b in shared_bundles:
        for h1 in bundle_to_helices_i[b]:
            for h2 in bundle_to_helices_j[b]:
                if design.are_neighbors(h1, h2):
                    potential_neighbors.append((h1, h2))

    valid_to_combine = len(potential_neighbors) > 0
    return valid_to_combine, potential_neighbors

def _initialize_graph_from_design_and_optimal_connections(optimal_connections, design: HadoNucleotideModel,
                                                          helix_orders):
    """ Creates a networkX graph of the design """
    helix_to_bundle = design.get_helix_to_bundle()
    scaffold_dirs = design.get_scaffold_directions()
    idx_edge_map = design.get_idx_edge_map()
    bundle_ids = np.unique(helix_to_bundle)
    graph = nx.Graph()
    node_map = {}
    node_count = 0

    # First we will add all helices as nodes
    for bi in bundle_ids:
        all_helices = np.where(helix_to_bundle == bi)[0]
        all_dirs = scaffold_dirs[all_helices]
        v0, v1 = idx_edge_map[bi]

        for h, scaf_dir in zip(all_helices, all_dirs):
            v0_is_unique, v1_is_unique = design.check_free_ends(h)

            # Add 2 nodes, one at each end of v0 and v1 and connect with edge:
            graph.add_node(node_count, helix=h, bundle=bi, scaf_dir=scaf_dir, internal=False, vertex=v0,
                           is_unique=v0_is_unique, is_connected=False)
            graph.add_node(node_count+1, helix=h, bundle=bi, scaf_dir=scaf_dir, internal=False, vertex=v1,
                           is_unique=v1_is_unique, is_connected=False)
            node_map[node_count] = h
            node_map[node_count+1] = h
            graph.add_edge(node_count, node_count+1)
            node_count += 2

    # Next add optimal connect edges:
    for oc in optimal_connections:
        h1, h2 = oc
        shared_vertex = design.get_shared_vertex(helix_to_bundle[h1], helix_to_bundle[h2])
        node_h1 = [n for n, data in graph.nodes(data=True) if data['helix'] == h1 and data['vertex'] == shared_vertex]
        node_h2 = [n for n, data in graph.nodes(data=True) if data['helix'] == h2 and data['vertex'] == shared_vertex]
        if len(node_h1) != 1 or len(node_h2) != 1:
            raise RuntimeError('ERROR: Invalid graph construction during connection algorithm.')
        node_h1, node_h2 = node_h1[0], node_h2[0]
        graph.add_edge(node_h1, node_h2)
        graph.nodes[node_h1]['is_connected'] = True
        graph.nodes[node_h2]['is_connected'] = True

    # Finally add internal end connections based on free-end:
    for bi in bundle_ids:
        helices_bi = np.where(helix_to_bundle == bi)[0]
        h_test = helices_bi[0]
        v0_is_unique, v1_is_unique = design.check_free_ends(h_test)
        if v0_is_unique and v1_is_unique:
            raise RuntimeError('ERROR: One end must not be unique for connections')
        if v0_is_unique or v1_is_unique:
            # If there are unique vertices, must connect the ends:
            helices_in_order = helix_orders[bi]
            unconnected_nodes = [n for n, data in graph.nodes(data=True) if
                                 data['bundle'] == bi and not data['is_connected']]
            if len(unconnected_nodes) != len(helices_bi):
                raise RuntimeError('ERROR: Invalid graph construction during connection algorithm.')

            for h in range(0, len(helices_in_order), 2):
                helix0, helix1 = helices_in_order[h], helices_in_order[h+1]
                node_h1 = [n for n, data in graph.nodes(data=True) if
                           data['helix'] == helix0 and not data['is_connected']]
                node_h2 = [n for n, data in graph.nodes(data=True) if
                           data['helix'] == helix1 and not data['is_connected']]
                if len(node_h1) != 1 or len(node_h2) != 1:
                    raise RuntimeError('ERROR: Invalid graph construction during connection algorithm.')
                node_h1, node_h2 = node_h1[0], node_h2[0]
                if node_h1 not in unconnected_nodes or node_h2 not in unconnected_nodes:
                    raise RuntimeError('ERROR: Invalid graph construction during connection algorithm.')
                graph.add_edge(node_h1, node_h2)

    # At this point, each node MUST have 2 edges (one internally to the helix bundle, one externally):
    if not all(deg == 2 for _, deg in graph.degree()):
        raise RuntimeError("ERROR: Each node should have exactly 2 edges after graph construction.")
    num_initial_cycles = nx.number_connected_components(graph)
    return graph, num_initial_cycles

def _rotate_bundle(grid_elements, local_cs, angle):
    """ Rotates the grid_elements CCW about their local Z-axis by the defined angle """
    axis = local_cs[3] - local_cs[0]  # Z-axis - origin center
    axis /= np.linalg.norm(axis)
    theta = -np.radians(angle)  # Convert from CW degrees -> CCW Radians
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    r = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    translated = grid_elements - local_cs[0]
    rotated = translated @ r.T
    rotated_elements = rotated + local_cs[0]
    return rotated_elements

def plot_positions(positions, base_axes=None, connections=None, plot_cylinders=False, **kwargs):
    return build_bundle_positions_figure(
        positions,
        base_axes=base_axes,
        connections=connections,
        plot_cylinders=plot_cylinders,
        **kwargs,
    )

def _animate_rotations(all_states, design: HadoNucleotideModel, diagnostics=None):
    """ Creates an animation showing how the optimal initial state is found """
    unique_states = []
    seen = set()

    for d in all_states:
        # Convert dict to tuple of sorted items so order doesn't matter
        key = tuple(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            unique_states.append(d)

    # First show the cylinders:
    base_positions_and_axes = decompose_design_into_bundles(design)
    base_positions, base_axes = base_positions_and_axes
    fig = plot_positions(base_positions, base_axes=base_axes, plot_cylinders=True, design=design)
    if diagnostics is not None:
        diagnostics.record_figure('connections', 'base_cylinders', fig)
    fig = plot_positions(base_positions, base_axes=base_axes, plot_cylinders=True, design=design, cylinder_color=True)
    if diagnostics is not None:
        diagnostics.record_figure('connections', 'base_cylinders_gray', fig)
    fig = plot_positions(base_positions, design=design)
    if diagnostics is not None:
        diagnostics.record_figure('connections', 'base_positions', fig)

    # Then we begin showing the unique states found
    for idx, us in enumerate(unique_states):
        rotated_positions = get_rotated_positions(design, base_positions_and_axes, us)
        _, flat_optimal_conditions = get_optimal_connections(design, us)
        fig = plot_positions(rotated_positions, connections=flat_optimal_conditions, design=design)
        if diagnostics is not None:
            diagnostics.record_figure('connections', f'rotation_state_{idx}', fig)

def _map_helices_in_order(design: HadoNucleotideModel):
    """ Creates dictionaries of helices in a CCW fashion for easy design scoring """
    num_bundles = len(design.geometry.edges)
    main_map = {}
    helix_to_bundle = design.get_helix_to_bundle()
    grid_locations = design.get_helix_bundle_grid_locations()
    for i in range(num_bundles):
        bundle_i_map = {}
        helices_in_bundle_unordered = np.where(helix_to_bundle == i)[0]
        grid_positions_at_i = grid_locations[helices_in_bundle_unordered]

        # Center the points to compute angles
        center = grid_positions_at_i.mean(axis=0)
        rel_positions = grid_positions_at_i - center
        angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0])

        sorted_indices = np.argsort(angles)  # CCW order
        helices_ccw_order = helices_in_bundle_unordered[sorted_indices]
        for h in helices_ccw_order:
            bundle_i_map[h] = None
        main_map[i] = bundle_i_map
    return main_map

def _preprocess_helices(design: HadoNucleotideModel, optimal_connections):
    """ Places helices in CCW order for each bundle """
    def _ccw_sort_by_position(d, grid_positions):
        key_positions = {k: np.array(grid_positions[k]) for k in d.keys()}
        positions = np.array(list(key_positions.values()))
        centroid = np.mean(positions, axis=0)

        def angle_from_centroid(pos):
            return np.arctan2(pos[1] - centroid[1], pos[0] - centroid[0])

        sorted_keys = sorted(d.keys(), key=lambda k: angle_from_centroid(key_positions[k]))
        key_to_idx = {k: i for i, k in enumerate(sorted_keys)}
        min_key = min(sorted_keys)
        min_idx = key_to_idx[min_key]
        sorted_keys = sorted_keys[min_idx:] + sorted_keys[:min_idx]
        return [(k, d[k]) for k in sorted_keys]

    helix_to_bundle = design.get_helix_to_bundle()
    grid_locations = design.get_helix_bundle_grid_locations()
    num_bundles = len(design.geometry.edges)

    all_sorted = []
    for b in range(num_bundles):
        helices_in_config = np.where(helix_to_bundle == b)[0]
        helix_to_next_bundle = {i: [] for i in helices_in_config}
        for i in optimal_connections:
            if i[0] in helices_in_config:
                helix_to_next_bundle[i[0]].append(helix_to_bundle[i[1]])
            elif i[1] in helices_in_config:
                helix_to_next_bundle[i[1]].append(helix_to_bundle[i[0]])
        new_sorted_helices = _ccw_sort_by_position(helix_to_next_bundle, grid_locations)
        all_sorted.append([x[0] for x in new_sorted_helices])

    return all_sorted

def _get_hungarian_cost(positions, design: HadoNucleotideModel, constrained_connections: set = None):
    """ Uses the sender / reciever helices at a given vertex to define the closest (or furthest) points to use in the
    mitering procedure
    """
    # Collect all senders / reciefvers from the dictionary
    all_senders = []
    all_receivers = []
    sender_bundle_ids = []
    receiver_bundle_ids = []
    helix_to_bundle = design.get_helix_to_bundle()

    for b, (senders, receivers, global_senders, global_receivers) in positions.items():
        if senders.size > 0:
            all_senders.append(senders)
            sender_bundle_ids.extend(global_senders)
        if receivers.size > 0:
            all_receivers.append(receivers)
            receiver_bundle_ids.extend(global_receivers)

    senders_array = np.vstack(all_senders)
    receivers_array = np.vstack(all_receivers)

    # Compute pairwise distances and solve via linear sum assignment
    cost_matrix = cdist(senders_array, receivers_array, metric='euclidean')

    # Update cost-matrix to prevent senders / recievers from connecting within the same bundle
    M = cost_matrix.shape[0]
    mapping = {}
    for i in range(M):
        for j in range(M):
            b1, b2 = sender_bundle_ids[i], receiver_bundle_ids[j]
            mapping[(b1, b2)] = (i, j)
            if helix_to_bundle[b1] == helix_to_bundle[b2]:
                cost_matrix[i, j] = 1e6  # large cost to block same-bundle connections

            if constrained_connections is not None:
                for c in constrained_connections:
                    if (b1, b2) in c or (b2, b1) in c:
                        cost_matrix[i, j] = 1e6

    return cost_matrix, (sender_bundle_ids, receiver_bundle_ids), mapping

def _eval_hungarian(cost_matrix, all_ids):
    # Solve linear sum assignment via Hungarian
    sender_bundle_ids, receiver_bundle_ids = all_ids
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reconstruct and return global mapping:
    matched_bundle_pairs = [(sender_bundle_ids[r], receiver_bundle_ids[c])
                            for r, c in zip(row_ind, col_ind)]
    matched_distances = cost_matrix[row_ind, col_ind]
    return matched_bundle_pairs, matched_distances

def _solve_via_hungarian(positions, design, constrained_connections=None, **kwargs):
    def _num_connections(_pairs):
        graph = nx.Graph()
        bundles_connected = set()
        all_bundles = set()
        for _p in _pairs:
            _b1, _b2 = h2b[_p[0]], h2b[_p[1]]
            bundles_connected.add((_b1, _b2))
            all_bundles.add(_b1)
            all_bundles.add(_b2)
            graph.add_edge(_b1, _b2)
        return list(nx.connected_components(graph)), bundles_connected, all_bundles

    def _store_local(local_pos, helix_idx):
        bundle = h2b[helix_idx]
        ps, pr, s, r = positions[bundle]
        if helix_idx in s:
            idx = np.where(s == helix_idx)[0][0]
            local_pos[bundle][0].append(ps[idx])
            local_pos[bundle][2].append(helix_idx)
        elif helix_idx in r:
            idx = np.where(r == helix_idx)[0][0]
            local_pos[bundle][1].append(pr[idx])
            local_pos[bundle][3].append(helix_idx)
        else:
            raise ValueError('ERROR: Helix index not found in list.')

    # Evaluate current cost matrix
    h2b = design.get_helix_to_bundle()
    cost_matrix, global_indices, cost_map = _get_hungarian_cost(positions, design, constrained_connections)
    pairs, distances = _eval_hungarian(cost_matrix, global_indices)
    components, all_connections, bundles = _num_connections(pairs)

    min_hungarian_threshold = kwargs.get('min_hungarian_threshold', 0.2)
    if not 0 < min_hungarian_threshold < 1:
        raise ValueError('ERROR: min_hungarian_threshold should be between 0 and 1.')

    n_to_reassign = int(design.geometry.n_per_edge[0] * min_hungarian_threshold)
    n_to_reassign = max(1, n_to_reassign)

    max_retries = kwargs.get('max_hungarian_retries', 25)
    iterations = 0
    pair_to_dist = {tuple(p): d for p, d in zip(pairs, distances)}


    # Loop to find a solution that results in a single connected component, which ensure a fully connected
    # design that can be continously scaffolded
    while len(components) > 1 and iterations < max_retries:
        measured = {c: [] for c in list(all_connections)}

        for p in pairs:
            i, j = tuple(p)
            b1, b2 = h2b[i], h2b[j]
            dist_val = pair_to_dist[(i, j)]

            if (b1, b2) in measured:
                measured[(b1, b2)].append(((i, j), dist_val))
            else:
                if (b2, b1) not in measured:
                    measured[(b2, b1)] = []
                measured[(b2, b1)].append(((i, j), dist_val))

        local_positions = {i: ([], [], [], []) for i in bundles}
        swap_pairs = set()

        for key, values in measured.items():
            sorted_vals = sorted(values, key=lambda x: (x[1], x[0]), reverse=True)
            for so in sorted_vals[:n_to_reassign]:
                (sender, receiver), og_score = so
                swap_pairs.add((sender, receiver))
                _store_local(local_positions, sender)
                _store_local(local_positions, receiver)

        for k, v in local_positions.items():
            local_positions[k] = (np.array(v[0]), np.array(v[1]), np.array(v[2]), np.array(v[3]))

        bridge_cost, bridge_global, bridge_map = _get_hungarian_cost(local_positions, design, constrained_connections)
        bridge_s, bridge_r = bridge_global

        comp_map = {}
        for comp_idx, comp_set in enumerate(components):
            for b in comp_set:
                comp_map[b] = comp_idx

        # Block connections that resolve to same component
        for i, ls in enumerate(bridge_s):
            for j, lr in enumerate(bridge_r):
                if comp_map.get(h2b[ls]) == comp_map.get(h2b[lr]):
                    bridge_cost[i, j] = 1e6

        bridge_pairs, bridge_distances = _eval_hungarian(bridge_cost, bridge_global)

        next_pairs = [p for p in pairs if tuple(p) not in swap_pairs and (p[1], p[0]) not in swap_pairs]
        next_pairs.extend(bridge_pairs)
        if len(pairs) != len(next_pairs):
            raise RuntimeError('ERROR: Number of pairs should remain the same after reassignment.')

        # Update the pairs to match new bridge solution
        for sp in swap_pairs:
            pair_to_dist.pop(sp, None)
            pair_to_dist.pop((sp[1], sp[0]), None)

        for bp, bd in zip(bridge_pairs, bridge_distances):
            pair_to_dist[tuple(bp)] = bd

        pairs = next_pairs
        components, all_connections, bundles = _num_connections(pairs)
        iterations += 1

    # We do not need to check len(components) == 1 here as:
    # 1) When finding the initial state, we may have physically impossible connecctions preventing a single component
    #    but these also have veyr high costs (distances) thus they never end up getting seelcted
    # 2) When populating the helices, we use an outer_loop because when N=2 helices there are some cases where
    #    no matter how we connect, there is not a single component. This requires a flipping of helices and adding
    #    constrained_connections to resolve (resulting in the outer loop in optimize_connections). If a design ends
    #    up being invalid, that outer loop will catch and raise an error (so again, no need to here)

    distances = [pair_to_dist[tuple(p)] for p in pairs]
    return pairs, distances
