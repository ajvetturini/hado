from hado.core.utils import Geometry, ScaffoldArgs
import networkx as nx


def perform_scaffold_routing(mesh_data: Geometry, scaffold_args: ScaffoldArgs) -> dict:
    """
    Perform scaffold routing on input Geometry data. Overall, in most cases for hollowframe structures, there are an
    even N per edge specified, and this function trivially returns the edge_design_dict.

    However, in the case of ODD N per edge, we use a cycle-splicing heuristic algorithm to stitch together
    a scaffold routing. This algorithm prevents re-using edges (as would be the case in say chinese postman tour).
    The user must specify either "overfill" or "underfill" that will either over or under-estimate the specified N by
    1. For example, if N was 3 and the user specified "overfill" in scaffold_args, then N will be 4 in the routed
    solution for all edges where the heuristic could not find N = 3. If "underfill" then N will be 2.
    """
    # All even n_per_edge values are guranteed by MST algorithm, thus simply return the geometry
    edges = mesh_data.edges
    edge_design_dict = {(int(edges[i][0]), int(edges[i][1])): {'M': 0, 'N': 0} for i in range(len(edges))}
    if all(i % 2 == 0 for i in mesh_data.n_per_edge):
        for i in range(len(edges)):
            edge = (int(edges[i][0]), int(edges[i][1]))
            even_split = mesh_data.n_per_edge[i] // 2
            edge_design_dict[edge]['N'] = even_split
            edge_design_dict[edge]['M'] = even_split
        return edge_design_dict

    # Odd N-per-edge search that can be expanded in the future, but for now is left as-is as there
    # isn't an obvious functional need for this currently (at least for DNA origami nanostructures)
    if not all(i == mesh_data.n_per_edge[0] for i in mesh_data.n_per_edge):
        raise NotImplementedError(
            "Currently, only CONSTANT odd N per edge is supported. If there is a functional need for more complex "
            "odd combinations, then please open a GitHub issue as this can be accommodated but is not urgent."
        )
    target_n = mesh_data.n_per_edge[0]

    # The fall-back, base case is every edge gets 2 helices until filled, then add on the best eulerian cycle
    # to add 1 helix to the target
    fallback_case = _get_fallback(mesh_data, target_n, scaffold_args, edge_design_dict)

    try:
        heuristic_case = _attempt_heuristic_reconstruct(mesh_data, target_n, scaffold_args, edge_design_dict)
        return _best_case(heuristic_case, fallback_case, target_n)
    except Exception as e:
        return fallback_case

def _best_case(heuristic_case: dict, fallback_case: dict, target_n: int):
    """ Compares the heuristic and fallback cases to the target N per edge and whichever has more edges that
    match the target is returned """
    assert len(heuristic_case) == len(fallback_case), "Heuristic and fallback cases must have same number of edges"
    heuristic_score = sum(1 for k in heuristic_case if heuristic_case[k]['M'] + heuristic_case[k]['N'] == target_n)
    fallback_score = sum(1 for k in fallback_case if fallback_case[k]['M'] + fallback_case[k]['N'] == target_n)

    if heuristic_score >= fallback_score:
        return heuristic_case
    else:
        return fallback_case

def _attempt_heuristic_reconstruct(mesh: Geometry, target_n: int, scaffold_args: ScaffoldArgs, edge_design_dict: dict):
    """
    Constructs a scaffold route by decomposing the mesh into edge-disjoint cycles and iteratively splicing
    them into a single continuous tour.

    This heuristic attempts to approximate an Eulerian tour (which would theoretically traverse each edge once and
    return to the start vertex) by:
    1. Iteratively stripping cycles from the graph until no cycles remain.
    2. Greedily merging intersecting cycles into a "master tour" starting with the largest cycle
    3. Mapping the resulting tour to edge traversals (M/N counts) that is returned for the next step of automation
    """
    base_val = int((target_n - 1) / 2)
    for k in edge_design_dict:
        edge_design_dict[k]['M'] = base_val
        edge_design_dict[k]['N'] = base_val

    G_work = _build_graph(mesh)
    found_cycles = []

    while G_work.number_of_edges() > 0:
        try:
            cycle_edges = nx.find_cycle(G_work, orientation='ignore')
        except nx.NetworkXNoCycle:
            break

        cycle_nodes = [e[0] for e in cycle_edges]
        found_cycles.append(cycle_nodes)
        G_work.remove_edges_from(cycle_edges)  # Remove these edges so that they're not used again
        G_work.remove_nodes_from(list(nx.isolates(G_work)))

    if not found_cycles:
        raise Exception("Heuristic failed: No cycles found in mesh.")

    # Splice cycles together
    found_cycles.sort(key=len, reverse=True)
    master_tour = found_cycles.pop(0)

    progress = True
    while progress and found_cycles:
        progress = False
        for i in range(len(found_cycles) - 1, -1, -1):
            candidate_cycle = found_cycles[i]

            common_nodes = set(master_tour).intersection(set(candidate_cycle))
            if common_nodes:
                pivot_node = common_nodes.pop()
                master_tour = _splice_cycles(master_tour, candidate_cycle, pivot_node)
                found_cycles.pop(i)
                progress = True

    # Apply tour after splicing
    visited_edges_count = 0
    unvisited_edges = set(tuple(i) for i in mesh.edges)
    for i in range(len(master_tour)):
        u = master_tour[i]
        v = master_tour[(i + 1) % len(master_tour)]  # Wrap around to close loop
        if (u, v) in edge_design_dict:
            edge_design_dict[(u, v)]['M'] += 1
            visited_edges_count += 1
            unvisited_edges.remove((u, v))
        elif (v, u) in edge_design_dict:
            edge_design_dict[(v, u)]['N'] += 1
            visited_edges_count += 1
            unvisited_edges.remove((v, u))
        else:
            raise Exception("Heuristic failed: Edge does not belong to mesh in tour reconstruction.")

    if scaffold_args.overfill_or_underfill.lower() == 'overfill':
        for i in unvisited_edges:
            edge_design_dict[i]['M'] += 1
            edge_design_dict[i]['N'] += 1

    return edge_design_dict


def _splice_cycles(tour_a: list, tour_b: list, pivot: int) -> list:
    """  Inserts tour_b into tour_a at the pivot node, for example:
    tour_a: [1, 2, 3, 1]
    tour_b: [3, 4, 5, 3]
    pivot: 3
    Result: [1, 2, 3, 4, 5, 3, 1]
    """
    idx = tour_b.index(pivot)
    rotated_b = tour_b[idx:] + tour_b[:idx]

    insert_idx = tour_a.index(pivot)
    new_tour = tour_a[:insert_idx] + rotated_b + tour_a[insert_idx:]

    return new_tour


def _get_fallback(mesh: Geometry, target_n: int, scaffold_args: ScaffoldArgs, edge_design_dict: dict):
    """ Defines a fall-back case for the geometry if the heuristic-driven solution can not find anything better """
    set_val = int((target_n - 1) / 2)  # target_n is odd, divide by 2 for one of each M / N below
    edge_dict = {k: {'M': set_val, 'N': set_val} for k in edge_design_dict.keys()}
    graph = _build_graph(mesh)
    closure, paths = _metric_closure(graph)
    vertices_ordered = _christofides_cycle(closure)

    unvisited_edges = set(tuple(i) for i in mesh.edges)

    def _update_physical_edge(_u, _v):
        if (_u, _v) in edge_dict:
            edge_dict[(_u, _v)]['M'] += 1
            if (_u, _v) in unvisited_edges: unvisited_edges.remove((_u, _v))
        elif (_v, _u) in edge_dict:
            edge_dict[(_v, _u)]['N'] += 1
            if (_v, _u) in unvisited_edges: unvisited_edges.remove((_v, _u))
        else:
            raise Exception(f'ERROR: Physical edge ({_u}, {_v}) not found in edge_dict')

    for v in range(len(vertices_ordered) - 1):
        v1, v2 = vertices_ordered[v], vertices_ordered[v + 1]
        path = paths[v1][v2]

        for j in range(len(path) - 1):
            p1, p2 = path[j], path[j + 1]
            _update_physical_edge(p1, p2)

    if scaffold_args.overfill_or_underfill.lower() == 'overfill':
        for i in unvisited_edges:
            edge_dict[i]['M'] += 1
            edge_dict[i]['N'] += 1

    return edge_dict


def _christofides_cycle(closure_graph):
    """ Estimates Hamiltonian cycle (visits each vertex once) using Christofides solver to the TSP """
    cycle = nx.algorithms.approximation.traveling_salesman_problem(
        closure_graph,
        weight="weight",
        cycle=True,
        method=nx.algorithms.approximation.christofides,
    )

    if cycle[0] != cycle[-1]:
        raise RuntimeError('ERROR: Cycle not found.')
    return cycle


def _metric_closure(G):
    """ Construct the metric closure of a connected weighted graph G. Metric closure is a complete graph where
    the weight of each edge (u, v) is the shortest path distance between u and v in G.

    Returns a complete graph K where weight(u, v) is the shortest path distance in G
    """
    if not nx.is_connected(G):
        raise ValueError("Mesh graph must be connected")

    all_paths = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    closure_graph = nx.Graph()
    for u in G.nodes:
        closure_graph.add_node(u)

    for u in G.nodes:
        for v in G.nodes:
            if u == v:
                continue
            closure_graph.add_edge(u, v, weight=all_lengths[u][v])

    return closure_graph, all_paths


def _build_graph(mesh: Geometry):
    graph = nx.Graph()

    for i, v in enumerate(mesh.vertices):
        graph.add_node(i, x=v[0], y=v[1], z=v[2])

    for i, (u, v) in enumerate(mesh.edges):
        w = mesh.edge_lengths_nm[i]
        graph.add_edge(u, v, weight=w)

    return graph
