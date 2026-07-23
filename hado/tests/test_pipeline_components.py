from pathlib import Path
import unittest
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from hado.core.automation.autostaple import autostaple_hollowframe, simple_autobreak
from hado.core.automation.autostaple.breakpoint_labels import _label_nodes_for_breaks
from hado.core.automation.autostaple.staple_bundle_graph import (
    _apply_verified_staple_xovers_to_graph,
    _convert_graph_to_xovers_list,
)
from hado.core.automation.connections.connect_bundles import (
    _solve_via_hungarian,
    find_best_initial_state,
    get_rotated_positions,
    decompose_design_into_bundles,
)
from hado.core.automation.diagnostics.visualization import (
    build_breakpoint_heatmap,
    build_bundle_positions_figure,
    build_honeycomb_grid_figure,
)
from hado.core.automation.routing.honeycomb_xsect import get_cross_section, select_honeycomb_ring_by_diameter
from hado.core.automation.routing.lattice import get_lattice_config
from hado.core.automation.pipeline.manager import HadoManager
from hado.core.automation.model.nucleotide_model import HadoNucleotideModel, initialize_base_design
from hado.core.automation.pipeline.config import PipelineConfig
from hado.core.automation.pipeline.types import PipelineDiagnostics
from hado.core.automation.model.scaffold_crossover_decoder import ScaffoldCrossoverDecoder
from hado.core.automation.routing.scaffold_routing import perform_scaffold_routing
from hado.core.automation.mitering.scaffold_overhangs import add_scaffold_3p_overhangs
from hado.core.export.oxdna import _old_oxdna_format
from hado.core.automation.adapters.oxdna import OxDNASystem, _OXDNA_NM_TO_UNITS, build_oxdna
from hado.core.utils import Geometry, ScaffoldArgs, StapleArgs


class MinimalDesign:
    def __init__(self, scaffold_nucleotides, staple_dirs=None, free_ends=None):
        self._scaffold_nucleotides = np.array(scaffold_nucleotides, dtype=np.bool_)
        self._staple_dirs = np.array(staple_dirs or [True] * len(scaffold_nucleotides), dtype=np.bool_)
        self._free_ends = free_ends or {}

    def get_scaffold_nucleotides(self):
        return self._scaffold_nucleotides.copy()

    def get_staple_directions(self):
        return self._staple_dirs.copy()

    def check_free_ends(self, helix):
        return self._free_ends.get(helix, (False, False))


class MinimalOxDNAGeometry:
    edges = [(0, 1)]


class MinimalOxDNAModel:
    def __init__(self):
        active_nts = np.zeros((1, 20), dtype=np.bool_)
        active_nts[0, 5:15] = True

        self.geometry = MinimalOxDNAGeometry()
        self._active_nts = active_nts
        self._scaffold_path = [(0, 5), (0, 6), (0, 7)]
        self._staple_paths = [[(0, 7), (0, 6), (0, 5)]]
        self.scaffold_args = ScaffoldArgs()

    def get_axial_rise(self):
        return 1.0

    def get_helix_bundle_grid_locations(self):
        return np.array([[2.0, 3.0]])

    def get_scaffold_nucleotides(self):
        return self._active_nts.copy()

    def get_staple_nucleotides(self):
        return self._active_nts.copy()

    def get_helix_to_bundle(self):
        return np.array([0])

    def get_final_rotations(self):
        return {0: 0.0}

    def get_point(self, idx):
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -10.0]][idx])

    def get_ordered_scaffold_nts(self):
        return list(self._scaffold_path)

    def get_ordered_staple_nts(self):
        return [list(path) for path in self._staple_paths]

    def get_staple_color_palette(self):
        return ["#336699"]

    def get_scaffold_direction(self, helix):
        return True

    def get_staple_direction(self, helix):
        return False

    def get_backbone_rotation_angle(self, nt_position, forward):
        return 0.0


class PipelineRefactorTests(unittest.TestCase):
    @staticmethod
    def _triangle_manager():
        default_input = Path(__file__).parents[1] / "app" / "pages" / "default_inputs" / "triangle.json"
        manager = HadoManager.load(default_input)
        return manager

    def test_oxdna_centers_are_anchored_to_edge_midpoint(self):
        model = MinimalOxDNAModel()
        system = OxDNASystem.__new__(OxDNASystem)
        system.model = model
        system.helix_frames = {}
        system.centers = {}

        system._build_helix_frames_and_centers()

        centers_nm = np.array([
            system.centers[(0, nt_idx)] for nt_idx in range(5, 15)
        ]) / _OXDNA_NM_TO_UNITS

        np.testing.assert_allclose(
            system.helix_frames[0]["midpoint_nm"],
            np.array([2.0, 3.0, -5.0]),
        )
        np.testing.assert_allclose(
            centers_nm.mean(axis=0),
            np.array([2.0, 3.0, -5.0]),
        )

    def test_build_oxdna_uses_model_ordered_paths_directly(self):
        class FallbackSequenceManager:
            verbose = False

            def get_sequences(self, model):
                raise RuntimeError("force fallback sequences")

        model = MinimalOxDNAModel()
        system = build_oxdna(model, FallbackSequenceManager())

        self.assertEqual(system.design_scaf_path, model.get_ordered_scaffold_nts())
        self.assertEqual(system.design_staple_paths, model.get_ordered_staple_nts())
        self.assertEqual(system.strand_paths, [model.get_ordered_scaffold_nts(), *model.get_ordered_staple_nts()])
        self.assertEqual([system.dat[i]["offset"] for i in range(3)], [5, 6, 7])
        self.assertEqual([system.dat[i]["offset"] for i in range(3, 6)], [7, 6, 5])

    def test_oxdna_derives_staple_sequences_from_ordered_paths(self):
        class ShortSequenceManager:
            verbose = False

            def get_sequences(self, model):
                return True, [
                    ["Name", "Start", "End", "Sequence"],
                    ["scaffold", "", "", "ACG"],
                    ["staple", "", "", "TT"],
                ]

        system = OxDNASystem.__new__(OxDNASystem)
        system.manager = ShortSequenceManager()
        system.model = object()
        system.strand_paths = [
            [(0, 0), (0, 1), (0, 2)],
            [(0, 2), (0, 1), (0, 0)],
        ]

        sequences = system._get_strand_sequences()

        self.assertEqual([len(sequence) for sequence in sequences], [3, 3])
        self.assertEqual(
            sequences[1],
            "".join(
                {"A": "T", "C": "G", "G": "C", "T": "A"}[base]
                for base in reversed(sequences[0])
            ),
        )

    def test_oxview_strand_endpoints_reference_actual_terminal_monomers(self):
        class FallbackSequenceManager:
            verbose = False

            def get_sequences(self, model):
                raise RuntimeError("force fallback sequences")

        system = build_oxdna(MinimalOxDNAModel(), FallbackSequenceManager())

        for strand in system.strands:
            monomers = strand["monomers"]
            self.assertEqual(strand["end5"], monomers[0]["id"])
            self.assertEqual(strand["end3"], monomers[-1]["id"])
            self.assertNotIn("n5", monomers[0])
            self.assertNotIn("n3", monomers[-1])

    def test_lattice_config_matches_honeycomb_defaults(self):
        lattice = get_lattice_config("dna_honeycomb")

        self.assertEqual(lattice.grid_type, "honeycomb")
        self.assertEqual(lattice.period, 21)
        self.assertAlmostEqual(lattice.axial_rise, 0.34)
        self.assertAlmostEqual(lattice.helix_spacing, 2.375)

    def test_pipeline_config_preserves_unknown_kwargs(self):
        config = PipelineConfig.from_kwargs({
            "skip_stapling": True,
            "angle_step_size": 30,
            "custom_option": "kept",
        })

        self.assertTrue(config.control.skip_stapling)
        self.assertEqual(config.connection_optimization.angle_step_size, 30)
        self.assertEqual(config.extra["custom_option"], "kept")
        self.assertEqual(config.to_kwargs()["custom_option"], "kept")

    def test_user_input_validation_raises_explicit_exceptions(self):
        with self.assertRaises(ValueError):
            ScaffoldArgs(overfill_or_underfill="invalid")

        with self.assertRaises(ValueError):
            StapleArgs(min_length_after_break=80, max_length_after_break=60)

        with self.assertRaises(ValueError):
            Geometry([[0, 0, 0]], [[0, 2]], 2)

    def test_autostaple_package_exports_public_entry_points(self):
        self.assertEqual(
            autostaple_hollowframe.__module__,
            "hado.core.automation.autostaple.hollowframe",
        )
        self.assertEqual(
            simple_autobreak.__module__,
            "hado.core.automation.autostaple.autobreak",
        )

    def test_refactored_boundary_checks_raise_explicit_exceptions(self):
        with self.assertRaises(ValueError):
            HadoManager.load(data={"design_name": "missing-fields"})

        with self.assertRaises(ValueError):
            get_cross_section(0, 1, 2.375)

        with self.assertRaises(ValueError):
            get_cross_section(1, 1, -1)

    def test_visualization_builders_return_figures_without_showing(self):
        bundle_fig = build_bundle_positions_figure(np.array([[0, 0, 0], [1, 0, 0]]))
        self.assertEqual(len(bundle_fig.data), 1)

        heatmap_fig = build_breakpoint_heatmap([
            {"helix": 0, "nt_position": 1, "valid_breakpoint": True},
            {"helix": 0, "nt_position": 2, "valid_breakpoint": False},
        ])
        self.assertEqual(heatmap_fig.layout.title.text, "Valid Breakpoint Heatmap")

        honeycomb_fig = build_honeycomb_grid_figure([(0, 0)], [(1, 0)])
        self.assertEqual(len(honeycomb_fig.axes), 1)
        plt.close(honeycomb_fig)

    def test_pipeline_diagnostics_can_collect_figures(self):
        diagnostics = PipelineDiagnostics()
        fig = build_bundle_positions_figure(np.array([[0, 0, 0]]))
        diagnostics.record_figure('connections', 'sample', fig)
        self.assertIs(diagnostics.figures['connections']['sample'], fig)

    def test_even_scaffold_routing_splits_edges_evenly(self):
        geometry = Geometry(
            vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 0]],
            edges=[[0, 1], [1, 2], [2, 0]],
            n_per_edge=4,
        )

        routing = perform_scaffold_routing(geometry, ScaffoldArgs())

        self.assertEqual(len(routing), 3)
        for edge_definition in routing.values():
            self.assertEqual(edge_definition['M'], 2)
            self.assertEqual(edge_definition['N'], 2)

    def test_geometry_resolves_edge_thickness_to_honeycomb_helix_counts(self):
        geometry = Geometry(
            vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 0]],
            edges=[[0, 1], [1, 2], [2, 0]],
            edge_thickness_nm=11.7,
        )

        self.assertEqual(geometry.n_per_edge, [8, 8, 8])
        self.assertEqual(geometry.edge_thickness_nm, [11.7, 11.7, 11.7])
        self.assertAlmostEqual(geometry.edge_thickness_actual_nm[0], 11.625)
        self.assertEqual(geometry.to_dict()["edge_thickness_nm"], [11.7, 11.7, 11.7])

        reloaded = Geometry(**geometry.to_dict())
        self.assertEqual(reloaded.n_per_edge, geometry.n_per_edge)
        self.assertEqual(reloaded.edge_thickness_nm, geometry.edge_thickness_nm)

    def test_geometry_rejects_ambiguous_edge_width_inputs(self):
        with self.assertRaises(ValueError):
            Geometry(
                vertices=[[0, 0, 0], [10, 0, 0]],
                edges=[[0, 1]],
                n_per_edge=4,
                edge_thickness_nm=11.7,
            )

    def test_honeycomb_ring_selector_uses_closest_diameter(self):
        lattice = get_lattice_config("dna_honeycomb")
        selection = select_honeycomb_ring_by_diameter(
            15.0,
            lattice.helix_spacing,
            helix_diameter=lattice.diameter,
        )

        self.assertEqual(selection["n_per_edge"], 16)
        self.assertAlmostEqual(selection["actual_diameter"], 14.692318727627262, places=6)

    def test_mixed_odd_scaffold_routing_raises_not_implemented(self):
        geometry = Geometry(
            vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 0]],
            edges=[[0, 1], [1, 2], [2, 0]],
            n_per_edge=[3, 5, 3],
        )

        with self.assertRaises(NotImplementedError):
            perform_scaffold_routing(geometry, ScaffoldArgs())

    def test_connection_optimizer_rejects_invalid_hungarian_threshold(self):
        manager = self._triangle_manager()
        routing = perform_scaffold_routing(manager.geometry, manager.scaffold_args)
        design = initialize_base_design(
            manager.geometry,
            manager.scaffold_args,
            manager.staple_args,
            routing,
        )
        best_state, _ = find_best_initial_state(design)
        base_positions_and_axes = decompose_design_into_bundles(design)
        rotated_positions = get_rotated_positions(design, base_positions_and_axes, best_state)
        helix_to_bundle = design.get_helix_to_bundle()

        positions = None
        for vertex in range(len(design.geometry.vertices)):
            bundles_at_vertex = list(design.get_bundles_at_vertex(vertex))
            if len(bundles_at_vertex) <= 1:
                continue

            candidate = {}
            for bundle in bundles_at_vertex:
                indices = np.where(helix_to_bundle == bundle)[0]
                bundle_points = rotated_positions[indices]
                global_senders, local_senders = design.get_sender_indices(bundle, vertex)
                global_receivers, local_receivers = design.get_receiver_indices(bundle, vertex)
                candidate[int(bundle)] = (
                    bundle_points[local_senders],
                    bundle_points[local_receivers],
                    global_senders,
                    global_receivers,
                )
            positions = candidate
            break

        self.assertIsNotNone(positions)
        with self.assertRaises(ValueError):
            _solve_via_hungarian(positions, design, min_hungarian_threshold=0.0)

    def test_breakpoint_labeling_marks_nodes_and_records_heatmap(self):
        graph = nx.path_graph([(0, i) for i in range(10)])
        design = MinimalDesign([[True] * 10])
        staple_args = StapleArgs(min_run_post_xover=2, min_run_post_bundle_connection=2)
        diagnostics = PipelineDiagnostics()

        labeled = _label_nodes_for_breaks(
            graph,
            all_scaffold_xovers=[[0, 4, 0, -1]],
            all_staple_xovers=np.empty((0, 4), dtype=np.int64),
            design=design,
            new_staple_nts=np.array([[True] * 10], dtype=np.bool_),
            staple_args=staple_args,
            show_breakpoint_labels=True,
            diagnostics=diagnostics,
        )

        self.assertFalse(labeled.nodes[(0, 0)]['valid_breakpoint'])
        self.assertFalse(labeled.nodes[(0, 3)]['valid_breakpoint'])
        self.assertTrue(labeled.nodes[(0, 2)]['valid_breakpoint'])
        self.assertIn('autostaple', diagnostics.figures)
        self.assertIn('breakpoint_heatmap', diagnostics.figures['autostaple'])

    def test_short_open_staples_are_flagged_for_repair(self):
        graph = nx.path_graph([(0, i) for i in range(4)])
        nx.set_node_attributes(graph, True, 'valid_breakpoint')
        design = MinimalDesign([[True] * 4])
        staple_args = StapleArgs()

        staple_breaks, staples_to_repair, total_cost = simple_autobreak(graph, design, staple_args)

        self.assertEqual(staple_breaks, [])
        self.assertEqual(total_cost, 0)
        self.assertEqual(len(staples_to_repair), 1)
        self.assertEqual(set(staples_to_repair[0]), set(graph.nodes))

    def test_exact_bound_open_staples_are_not_broken(self):
        graph = nx.path_graph([(0, i) for i in range(20)])
        nx.set_node_attributes(graph, True, 'valid_breakpoint')
        design = MinimalDesign([[True] * 20])
        staple_args = StapleArgs(min_length_after_break=20, max_length_after_break=60)

        staple_breaks, staples_to_repair, total_cost = simple_autobreak(graph, design, staple_args)

        self.assertEqual(staple_breaks, [])
        self.assertEqual(staples_to_repair, [])
        self.assertEqual(total_cost, 0)

    def test_free_end_breakpoints_wait_until_after_first_crossover(self):
        graph = nx.path_graph([(0, i) for i in range(6)])
        graph.add_edge((0, 5), (1, 5))
        graph.add_edges_from([((1, i), (1, i + 1)) for i in range(5, 9)])
        design = MinimalDesign(
            [[True] * 10, [True] * 10],
            free_ends={0: (True, False)},
        )
        staple_args = StapleArgs(min_run_post_xover=1, min_run_post_bundle_connection=1)

        labeled = _label_nodes_for_breaks(
            graph,
            all_scaffold_xovers=np.empty((0, 4), dtype=np.int64),
            all_staple_xovers=np.array([[0, 5, 1, -1]], dtype=np.int64),
            design=design,
            new_staple_nts=np.array([[True] * 10, [True] * 10], dtype=np.bool_),
            staple_args=staple_args,
            show_breakpoint_labels=False,
        )

        self.assertTrue(all(not labeled.nodes[(0, i)]['valid_breakpoint'] for i in range(6)))
        self.assertTrue(labeled.nodes[(1, 6)]['valid_breakpoint'])

    def test_scaffold_3p_overhangs_update_only_scaffold_sender_end(self):
        design = HadoNucleotideModel.__new__(HadoNucleotideModel)
        design._scaffold_dirs = np.array([True, False], dtype=np.bool_)
        design._scaffold_nucleotides = np.zeros((2, 8), dtype=np.bool_)
        design._scaffold_nucleotides[0, 1:4] = True
        design._scaffold_nucleotides[1, 3:6] = True
        design._staple_nucleotides = design._scaffold_nucleotides.copy()
        design._scaffold_crossovers = np.array([[0, 3, 1, 5]], dtype=np.int64)
        design._helix_to_bundle = np.array([0, 1], dtype=np.int64)
        design._idx_edge_map = np.array([[0, 1], [1, 2]], dtype=np.int64)
        final_positions = {
            0: [(1, np.array([0.0, 0.0, 0.0]))],
            1: [(1, np.array([1.2, 0.0, 0.0]))],
        }

        added = add_scaffold_3p_overhangs(
            design,
            final_positions,
            StapleArgs(polyt_bulge_dist=0.5, max_staple_spacer_length=4),
        )

        self.assertEqual(added, 2)
        self.assertTrue(np.all(design.get_scaffold_nucleotides()[0, 4:6]))
        self.assertFalse(np.any(design.get_staple_nucleotides()[0, 4:6]))
        np.testing.assert_array_equal(design.get_scaffold_crossovers(), np.array([[0, 5, 1, 5]]))

    def test_breakpoint_labeling_bounds_to_double_stranded_region(self):
        graph = nx.path_graph([(0, i) for i in range(2, 8)])
        design = MinimalDesign([[True] * 10])
        staple_nts = np.zeros((1, 10), dtype=np.bool_)
        staple_nts[0, 2:8] = True
        staple_args = StapleArgs(min_run_post_xover=1, min_run_post_bundle_connection=2)

        labeled = _label_nodes_for_breaks(
            graph,
            all_scaffold_xovers=np.empty((0, 4), dtype=np.int64),
            all_staple_xovers=np.empty((0, 4), dtype=np.int64),
            design=design,
            new_staple_nts=staple_nts,
            staple_args=staple_args,
            show_breakpoint_labels=False,
        )

        self.assertFalse(labeled.nodes[(0, 2)]['valid_breakpoint'])
        self.assertFalse(labeled.nodes[(0, 3)]['valid_breakpoint'])
        self.assertTrue(labeled.nodes[(0, 4)]['valid_breakpoint'])
        self.assertFalse(labeled.nodes[(0, 7)]['valid_breakpoint'])

    def test_staple_args_emit_advisory_warnings_for_unusual_ranges(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            StapleArgs(
                min_length_after_break=14,
                max_length_after_break=61,
                min_dist_between_xovers=11,
            )

        messages = [str(item.message) for item in caught]
        self.assertTrue(any("Min distance between xovers" in message for message in messages))
        self.assertTrue(any("Min staple length should be between" in message for message in messages))
        self.assertTrue(any("Max staple length should be between" in message for message in messages))

    def test_verified_staple_xovers_are_applied_to_break_graph(self):
        graph = nx.Graph()
        graph.add_edges_from([
            ((0, 0), (0, 1)),
            ((0, 1), (0, 2)),
            ((1, 0), (1, 1)),
            ((1, 1), (1, 2)),
        ])

        _apply_verified_staple_xovers_to_graph(
            graph,
            np.array([[0, 1, 1, -1], [1, 2, 0, -1]], dtype=np.int64),
        )

        self.assertFalse(graph.has_edge((0, 1), (0, 2)))
        self.assertFalse(graph.has_edge((1, 1), (1, 2)))
        self.assertTrue(graph.has_edge((0, 1), (1, 1)))
        self.assertTrue(graph.has_edge((0, 2), (1, 2)))

    def test_staple_crossover_conversion_groups_edges_by_position(self):
        graph = nx.Graph()
        graph.add_edge((0, 10), (1, 10))
        graph.add_edge((0, 20), (1, 20))
        graph.add_edge((0, 11), (1, 11))
        graph.add_edge((0, 21), (1, 21))

        xovers = _convert_graph_to_xovers_list(
            graph,
            staple_dirs=np.array([True, False], dtype=np.bool_),
            local_to_global={0: 0, 1: 1},
        )

        self.assertEqual(
            xovers.tolist(),
            [[0, 10, 1, -1], [1, 11, 0, -1], [0, 20, 1, -1], [1, 21, 0, -1]],
        )

    def test_scaffold_decoder_sorts_internal_middle_pairs_consistently(self):
        sorted_data = ScaffoldCrossoverDecoder._sort_crossovers([
            (0, 1),
            (2, 3, 'INTERNAL'),
            (4, 5, 'INTERNAL_MIDDLE'),
            (5, 4, 'INTERNAL_MIDDLE'),
        ])

        self.assertEqual(sorted_data['END'], [(0, 1)])
        self.assertEqual(sorted_data['INTERNAL'], [(2, 3, 'INTERNAL')])
        self.assertEqual(len(sorted_data['INTERNAL_MIDDLE']), 1)
        self.assertEqual(
            sorted_data['INTERNAL_MIDDLE'][0],
            ((4, 5, 'INTERNAL_MIDDLE'), (5, 4, 'INTERNAL_MIDDLE')),
        )

    def test_scadnano_adapter_sets_design_on_model(self):
        manager = self._triangle_manager()
        manager.run()
        design = manager.nucleotide_level_model

        design.set_scadnano(manager.scaffold_args.scaffold_sequence, manager.staple_args.unpaired_sequence)
        sc_design = design.get_sc_design()

        self.assertFalse(design.scadnano_not_set())
        self.assertEqual(len(sc_design.helices), design.get_scaffold_nucleotides().shape[0])

    def test_manager_run_stores_model_and_prepares_exports(self):
        manager = self._triangle_manager()

        manager.run()
        design = manager.nucleotide_level_model

        self.assertIs(manager.get_nucleotide_model(), design)
        self.assertFalse(design.scadnano_not_set())
        self.assertFalse(design.oxdna_not_set())
        self.assertIsInstance(manager.get_cadnano_json(), dict)

        data = manager.to_json()
        model_data = data['nucleotide_level_model']
        self.assertIsNotNone(model_data)
        self.assertNotIn('_sc_design', model_data)
        self.assertNotIn('_oxdna_system', model_data)

        loaded_manager = HadoManager.load(data=data)
        self.assertIsInstance(loaded_manager.get_cadnano_json(), dict)

    def test_manager_scaffold_sequence_change_invalidates_prepared_exports(self):
        manager = self._triangle_manager()
        manager.run()
        design = manager.nucleotide_level_model

        manager.set_new_scaffold_sequence("ACGT" * 3000)

        self.assertTrue(design.scadnano_not_set())
        self.assertTrue(design.oxdna_not_set())

    def test_old_oxdna_topology_uses_3prime_then_5prime_neighbor_order(self):
        oxview_json = {
            'box': [10, 10, 10],
            'systems': [{
                'id': 0,
                'strands': [{
                    'id': 1,
                    'monomers': [
                        {'id': 0, 'type': 'A', 'p': [0, 0, 0], 'a1': [1, 0, 0], 'a3': [0, 0, 1], 'n3': 1},
                        {'id': 1, 'type': 'C', 'p': [1, 0, 0], 'a1': [1, 0, 0], 'a3': [0, 0, 1], 'n5': 0, 'n3': 2},
                        {'id': 2, 'type': 'G', 'p': [2, 0, 0], 'a1': [1, 0, 0], 'a3': [0, 0, 1], 'n5': 1},
                    ],
                }],
            }],
        }

        top, _ = _old_oxdna_format(oxview_json)

        self.assertEqual(top[0], '3 1\n')
        self.assertEqual(top[1], '1 A 1 -1\n')
        self.assertEqual(top[2], '1 C 2 0\n')
        self.assertEqual(top[3], '1 G -1 1\n')


    def test_model_exposes_correctly_spelled_staple_color_palette_alias(self):
        geometry = Geometry([[0, 0, 0], [20, 0, 0]], [[0, 1]], 2)
        design = initialize_base_design(
            geometry,
            ScaffoldArgs(),
            StapleArgs(),
            {(0, 1): {"M": 1, "N": 1}},
        )

        preferred = design.get_staple_color_palette()
        legacy = design.get_staple_color_pallete()

        self.assertEqual(preferred, legacy)
        self.assertIsNot(preferred, design._staple_color_palette)

    def test_ordered_scaffold_traversal_with_explicit_start_point(self):
        design = HadoNucleotideModel.__new__(HadoNucleotideModel)
        design._scaffold_start_point = np.array([0, 0])
        design._scaffold_dirs = np.array([True, False], dtype=np.bool_)
        design._scaffold_nucleotides = np.array([
            [True, True],
            [True, True],
        ], dtype=np.bool_)
        design._scaffold_crossovers = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
        ])

        self.assertEqual(
            design.get_ordered_scaffold_nts(),
            [(0, 0), (0, 1), (1, 1), (1, 0)],
        )

    def test_ordered_open_staple_traversal_preserves_crossover_nucleotides(self):
        design = HadoNucleotideModel.__new__(HadoNucleotideModel)
        design._staple_dirs = np.array([True, False], dtype=np.bool_)
        design._staple_nucleotides = np.array([
            [True, False, True, True],
            [True, True, True, True],
        ], dtype=np.bool_)
        design._staple_breaks = np.array([[0, 0, 2]])
        design._staple_crossovers = np.array([
            [0, 3, 1, 3],
            [1, 0, 0, 0],
        ])

        self.assertEqual(
            design.get_ordered_staple_nts(),
            [[(0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0), (0, 0)]],
        )

    def test_ordered_staple_traversal_stops_at_natural_run_end(self):
        design = HadoNucleotideModel.__new__(HadoNucleotideModel)
        design._staple_dirs = np.array([False, True], dtype=np.bool_)
        design._staple_nucleotides = np.array([
            [False, True, True, True, False],
            [False, True, True, True, False],
        ], dtype=np.bool_)
        design._staple_breaks = np.empty((0, 3), dtype=np.int64)
        design._staple_crossovers = np.array([
            [0, 1, 1, 1],
        ])

        self.assertEqual(
            design.get_ordered_staple_nts(),
            [[(0, 3), (0, 2), (0, 1), (1, 1), (1, 2), (1, 3)]],
        )

    def test_ordered_circular_staple_traversal_uses_stable_start(self):
        design = HadoNucleotideModel.__new__(HadoNucleotideModel)
        design._staple_dirs = np.array([True, False], dtype=np.bool_)
        design._staple_nucleotides = np.array([
            [True, True],
            [True, True],
        ], dtype=np.bool_)
        design._staple_breaks = np.empty((0, 3), dtype=np.int64)
        design._staple_crossovers = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
        ])

        staples = design.get_ordered_staple_nts()

        self.assertEqual(len(staples), 1)
        self.assertEqual(set(staples[0]), {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_scaffold_crossover_options_filter_active_boundaries_by_nt_index(self):
        design = HadoNucleotideModel.__new__(HadoNucleotideModel)
        design.staple_args = StapleArgs(min_run_post_bundle_connection=2)
        design._period = 10
        design._scaffold_dirs = np.array([True, False], dtype=np.bool_)
        design._scaffold_nucleotides = np.array([
            [True] * 10,
            [True] * 10,
        ], dtype=np.bool_)

        h0_to_h1 = np.full(10, -1, dtype=np.int32)
        h1_to_h0 = np.full(10, -1, dtype=np.int32)
        h0_to_h1[[1, 2, 5, 6, 8, 9]] = 1
        h1_to_h0[[1, 2, 5, 6, 8, 9]] = 0
        design._scaffold_xover_map = {
            (0, 1): h0_to_h1,
            (1, 0): h1_to_h0,
        }

        self.assertEqual(design.get_all_valid_scaffold_crossover_options(0, 1), {(5, 6)})

    def test_square_scaffold_crossover_options_preserve_wraparound_pairing(self):
        design = HadoNucleotideModel.__new__(HadoNucleotideModel)
        design.staple_args = StapleArgs(min_run_post_bundle_connection=1)
        design._period = 32
        design._scaffold_dirs = np.array([True, False], dtype=np.bool_)
        design._staple_dirs = np.array([False, True], dtype=np.bool_)
        design._scaffold_nucleotides = np.array([
            [True] * 32,
            [True] * 32,
        ], dtype=np.bool_)

        h0_to_h1 = np.full(32, -1, dtype=np.int32)
        h1_to_h0 = np.full(32, -1, dtype=np.int32)
        h0_to_h1[[0, 10, 11, 20, 21, 31]] = 1
        h1_to_h0[[0, 10, 11, 20, 21, 31]] = 0
        design._scaffold_xover_map = {
            (0, 1): h0_to_h1,
            (1, 0): h1_to_h0,
        }
        design._staple_xover_map = {
            (0, 1): h0_to_h1,
            (1, 0): h1_to_h0,
        }

        np.testing.assert_array_equal(
            design.get_all_scaffold_crossover_options(0, 1),
            np.array([10, 20, 31], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            design.get_all_scaffold_crossover_options(1, 0),
            np.array([11, 21, 0], dtype=np.int32),
        )
        self.assertEqual(design.get_nearest_scaffold_crossover_index(0, 1, 30), 31)
        self.assertEqual(design.get_nearest_scaffold_crossover_index(1, 0, 30), 0)

    def test_staple_crossover_conversion_supports_period_wraparound(self):
        graph = nx.Graph()
        graph.add_edge((0, 31), (1, 31))
        graph.add_edge((0, 0), (1, 0))

        xovers = _convert_graph_to_xovers_list(
            graph,
            staple_dirs=np.array([True, False], dtype=np.bool_),
            local_to_global={0: 0, 1: 1},
            period=32,
        )

        self.assertEqual(
            xovers.tolist(),
            [[0, 31, 1, -1], [1, 0, 0, -1]],
        )

    def test_verified_staple_xovers_support_period_wraparound(self):
        graph = nx.Graph()
        graph.add_edges_from([
            ((0, 31), (0, 0)),
            ((1, 31), (1, 0)),
        ])

        _apply_verified_staple_xovers_to_graph(
            graph,
            np.array([[0, 31, 1, -1], [1, 0, 0, -1]], dtype=np.int64),
            period=32,
        )

        self.assertFalse(graph.has_edge((0, 31), (0, 0)))
        self.assertFalse(graph.has_edge((1, 31), (1, 0)))
        self.assertTrue(graph.has_edge((0, 31), (1, 31)))
        self.assertTrue(graph.has_edge((0, 0), (1, 0)))

    def test_manager_run_can_return_diagnostics(self):
        manager = self._triangle_manager()

        diagnostics = manager.run(
            skip_mitering=True,
            skip_stapling=True,
            return_diagnostics=True,
        )
        design = manager.nucleotide_level_model

        self.assertEqual(design.get_period(), 21)
        self.assertIn("routing", diagnostics.stage_results)
        self.assertIn("scaffold_crossovers", diagnostics.stage_results)
        self.assertEqual(diagnostics.figures, {})
        self.assertIn("Skipping mitering step as specified by user...", diagnostics.info_messages)
        self.assertIn("Skipping stapling step as specified by user...", diagnostics.info_messages)


if __name__ == "__main__":
    unittest.main()
