import json
from pathlib import Path

import numpy as np
import pytest

from hado.core.automation.model.nucleotide_model import initialize_base_design
from hado.core.automation.pipeline.config import PipelineConfig
from hado.core.automation.pipeline.manager import HadoManager
from hado.core.automation.routing.cross_sections import get_cross_section
from hado.core.automation.routing.scaffold_routing import perform_scaffold_routing
from hado.core.automation.routing.square_xsect import (
    get_square_mapping_cadnano,
    get_square_ring_diameter,
    set_hollow_square,
)
from hado.core.export.cadnano import CaDNAnoWriter, get_cadnano_json
from hado.core.utils import Geometry, ScaffoldArgs, StapleArgs


def test_square_cross_section_builds_plus_sign_helix_count():
    spacing = 3.25
    evens, odds, metadata = set_hollow_square(9, 9, spacing)

    assert len(evens) == 9
    assert len(odds) == 9
    assert metadata["num_helices"] == 18
    assert get_square_ring_diameter(evens, odds) > 0

    all_points = np.array(evens + odds)
    distances = np.linalg.norm(all_points[:, None, :] - all_points[None, :, :], axis=-1)
    neighbor_counts = np.isclose(distances, spacing).sum(axis=1)
    assert np.all(neighbor_counts >= 2)


def test_square_cross_section_dispatches_from_registry():
    evens, odds = get_cross_section(9, 9, 3.25, lattice_type="square")

    assert len(evens) == 9
    assert len(odds) == 9


def test_plus_sign_default_input_builds_square_base_design():
    input_path = Path(__file__).parents[1] / "app" / "pages" / "default_inputs" / "plus_sign.json"
    data = json.loads(input_path.read_text())
    data["scaffold_args"]["lattice_type"] = "dna_square"

    geometry = Geometry(**data["geometry"])
    scaffold_args = ScaffoldArgs(**data["scaffold_args"])
    staple_args = StapleArgs(**data["staple_args"])
    routing = perform_scaffold_routing(geometry, scaffold_args)
    model = initialize_base_design(geometry, scaffold_args, staple_args, routing)

    assert model._grid_type == "square"
    assert model.get_period() == 32
    assert model.get_scaffold_nucleotides().shape[0] == 72


def test_square_cadnano_mapping_preserves_local_lattice_neighbors():
    evens, odds, _ = set_hollow_square(9, 9, 3.25)
    mapping = get_square_mapping_cadnano(3.25, [evens + odds], precision=5)

    assert set(mapping) == {0}
    rows_and_cols = np.array(list(mapping[0].values()))
    assert rows_and_cols.shape == (18, 2)
    assert len({tuple(point) for point in rows_and_cols}) == 18


def test_pipeline_config_can_disable_export_preparation():
    config = PipelineConfig.from_kwargs({"prepare_exports": False})

    assert not config.control.prepare_exports
    assert not config.to_kwargs()["prepare_exports"]


def test_hado_manager_switches_plus_sign_to_square_lattice():
    manager, _ = HadoManager.load_default('plus_sign')

    manager.set_lattice_type("dna_square")

    assert manager.scaffold_args.lattice_type == "dna_square"
    assert manager.get_nucleotide_model() is None


def test_hado_manager_runs_square_plus_sign_pipeline_without_exports():
    manager, _ = HadoManager.load_default('plus_sign')
    manager.set_lattice_type("dna_square")

    diagnostics = manager.run(
        skip_stapling=True,
        return_diagnostics=True,
        prepare_exports=False,
    )
    design = manager.nucleotide_level_model

    assert design._grid_type == "square"
    assert design.get_period() == 32
    assert design.scadnano_not_set()
    assert "scaffold_crossovers" in diagnostics.stage_results


def test_square_cadnano_mapping_preserves_parity_across_packed_bundles():
    evens, odds, _ = set_hollow_square(9, 9, 3.25)
    mapping = get_square_mapping_cadnano(3.25, [evens + odds, evens + odds], precision=5)

    for bundle_mapping in mapping.values():
        for point in evens:
            col, row = bundle_mapping[tuple(np.round(point, decimals=5))]
            assert (col + row) % 2 == 0
        for point in odds:
            col, row = bundle_mapping[tuple(np.round(point, decimals=5))]
            assert (col + row) % 2 == 1


def test_custom_cross_section_builds_base_design_and_disables_public_cadnano():
    custom_cross_section = {"evens": [(0.0, 1.5)], "odds": [(0.0, -1.5)]}
    geometry = Geometry([[0, 0, 0], [20, 0, 0]], [[0, 1]], n_per_edge=2)
    scaffold_args = ScaffoldArgs(lattice_type="dna_square", custom_cross_section=custom_cross_section)
    model = initialize_base_design(geometry, scaffold_args, StapleArgs(), {(0, 1): {"M": 1, "N": 1}})

    assert scaffold_args.has_custom_cross_section()
    assert model.get_scaffold_nucleotides().shape[0] == 2
    np.testing.assert_allclose(model.get_helix_bundle_grid_locations(), np.array([[0.0, 1.5], [0.0, -1.5]]))

    with pytest.raises(ValueError, match="custom cross-sections"):
        get_cadnano_json("custom", model)

    writer = CaDNAnoWriter("custom", model, False, allow_custom_cross_section=True)
    assert len(writer.get_json_data()["vstrands"]) == 2


def test_custom_cross_section_generator_is_validated():
    def generator(M, N, L):
        return [(0.0, L)], [(0.0, -L)]

    evens, odds = get_cross_section(1, 1, 3.25, lattice_type="square", cross_section_generator=generator)

    assert evens == [(0.0, 3.25)]
    assert odds == [(0.0, -3.25)]

    with pytest.raises(ValueError, match="Expected 2 even-running"):
        get_cross_section(2, 2, 3.25, lattice_type="square", cross_section_generator=generator)
