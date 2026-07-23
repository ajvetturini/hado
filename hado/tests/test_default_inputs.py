from hado import HadoManager, Geometry
import numpy as np
import pytest
from pathlib import Path
import os

FILEPATH = Path('../app/pages/default_inputs').resolve()
SCALES = [1.0]

def get_test_files():
    if not FILEPATH.exists():
        return []

    all_files = [f for f in FILEPATH.iterdir() if f.suffix in ['.json', '.hado']]

    if os.environ.get("TEST_MODE", "quick").lower() == "quick":
        return [all_files[0]]

    # For full run use the command for your system below. The full test will take about 20 minutes and will perform
    # basically 30-40 different input meshes (see default_inputs in app/pages) across scales 1.0 and 1.5 (to perturb
    # features slightly and test robustness).
    #   $env:TEST_MODE="full"; pytest .\test_default_inputs.py  (Windows)
    #   TEST_MODE=full pytest test_default_inputs.py            (Linux/Mac)
    return all_files

def _scale(mgr, scale):
    """Scales vertices by scale."""
    temp = np.asarray(mgr.geometry.vertices, dtype=float)
    new_vertices = scale * temp
    new_geometry = Geometry(new_vertices, mgr.geometry.edges, mgr.geometry.n_per_edge)
    mgr.geometry = new_geometry


@pytest.mark.parametrize("file_path", get_test_files(), ids=lambda p: p.name)
@pytest.mark.parametrize("scale", SCALES)
def test_hado_default_inputs(file_path, scale):
    print(f'Testing {file_path} with scale {scale}...')
    manager, _ = HadoManager.load(file_path)
    _scale(manager, scale)
    try:
        manager.run()
        model = manager.nucleotide_level_model
        assert model is not None, f"Model at filepath {file_path} with scale {scale} execution returned None"
    except Exception as e:
        pytest.fail(f"Execution failed for {file_path.name} at scale {scale}. Error: {e}")