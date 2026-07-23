import os
import numpy as np
from hado import HadoManager, Geometry

def _scale(mgr, scale):
    temp = np.asarray(mgr.geometry.vertices, dtype=float)
    new_vertices = scale * temp
    new_geometry = Geometry(new_vertices, mgr.geometry.edges, mgr.geometry.n_per_edge)
    mgr.geometry = new_geometry

scale_these = [
]

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.json'):
        f = os.path.join(os.path.dirname(__file__), file)
        try:
            manager, _ = HadoManager.load(f)
        except Exception as e:
            print(f'ERROR reading in {file}: {e}')
            continue

        if manager.design_name in scale_these:
            _scale(manager, scale=1.15)
        manager.save()


