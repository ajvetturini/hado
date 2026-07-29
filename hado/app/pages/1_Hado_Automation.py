import tempfile
from pathlib import Path

import streamlit as st

from hado.app.utils import apply_page_width, DEFAULT_SURFACE_MESHES, DEFAULT_OPEN_MESHES
from hado.app.pages.mesh_gui import (
    DEFAULT_EDGE_THICKNESS_NM,
    DEFAULT_N_PER_EDGE,
    DEFAULT_OPEN_EDGE_THICKNESS_NM,
    DEFAULT_OPEN_N_PER_EDGE,
    EMPTY_GEOMETRY_STATE,
    render_mesh_designer,
)
from hado.core.automation.pipeline.manager import HadoManager
from hado.core.utils import Geometry



def show_input_page():
    apply_page_width()
    st.header("Mesh Designer")
    if st.session_state.get("designer_mode", False):
        render_mesh_designer()
    else:
        _render_upload_section()


def _render_upload_section():
    """Display the mesh upload / selection entry point."""
    st.subheader("Upload mesh (or start from scratch) to start!")

    with st.container():
        mesh_file = st.file_uploader(
            "Upload Mesh File (.ply, .obj, .lm, or .hado save states)",
            type=["ply", "obj", "lm", "json", "hado"],
            help="Please upload a valid .ply, obj, .lm, or hado (json)",
        )

        cc1, cc2 = st.columns(2)
        with cc1:
            default_mesh1 = st.selectbox(
                "Or select a standard **closed surface** mesh",
                options=["None"] + list(DEFAULT_SURFACE_MESHES.keys()),
                index=0,
            )
        with cc2:
            default_mesh2 = st.selectbox(
                "Or select a default **open wireframe** model",
                options=["None"] + list(DEFAULT_OPEN_MESHES.keys()),
                index=0,
            )

        if mesh_file:
            try:
                if _handle_file_upload(mesh_file):
                    st.rerun()
            except Exception as e:
                st.error(e)

        if default_mesh1 != "None" or default_mesh2 != "None":
            try:
                if default_mesh1 != "None":
                    _handle_default_mesh_selection(DEFAULT_SURFACE_MESHES[default_mesh1])
                else:
                    _handle_default_mesh_selection(DEFAULT_OPEN_MESHES[default_mesh2])
                st.rerun()
            except Exception as e:
                st.error(e)

        if st.button("Start from scratch"):
            _handle_start_from_scratch()
            st.rerun()


def _check_verts_n_edges(vertices, edges):
    vertices = {i: 0 for i in range(len(vertices))}
    for edge in edges:
        i, j = edge
        vertices[i] += 1
        vertices[j] += 1

    return not any(i == 1 for i in vertices.values())


def _first_width_value(geometry: dict, key: str, default):
    values = geometry.get(key)
    if isinstance(values, (list, tuple)) and len(values) > 0:
        return values[0]
    return default


def _activate_designer_state(
    geometry: dict,
    basename: str,
    validated_manager: dict | None = None,
    run_verified: bool = False,
    edge_width_mode: str | None = None,
):
    if edge_width_mode is None:
        edge_width_mode = "thickness" if "edge_thickness_nm" in geometry else "helices"

    st.session_state["geometry"] = geometry
    st.session_state["basename"] = basename
    st.session_state["validated_manager"] = validated_manager
    st.session_state["run_verified"] = run_verified
    st.session_state["designer_mode"] = True
    st.session_state["error_msgs"] = None
    st.session_state["scale"] = 1.0
    st.session_state["reset_scale"] = True
    st.session_state["edge_width_mode"] = edge_width_mode
    st.session_state["edge_thickness_nm_input"] = float(
        _first_width_value(geometry, "edge_thickness_nm", DEFAULT_EDGE_THICKNESS_NM)
    )
    st.session_state["n_per_edge_input"] = int(
        _first_width_value(geometry, "n_per_edge", DEFAULT_N_PER_EDGE)
    )


@st.cache_data
def _load_default_mesh_state(default_mesh_path: str):
    manager = HadoManager.load(default_mesh_path)
    model = manager.nucleotide_level_model
    return {
        "geometry": manager.geometry.to_dict(),
        "basename": manager.design_name,
        "validated_manager": manager.to_json(),
        "run_verified": model is not None,
    }


def _handle_file_upload(mesh_file) -> bool:
    """Process an uploaded mesh file or a saved HADO state."""
    path = Path(mesh_file.name)
    suffix = path.suffix.lower()
    basename = path.stem.lower()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(mesh_file.getvalue())
            tmp_path = Path(tmp_file.name)

        if suffix in {".json", ".hado"}:
            manager, nucleotide_model = HadoManager.load_ui(tmp_path)
            _activate_designer_state(
                geometry=manager.geometry.to_dict(),
                basename=basename,
                validated_manager=manager.to_json(),
                run_verified=nucleotide_model is not None,
            )
            return True

        geometry = Geometry.read_in_mesh(tmp_path)
        is_closed_surface = _check_verts_n_edges(geometry.vertices, geometry.edges)
        default_edge_thickness = DEFAULT_EDGE_THICKNESS_NM if is_closed_surface else DEFAULT_OPEN_EDGE_THICKNESS_NM
        default_n_per_edge = DEFAULT_N_PER_EDGE if is_closed_surface else DEFAULT_OPEN_N_PER_EDGE
        manager = HadoManager(basename, geometry)
        _activate_designer_state(
            geometry={
                "vertices": geometry.vertices,
                "edges": geometry.edges,
                "edge_thickness_nm": [default_edge_thickness] * len(geometry.edges),
            },
            basename=basename,
            edge_width_mode="helices",
            validated_manager=manager.to_json(),
            run_verified=False,
        )
        st.session_state["n_per_edge_input"] = default_n_per_edge
        return True

    except Exception as e:
        if suffix in {".json", ".hado"}:
            st.error(f"ERROR: Invalid JSON file data due to {e}")
        else:
            st.error(f"ERROR: Invalid mesh input data (ply, obj, or lm) due to {e}")
        return False
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


def _handle_default_mesh_selection(default_mesh_path: Path):
    """Process a default mesh selection and update session state."""
    state = _load_default_mesh_state(str(default_mesh_path))
    _activate_designer_state(
        geometry=state["geometry"],
        basename=state["basename"],
        validated_manager=state["validated_manager"],
        run_verified=state["run_verified"],
    )


def _handle_start_from_scratch():
    _activate_designer_state(dict(EMPTY_GEOMETRY_STATE), "")


if __name__ == "__main__":
    show_input_page()
