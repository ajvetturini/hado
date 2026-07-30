import streamlit as st
from pathlib import Path

def initialize_session_state():
    init_keys = [
        "initialized_state",
        "designer_mode",
        "validated_manager",
        "scale",
        "reset_scale",
        "error_msgs",
        "run_verified",
    ]
    init_values = [
        True,
        False,
        None,
        1.00,
        True,
        None,
        False,
    ]
    for k, v in zip(init_keys, init_values):
        if k not in st.session_state:
            st.session_state[k] = v

def reset_session():
    """Clears the entire session state and re-initializes."""
    for k in st.session_state.keys():
        del st.session_state[k]
    initialize_session_state()

def apply_page_width():
    # I noticed some weird clipping with a full 100% width for my plotly charts
    # 95% seems good
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 95vw;
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


APP_DIR = Path(__file__).parent
INPUT_DIR = APP_DIR / "pages" / "default_inputs"

# Below meshes from ATHENA: https://github.com/lcbb/athena/tree/master/sample_inputs
DEFAULT_SURFACE_MESHES = {
    "Cube": INPUT_DIR / "cube.hado",
    "Tetrahedron": INPUT_DIR / "tetrahedron.hado",
    "Octahedron": INPUT_DIR / "octahedron.hado",
    "Triangle": INPUT_DIR / "triangle.hado",
    "Square": INPUT_DIR / "square.hado",
    "Triangle Mesh": INPUT_DIR / "triangle_mesh.hado",
    "Square Mesh": INPUT_DIR / "square_mesh.hado",
    "Hexagonal Prism": INPUT_DIR / "hexagonal_prism.hado",
    "Octet Truss": INPUT_DIR / "octet_truss.hado",
    "Icosahedron": INPUT_DIR / "icosahedron.hado",
    "Truncated Icosahedron": INPUT_DIR / "truncated_icosahedron.hado",
    "Truncated Tetrahedron": INPUT_DIR / "truncated_tetrahedron.hado",
    "Pentagonal Bipyramid": INPUT_DIR / "pentagonal_bipyramid.hado",
    "Triakis Octahedron": INPUT_DIR / "triakis_octahedron.hado",
    "Nested Cube": INPUT_DIR / "nested_cube.hado",
    "Annulus Mesh": INPUT_DIR / "annulus_mesh.hado",
    "Star Mesh": INPUT_DIR / "star_mesh.hado",
    "Asymmetric Triangle": INPUT_DIR / "asymmetric_triangle.hado",
    "Letter O": INPUT_DIR / "letter_o.hado",
}

# Below were hand-designed
DEFAULT_OPEN_MESHES = {
    "Tetrapod": INPUT_DIR / "tetrapod.hado",
    "Arrow": INPUT_DIR / "arrow.hado",
    "Y Junction": INPUT_DIR / "y_junction.hado",
    "One Way": INPUT_DIR / "one_way.hado",
    "Body Center Cubic": INPUT_DIR / "bcc.hado",
    "Plus Sign": INPUT_DIR / "plus_sign.hado",
    "Letter A": INPUT_DIR / "letter_a.hado",
    "Letter T": INPUT_DIR / "letter_t.hado",
    "Letter G": INPUT_DIR / "letter_g.hado",
    "Letter C": INPUT_DIR / "letter_c.hado",
    "Letter X": INPUT_DIR / "letter_x.hado",
    "Letter V": INPUT_DIR / "letter_v.hado",
    "Letter F": INPUT_DIR / "letter_f.hado",
    "Letter N": INPUT_DIR / "letter_n.hado",
    "Letter M": INPUT_DIR / "letter_m.hado",
    "Letter U": INPUT_DIR / "letter_u.hado",
}
