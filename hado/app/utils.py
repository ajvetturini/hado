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
    "Cube": INPUT_DIR / "cube.json",
    "Tetrahedron": INPUT_DIR / "tetrahedron.json",
    "Octahedron": INPUT_DIR / "octahedron.json",
    "Triangle": INPUT_DIR / "triangle.json",
    "Square": INPUT_DIR / "square.json",
    "Triangle Mesh": INPUT_DIR / "triangle_mesh.json",
    "Square Mesh": INPUT_DIR / "square_mesh.json",
    "Hexagonal Prism": INPUT_DIR / "hexagonal_prism.json",
    "Octet Truss": INPUT_DIR / "octet_truss.json",
    "Icosahedron": INPUT_DIR / "icosahedron.json",
    "Truncated Icosahedron": INPUT_DIR / "truncated_icosahedron.json",
    "Truncated Tetrahedron": INPUT_DIR / "truncated_tetrahedron.json",
    "Pentagonal Bipyramid": INPUT_DIR / "pentagonal_bipyramid.json",
    "Triakis Octahedron": INPUT_DIR / "triakis_octahedron.json",
    "Nested Cube": INPUT_DIR / "nested_cube.json",
    "Annulus Mesh": INPUT_DIR / "annulus_mesh.json",
    "Star Mesh": INPUT_DIR / "star_mesh.json",
    "Asymmetric Triangle": INPUT_DIR / "asymmetric_triangle.json",
    "Letter O": INPUT_DIR / "letter_o.json",
}

# Below were hand-designed
DEFAULT_OPEN_MESHES = {
    "Tetrapod": INPUT_DIR / "tetrapod.json",
    "Arrow": INPUT_DIR / "arrow.json",
    "Y Junction": INPUT_DIR / "y_junction.json",
    "One Way": INPUT_DIR / "one_way.json",
    "Body Center Cubic": INPUT_DIR / "bcc.json",
    "Plus Sign": INPUT_DIR / "plus_sign.json",
    "Letter A": INPUT_DIR / "letter_a.json",
    "Letter T": INPUT_DIR / "letter_t.json",
    "Letter G": INPUT_DIR / "letter_g.json",
    "Letter C": INPUT_DIR / "letter_c.json",
    "Letter X": INPUT_DIR / "letter_x.json",
    "Letter V": INPUT_DIR / "letter_v.json",
    "Letter F": INPUT_DIR / "letter_f.json",
    "Letter N": INPUT_DIR / "letter_n.json",
    "Letter M": INPUT_DIR / "letter_m.json",
    "Letter U": INPUT_DIR / "letter_u.json",
}