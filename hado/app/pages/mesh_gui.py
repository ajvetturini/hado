import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from st_oxview import oxview_from_json

import re
import json
import io
import csv
import zipfile
from collections import Counter, deque
from math import isfinite
from numbers import Real

from hado.app.utils import reset_session
from hado.app.scaffold_geometry import get_scaffold_sequence
from hado.core.utils import (
    Geometry,
    MAX_NODES_FOR_CROSS_SECTION_SEARCH,
    MIN_ALLOWABLE_EDGE_LENGTH_NM,
    ScaffoldArgs,
    StapleArgs,
)
from hado.core.automation.pipeline.manager import HadoManager

DEFAULT_SCAFFOLD_IDS = ("m13", "p7308", "p7560", "p7704", "p8064", "p8100", "p8634")
DEFAULT_N_PER_EDGE = 6
DEFAULT_OPEN_N_PER_EDGE = 18
DEFAULT_EDGE_THICKNESS_NM = 8.75
DEFAULT_OPEN_EDGE_THICKNESS_NM = 19.45
EDGE_WIDTH_MODE_LABELS = {
    "thickness": "Edge thickness (nm)",
    "helices": "N helices",
}
EMPTY_GEOMETRY_STATE = {"vertices": [], "edges": [], "edge_thickness_nm": []}


def render_mesh_designer():
    _ensure_designer_session_state()
    vals = _load_values()
    validated_state = st.session_state.get("validated_manager")
    validated_manager = _load_manager_state(validated_state)
    validated_model = validated_manager.nucleotide_level_model
    input_disabled = st.session_state["run_verified"]

    c1, c2 = st.columns([1.2, 2], gap="medium")
    with c1:
        form_values = _render_input_panel(vals, input_disabled)
    with c2:
        display_dict = _render_display_options()
        _render_preview_panel(form_values, display_dict, input_disabled, validated_manager, validated_model)

    _render_run_section(validated_manager)

    if st.session_state["run_verified"] and validated_manager and validated_model:
        _render_nucleotide_model_section(validated_manager, validated_model, validated_state)


def _ensure_designer_session_state():
    st.session_state.setdefault("geometry", dict(EMPTY_GEOMETRY_STATE))
    st.session_state.setdefault("basename", "")
    st.session_state.setdefault("validated_manager", None)
    st.session_state.setdefault("error_msgs", None)
    st.session_state.setdefault("run_verified", False)
    st.session_state.setdefault("edge_width_mode", "thickness")
    st.session_state.setdefault("edge_thickness_nm_input", _first_geometry_width_value(
        st.session_state["geometry"], "edge_thickness_nm", DEFAULT_EDGE_THICKNESS_NM
    ))
    st.session_state.setdefault("n_per_edge_input", int(_first_geometry_width_value(
        st.session_state["geometry"], "n_per_edge", DEFAULT_N_PER_EDGE
    )))

    if st.session_state.get("reset_scale", False):
        st.session_state["scale"] = 1.0
        st.session_state["reset_scale"] = False
    else:
        st.session_state.setdefault("scale", 1.0)


def _first_geometry_width_value(geometry_state: dict, key: str, default: float) -> float:
    values = geometry_state.get(key) if geometry_state else None
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)) and len(values) > 0:
        return values[0]
    if isinstance(values, Real) and not isinstance(values, bool):
        return values
    return default


def _render_edge_width_controls(input_disabled: bool) -> str:
    current_mode = st.session_state.get("edge_width_mode", "helices")
    if current_mode not in EDGE_WIDTH_MODE_LABELS:
        current_mode = "helices"

    mode = st.radio(
        "**Edge width input**",
        options=list(EDGE_WIDTH_MODE_LABELS),
        format_func=lambda option: EDGE_WIDTH_MODE_LABELS[option],
        index=list(EDGE_WIDTH_MODE_LABELS).index(current_mode),
        horizontal=True,
        disabled=input_disabled,
        key="edge_width_mode",
    )

    if mode == "thickness":
        st.number_input(
            "**Edge thickness (nm)**",
            min_value=5.0,
            max_value=35.0,
            step=0.5,
            format="%.2f",
            key="edge_thickness_nm_input",
            disabled=input_disabled,
        )
    else:
        st.number_input(
            "**N per edge**",
            min_value=2,
            max_value=MAX_NODES_FOR_CROSS_SECTION_SEARCH,
            step=1,
            key="n_per_edge_input",
            disabled=input_disabled,
        )

    return mode


def _render_input_panel(vals, input_disabled: bool) -> dict:
    with st.container(border=True):
        if input_disabled:
            st.write(
                "**NOTE** You must clear the nucleotide-model if you wish to re-edit the input values "
                "(this option is underneath `Export` towards bottom of web-page)"
            )
        name = st.text_input("**Design name**", value=vals["name"], key="name", disabled=input_disabled)

        t1, t2, t3 = st.tabs(["Geometry", "Scaffold", "Staples"])

        with t1:
            st.subheader("Vertex Coordinates")
            v_df = pd.DataFrame(st.session_state["geometry"]["vertices"], columns=["X", "Y", "Z"])
            edited_v = st.data_editor(
                v_df,
                num_rows="dynamic",
                width="stretch",
                hide_index=False,
                key="v_edit",
                disabled=input_disabled,
            )

            st.subheader("Edge Connections")
            e_df = pd.DataFrame(st.session_state["geometry"]["edges"], columns=["V1", "V2"])
            edited_e = st.data_editor(
                e_df,
                num_rows="dynamic",
                width="stretch",
                hide_index=False,
                key="e_edit",
                disabled=input_disabled,
            )

            scale = st.number_input(
                "**Scale design** (press Update to apply)",
                min_value=0.1,
                max_value=100.0,
                key="scale",
                disabled=input_disabled,
            )

            edge_width_mode = _render_edge_width_controls(input_disabled)
            n_per_edge = int(st.session_state.get("n_per_edge_input", DEFAULT_N_PER_EDGE))
            edge_thickness_nm = float(st.session_state.get("edge_thickness_nm_input", DEFAULT_EDGE_THICKNESS_NM))

        with t2:
            st.markdown('Remember to press **Update** after changing any parameters below!')
            scaffold_seq = st.text_input(
                "**Scaffold Sequence**",
                value=vals["seq"],
                key="scaf_seq",
                help=(
                    "Enter a standard scaffold name (m13, p7308, p7560, p7704, p8064, p8100, or p8634) "
                    "or paste in a custom DNA scaffold sequence."
                ),
                disabled=input_disabled,
            )
            fill_modes = ["overfill", "underfill"]
            over_or_underfill = st.selectbox(
                "**Odd N Fill Mode**",
                fill_modes,
                index=fill_modes.index(vals["over_or_underfill"]),
                key="over_or_underfill",
                disabled=input_disabled,
            )

        with t3:
            st.markdown('Remember to press **Update** after changing any parameters below!')
            st.write(
                "**NOTE**: It is recommended to leave these as default unless you are explicitly testing these "
                "parameters. Setting to unintended values can lead to wacky results / failures."
            )
            col_a, col_b = st.columns(2)

            with col_a:
                min_length = st.number_input(
                    "Min Length",
                    value=vals["min_len"],
                    min_value=16,
                    max_value=100,
                    key="min_len",
                    disabled=input_disabled,
                )
                target_length = st.number_input(
                    "Target Length",
                    value=vals["target_len"],
                    min_value=16,
                    max_value=100,
                    key="target_len",
                    disabled=input_disabled,
                )
                target_miter = st.number_input(
                    "Target Miter Distance",
                    min_value=0.50,
                    max_value=7.50,
                    value=vals["miter"],
                    key="miter",
                    disabled=input_disabled,
                )
                default_blunt_end_length = st.number_input(
                    "Default blunt end length",
                    min_value=0,
                    max_value=10,
                    value=vals["bluntend"],
                    key="bluntend",
                    disabled=input_disabled,
                )
                min_run_post_bundle_connection = st.number_input(
                    "Minimum run distance post-binding",
                    value=vals["min_run_post_bundle_connection"],
                    min_value=1,
                    max_value=10,
                    step=1,
                    key="min_run_post_bundle_connection",
                    disabled=input_disabled,
                )
                min_run_post_xover = st.number_input(
                    "Minimum run distance post staple crossover",
                    value=vals["min_run_post_xover"],
                    min_value=1,
                    max_value=10,
                    step=1,
                    key="min_run_post",
                    disabled=input_disabled,
                )
                unpaired_options = ["T", "A", "C", "G"]
                unpaired_sequence = st.selectbox(
                    "Unpaired ssDNA sequence",
                    unpaired_options,
                    index=unpaired_options.index(vals["unpaired_sequence"]),
                    key="unpaired_seq",
                    disabled=input_disabled,
                )

            with col_b:
                max_length = st.number_input(
                    "Max Length",
                    value=vals["max_length"],
                    min_value=42,
                    max_value=100,
                    key="max_len",
                    disabled=input_disabled,
                )
                random_seed = st.number_input("Random Seed", value=vals["seed"], key="seed", disabled=input_disabled)
                polyt_spacer = st.number_input(
                    "Spacer Relaxation Distance",
                    value=vals["polyt_relaxation"],
                    min_value=0.10,
                    max_value=1.00,
                    step=0.01,
                    disabled=input_disabled,
                )
                max_spacer = st.number_input(
                    "Max ssDNA spacer overhang length",
                    value=vals["max_spacer"],
                    min_value=0,
                    max_value=10,
                    step=1,
                    disabled=input_disabled,
                )
                min_dist_between_xovers = st.number_input(
                    "Minimum distance between crossovers",
                    value=vals["min_dist_between_xovers"],
                    min_value=1,
                    max_value=10,
                    step=1,
                    key="min_run_between",
                    disabled=input_disabled,
                )
                flush_distance = st.number_input(
                    "Number of spacers to make flush to",
                    value=vals["flush_distance"],
                    min_value=1,
                    max_value=10,
                    step=1,
                    key="flush_dist",
                    disabled=input_disabled,
                )
                only_add = st.checkbox(
                    "Only add staple crossovers?",
                    value=vals["only_add"],
                    help=(
                        "If checked, the autostaple algorithm will not perform auto-breaking, leaving very long staples."
                    ),
                    disabled=input_disabled,
                )

    return {
        "name": name,
        "vertices_df": edited_v,
        "edges_df": edited_e,
        "scale": scale,
        "edge_width_mode": edge_width_mode,
        "n_per_edge": int(n_per_edge),
        "edge_thickness_nm": float(edge_thickness_nm),
        "scaffold_seq": scaffold_seq,
        "over_or_underfill": over_or_underfill,
        "min_length": int(min_length),
        "target_length": int(target_length),
        "target_miter": float(target_miter),
        "default_blunt_end_length": int(default_blunt_end_length),
        "min_run_post_bundle_connection": int(min_run_post_bundle_connection),
        "min_run_post_xover": int(min_run_post_xover),
        "unpaired_sequence": unpaired_sequence,
        "max_length": int(max_length),
        "random_seed": int(random_seed),
        "polyt_spacer": float(polyt_spacer),
        "max_spacer": int(max_spacer),
        "min_dist_between_xovers": int(min_dist_between_xovers),
        "flush_distance": int(flush_distance),
        "only_add": only_add,
    }


def _render_display_options() -> dict:
    display_dict = {}
    with st.expander("**Display Options**", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_dict["node_color"] = st.color_picker("Vertex Color", value="#ab7ce0", key="node_color")
        with col2:
            display_dict["edge_color"] = st.color_picker("Edge Color", value="#000000", key="edge_color")
        with col3:
            display_dict["edge_label_color"] = st.color_picker(
                "Edge Label Color", value="#ff7f0e", key="edge_label_color"
            )
        with col4:
            display_dict["mesh_color"] = st.color_picker("Mesh color", value="#ab7ce0", key="mesh_color")

        display_dict["edge_width"] = st.slider("Edge Width", min_value=2, max_value=16, value=4, key="edge_width")
        display_dict["font_size"] = st.slider("Font Size", min_value=10, max_value=30, value=20, key="font_size")
        display_dict["marker_size"] = st.slider(
            "Marker Size", min_value=10, max_value=20, value=14, key="marker_size"
        )
        display_dict["mesh_scale"] = st.slider("Mesh Scale", min_value=1, max_value=100, value=10, key="mesh_scale")

        col1, col2, col3 = st.columns(3)
        with col1:
            display_dict["show_vertex_labels"] = st.checkbox(
                "Show vertex labels?", value=True, key="show_vertex_labels", help="Wireframe mode only"
            )
        with col2:
            display_dict["show_edge_labels"] = st.checkbox(
                "Show edge labels?", value=True, key="show_edge_labels", help="Wireframe mode only"
            )
        with col3:
            display_dict["show_background_grid"] = st.checkbox(
                "Show background grid?", value=True, key="show_background_grid"
            )

        col4, col5, col6 = st.columns(3)
        with col4:
            display_dict["use_mesh"] = st.checkbox("Show Mesh?", value=False, key="show_mesh")
        with col5:
            display_dict["show_lengths"] = st.checkbox(
                "Show Edge Lengths?", value=True, key="show_edge_lengths", help="Wireframe mode only"
            )
        with col6:
            display_dict["show_in_nts"] = st.checkbox(
                "Show in # of nucleotides?",
                value=True,
                disabled=not display_dict["show_lengths"],
                key="show_in_nts",
                help="Wireframe mode only",
            )
    return display_dict


def _render_preview_panel(form_values, display_dict, input_disabled: bool, validated_manager, validated_model):
    with st.container(border=True):
        cc1, cc2 = st.columns([1, 1.5])

        with cc1:
            if st.button("Update", width="stretch", type="primary", disabled=input_disabled):
                _handle_update(form_values)

        with cc2:
            st.info(f"**Estimated Nucleotides:** {_estimate_nt_message(validated_manager, validated_model)}")

        if isinstance(st.session_state.get("error_msgs"), list):
            st.write("ERROR in previous update attempt:")
            for msg in st.session_state["error_msgs"]:
                st.error(msg)

    with st.container(border=True):
        fig, error = _get_preview_figure(display_dict)
        if error:
            st.warning(error)
        elif fig is not None:
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Adjust geometry and click 'Update' to view.")


def _handle_update(form_values):
    st.session_state["reset_scale"] = True
    try:
        vertices = _normalize_vertices(form_values["vertices_df"].values.tolist())
        edges = _normalize_edges(form_values["edges_df"].values.tolist())
    except ValueError as exc:
        st.session_state["validated_manager"] = None
        st.session_state["run_verified"] = False
        st.session_state["error_msgs"] = [str(exc)]
        st.rerun()

    if form_values["scale"] != 1.0:
        verts = np.array(vertices, dtype=float)
        centroid = np.mean(verts, axis=0)
        new_vertices = (verts - centroid) * form_values["scale"] + centroid
        vertices = new_vertices.tolist()

    geometry_state = _geometry_state_from_inputs(
        vertices,
        edges,
        form_values["edge_width_mode"],
        form_values["n_per_edge"],
        form_values["edge_thickness_nm"],
    )
    is_valid, error_msgs = _validate(vertices, edges, form_values["scaffold_seq"], form_values["name"])
    st.session_state["geometry"] = geometry_state
    st.session_state["run_verified"] = False

    if is_valid:
        try:
            manager = _build_manager(form_values, geometry_state)
            st.session_state["validated_manager"] = manager.to_json()
            st.session_state["error_msgs"] = None
        except Exception as exc:
            st.session_state["validated_manager"] = None
            st.session_state["error_msgs"] = [str(exc)]
    else:
        #st.session_state["validated_manager"] = None
        st.session_state["error_msgs"] = error_msgs

    st.rerun()


def _build_manager(form_values, geometry_state: dict) -> HadoManager:
    geometry = Geometry(**geometry_state)
    scaffold_args = ScaffoldArgs(
        scaffold_sequence=form_values["scaffold_seq"],
        overfill_or_underfill=form_values["over_or_underfill"],
    )
    staple_args = StapleArgs(
        min_length_after_break=form_values["min_length"],
        max_length_after_break=form_values["max_length"],
        only_add=form_values["only_add"],
        random_seed=form_values["random_seed"],
        target_miter_distance=form_values["target_miter"],
        polyt_bulge_dist=form_values["polyt_spacer"],
        unpaired_sequence=form_values["unpaired_sequence"],
        default_blunt_end_length=form_values["default_blunt_end_length"],
        max_staple_spacer_length=form_values["max_spacer"],
        min_run_post_bundle_connection=form_values["min_run_post_bundle_connection"],
        min_dist_between_xovers=form_values["min_dist_between_xovers"],
        min_run_post_xover=form_values["min_run_post_xover"],
        target_staple_length=form_values["target_length"],
        make_flush=True,
        flush_distance=form_values["flush_distance"],
    )
    return HadoManager(form_values["name"], geometry, scaffold_args, staple_args)


def _render_run_section(validated_manager: HadoManager | None):
    st.divider()
    st.header("Run Automation")
    i, j, k = st.columns([1, 1, 1])

    with i:
        _, center, _ = st.columns([1, 1, 1])
        with center:
            run_button = st.button(
                "Run!",
                disabled=st.session_state["run_verified"],
                help=(
                    "You need to clear the existing nucleotide-level model before running again."
                    if st.session_state["run_verified"]
                    else "Run the design automation with the validated inputs."
                ),
            )

        if run_button:
            with st.spinner("Running..."):
                if validated_manager is None:
                    st.error("You must first select the `Update` button to validate the inputs.")
                    return

                try:
                    validated_manager.run()
                    st.session_state["validated_manager"] = validated_manager.to_json()
                    st.session_state["run_verified"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"{e}")

    with j:
        _, center, _ = st.columns([1, 2, 1])
        with center:
            download_payload = _get_hado_download_payload()
            disabled = validated_manager is None
            json_string = json.dumps(download_payload, indent=2) if download_payload else ""
            fname = f"{validated_manager.design_name}.hado" if validated_manager else "empty.hado"
            st.download_button(
                label="Download .hado",
                file_name=fname,
                mime="application/json",
                data=json_string,
                disabled=disabled,
            )

    with k:
        _, center, _ = st.columns([1, 2, 1])
        with center:
            if st.button("Select new mesh", help="Removes all edge / vertex input and returns to the mesh selector page."):
                reset_session()
                st.rerun()


def _render_nucleotide_model_section(manager: HadoManager, model, validated_state: dict):
    st.divider()
    st.header("Nucleotide-level Model")
    _render_run_input_summary(manager, model)
    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.subheader("Scaffold + Staple Statistics")
            worked, scaffold_nt_count, lengths = _extract_sequence_metrics(manager, model)
            if not worked:
                st.warning(
                    f"NOTE: The number of scaffold nucleotides (={scaffold_nt_count}) required is too large for "
                    "sequencing to take place with the input scaffold. Staple statistics still shown, but the "
                    "sequences file can not be exported."
                )
            st.table(_create_staple_stats_table(lengths, scaffold_nt_count))

            st.divider()
            st.subheader("Export")
            oxview_json = _render_export_buttons(manager, model, worked, validated_state)

            st.divider()
            if st.button(
                "Clear nucleotide model",
                width="stretch",
                help=(
                    "Delete the current nucleotide-level model so you can re-edit the mesh / parameters without "
                    "starting from scratch."
                ),
            ):
                st.session_state["validated_manager"] = manager.to_json()
                st.session_state["run_verified"] = False
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center;'>oxView Preview</h4>", unsafe_allow_html=True)
            st.markdown(
                "The oxView preview is enabled by st_oxview Python package "
                "[here](https://github.com/lucandia/st_oxview).",
                unsafe_allow_html=True,
            )
            try:
                oxview_from_json(scene_data=oxview_json, width=1000, height=500, key=None)
            except Exception as e:
                st.error(f"Error rendering oxView json file: {e}")


def _summary_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "Value" in df:
        df["Value"] = df["Value"].astype(str)
    return df


def _format_summary_values(values, precision: int | None = None) -> str:
    if values is None:
        return "None"
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values]
    if len(values) == 0:
        return "None"

    formatted = []
    for value in values:
        if precision is None:
            formatted.append(str(int(value)))
        else:
            formatted.append(f"{float(value):.{precision}f}")

    counts = Counter(formatted)
    parts = [f"{value} x{count}" if count > 1 else value for value, count in sorted(counts.items())]
    if len(parts) > 6:
        return ", ".join(parts[:6]) + f", ... ({len(parts)} distinct)"
    return ", ".join(parts)


def _summarize_cross_sections(model) -> str:
    xsect_definitions = getattr(model, "edge_xsect_definitions", None)
    if not xsect_definitions:
        return "Unavailable"

    labels = []
    for xsect in xsect_definitions.values():
        labels.append(f"M={int(xsect['M'])}, N={int(xsect['N'])}")

    counts = Counter(labels)
    return ", ".join(
        f"{label} x{count}" if count > 1 else label
        for label, count in sorted(counts.items())
    )


def _scaffold_summary_label(scaffold_sequence: str) -> str:
    label = _process_input_seq(scaffold_sequence)
    if label in DEFAULT_SCAFFOLD_IDS:
        return label
    return f"Custom sequence ({len(scaffold_sequence)} nt)"


def _render_run_input_summary(manager: HadoManager, model) -> None:
    with st.expander("Run input summary", expanded=False):
        geometry = manager.geometry
        if geometry.edge_thickness_nm is not None:
            width_rows = [
                {"Setting": "Input mode", "Value": "Edge thickness (nm)"},
                {"Setting": "Requested edge thickness (nm)", "Value": _format_summary_values(geometry.edge_thickness_nm, 2)},
                {
                    "Setting": "Selected honeycomb ring diameter (nm)",
                    "Value": _format_summary_values(geometry.edge_thickness_actual_nm, 2),
                },
                {"Setting": "Resolved N helices per edge", "Value": _format_summary_values(geometry.n_per_edge)},
                {"Setting": "Cross-section M/N", "Value": _summarize_cross_sections(model)},
            ]
        else:
            width_rows = [
                {"Setting": "Input mode", "Value": "N helices per edge"},
                {"Setting": "N helices per edge", "Value": _format_summary_values(geometry.n_per_edge)},
                {"Setting": "Cross-section M/N", "Value": _summarize_cross_sections(model)},
            ]

        scaffold_rows = [
            {"Setting": "Scaffold sequence", "Value": _scaffold_summary_label(manager.scaffold_args.scaffold_sequence)},
            {"Setting": "Scaffold length", "Value": f"{len(manager.scaffold_args.scaffold_sequence)} nt"},
            {"Setting": "Odd N fill mode", "Value": manager.scaffold_args.overfill_or_underfill},
        ]

        staple = manager.staple_args
        staple_rows = [
            {"Setting": "Min staple length", "Value": staple.min_length_after_break},
            {"Setting": "Target staple length", "Value": staple.target_staple_length},
            {"Setting": "Max staple length", "Value": staple.max_length_after_break},
            {"Setting": "Random seed", "Value": staple.random_seed},
            {"Setting": "Target miter distance", "Value": f"{staple.target_miter_distance:.2f}"},
            {"Setting": "Default blunt-end length", "Value": staple.default_blunt_end_length},
            {"Setting": "Minimum run post-binding", "Value": staple.min_run_post_bundle_connection},
            {"Setting": "Minimum run post-staple crossover", "Value": staple.min_run_post_xover},
            {"Setting": "Minimum distance between crossovers", "Value": staple.min_dist_between_xovers},
            {"Setting": "Spacer relaxation distance", "Value": f"{staple.polyt_bulge_dist:.2f}"},
            {"Setting": "Max ssDNA spacer overhang length", "Value": staple.max_staple_spacer_length},
            {"Setting": "Flush distance", "Value": staple.flush_distance},
            {"Setting": "Unpaired sequence", "Value": staple.unpaired_sequence},
            {"Setting": "Only add staple crossovers", "Value": str(staple.only_add)},
        ]

        t1, t2, t3 = st.tabs(["Cross-section", "Scaffold", "Staples"])
        with t1:
            st.table(_summary_dataframe(width_rows))
        with t2:
            st.table(_summary_dataframe(scaffold_rows))
        with t3:
            st.table(_summary_dataframe(staple_rows))


def _geometry_state_from_inputs(
    vertices: list,
    edges: list,
    edge_width_mode: str,
    n_per_edge: int | list | None = None,
    edge_thickness_nm: float | list | None = None,
) -> dict:
    if edge_width_mode == "thickness":
        if isinstance(edge_thickness_nm, Real) and not isinstance(edge_thickness_nm, bool):
            values = [float(edge_thickness_nm)] * len(edges) if len(edges) > 0 else [float(edge_thickness_nm)]
        else:
            values = [float(i) for i in edge_thickness_nm]
        return {"vertices": vertices, "edges": edges, "edge_thickness_nm": values}

    if isinstance(n_per_edge, int):
        values = [n_per_edge] * len(edges) if len(edges) > 0 else [n_per_edge]
    else:
        values = [int(i) for i in n_per_edge]
    return {"vertices": vertices, "edges": edges, "n_per_edge": values}


def _load_manager_state(state: dict | None):
    if not state:
        return None, None
    return HadoManager.load(data=state)


def _estimate_nt_message(manager: HadoManager | None, model):
    if manager is None:
        return "No valid geometry defined yet."
    try:
        return manager.get_estimate_num_nts(model)
    except Exception as e:
        if isinstance(e, ValueError):
            return f"ERROR estimating: {e} (are your edges overly short?)."
        if isinstance(e, RuntimeError):
            return f"ERROR estimating: {e} (this may be a scaffold routing error!)."
        return f"ERROR estimating: {e}"


def _get_hado_download_payload() -> dict | None:
    validated_manager = st.session_state.get("validated_manager")
    if not validated_manager:
        return None

    payload = dict(validated_manager)
    return payload


@st.cache_data
def _build_preview_figure(geometry_dict: dict, display_dict: dict):
    geometry = Geometry(**geometry_dict)
    if display_dict["use_mesh"]:
        return _create_mesh_plot(geometry, display_dict)
    return _create_wireframe_plot(geometry, display_dict)


def _get_preview_figure(display_dict: dict):
    geometry_dict = st.session_state.get("geometry")
    if not geometry_dict or not geometry_dict.get("vertices") or not geometry_dict.get("edges"):
        return None, None

    try:
        return _build_preview_figure(geometry_dict, display_dict), None
    except Exception as exc:
        return None, f"Preview unavailable until geometry validates cleanly: {exc}"


def _normalize_vertices(vertices: list) -> list:
    cleaned = []
    for vertex in vertices:
        if len(vertex) != 3:
            raise ValueError("Each vertex must be defined by exactly 3 coordinates.")
        if not all(isfinite(float(coord)) for coord in vertex):
            # This is the final row (which might be all Nones) that simply get's ignored
            # raise ValueError("Vertex must be defined by 3 finite numeric values.")
            continue
        cleaned.append([float(coord) for coord in vertex])
    return cleaned


def _normalize_edges(edges: list) -> list:
    cleaned = []
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Each edge must be defined between 2 vertices (vi, vj).")
        coerced = []
        for idx in edge:
            try:
                if not float(idx).is_integer():
                    raise ValueError("Edge vertex indices must be integers.")
            except TypeError as e:
                raise  ValueError(f"Edge: {edge} is ill-defined")

            coerced.append(int(idx))
        cleaned.append(coerced)
    return cleaned


def _extract_sequence_metrics(manager: HadoManager, model) -> tuple[bool, int, list]:
    worked, sequences = manager.get_sequences(model)
    scaffold_nt_count = int(np.sum(model.get_scaffold_nucleotides()))
    if not worked:
        lengths = [len(i) for i in sequences]
        return worked, scaffold_nt_count, lengths

    lengths = [len(i[3]) for i in sequences[2:]]
    return worked, len(sequences[1][3]), lengths


def _create_staple_stats_table(lengths: list, scaffold_nt_count: int | None = None) -> pd.DataFrame:
    if lengths:
        avg = float(np.mean(lengths))
        std = float(np.std(lengths))
        min_len = int(np.min(lengths))
        max_len = int(np.max(lengths))
    else:
        avg = std = 0.0
        min_len = max_len = 0

    metrics = []
    values = []
    if scaffold_nt_count is not None:
        metrics.append("Number Scaffold Nucleotides Used")
        values.append(f"{int(scaffold_nt_count)}")

    metrics.extend([
        "Total Staple Count",
        "Average Staple Length",
        "Staple Standard Deviation",
        "Minimum Staple Length",
        "Maximum Staple Length",
    ])
    values.extend([
        f"{int(len(lengths))}",
        f"{avg:.3f}",
        f"{std:.3f}",
        f"{min_len}",
        f"{max_len}",
    ])

    table = pd.DataFrame({"Metric": metrics, "Value (bp)": values})
    table.index = [""] * len(table)
    return table


@st.cache_data(show_spinner="Preparing export files...")
def _compute_exports(cache_key, use_old, oxview_json, worked):
    manager = HadoManager.load(data=cache_key)
    cadnano_json = manager.get_cadnano_json()
    scadnano_str = manager.get_scadnano_json()
    scadnano_json = json.loads(scadnano_str)  # I have no idea why, but i have to loads then dumps later to work..?

    if oxview_json is None:
        oxview_json, (top, dat) = manager.get_oxview_json_and_oxdna_strings(
            use_old_top=use_old,
        )
    else:
        (top, dat) = manager.get_oxdna_strings(
            use_old_top=use_old,
        )

    if worked:
        _, sequences = manager.get_sequences()
    else:
        sequences = []

    cadnano_str = json.dumps(cadnano_json, indent=2)
    scadnano_str = json.dumps(scadnano_json, indent=2)
    oxview_str = json.dumps(oxview_json, indent=2)

    top_str = "\n".join(top)
    dat_str = "\n".join(dat)

    seq_buffer = io.StringIO()
    writer = csv.writer(seq_buffer)
    if sequences and len(sequences) > 0:
        writer.writerows(sequences)
    sequences_csv = seq_buffer.getvalue()

    return cadnano_str, scadnano_str, oxview_str, top_str, dat_str, sequences_csv, oxview_json

def _render_export_buttons(manager: HadoManager, model, worked, validated_state: dict):
    """Export design to standard CAD files with buttons."""
    use_old = st.checkbox("Use old oxDNA topology?", value=True)
    name = manager.design_name
    oxview_json = validated_state.get("custom_oxview") if validated_state else None
    data = manager.to_json()
    (cadnano_str, scadnano_str, oxview_str, top_str, dat_str, sequences_csv, oxview_json) = _compute_exports(
        data,
        use_old,
        oxview_json,
        worked,
    )

    t1, t2, t3, t4, t5 = st.tabs(["caDNAno", "scadnano", "sequences", "oxView", "oxDNA"])

    with t1:
        _, a, _ = st.columns([1, 2, 1])
        with a:
            st.download_button(
                "Download cadnano",
                data=cadnano_str,
                file_name=f"{name}_cadnano.json",
                mime="application/json",
                width="stretch",
            )

    with t2:
        _, a, _ = st.columns([1, 2, 1])
        with a:
            st.download_button(
                "Download scadnano",
                data=scadnano_str,
                file_name=f"{name}_scadnano.sc",
                mime="application/json",
                width="stretch",
            )

    with t3:
        _, a, _ = st.columns([1, 2, 1])
        with a:
            st.download_button(
                "Download sequences (CSV)",
                data=sequences_csv,
                file_name=f"{name}_sequences.csv",
                mime="text/csv",
                width="stretch",
                disabled=not worked,
            )

    with t4:
        _, a, _ = st.columns([1, 2, 1])
        with a:
            st.download_button(
                "Download oxView",
                data=oxview_str,
                file_name=f"{name}_oxview.oxview",
                mime="application/json",
                width="stretch",
            )

    with t5:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr(f"{name}.top", top_str)
            zf.writestr(f"{name}.dat", dat_str)
        zip_buffer.seek(0)
        _, a, _ = st.columns([1, 1, 1])
        with a:
            st.download_button(
                "Download oxDNA",
                data=zip_buffer,
                file_name=f"{name}_oxdna.zip",
                mime="application/zip",
                width="stretch",
            )

    return oxview_json

def _process_input_seq(seq):
    """Return the scaffold alias when the full sequence matches a bundled default scaffold."""
    for scaffold_id in DEFAULT_SCAFFOLD_IDS:
        if seq == get_scaffold_sequence(scaffold_id):
            return scaffold_id
    return seq

def _load_values():
    """Load form defaults from the validated manager state when available."""
    manager = _load_manager_state(st.session_state.get("validated_manager"))
    if manager is not None:
        return {
            "name": manager.design_name,
            "seq": _process_input_seq(manager.scaffold_args.scaffold_sequence),
            "over_or_underfill": manager.scaffold_args.overfill_or_underfill,
            "min_len": manager.staple_args.min_length_after_break,
            "target_len": manager.staple_args.target_staple_length,
            "miter": manager.staple_args.target_miter_distance,
            "bluntend": manager.staple_args.default_blunt_end_length,
            "min_run_post_bundle_connection": manager.staple_args.min_run_post_bundle_connection,
            "min_run_post_xover": manager.staple_args.min_run_post_xover,
            "unpaired_sequence": manager.staple_args.unpaired_sequence,
            "max_length": manager.staple_args.max_length_after_break,
            "seed": manager.staple_args.random_seed,
            "polyt_relaxation": manager.staple_args.polyt_bulge_dist,
            "max_spacer": manager.staple_args.max_staple_spacer_length,
            "min_dist_between_xovers": manager.staple_args.min_dist_between_xovers,
            "flush_distance": manager.staple_args.flush_distance,
            "only_add": manager.staple_args.only_add,
        }

    temp = StapleArgs()
    return {
        "name": st.session_state.get("basename", ""),
        "seq": "m13",
        "over_or_underfill": "overfill",
        "min_len": temp.min_length_after_break,
        "target_len": temp.target_staple_length,
        "miter": temp.target_miter_distance,
        "bluntend": temp.default_blunt_end_length,
        "min_run_post_bundle_connection": temp.min_run_post_bundle_connection,
        "min_run_post_xover": temp.min_run_post_xover,
        "unpaired_sequence": temp.unpaired_sequence,
        "max_length": temp.max_length_after_break,
        "seed": temp.random_seed,
        "polyt_relaxation": temp.polyt_bulge_dist,
        "max_spacer": temp.max_staple_spacer_length,
        "min_dist_between_xovers": temp.min_dist_between_xovers,
        "flush_distance": temp.flush_distance,
        "only_add": temp.only_add,
    }

def _validate(vertices, edges, scaffold_sequence, name):
    """Validate the editable mesh inputs before constructing a manager."""
    errors = []
    num_vertices = len(vertices)
    if num_vertices == 0:
        return False, ["No vertices defined."]

    if len(edges) == 0:
        return False, ["No edges defined."]

    for vertex in vertices:
        if len(vertex) != 3 or not all(
            isinstance(coord, Real) and not isinstance(coord, bool) and isfinite(float(coord))
            for coord in vertex
        ):
            return False, ["Vertex must be defined by 3 finite numeric values."]

    for edge in edges:
        if len(edge) != 2:
            return False, ["Edge must be defined between 2 vertices (vi, vj)."]
        if not all(
            isinstance(idx, Real) and not isinstance(idx, bool) and float(idx).is_integer()
            for idx in edge
        ):
            return False, ["Edge must be defined between 2 integer vertex indices (vi, vj)."]

    unique_v = set(tuple(v) for v in vertices)
    if len(unique_v) != num_vertices:
        errors.append("Duplicate vertices detected. All vertex coordinates must be unique.")

    valid_indices = True
    for i, edge in enumerate(edges):
        if any(idx >= num_vertices or idx < 0 for idx in edge):
            errors.append(f"Edge {i} {edge} references a vertex index that does not exist.")
            valid_indices = False

    if not valid_indices:
        return False, errors

    for i, edge in enumerate(edges):
        v1 = np.array(vertices[edge[0]])
        v2 = np.array(vertices[edge[1]])
        dist = np.linalg.norm(v1 - v2)
        if dist < MIN_ALLOWABLE_EDGE_LENGTH_NM:
            errors.append(f"Edge {i} too short ({dist:.2f} nm). Minimum required: {MIN_ALLOWABLE_EDGE_LENGTH_NM:.2f} nm.")

    used_indices = {idx for edge in edges for idx in edge}
    if len(used_indices) != num_vertices:
        unused_vertices = [i for i in range(num_vertices) if i not in used_indices]
        errors.append(f"Unused vertices not belonging to any edge found: {unused_vertices}")

    adj = {i: [] for i in range(num_vertices)}
    for v1, v2 in edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    visited = {0}
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    if len(visited) != num_vertices:
        errors.append("The geometry is disjoint. It must consist of a single connected component.")

    if scaffold_sequence.lower() not in DEFAULT_SCAFFOLD_IDS:
        valid_seq = set(scaffold_sequence.upper()).issubset({"A", "C", "G", "T"})
        if not valid_seq:
            errors.append("The provided scaffold sequence contains characters other than A C G or T.")

    if not isinstance(name, str):
        errors.append("The name is not a string.")
    elif not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        errors.append("Name contains invalid characters (only alphanumeric and _ allowed; no spaces)")

    return len(errors) == 0, errors

def _create_wireframe_plot(geometry: Geometry, display_dict, axial_rise: float = 0.34):
    fig = go.Figure()
    vertices, edges = geometry.vertices, geometry.edges
    edge_lengths = geometry.edge_lengths_nm
    edge_lengths_bp = np.array(edge_lengths) / axial_rise
    if len(vertices) > 0:
        vx, vy, vz = zip(*vertices)
        indices = list(range(len(vertices)))
        if display_dict['show_vertex_labels']:
            fig.add_trace(go.Scatter3d(
                x=vx, y=vy, z=vz,
                mode='markers+text',
                marker=dict(size=display_dict['marker_size'], color=display_dict['node_color']),
                name='Vertices',
                text=[f"{i}" for i in indices],
                textposition='middle center',
                textfont=dict(color='black', size=display_dict['font_size'], family='Arial', ),
                hovertemplate='Node #%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>',
                customdata=indices
            ))

    else:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None]))

    edge_x, edge_y, edge_z = [], [], []
    mx, my, mz = [], [], []
    valid_edges_for_plot = []

    if edges is not None and len(edges) > 0:
        max_vertex_index = len(vertices) - 1
        for edge in edges:
            if 0 <= edge[0] <= max_vertex_index and 0 <= edge[1] <= max_vertex_index:
                valid_edges_for_plot.append(edge)

    if valid_edges_for_plot:
        for edge in valid_edges_for_plot:
            p0 = vertices[edge[0]]
            p1 = vertices[edge[1]]
            edge_x.extend([p0[0], p1[0], None])
            edge_y.extend([p0[1], p1[1], None])
            edge_z.extend([p0[2], p1[2], None])
            mx.append((p0[0] + p1[0]) / 2)
            my.append((p0[1] + p1[1]) / 2)
            mz.append((p0[2] + p1[2]) / 2)

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color=display_dict['edge_color'], width=display_dict['edge_width']),
            name='Edges',
            hoverinfo='skip'
        ))

        if display_dict['show_edge_labels']:
            if display_dict['show_lengths']:
                if display_dict['show_in_nts']:
                    edge_text = [f"<b>{edge[0]}-{edge[1]}<br>{int(edge_lengths_bp[int(i)]):d} bp</b>" for i, edge in
                                 enumerate(valid_edges_for_plot)]
                else:
                    edge_text = [f"<b>{edge[0]}-{edge[1]}<br>Edge {int(i)}</b><br>{edge_lengths[int(i)]:.2f} nm</b>"
                                 for i, edge in enumerate(valid_edges_for_plot)]
            else:
                edge_text = [f"<b>{edge[0]}-{edge[1]}<br>Edge {int(i)}</b>" for i, edge in
                             enumerate(valid_edges_for_plot)]

            fig.add_trace(go.Scatter3d(
                x=mx, y=my, z=mz,
                mode='text',
                text=edge_text,
                textposition='middle center',
                showlegend=False,
                hoverinfo='text',
                textfont=dict(color=display_dict['edge_label_color'], size=display_dict['font_size'], family='Arial', ),
            ))

    if display_dict['show_background_grid']:
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=False,
            height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                aspectmode='data'
            ),
            showlegend=False,
            height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
    return fig

def _create_mesh_plot(geometry: Geometry, display_dict):
    vertices, edges = geometry.vertices, geometry.edges
    N_per_edge = geometry.n_per_edge
    diameter = display_dict['mesh_scale'] / 50
    fig = go.Figure()

    for ct, edge in enumerate(edges):
        if 0 <= edge[0] < len(vertices) and 0 <= edge[1] < len(vertices):
            p1 = np.array(vertices[edge[0]])
            p2 = np.array(vertices[edge[1]])
            length = np.linalg.norm(p2 - p1)
            center = (p1 + p2) / 2
            direction = (p2 - p1) / length if length > 0 else np.array([1, 0, 0])  # Handle zero length

            t = np.linspace(0, 2 * np.pi, 15)
            h = np.linspace(-length / 2, length / 2, 2)
            radius = diameter / 2

            radius *= N_per_edge[ct]  # Scale by N for visual comparison
            if radius == 0:
                continue

            x = radius * np.outer(np.cos(t), np.ones(len(h)))
            y = radius * np.outer(np.sin(t), np.ones(len(h)))
            z = np.outer(np.ones(np.size(t)), h)

            v = direction
            a = np.array([0, 0, 1])
            if not np.allclose(v, a):
                cross_product = np.cross(a, v)
                norm_cross = np.linalg.norm(cross_product)
                if norm_cross > 1e-6:
                    cross_product /= norm_cross
                    theta = np.arccos(np.dot(a, v))
                    k = cross_product
                    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
                    for i in range(x.shape[0]):
                        for j in range(x.shape[1]):
                            rotated = R @ np.array([x[i, j], y[i, j], z[i, j]])
                            x[i, j], y[i, j], z[i, j] = rotated

            x += center[0]
            y += center[1]
            z += center[2]
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, display_dict['mesh_color']],
                                                                [1, display_dict['mesh_color']]],
                                     showscale=False, name=f'{edge[0]}-{edge[1]}'))

    if display_dict['show_background_grid']:
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            showlegend=False,
            height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
    else:
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                aspectmode='data'
            ),
            showlegend=False,
            height=600,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

    return fig
