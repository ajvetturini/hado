from __future__ import annotations

import numpy as np
import scadnano as sc

from hado.core.automation.pipeline.types import emit_runtime_message


def build_scadnano_design(model, scaf_seq: str, unpaired_seq: str, diagnostics=None, verbose: bool = False):
    """Create a scadnano.Design from the current HADO nucleotide model."""
    from hado.core.export import CaDNAnoWriter
    # NOTE: I already wrote the caDNAno writer prior to using scadnano so I find simply importing the
    #       base cadnano design easiest for converting -> scadnano instead of manually writing all that code
    #       The reason I chose cadnano originally was simplicity of output format
    
    try:
        writer = CaDNAnoWriter('', model, False, diagnostics=diagnostics, allow_custom_cross_section=True)
        json_data = writer.get_json_data()
        sc_design = sc.Design.from_cadnano_v2(json_dict=json_data)
        if model.scaffold_args.has_custom_cross_section():
            _apply_custom_cross_section_positions(model, sc_design, writer.design_to_cadnano_num)
    except Exception as exc:
        emit_runtime_message(
            f'ERROR: Unable to convert to scadnano file: {exc}',
            diagnostics=diagnostics,
            verbose=verbose,
            warning=True,
        )
        return None

    # NOTE: I could not get a graph / understanding of the sc.HelixGroup functionality and the bundles were always
    #       weirdly-rotated. I created my own oxDNA / oxView exporter to circumvent using it, but I do leave my old
    #       hacky-fix here if I end up figuring out how to set the HelixGroup correctly.

    # grid = sc.Grid.none
    # helix_groups = create_helix_groups_by_scaling_vertices(model, grid, sc_design, 1.5, writer.design_to_cadnano_num)
    # sc_design.groups = helix_groups
    # helix_to_bundle = model.get_helix_to_bundle()
    # grid_locations = model.get_helix_bundle_grid_locations()
    # for helix_idx, bundle_idx, grid_location in zip(sc_design.helices, helix_to_bundle, grid_locations):
    #     grid_location = (float(grid_location[0]), float(grid_location[1]))
    #     sc_design.helices[helix_idx].group = str(bundle_idx)
    #     sc_design.helices[helix_idx].grid_position = None
    #     sc_design.helices[helix_idx].position = sc.Position3D(x=grid_location[0], y=grid_location[1], z=0.0)

    found_scaffold = None
    for strand in sc_design.strands:
        if strand.is_scaffold:
            found_scaffold = strand
            break
    if found_scaffold is None:
        raise ValueError('ERROR: Unable to identify scaffold Strand.')

    scadnano_module = sc.scadnano if hasattr(sc, 'scadnano') else sc
    scadnano_module.__dict__['DNA_base_wildcard'] = unpaired_seq
    sc_design.assign_dna(strand=found_scaffold, sequence=scaf_seq, assign_complement=True)
    return sc_design


def _apply_custom_cross_section_positions(model, sc_design: sc.Design, design_to_cadnano_num):
    helix_to_bundle = model.get_helix_to_bundle()
    grid_locations = model.get_helix_bundle_grid_locations()
    sc_design.groups = create_helix_groups_by_scaling_vertices(
        model,
        sc.Grid.none,
        sc_design,
        1.5,
        design_to_cadnano_num,
    )
    for design_helix_idx, bundle_idx in enumerate(helix_to_bundle):
        cadnano_helix_idx = design_to_cadnano_num[design_helix_idx]
        grid_location = grid_locations[design_helix_idx]
        helix = sc_design.helices[cadnano_helix_idx]
        helix.group = str(int(bundle_idx))
        helix.grid_position = None
        helix.position = sc.Position3D(x=float(grid_location[0]), y=float(grid_location[1]), z=0.0)

def create_helix_groups_by_scaling_vertices(model, grid: sc.Grid, sc_design: sc.Design,
                                            scale: float, design_to_cadnano_num):
    """Create scadnano HelixGroup objects from HADO geometry."""
    geom = sc_design.geometry
    imap = model.get_idx_edge_map()
    helix_to_bundle = model.get_helix_to_bundle()
    roll_angles = model.get_final_rotations()

    original_vertices_np = model.get_all_vertices()
    scaled_vertices_list = [vertex * scale for vertex in original_vertices_np]

    helix_groups = {}
    all_helices = sc_design.helices
    design_to_cadnano_bundle_number = design_to_cadnano_num
    for edge_idx in range(imap.shape[0]):
        v1_idx, v2_idx = imap[edge_idx]
        p1_scaled = scaled_vertices_list[v1_idx]
        p2_scaled = scaled_vertices_list[v2_idx]

        scaled_edge_vector = p2_scaled - p1_scaled
        scaled_edge_length = np.linalg.norm(scaled_edge_vector)

        if scaled_edge_length == 0:
            continue

        ux, uy, uz = scaled_edge_vector / scaled_edge_length

        pitch_rad = np.arctan2(-uy, np.sqrt(ux**2 + uz**2))
        yaw_rad = np.arctan2(ux, uz)
        angle = float(roll_angles[edge_idx]) if (roll_angles is not None and edge_idx in roll_angles) else 0.0

        view_order = []
        helices_in_bundle_i = np.where(helix_to_bundle == edge_idx)[0]
        for helix_idx in helices_in_bundle_i:
            helix = all_helices[design_to_cadnano_bundle_number[helix_idx]]
            view_order.append(int(helix.idx))

        position = p1_scaled
        new_bundle = sc.HelixGroup(
            position=sc.Position3D(
                x=float(position[0]),
                y=-float(position[1]),
                z=float(position[2]),
            ),
            pitch=float(np.degrees(pitch_rad)),
            yaw=float(np.degrees(yaw_rad)),
            roll=float(angle),
            grid=grid,
            geometry=geom,
            helices_view_order=view_order,
        )
        helix_groups[str(edge_idx)] = new_bundle

    return helix_groups
