from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_honeycomb_grid_figure(e_vertices, o_vertices):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(*zip(*e_vertices), color="blue", alpha=0.8, label="All E")
    ax.scatter(*zip(*o_vertices), color="red", alpha=0.8, label="All O")
    ax.set_aspect("equal", adjustable="box")
    return fig


def build_honeycomb_symmetry_graph_figure(e_vertices, o_vertices, starts, ends):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(*zip(*e_vertices), color="blue", alpha=0.8, label="All E")
    ax.scatter(*zip(*o_vertices), color="red", alpha=0.8, label="All O")
    ax.scatter(*zip(*starts), color="black", alpha=0.8, label="Start S")
    ax.scatter(*zip(*ends), color="orange", alpha=0.8, label="Start R")
    ax.set_aspect("equal", adjustable="box")
    return fig


def build_cross_section_points_figure(evens, odds):
    fig, ax = plt.subplots()
    ax.plot([p[0] for p in evens], [p[1] for p in evens], "bo")
    ax.plot([p[0] for p in odds], [p[1] for p in odds], "ro")
    ax.set_aspect("equal", adjustable="box")
    return fig


def build_mitering_figure(design, rotated_positions: np.ndarray | dict, connections: np.ndarray | dict,
                          show_colors: bool = False, cylinder_radius: float = 0.5, cylinder_res: int = 10,
                          visual_override: bool = False):
    """Build a Plotly figure for mitering/debug visualization."""
    fig = go.Figure()

    colors_to_use = ['purple', 'cyan', 'orange', 'red', 'blue', 'magenta', 'yellow', 'brown', 'pink', 'gray']
    rotated_color_to_use = {}
    idx_edge_map = design.get_idx_edge_map()
    helix_to_bundle = design.get_helix_to_bundle()
    scaffold_dirs = design.get_scaffold_directions()

    for edge_ct, edge in enumerate(idx_edge_map):
        v1, v2 = design.get_point(edge[0]), design.get_point(edge[1])
        color = 'black' if not show_colors else colors_to_use[edge_ct % len(colors_to_use)]
        rotated_color_to_use[edge_ct] = color
        fig.add_trace(go.Scatter3d(x=[v1[0], v2[0]], y=[v1[1], v2[1]], z=[v1[2], v2[2]], mode='lines',
                                   line=dict(width=10, color=color), showlegend=False))

    if show_colors:
        helix_positions_per_bundle = {i: [[], []] for i in rotated_color_to_use.keys()}

        if isinstance(rotated_positions, np.ndarray):
            for row_ct, row in enumerate(rotated_positions):
                helix_positions_per_bundle[helix_to_bundle[row_ct]][0].append(row)
                grid_type = 'circle' if scaffold_dirs[row_ct] else 'square'
                helix_positions_per_bundle[helix_to_bundle[row_ct]][1].append(grid_type)
        else:
            for k, v in rotated_positions.items():
                helix_positions_per_bundle[helix_to_bundle[k]][0].append(v)
                grid_type = 'circle' if scaffold_dirs[k] else 'square'
                helix_positions_per_bundle[helix_to_bundle[k]][1].append(grid_type)

        for k, v in helix_positions_per_bundle.items():
            vals_list, symbols_list = v
            if not vals_list:
                continue

            vals = np.vstack(vals_list)
            color = rotated_color_to_use[k]
            edge = idx_edge_map[k]
            v1 = np.array(design.get_point(edge[0]))
            v2 = np.array(design.get_point(edge[1]))

            centroid = np.mean(vals, axis=0)
            if np.linalg.norm(centroid - v1) < np.linalg.norm(centroid - v2):
                v_start_ref, v_end_ref = v1, v2
                if visual_override:
                    v_start_ref, v_end_ref = v2, v1
            else:
                v_start_ref, v_end_ref = v2, v1
                if visual_override:
                    v_start_ref, v_end_ref = v1, v2

            axis_vec = v_end_ref - v_start_ref
            axis_len = np.linalg.norm(axis_vec)
            if axis_len == 0:
                continue
            axis_unit = axis_vec / axis_len

            bx, by, bz, bi, bj, bk = [], [], [], [], [], []
            idx_offset = 0
            for start_pt in vals:
                vec_to_end = v_end_ref - start_pt
                proj_len = np.dot(vec_to_end, axis_unit)
                end_pt = start_pt + (axis_unit * proj_len)
                cx, cy, cz, ci, cj, ck = get_cylinder_mesh(start_pt, end_pt, radius=cylinder_radius, res=cylinder_res)

                if len(cx) > 0:
                    bx.extend(cx)
                    by.extend(cy)
                    bz.extend(cz)
                    bi.extend(ci + idx_offset)
                    bj.extend(cj + idx_offset)
                    bk.extend(ck + idx_offset)
                    idx_offset += len(cx)

            fig.add_trace(go.Mesh3d(
                x=bx, y=by, z=bz, i=bi, j=bj, k=bk,
                color=color, flatshading=True,
                name=f"Bundle {k}", showlegend=True, legendgroup=str(k),
            ))

    else:
        if isinstance(rotated_positions, dict):
            temp = np.array([list(rotated_positions.values())]).squeeze()
            fig.add_trace(go.Scatter3d(x=temp[:, 0], y=temp[:, 1], z=temp[:, 2],
                                       mode='markers', marker=dict(symbol='circle', color='black'), showlegend=False))
        else:
            fig.add_trace(go.Scatter3d(x=rotated_positions[:, 0], y=rotated_positions[:, 1], z=rotated_positions[:, 2],
                                       mode='markers', marker=dict(symbol='circle', color='black'), showlegend=False))

    if isinstance(connections, dict):
        all_connections = np.vstack(list(connections.values()))
    else:
        all_connections = np.array(connections)

    for p1, p2 in all_connections:
        v1 = rotated_positions[int(p1)]
        v2 = rotated_positions[int(p2)]
        fig.add_trace(go.Scatter3d(
            x=[v1[0], v2[0]],
            y=[v1[1], v2[1]],
            z=[v1[2], v2[2]],
            mode='lines',
            legendgroup='connected_edges',
            line=dict(width=6, color='black'),
            showlegend=True,
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False,
                       backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False,
                       backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False,
                       backgroundcolor='rgba(0,0,0,0)'),
            dragmode='turntable',
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def build_bundle_positions_figure(positions, base_axes=None, connections=None, plot_cylinders=False, **kwargs):
    """Build a Plotly figure for bundle positions and optional connection traces."""
    fig = go.Figure()
    colors = ['purple', 'orange', 'cyan', 'red', 'blue', 'magenta', 'yellow', 'brown', 'pink', 'gray']
    if base_axes is not None and plot_cylinders:
        # If base_axes is provided and plot_cylinders is set True, we use the Z-axis (defined by base_axes[0] ->
        # base_axes[3]) to plot a cylinder trace for each point in positions:
        # Overall, this is a purely visual representation and uses a gross simplification just to show
        design = kwargs.get('design')
        radius = kwargs.get('radius', 1.5)
        length = kwargs.get('length', 40)
        resolution = kwargs.get('resolution', 15)
        spacer = kwargs.get('space_distance', 35)
        theta = np.linspace(0, 2 * np.pi, resolution)
        z = np.linspace(spacer, length + spacer, 2)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_unit = radius * np.cos(theta_grid)
        y_unit = radius * np.sin(theta_grid)
        for c, (cylinder_centers, grid_axis) in enumerate(zip(positions, base_axes)):
            # Z-axis vector for cylinder orientation
            z_axis = grid_axis[3] - grid_axis[0]
            z_axis /= np.linalg.norm(z_axis)

            # pick an arbitrary vector not parallel to z_axis to form basis
            arbitrary = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            x_axis = np.cross(arbitrary, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            for center in cylinder_centers:
                # transform unit cylinder points into world coordinates
                pts = (np.outer(x_unit.flatten(), x_axis) +
                       np.outer(y_unit.flatten(), y_axis) +
                       np.outer(z_grid.flatten(), z_axis))
                pts += center

                X = pts[:, 0].reshape(x_unit.shape)
                Y = pts[:, 1].reshape(x_unit.shape)
                Z = pts[:, 2].reshape(x_unit.shape)

                # Build faces for mesh
                verts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
                faces = []
                for i in range(resolution):
                    for j in range(1):
                        p0 = j * resolution + i
                        p1 = j * resolution + (i + 1) % resolution
                        p2 = (j + 1) * resolution + i
                        p3 = (j + 1) * resolution + (i + 1) % resolution
                        faces.append([p0, p2, p1])
                        faces.append([p1, p2, p3])
                i_faces, j_faces, k_faces = np.array(faces).T

                if kwargs.get('cylinder_color', False):
                    trace_color = 'darkgray'
                else:
                    trace_color = colors[c]
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=i_faces, j=j_faces, k=k_faces,
                    color=trace_color,
                    opacity=1.0,
                    flatshading=True,
                    showscale=False
                ))

    elif isinstance(positions, dict):
        for k, vals in positions.items():
            p1, p2, _, _ = vals
            vals = np.vstack((p1, p2))
            bundle_k_helices = go.Scatter3d(x=vals[:, 0], y=vals[:, 1], z=vals[:, 2], showlegend=False, mode='markers',
                                            marker=dict(size=10))
            fig.add_trace(bundle_k_helices)
    else:
        if kwargs.get('design', None):
            # If a design was passed in, we will plot by group-number:
            colors = ['red', 'cyan', 'orange', 'magenta']
            design = kwargs.get('design')
            helix_to_bundle = design.get_helix_to_bundle()
            scaffold_dirs = design.get_scaffold_directions()

            positions = np.vstack(positions)
            for i in np.unique(helix_to_bundle):
                indices = np.where(helix_to_bundle == i)[0]
                s, r = [], []
                for j in indices:
                    if scaffold_dirs[j]:
                        s.append(positions[j])
                    else:
                        r.append(positions[j])
                s, r = np.array(s), np.array(r)
                sender_helices = go.Scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2],
                                                showlegend=False, mode='markers', marker=dict(size=10, symbol='circle',
                                                                                              color=colors[i]))
                receiver_helices = go.Scatter3d(x=r[:, 0], y=r[:, 1], z=r[:, 2],
                                                showlegend=False, mode='markers', marker=dict(size=10, symbol='square',
                                                                                              color=colors[i]))
                fig.add_trace(sender_helices)
                fig.add_trace(receiver_helices)

        else:
            temp = np.array(positions)
            bundle_k_helices = go.Scatter3d(x=temp[:, 0], y=temp[:, 1], z=temp[:, 2], showlegend=False,
                                            mode='markers', marker=dict(size=10))
            fig.add_trace(bundle_k_helices)

    if connections is not None:
        positions = np.vstack(positions)
        edge_x, edge_y, edge_z = [], [], []
        for c in connections:
            if len(c) == 3:
                continue
            p1, p2 = positions[c[0]], positions[c[1]]
            edge_x.extend([p1[0], p2[0], None])
            edge_y.extend([p1[1], p2[1], None])
            edge_z.extend([p1[2], p2[2], None])
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=2, color='gray'),
            showlegend=False,
            hoverinfo='none'
        )
        fig.add_trace(edge_trace)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                title='', showgrid=False, zeroline=False, showticklabels=False,
                backgroundcolor='rgba(0,0,0,0)'
            ),
            yaxis=dict(
                title='', showgrid=False, zeroline=False, showticklabels=False,
                backgroundcolor='rgba(0,0,0,0)'
            ),
            zaxis=dict(
                title='', showgrid=False, zeroline=False, showticklabels=False,
                backgroundcolor='rgba(0,0,0,0)'
            ),
            dragmode='turntable',
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.837, y=0.65, z=1.25),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def build_breakpoint_heatmap(viz_data):
    """Build a Plotly heatmap for valid/invalid autostaple breakpoints."""
    df = pd.DataFrame(viz_data)
    heatmap_data = df.pivot(
        index="helix",
        columns="nt_position",
        values="valid_breakpoint",
    )
    heatmap_data = heatmap_data.sort_index(ascending=False)
    z = heatmap_data.astype(float).values
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[
                [0.0, "red"],
                [1.0, "green"],
            ],
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        title="Valid Breakpoint Heatmap",
        xaxis_title="Nucleotide Position",
        yaxis_title="Helix",
        yaxis_autorange="reversed",
        template="simple_white",
    )
    return fig

def get_cylinder_mesh(p1, p2, radius=0.5, res=8):
    """Generate mesh data for a cylinder from p1 to p2."""
    v = p2 - p1
    mag = np.linalg.norm(v)
    if mag == 0:
        return [], [], [], [], [], []
    v = v / mag

    not_v = np.array([1, 0, 0])
    if np.allclose(v, not_v) or np.allclose(v, -not_v):
        not_v = np.array([0, 1, 0])
    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)

    t = np.linspace(0, 2 * np.pi, res, endpoint=False)
    x_c = radius * (np.outer(np.cos(t), n1[0]) + np.outer(np.sin(t), n2[0]))
    y_c = radius * (np.outer(np.cos(t), n1[1]) + np.outer(np.sin(t), n2[1]))
    z_c = radius * (np.outer(np.cos(t), n1[2]) + np.outer(np.sin(t), n2[2]))

    vertices = np.vstack([p1 + np.stack([x_c.flat, y_c.flat, z_c.flat], axis=1),
                          p2 + np.stack([x_c.flat, y_c.flat, z_c.flat], axis=1)])

    indices = []
    for i in range(res):
        j = (i + 1) % res
        indices += [[i, i + res, j + res], [i, j + res, j]]

    indices = np.array(indices)
    return vertices[:, 0], vertices[:, 1], vertices[:, 2], indices[:, 0], indices[:, 1], indices[:, 2]
