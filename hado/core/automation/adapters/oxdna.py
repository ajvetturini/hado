from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from hado.core.automation.pipeline.types import emit_runtime_message

# oxDNA conversion constants
_OXDNA_NM_TO_UNITS = 1.0 / 0.8518
_OXDNA_BASE_DIST = 0.6
_GROOVE_GAMMA_DEG = 20.0

# Strictly for incomplete monomer sequence data. Full fallback strands use a balanced deterministic sequence below.
_DEFAULT_BASE = "T"
_BASE_COMPLEMENT = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
}
_FALLBACK_SEQUENCE_SEED = 8
_FALLBACK_BASES = np.array(["A", "C", "G", "T"])


def build_oxdna(model, manager):
    """Create an oxDNA system directly from the hado nucleotide model."""
    scaf_path = model.get_ordered_scaffold_nts()
    staple_paths = model.get_ordered_staple_nts()
    return OxDNASystem(model, manager, scaf_path, staple_paths)


class OxDNASystem:
    def __init__(self, model, manager, scaf_path, staple_paths):
        self.model = model
        self.manager = manager
        self.design_scaf_path = self._normalize_path(scaf_path)
        self.design_staple_paths = [self._normalize_path(path) for path in staple_paths]
        self.scaf_path = self.design_scaf_path
        self.staple_paths = self.design_staple_paths
        self.strand_paths = [self.design_scaf_path, *self.design_staple_paths]

        self.helix_frames = {}
        self.helix_phase_offsets = {}
        self.centers = {}
        self.dat = {}
        self.strands = []
        self.box = np.ones(3, dtype=float)
        self._init_system()

    def _relax_helix_phase_offsets(self):
        """Return per-helix phase offsets that minimize internal crossover strain."""
        phase_offsets = {int(helix): 0.0 for helix in self.helix_frames}

        for helix in self.helix_frames:
            relative_angles = []
            for other_helix, offset, strand_direction in self._internal_crossover_addresses_for_helix(helix):
                if other_helix not in self.helix_frames:
                    continue

                current_angle = np.degrees(self._get_backbone_angle(offset, strand_direction)) % 360.0
                target_angle = self._angle_from_helix_to_helix(helix, other_helix)
                relative_angles.append((current_angle, target_angle))

            if relative_angles:
                phase_offsets[int(helix)] = self._minimum_strain_angle(relative_angles)

        return phase_offsets

    def _internal_crossover_addresses_for_helix(self, helix: int):
        """Yield (other_helix, offset, strand_direction) for same-bundle crossovers."""
        helix = int(helix)
        helix_to_bundle = self.model.get_helix_to_bundle()
        scaffold_crossovers, staple_crossovers = self.model.get_internal_crossovers_on_helix(helix)

        scaffold_direction = bool(self.model.get_scaffold_direction(helix))
        for other_helix, offset in scaffold_crossovers:
            other_helix = int(other_helix)
            if helix_to_bundle[other_helix] == helix_to_bundle[helix]:
                yield other_helix, int(offset), scaffold_direction

        staple_direction = bool(self.model.get_staple_direction(helix))
        for other_helix, offset in staple_crossovers:
            other_helix = int(other_helix)
            if helix_to_bundle[other_helix] == helix_to_bundle[helix]:
                yield other_helix, int(offset), staple_direction

    def _angle_from_helix_to_helix(self, helix: int, other_helix: int) -> float:
        """Return the HADO backbone angle pointing from helix toward other_helix."""
        helix_frame = self.helix_frames[int(helix)]
        other_frame = self.helix_frames[int(other_helix)]

        delta = np.asarray(other_frame["midpoint_nm"], dtype=float) - np.asarray(
            helix_frame["midpoint_nm"],
            dtype=float,
        )
        x_component = float(np.dot(delta, helix_frame["x_axis"]))
        y_component = float(np.dot(delta, helix_frame["y_axis"]))

        # _get_a1_vector uses cos(theta) * x_axis - sin(theta) * y_axis, so
        # the local Y component has the opposite sign from atan2's usual frame.
        return float(np.degrees(np.arctan2(-y_component, x_component)) % 360.0)

    def _minimum_strain_angle(self, relative_angles) -> float:
        """
        scadnano-style circular least-squares angle.

        relative_angles contains (current_angle, target_angle) pairs. The returned
        phase offset is the circular angle that best rotates all current angles
        onto their corresponding target angles.
        """
        adjusted_angles = [
            (float(current_angle) - float(target_angle)) % 360.0
            for current_angle, target_angle in relative_angles
        ]
        return float((-self._average_angle(adjusted_angles)) % 360.0)

    @staticmethod
    def _average_angle(angles) -> float:
        angles_rad = np.deg2rad(np.asarray(angles, dtype=float))
        mean_sin = float(np.mean(np.sin(angles_rad)))
        mean_cos = float(np.mean(np.cos(angles_rad)))
        if np.isclose(mean_sin, 0.0) and np.isclose(mean_cos, 0.0):
            return 0.0
        return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)

    def _init_system(self):
        """Convert the hado nucleotide model into oxDNA monomer data."""
        self._build_helix_frames_and_centers()
        if self.model.scaffold_args.lattice_type == 'dna_honeycomb':
            self.helix_phase_offsets = {helix: 0.0 for helix in self.helix_frames}
        else:
            self.helix_phase_offsets = self._relax_helix_phase_offsets()
        staple_colors = self.model.get_staple_color_palette()
        helix_to_bundle = self.model.get_helix_to_bundle()

        strand_sequences = self._get_strand_sequences()
        if len(strand_sequences) != len(self.strand_paths):
            raise RuntimeError("ERROR: oxDNA sequence count does not match strand path count.")

        monomer_ids_by_position = {}
        next_monomer_id = 0
        for path_index, path in enumerate(self.strand_paths):
            is_scaffold = path_index == 0
            for position in path:
                monomer_ids_by_position.setdefault(position, {})[is_scaffold] = next_monomer_id
                next_monomer_id += 1

        base_pair_ids = {}
        for ids_at_position in monomer_ids_by_position.values():
            scaffold_id = ids_at_position.get(True)
            staple_id = ids_at_position.get(False)
            if scaffold_id is not None and staple_id is not None:
                base_pair_ids[scaffold_id] = staple_id
                base_pair_ids[staple_id] = scaffold_id

        monomer_id = 0
        for strand_index, path in enumerate(self.strand_paths, start=1):
            sequence = strand_sequences[strand_index - 1]
            if len(sequence) != len(path):
                raise RuntimeError("ERROR: oxDNA sequence length does not match strand path length.")
            is_scaffold = strand_index == 1
            monomers = []

            for idx_in_strand, (helix, nt_idx) in enumerate(path):
                base = sequence[idx_in_strand]
                monomer = self._build_monomer(
                    monomer_id=monomer_id,
                    strand_id=strand_index,
                    helix=helix,
                    nt_idx=nt_idx,
                    base=base,
                    is_scaffold=is_scaffold,
                    base_pair_id=base_pair_ids.get(monomer_id),
                    idx_in_strand=idx_in_strand,
                    strand_length=len(path),
                    color=staple_colors[helix_to_bundle[helix]],
                    cluster=helix_to_bundle[helix],
                )
                self.dat[monomer_id] = monomer
                monomers.append(self._to_oxview_monomer(monomer))
                monomer_id += 1

            self.strands.append(
                {
                    "id": strand_index,
                    "class": "NucleicAcidStrand",
                    "end5": int(monomers[0]["id"]) if monomers else -1,
                    "end3": int(monomers[-1]["id"]) if monomers else -1,
                    "monomers": monomers,
                }
            )

        self.box = self.calc_bbox()

    def _build_helix_frames_and_centers(self):
        axial_rise = self.model.get_axial_rise()
        grid_locations = self.model.get_helix_bundle_grid_locations()
        active_nts = np.logical_or(
            self.model.get_scaffold_nucleotides().astype(bool),
            self.model.get_staple_nucleotides().astype(bool),
        )
        helix_to_bundle = self.model.get_helix_to_bundle()
        bundle_rotations = self.model.get_final_rotations()

        local_axis = np.array([0.0, 0.0, -1.0])
        local_x = np.array([1.0, 0.0, 0.0])
        local_y = np.array([0.0, 1.0, 0.0])

        for bundle_idx, edge in enumerate(self.model.geometry.edges):
            v1, v2 = edge
            p1 = self.model.get_point(v1)
            p2 = self.model.get_point(v2)
            edge_vector = p2 - p1
            edge_length = np.linalg.norm(edge_vector)
            if edge_length <= 0:
                continue

            edge_axis = edge_vector / edge_length
            roll_angle = self._get_bundle_rotation(bundle_rotations, bundle_idx)
            alignment_rot, _ = R.align_vectors(edge_axis.reshape(1, 3), local_axis.reshape(1, 3))
            roll_rot = R.from_euler("z", -roll_angle, degrees=True)
            combined_rot = alignment_rot * roll_rot

            x_axis = self._normalize(combined_rot.apply(local_x))
            y_axis = self._normalize(combined_rot.apply(local_y))
            helix_axis = self._normalize(combined_rot.apply(local_axis))

            helices_in_bundle = np.where(helix_to_bundle == bundle_idx)[0]
            edge_center = (p1 + p2) / 2.0
            midpoint_nt = self._get_edge_midpoint_nt(
                edge_length,
                axial_rise,
                active_nts.shape[1],
            )
            for helix in helices_in_bundle:
                grid_position = np.asarray(grid_locations[helix], dtype=float)
                helix_midpoint_nm = edge_center + combined_rot.apply(
                    np.array([grid_position[0], grid_position[1], 0.0], dtype=float)
                )
                helix_origin_nm = helix_midpoint_nm - (midpoint_nt * axial_rise * helix_axis)

                helix_key = int(helix)
                self.helix_frames[helix_key] = {
                    "origin_nm": helix_origin_nm,
                    "midpoint_nm": helix_midpoint_nm,
                    "midpoint_nt": midpoint_nt,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "helix_axis": helix_axis,
                    "axial_rise": axial_rise,
                }

                for nt_idx in np.flatnonzero(active_nts[helix]):
                    center_nm = helix_origin_nm + (float(nt_idx) * axial_rise * helix_axis)
                    self.centers[(helix_key, int(nt_idx))] = center_nm * _OXDNA_NM_TO_UNITS

    @staticmethod
    def _get_bundle_rotation(bundle_rotations, bundle_idx: int) -> float:
        if isinstance(bundle_rotations, dict):
            return float(bundle_rotations.get(bundle_idx, 0.0))
        if isinstance(bundle_rotations, (list, tuple, np.ndarray)) and bundle_idx < len(bundle_rotations):
            return float(bundle_rotations[bundle_idx])
        return 0.0

    @staticmethod
    def _get_edge_midpoint_nt(edge_length: float, axial_rise: float, nt_count: int) -> float:
        edge_nts = int(edge_length // axial_rise)
        start_nt = (int(nt_count) - edge_nts) // 2
        return start_nt + ((edge_nts - 1) / 2.0)

    @staticmethod
    def _normalize_path(path):
        return [(int(helix), int(nt_idx)) for helix, nt_idx in path]

    def _get_strand_sequences(self):
        fallback = self._get_fallback_strand_sequences()
        try:
            worked, rows = self.manager.get_sequences(self.model)
        except Exception as exc:
            emit_runtime_message(
                f"WARNING: Unable to sequence for oxDNA export due to error {exc}, "
                f"using balanced A/C/G/T fallback sequences",
                verbose=getattr(self.manager, "verbose", False),
                warning=True,
            )
            return fallback

        if not worked or rows is None or len(rows) < 2:
            emit_runtime_message(
                f"WARNING: Provided scaffold sequence too short for oxDNA export, "
                f"using balanced A/C/G/T fallback sequences",
                verbose=getattr(self.manager, "verbose", False),
                warning=True,
            )
            return fallback

        scaffold_sequence = rows[1][3] if len(rows[1]) >= 4 else ""
        if not isinstance(scaffold_sequence, str) or len(scaffold_sequence) != len(self.strand_paths[0]):
            emit_runtime_message(
                "WARNING: Scaffold sequence does not match the oxDNA scaffold path, "
                "using balanced A/C/G/T fallback sequences",
                verbose=getattr(self.manager, "verbose", False),
                warning=True,
            )
            return fallback

        scaffold_by_position = dict(zip(self.strand_paths[0], scaffold_sequence))
        staple_args = getattr(self.manager, "staple_args", None)
        unpaired_base = str(getattr(staple_args, "unpaired_sequence", _DEFAULT_BASE)).upper()
        if unpaired_base not in _BASE_COMPLEMENT:
            unpaired_base = _DEFAULT_BASE

        strand_sequences = [scaffold_sequence]
        for path in self.strand_paths[1:]:
            sequence = [
                _BASE_COMPLEMENT.get(scaffold_by_position.get(position, ""), unpaired_base)
                for position in path
            ]
            strand_sequences.append("".join(sequence))

        return strand_sequences

    def _get_fallback_strand_sequences(self):
        scaffold_sequence = self._get_balanced_fallback_sequence(len(self.strand_paths[0]))
        scaffold_by_position = dict(zip(self.strand_paths[0], scaffold_sequence))

        strand_sequences = [scaffold_sequence]
        for path in self.strand_paths[1:]:
            sequence = [
                _BASE_COMPLEMENT.get(scaffold_by_position.get(position, ""), _DEFAULT_BASE)
                for position in path
            ]
            strand_sequences.append("".join(sequence))
        return strand_sequences

    @staticmethod
    def _get_balanced_fallback_sequence(length: int) -> str:
        if length <= 0:
            return ""

        rng = np.random.default_rng(_FALLBACK_SEQUENCE_SEED)
        bases = []
        while len(bases) < length:
            bases.extend(rng.permutation(_FALLBACK_BASES).tolist())
        return "".join(bases[:length])

    def _get_backbone_angle(self, idx: int, strand_direction: bool):
        return self.model.get_backbone_rotation_angle(int(idx), bool(strand_direction))

    def _get_a1_vector(self, helix: int, nt_idx: int, strand_direction: bool):
        helix_frame = self.helix_frames[int(helix)]
        helix_axis = self._normalize(helix_frame["helix_axis"])
        phase_offset = self.helix_phase_offsets.get(int(helix), 0.0)

        # Pass is_scaffold to get the correct rotation angle
        theta_deg = np.degrees(self._get_backbone_angle(nt_idx, strand_direction)) + phase_offset
        backbone = (
                np.cos(np.deg2rad(theta_deg)) * helix_frame["x_axis"]
                - np.sin(np.deg2rad(theta_deg)) * helix_frame["y_axis"]
        )

        # Gamma groove correction depends on the physical progression direction of the strand
        gamma_correction = -_GROOVE_GAMMA_DEG if strand_direction else _GROOVE_GAMMA_DEG
        a1 = self._rotate_about_axis(backbone, helix_axis, gamma_correction)
        return self._normalize(-a1)

    def _build_monomer(self, monomer_id: int, strand_id: int, helix: int, nt_idx: int, base: str,
                       is_scaffold: bool, base_pair_id: int | None, idx_in_strand: int,
                       strand_length: int, color: str, cluster: int):
        center = self._get_center_oxdna_units(helix, nt_idx)
        helix_frame = self.helix_frames[helix]
        strand_direction = bool(
            self.model.get_scaffold_direction(helix)
            if is_scaffold else
            self.model.get_staple_direction(helix)
        )

        helix_axis = self._normalize(helix_frame["helix_axis"])
        a1 = self._get_a1_vector(helix, nt_idx, strand_direction)

        a3 = -helix_axis if strand_direction else helix_axis
        a3 = self._normalize(a3)

        position = center - (a1 * _OXDNA_BASE_DIST)

        monomer = {
            "id": int(monomer_id),
            "strand_id": int(strand_id),
            "helix": int(helix),
            "offset": int(nt_idx),
            "type": base if base else _DEFAULT_BASE,
            "center": center.tolist(),
            "p": position.tolist(),
            "a1": a1.tolist(),
            "a3": a3.tolist(),
            "v": [0., 0., 0.],
            "L": [0., 0., 0.],
            "color": color,
            "cluster": int(cluster),
        }
        if idx_in_strand != 0:
            monomer["n5"] = monomer_id - 1
        if base_pair_id is not None:
            monomer["bp"] = int(base_pair_id)
        if idx_in_strand != strand_length - 1:
            monomer["n3"] = monomer_id + 1
        return monomer

    def _get_center_oxdna_units(self, helix: int, nt_idx: int):
        key = (int(helix), int(nt_idx))
        if key not in self.centers:
            helix_frame = self.helix_frames[int(helix)]
            center_nm = helix_frame["origin_nm"] + (
                float(nt_idx) * helix_frame["axial_rise"] * helix_frame["helix_axis"]
            )
            self.centers[key] = center_nm * _OXDNA_NM_TO_UNITS
        return self.centers[key].copy()

    @staticmethod
    def _to_oxview_monomer(monomer: dict):
        oxview_monomer = {
            "id": monomer["id"],
            "p": monomer["p"],
            "a1": monomer["a1"],
            "a3": monomer["a3"],
            "class": "DNA",
            "type": monomer["type"],
            "color": monomer["color"],
            "cluster": monomer["cluster"],
        }
        if "n5" in monomer:
            oxview_monomer["n5"] = monomer["n5"]
        if "n3" in monomer:
            oxview_monomer["n3"] = monomer["n3"]
        if "bp" in monomer:
            oxview_monomer["bp"] = monomer["bp"]
        return oxview_monomer

    @staticmethod
    def _rotate_about_axis(vector, axis, angle_deg: float):
        axis = np.asarray(axis, dtype=float)
        vector = np.asarray(vector, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            return vector
        return R.from_rotvec(np.deg2rad(angle_deg) * (axis / axis_norm)).apply(vector)

    @staticmethod
    def _normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.asarray(vector, dtype=float)
        return np.asarray(vector, dtype=float) / norm

    def calc_bbox(self):
        """Calculate a cubic bounding box of 1.5X the max span of monomer centers"""
        if not self.dat:
            return np.ones(3, dtype=float)

        centers = np.array([entry["center"] for entry in self.dat.values()], dtype=float)
        mins = centers.min(axis=0)
        maxs = centers.max(axis=0)
        bbox = 1.5 * (maxs - mins)
        max_side = float(np.max(bbox))
        return np.array([max_side, max_side, max_side], dtype=float)

    def get_oxview_dict(self):
        return {
            "box": self.box.tolist(),
            "systems": [
                {
                    "id": 0,
                    "strands": self.strands,
                }
            ],
        }
