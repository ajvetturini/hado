import numpy as np
from collections import Counter
import ast

from hado.core.utils import Geometry, ScaffoldArgs, StapleArgs
from hado.core.automation.model.builder import build_initial_model_state
from hado.core.automation.model.crossover_maps import (
    fix_leading_zero_index,
    get_all_scaffold_crossover_options as _get_all_scaffold_crossover_options,
    get_all_staple_crossover_options as _get_all_staple_crossover_options,
    get_all_valid_scaffold_crossover_options as _get_all_valid_scaffold_crossover_options,
    get_nearest_scaffold_crossover_index as _get_nearest_scaffold_crossover_index,
    replace_scaffold_positions as _replace_scaffold_positions,
)
from hado.core.automation.model.strand_ordering import (
    get_ordered_scaffold_nts as _get_ordered_scaffold_nts,
    get_ordered_staple_nts as _get_ordered_staple_nts,
)

_fix_leading_zero_index = fix_leading_zero_index

class HadoNucleotideModel:

    def __init__(self,
                 geometry: Geometry,
                 scaffold_args: ScaffoldArgs,
                 staple_args: StapleArgs,
                 edge_xsect_definitions: dict,
                 restored_state: dict = False,
                 max_cross_section_size_override: bool = False,
                 diagnostics=None
                 ):
        self.geometry = geometry
        self.scaffold_args = scaffold_args
        self.staple_args = staple_args
        self.edge_xsect_definitions = edge_xsect_definitions

        if restored_state:
            for k, v in restored_state.items():
                if isinstance(v, list) and k not in ['_scaf_xovers', '_stap_xovers', '_staple_color_palette']:
                    v = np.array(v)
                elif k in ['_scaffold_xover_map', '_staple_xover_map', ]:
                    v = {ast.literal_eval(x): y for x, y in v.items()}
                elif k == '_bundle_rotations':
                    v = {int(x): int(y) for x, y in v.items()}
                elif k in ['_connected_helices', '_rotated_helix_positions']:
                    v = {int(x): y for x, y in v.items()}
                self.__setattr__(k, v)
        else:
            self._initialize_model(max_cross_section_size_override, diagnostics=diagnostics)
            self._connected_helices = None
            self._rotated_helix_positions = None  # These are sometimes set, but init to null
        self._sc_design = None  # Never stored in to_dict
        self._oxdna_system = None  # Never stored in to_dict


    def _initialize_model(self, max_cross_section_size_override: bool = False, diagnostics=None):
        """Prepares private data structures for a nucleotide-level model."""
        state = build_initial_model_state(
            self,
            max_cross_section_size_override=max_cross_section_size_override,
            diagnostics=diagnostics,
        )
        for attr, value in state.items():
            setattr(self, attr, value)

    def to_dict(self):
        state = {'edge_xsect_definitions': {str(x): y for x, y in self.edge_xsect_definitions.items()}}

        for name, value in self.__dict__.items():
            if name.startswith("_") and not name.startswith("__"):
                if name in ['_sc_design', '_oxdna_system']:
                    continue
                elif isinstance(value, np.ndarray):
                    state[name] = value.tolist()
                elif '_scaffold_xover_map' in name or '_staple_xover_map' in name:
                    temp = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            temp[str(k)] = v.tolist()
                        else:
                            temp[str(k)] = v
                    state[name] = temp
                elif name == '_bundle_rotations':
                    temp = {}
                    for k, v in value.items():
                        temp[int(k)] = int(v)
                    state[name] = temp
                else:
                    state[name] = value
        return state

    @classmethod
    def load_from_dict(cls, geo, scaf, stap, nucleotide_dict):
        if nucleotide_dict is None:
            return None
        edge_xsect = nucleotide_dict['edge_xsect_definitions']
        edge_xsect_fixed = {ast.literal_eval(x): y for x, y in edge_xsect.items()}
        model = cls(geo, scaf, stap, edge_xsect_fixed, nucleotide_dict)
        return model

    def set_scadnano(self, scaf_seq: str, unpaired_seq: str, diagnostics=None, verbose: bool = False):
        """Create and store a scadnano.Design for export/sequencing workflows."""
        from hado.core.automation.adapters.scadnano import build_scadnano_design
        self._sc_design = build_scadnano_design(self, scaf_seq, unpaired_seq, diagnostics=diagnostics, verbose=verbose)

    def set_oxdna_system(self, manager):
        """Create and store an oxDNA system for export to oxDNA / oxView for simulation and visualization purposes."""
        from hado.core.automation.adapters.oxdna import build_oxdna
        self._oxdna_system = build_oxdna(self, manager)

    def set_connected_helices(self, optimal_connections: dict):
        json_compatible = {}
        for k, v in optimal_connections.items():
            temp = []
            for pair in v:
                temp.append((int(pair[0]), int(pair[1])))
            json_compatible[int(k)] = temp
        self._connected_helices = json_compatible

    def set_rotated_helix_positions(self, final_positions: dict):
        rotated_positions = {}
        for k, v in final_positions.items():
            temp = []
            for i in v:
                a, b = i
                aa = int(a)
                bb = [float(b[0]), float(b[1]), float(b[2])]
                temp.append((aa, bb))
            rotated_positions[int(k)] = temp
        self._rotated_helix_positions = rotated_positions

    def scadnano_not_set(self):
        if self._sc_design is None:
            return True
        return False

    def oxdna_not_set(self):
        if self._oxdna_system is None:
            return True
        return False

    def get_sc_design(self):
        self._sc_design.relax_helix_rolls()
        return self._sc_design

    def get_oxdna_system(self):
        return self._oxdna_system

    def get_point(self, i):
        """ Returns the (x, y, z) of the vertex i """
        return np.array(self.geometry.vertices[i].copy())

    def get_all_vertices(self):
        return np.array(self.geometry.vertices.copy())

    def get_helix_to_bundle(self):
        return np.array(self._helix_to_bundle.copy())

    def get_idx_edge_map(self):
        return np.array(self._idx_edge_map.copy())

    def get_helix_bundle_grid_locations(self):
        return np.array(self._grid_locations.copy())

    def get_bundles_at_vertex(self, vertex):
        """ returns which indices of idx_edge_map belong to the current vertex """
        bundle_numbers = []
        for ct, i in enumerate(self._idx_edge_map):
            if i[0] == vertex or i[1] == vertex:
                bundle_numbers.append(ct)
        return bundle_numbers

    def get_staple_color_palette(self):
        return self._staple_color_palette.copy()

    def get_staple_color_pallete(self):
        return self.get_staple_color_palette()

    def get_scaffold_direction(self, helix: int):
        return self._scaffold_dirs[helix]

    def get_staple_direction(self, helix: int):
        return self._staple_dirs[helix]

    def get_backbone_rotation_angle(self, nt_position, forward):
        """ This is a CLOCKWISE angle from the x-axis """
        rotation = self._init_offset + self._twist_per_base * nt_position
        if not forward:
            rotation += self._minor_groove_ang
        rotation = rotation % 360
        return np.deg2rad(rotation)

    # def get_internal_crossovers_on_helix(self, helix: int):
    #     """ Returns crossovers from a helix to other helices within the same helix bundle for both scaffold
    #     and staples """
    #     scaf, stap = [], []
    #     for xo in self._scaffold_crossovers:
    #         h1, nt1, h2, nt2 = xo
    #         if h1 == helix and nt2 == -1:
    #             scaf.append((h2, nt1))
    #         elif h2 == helix and nt2 == -1:
    #             scaf.append((h1, nt1))
    #
    #     for xo in self._staple_crossovers:
    #         h1, nt1, h2, nt2 = xo
    #         if h1 == helix and nt2 == -1:
    #             stap.append((h2, nt1))
    #         elif h2 == helix and nt2 == -1:
    #             stap.append((h1, nt1))
    #
    #     return scaf, stap

    def get_internal_crossovers_on_helix(self, helix: int):
        """ Returns crossovers from a helix to other helices within the same helix bundle for both scaffold
         and staples """
        h2b = self.get_helix_to_bundle()
        helix = int(helix)
        helix_bundle = h2b[helix]

        def collect(crossovers):
            out = []
            for xo in crossovers:
                h1, nt1, h2, nt2 = [int(x) for x in xo]
                if h2b[h1] != helix_bundle or h2b[h2] != helix_bundle:
                    continue
                if h1 == helix:
                    out.append((h2, nt1))
                elif h2 == helix:
                    out.append((h1, nt1 if nt2 == -1 else nt2))
            return out

        return collect(self._scaffold_crossovers), collect(self._staple_crossovers)

    def get_sender_indices(self, node: int, vertex: int):
        """ Looks at all grid elements for a given node / vertex pair and determines which of the grid elements
        are able to be selected as the sender
        """
        h2b_indices = np.where(self._helix_to_bundle == node)[0]
        scaf_dirs = self._scaffold_dirs[h2b_indices]

        # Then store the True / False values using the directionality of the scaffold
        # This is only difference between receivers / standards (these conditionals are flipped)
        if self._idx_edge_map[node][0] == vertex:
            senders = np.where(~scaf_dirs)[0]
        elif self._idx_edge_map[node][1] == vertex:
            senders = np.where(scaf_dirs)[0]
        else:
            raise Exception('ERROR: Node / Vertex pair is invalid')

        return h2b_indices[senders], senders

    def get_receiver_indices(self, node: int, vertex: int):
        """ Looks at all grid elements for a given node / vertex pair and determines which of the grid elements
        are able to be selected as the sender
        """
        h2b_indices = np.where(self._helix_to_bundle == node)[0]
        scaf_dirs = self._scaffold_dirs[h2b_indices]

        # Then store the True / False values using the directionality of the scaffold
        # This is only difference between receivers / standards (these conditionals are flipped)
        if self._idx_edge_map[node][0] == vertex:
            receivers = np.where(scaf_dirs)[0]
        elif self._idx_edge_map[node][1] == vertex:
            receivers = np.where(~scaf_dirs)[0]
        else:
            raise Exception('ERROR: Node / Vertex pair is invalid')

        return h2b_indices[receivers], receivers

    def get_scaffold_directions(self):
        return np.array(self._scaffold_dirs.copy())

    def get_staple_directions(self):
        return np.array(self._staple_dirs.copy())

    def get_scaffold_nucleotides(self):
        return np.array(self._scaffold_nucleotides.copy())

    def get_scaffold_crossovers(self):
        return np.array(self._scaffold_crossovers.copy())

    def get_staple_crossovers(self):
        return np.array(self._staple_crossovers.copy())

    def get_staple_break_points(self):
        return np.array(self._staple_breaks.copy())

    def get_scaffold_start_point(self):
        return np.array(self._scaffold_start_point.copy())

    def get_staple_nucleotides(self):
        return np.array(self._staple_nucleotides.copy())

    def get_axial_rise(self):
        return float(self._axial_rise)

    def get_period(self):
        return int(self._period)

    def get_spacing_distance(self):
        return self._helix_spacing

    def get_diameter(self):
        return self._diameter

    def get_connected_helices(self):
        return self._connected_helices

    def get_rotated_helix_positions(self):
        rotated_positions = {}
        for k, v in self._rotated_helix_positions.items():
            temp = []
            for i in v:
                a, b = i
                aa = int(a)
                bb = np.array([float(b[0]), float(b[1]), float(b[2])])
                temp.append((aa, bb))
            rotated_positions[int(k)] = temp
        return rotated_positions

    def check_free_ends(self, helix):
        """ Determines if either the min / max end of a helix is a 'free' end / degree-1 end """
        helix_bundle = self._helix_to_bundle[helix]
        edge = self._idx_edge_map[helix_bundle]
        all_counts = Counter(self._idx_edge_map.flatten())
        free_min_end = True if all_counts[edge[0]] == 1 else False
        free_max_end = True if all_counts[edge[1]] == 1 else False
        return free_min_end, free_max_end

    def get_shared_vertex(self, node1: int, node2: int):
        """ Returns shared vertex between two nodes """
        vs1, vs2 = self._idx_edge_map[node1], self._idx_edge_map[node2]
        shared = set(vs1) & set(vs2)
        return shared.pop() if shared else None

    def are_neighbors(self, id1, id2):
        """ Checks if two helices which must belong to the same helix-bundle are able to crossover to eachother """
        dist, acceptable_length = self._distance_apart(id1, id2)
        if np.isclose(dist, acceptable_length):
            return True
        return False

    def set_scaffold_nucleotides(self, new_scaffold_nucleotides):
        temp = np.array(new_scaffold_nucleotides)
        if temp.shape != self._scaffold_nucleotides.shape:
            raise ValueError('ERROR: new scaffold nucleotides shape mismatch')
        self._scaffold_nucleotides = temp.copy()

    def set_staple_nucleotides(self, new_staple_nucleotides):
        temp = np.array(new_staple_nucleotides)
        if temp.shape != self._staple_nucleotides.shape:
            raise ValueError('ERROR: new staple nucleotides shape mismatch')
        self._staple_nucleotides = temp.copy()

    def set_scaffold_crossovers(self, new_scaffold_crossovers):
        self._scaffold_crossovers = np.array(new_scaffold_crossovers.copy())

    def set_staple_crossovers(self, new_staple_crossovers):
        self._staple_crossovers = np.array(new_staple_crossovers.copy())

    def set_staple_breaks(self, new_staple_breaks):
        self._staple_breaks = np.array(new_staple_breaks.copy())

    def set_scaffold_start_point(self, new_scaffold_start_point):
        self._scaffold_start_point = np.array(new_scaffold_start_point.copy())

    def set_bundle_rotations(self, bundle_rotations: dict):
        self._bundle_rotations = {int(i): int(j) for i, j in bundle_rotations.items()}

    def get_final_rotations(self):
        return self._bundle_rotations

    def get_ordered_scaffold_nts(self):
        if np.asarray(self._scaffold_start_point).size != 2:
            nts = self.get_scaffold_nucleotides()

            # Place the scaffold break point on helix 0 by default.
            scaffold_xovers, _ = self.get_internal_crossovers_on_helix(0)
            potential_options = np.where(nts[0])[0]

            for xo in scaffold_xovers:
                _, position = xo
                min_dist = self.staple_args.min_dist_between_xovers
                start = position - min_dist
                end = position + min_dist

                to_remove = np.arange(start, end)
                potential_options = potential_options[~np.isin(potential_options, to_remove)]

            selected_nt = potential_options[int(len(potential_options) // 2)]
            self.set_scaffold_start_point([0, selected_nt])
        return _get_ordered_scaffold_nts(self)

    def get_ordered_staple_nts(self):
        return _get_ordered_staple_nts(self)

    def get_nearest_scaffold_crossover_index(self, helix_from, helix_to, idx):
        return _get_nearest_scaffold_crossover_index(self, helix_from, helix_to, idx)

    def get_all_scaffold_crossover_options(self, helix_from, helix_to):
        return _get_all_scaffold_crossover_options(self, helix_from, helix_to)

    def get_all_valid_scaffold_crossover_options(self, helix_from, helix_to):
        return _get_all_valid_scaffold_crossover_options(self, helix_from, helix_to)

    def replace_scaffold_positions(self, h1, h2, oldnt1, oldnt2, newnt1, newnt2):
        return _replace_scaffold_positions(self, h1, h2, oldnt1, oldnt2, newnt1, newnt2)

    def get_all_staple_crossover_options(self, helix_from, helix_to):
        return _get_all_staple_crossover_options(self, helix_from, helix_to)

    def populate_scaffold_crossovers(self, scaffold_path: list, **kwargs):
        from hado.core.automation.model.scaffold_crossover_decoder import ScaffoldCrossoverDecoder

        return ScaffoldCrossoverDecoder(self).populate_scaffold_crossovers(scaffold_path, **kwargs)

    def _distance_apart(self, id1, id2):
        """ Calculates euclidean distance apart from two helices belonging to same helix bundle """
        if self._helix_to_bundle[id1] != self._helix_to_bundle[id2]:
            raise ValueError('ERROR: Indices must be on same helix bundle')
        gl1, gl2 = self._grid_locations[id1], self._grid_locations[id2]
        dist = np.linalg.norm(gl2 - gl1)
        return dist, self._helix_spacing


def initialize_base_design(geometry: Geometry,
                           scaffold_args: ScaffoldArgs,
                           staple_args: StapleArgs,
                           edge_xsect_definitions: dict,
                           **kwargs
                           ) -> HadoNucleotideModel:
    """ Initializes a base HadoNucleotideModel design given the geometry, scaffold args, and edge cross-section
    definitions found by the scaffold routing algorithm. Upon passing these arguments, the nucleotide-level model
    is constructed internally. This process is down in the HadoNucleotideModel._initialize_model() method.
    """
    max_cross_section_size_override = kwargs.get('max_cross_section_size_override', False)
    return HadoNucleotideModel(
        geometry,
        scaffold_args,
        staple_args,
        edge_xsect_definitions,
        max_cross_section_size_override=max_cross_section_size_override,
        diagnostics=kwargs.get('diagnostics'),
    )


