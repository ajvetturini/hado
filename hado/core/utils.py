from pathlib import Path
import colorsys

import numpy as np
import warnings

from hado.app.scaffold_geometry import get_scaffold_sequence

## GLOBAL PARAMETERS ##
MIN_ALLOWABLE_EDGE_LENGTH_NTS = 38  # Based on my own experimentation, shorter edges may have issues post-mitering
MIN_ALLOWABLE_EDGE_LENGTH_NM = MIN_ALLOWABLE_EDGE_LENGTH_NTS * 0.34  # in nm, assumes BDNA

# Auto-break can become really expensive for large designs, so this is to prevent performing this task unless
# user sets the kwarg flaw of "override_staple_autobreak_limit"
MAX_NUM_NTS_FOR_HADO_AUTOBREAK = 50000

# Default values restricting the autostaple constraint sweep function to ensure operations don't run for too long
# These likely will NOT impact you unless you are using the autostaple.sweep_staple_args function
MIN_TARGET, MAX_TARGET = 40, 50
MIN_RUN_POST_XOVER, MAX_RUN_POST_XOVER = 1, 7
MIN_POST_BUNDLE, MAX_POST_BUNDLE = 1, 7
MIN_DIST_BETWEEN, MAX_DIST_BETWEEN = 1, 7
MAX_DIFFERENCE, MAX_DIFFERENCE_TARGET = 3, 10

# Edge cross section definition can be overly-long for very large cross-sections. DNA origami is going to be limited
# to smaller cross-section sizes (e.g., 30HB) because the scaffold "gets used up" very quickly. The algorithms
# used for this are thus not super efficient (so for larger cross-section sizes, the algorithms will need to be
# optimized). I set this rather arbitrarly below based on waiting for ~30 seconds on my computer (m1 pro macbook)
MAX_NODES_FOR_CROSS_SECTION_SEARCH = 50

WCF_COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}  # watson-crick-franklin

class ScaffoldArgs:
    """
    ScaffoldArgs contains information about the geometry of the DNA scaffold used in making hollowframe nanostructures.

    The default values for this class presume honeycomb DNA origami (which is the design paradigm of hollowframe
    DNA origami nanostructures). Deviating from the default values specified below may lead to errant results.

    :param grid_style: The style of the internal grid lattice (e.g., 'hollow').
        Defaults to 'hollow' and currently only `hollow` is supported.
    :type grid_style: str, optional

    :param lattice_type: The type of scaffold + lattice configuration (e.g., 'dna_honeycomb').
        Defaults to 'dna_honeycomb'. Use 'dna_square' for square-grid caDNAno-style designs.
    :type lattice_type: str, optional

    :param overfill_or_underfill: strategy for filling.
        **Note:** Not used in hollowframe; applies only to odd-N per helix wireframe.
        Defaults to 'underfill'.
    :type overfill_or_underfill: str, optional
    """

    def __init__(self,
                 lattice_type: str = 'dna_honeycomb',
                 grid_style: str = 'hollow',
                 scaffold_sequence: str = 'm13',
                 overfill_or_underfill: str = 'overfill',
                 custom_cross_section=None,
                 cross_section_generator=None,
                 ):

        scaffold_name = scaffold_sequence.lower()
        if scaffold_name in ['m13', 'p7308', 'p7560', 'p7704', 'p8064', 'p8100', 'p8634']:
            seq = get_scaffold_sequence(scaffold_name)
            self.scaffold_sequence = seq
            self._original_scaf = scaffold_name

        else:
            scaffold_sequence = scaffold_sequence.upper()
            valid = set(scaffold_sequence).issubset({'A', 'C', 'G', 'T'})
            if not valid:
                raise ValueError('ERROR: Invalid custom scaffold sequence, must contain A C G or T only.')
            self.scaffold_sequence = scaffold_sequence
            self._original_scaf = None

        self.min_edge_length_in_bp = MIN_ALLOWABLE_EDGE_LENGTH_NTS
        self.overfill_or_underfill = overfill_or_underfill.lower()
        self.N_per_edge_threshold = 2 if self.overfill_or_underfill == 'underfill' else 1
        self.grid_style = grid_style.lower()
        self.lattice_type = lattice_type.lower()
        self.custom_cross_section = None
        self.cross_section_generator = None
        if custom_cross_section is not None:
            self.set_custom_cross_section(custom_cross_section)
        if cross_section_generator is not None:
            self.set_cross_section_generator(cross_section_generator)
        if self.overfill_or_underfill not in ["overfill", "underfill"]:
            raise ValueError("ERROR: Args.overfill_or_underfill can only be set to `overfill` or `underfilll`")
        if self.grid_style != "hollow":
            raise ValueError("ERROR: Only hollow-frame structures supported as of now.")
        if self.lattice_type not in ["dna_honeycomb", "dna_square"]:
            raise ValueError("ERROR: Only `dna_honeycomb` and `dna_square` grids are supported as of now.")

    def to_dict(self):
        """Return constructor-compatible scaffold settings for JSON serialization."""
        data = dict(self.__dict__)
        del data['N_per_edge_threshold']
        del data['min_edge_length_in_bp']

        if self._original_scaf is not None:
            data['scaffold_sequence'] = self._original_scaf

        del data['_original_scaf']
        if self.cross_section_generator is not None:
            data.pop('cross_section_generator', None)
        return data

    def set_custom_cross_section(self, custom_cross_section) -> None:
        from hado.core.automation.routing.cross_sections import serialize_custom_cross_section

        self.custom_cross_section = serialize_custom_cross_section(custom_cross_section)
        self.cross_section_generator = None

    def set_cross_section_generator(self, cross_section_generator) -> None:
        if not callable(cross_section_generator):
            raise TypeError('ERROR: cross_section_generator must be callable.')
        self.cross_section_generator = cross_section_generator
        self.custom_cross_section = None

    def clear_custom_cross_section(self) -> None:
        self.custom_cross_section = None
        self.cross_section_generator = None

    def has_custom_cross_section(self) -> bool:
        return self.custom_cross_section is not None or self.cross_section_generator is not None


class StapleArgs:
    """
    StapleArgs contains various parameters that influence the staple sequence design automation functions. These
    parameters are used to ensure things like staple and scaffold crossovers are not too close to each other and
    that the staple lengths are similarly length-ed post-autobreak.

    **Note** Autostapling requires a periodic lattice configuration with scaffold and staple crossover offsets.

    :param min_length_after_break: Minimum staple length to accept as valid in # of nucleotides, defaults to 20.
    :type min_length_after_break: int, optional

    :param max_length_after_break: Maximum staple length to accept as valid in # of nucleotides, defaults to 60.
    :type max_length_after_break: int, optional

    :param only_add: Optional flag to disable stapling if set to True, defaults to False.
    :type only_add: bool, optional

    :param random_seed: Random seed value for reproducibility. Defaults to 8.
    :type random_seed: int, optional

    :param target_miter_distance: A target distance between two connected DNA helices (i.e., distance between the
        3' and 5' ends that the scaffold is traversing through). Defaults to 3.0 nanometers (nm).
    :type target_miter_distance: float, optional

    :param polyt_bulge_dist: A value used to calculate the number of ssDNA staple nucleotides based on the final
        mitered distance. Defaults to 0.60 nanometers (nm).
    :type polyt_bulge_dist: float, optional

    :param unpaired_sequence: The sequence used for ssDNA staple overhangs
        Defaults to 'T'.
    :type unpaired_sequence: str, optional

    :param max_staple_spacer_length: Complement to polyt_bulge_dist by restricting the max number of overhangs.
        Defaults to 8 nucleotides.
    :type max_staple_spacer_length: int, optional

    :param default_blunt_end_length: Number of Thymine-overhangs to place at free-ends of DNA bundless to prevent blunt
        end stacking. Defaults to 3.
    :type default_blunt_end_length: int, optional

    :param min_run_post_bundle_connection: Minimum number of nucleotides that a staple must traverse before 
        crossing-over after connecting two adjacent bundles. Defaults to 5.
    :type min_run_post_bundle_connection: int, optional

    :param min_dist_between_xovers: Minimum distance between two crossovers to prevent kinetic traps.
        Defaults to 3 nucleotides.
    :type min_dist_between_xovers: int, optional

    :param min_run_post_xover: Minimum distance that a staple must traverse before being selected as a break point
        during autostapling. Defaults to 3 nucleotides.
    :type min_run_post_xover: int, optional

    :param target_staple_length: The target length used in the dynamic programming autobreak algorithm. Overall, this
        value is used to inform break points of long staples to bound their lengths betwen [min_length_after_break,
        max_length_after_break]. Defaults to 42 nucleotides.
    :type target_staple_length: int, optional

    :param make_flush: If a hollowframe design, make all the free 3' and 5' ends at pendant vertices flush at the same
        nucleotide position. This sets all 3' and 5' ends to be at a distance of `flush_distance` from the longest
        point. Defaults to True.
    :type make_flush: bool, optional

    :param flush_distance: Within a helix bundle the scaffold crossovers at the end depend on the helix location within
        the cross-section. This flush_distance value will place the 3' and 5' ends of staples at the same position of
        flush_distance from the longest point in the bundle. This only affects hollowframe structures (not wireframes
        where each vertex has at least 2 edges). Defaults to 1 nucleotides.
    :type flush_distance: int, optional
    """

    def __init__(self,
                 min_length_after_break: int = 20,
                 max_length_after_break: int = 60,
                 only_add: bool = False,
                 random_seed: int = 8,
                 target_miter_distance: float = 2.0,  # Note: This value was set using oxDNA simulations
                 polyt_bulge_dist: float = 0.60,      # Note: This value was set using oxDNA simulations
                 unpaired_sequence: str = 'T',
                 default_blunt_end_length: int = 5,
                 max_staple_spacer_length: int = 8,
                 min_run_post_bundle_connection: int = 5,
                 min_dist_between_xovers: int = 3,
                 min_run_post_xover: int = 3,
                 target_staple_length: int = 42,
                 make_flush: bool = True,
                 flush_distance: int = 1,
                 ):

        self.min_length_after_break = min_length_after_break
        self.max_length_after_break = max_length_after_break
        self.random_seed = random_seed
        self.max_staple_spacer_length = max_staple_spacer_length
        if max_staple_spacer_length <= 0:
            raise ValueError("ERROR: Max staple space length should be be at least 0, but ideally should not be "
                             "modified from 8 as that is un-tested experimentally.")

        self.min_dist_between_xovers = min_dist_between_xovers
        self.min_run_post_xover = min_run_post_xover
        self.target_staple_length = target_staple_length

        if not 0.1 < polyt_bulge_dist <= 1.0:
            raise ValueError("ERROR: Polyt bulge distance should not be drastically modified from 0.60 nm "
                             "unless testing it.")
        self.polyt_bulge_dist = polyt_bulge_dist

        if not 0 < default_blunt_end_length <= 7:
            raise ValueError("ERROR: Blunt end should be an integer bounded [0, 7]")
        self.default_blunt_end_length = default_blunt_end_length
        self.only_add = only_add

        if not 0 < min_length_after_break < max_length_after_break:
            raise ValueError("ERROR: Min staple length should be larger than 0 but less than the max staple length.")
        if min_dist_between_xovers <= 0 or min_run_post_xover <= 0:
            raise ValueError("ERROR: Min distance between crossover and the min run post crossover values "
                             "should be larger than 0")
        if min_dist_between_xovers > 10 or min_run_post_xover > 10:
            warnings.warn(
                'Min distance between xovers / min run post xover is set quite high; did you mean to do this?',
                stacklevel=2,
            )

        if min_length_after_break < 15 or min_length_after_break > 60:
            warnings.warn(
                'Min staple length should be between (15, 60) for 25 nmole synthesis via IDT',
                stacklevel=2,
            )
        if max_length_after_break < 15 or max_length_after_break > 60:
            warnings.warn(
                'Max staple length should be between (15, 60) for 25 nmole synthesis via IDT',
                stacklevel=2,
            )
        if min_run_post_bundle_connection <= 0:
            raise ValueError("ENSURE min_run_post_bundle_connection is set larger than 0")

        self.min_run_post_bundle_connection = min_run_post_bundle_connection
        self.target_miter_distance = target_miter_distance
        if not 0 < self.target_miter_distance <= 5:
            raise ValueError("ERROR: Miter threshold should not be set too large (> 5 nm) or too small (i.e, < 1 nm) "
                             "or else design automation may not lead to satisfactory design.")

        self.make_flush = make_flush
        if not 0 < flush_distance <= 5:
            raise ValueError("ERROR: Flush distance should be larger than 0 and less than 5 (5 is rather arbitrary, "
                             "but this value should not be set high")
        self.flush_distance = flush_distance

        if unpaired_sequence.upper() not in ["A", "C", "G", "T"]:
            raise ValueError("ERROR: Unpaired sequence can only be A C G or T (default: T)")
        self.unpaired_sequence = unpaired_sequence

    def to_dict(self):
        """Return constructor-compatible staple design settings for JSON serialization."""
        return dict(self.__dict__)


class Geometry:
    """
    A Geometry simply contains the vertices, edges, and desired number of DNA helices-per-edge for a given design.
    Note that the vertices MUST be specified in floating point numbers and are assumed to be in units of nanometers.

    You can also read in a mesh file (currently supports .ply and .obj files) to create a Geometry object. You could
    also specify a ".lm" file which is a simple format of vertices written as (v1, v2, v3) and edges as (vi, vj) per
    line.

    :param vertices: Vertices in [[X, Y, Z], ...] format. Units are assumed to be in nanometers (nm).
    :type vertices: list of lists (float) or np.ndarray of shape (n_vertices, 3)

    :param edges: Edges in [[vertex_index_1, vertex_index_2], ...] format. Vertex indices are 0-based and refer to
        the vertices list.
    :type edges: list of lists (int) or np.ndarray of shape (n_vertices, 2)

    :param n_per_edge: Desired number of DNA helices per edge. Can be specified as a single integer (e.g., 18) to
        apply the same number of helices to all edges, or as a list of integers (e.g., [18, 30, ...]) to specify
        different numbers of helices for each edge. If a list is provided, its length must match the number of edges.
    :type n_per_edge: int or list of ints, optional

    :param edge_thickness_nm: Desired honeycomb edge thickness in nanometers. This is interpreted as the outer
        diameter of the honeycomb ring cross-section. Provide either n_per_edge or edge_thickness_nm, not both.
    :type edge_thickness_nm: float or list of floats, optional
    """

    def __init__(self,
                 vertices: list,
                 edges: list,
                 n_per_edge: int | list | None = None,
                 edge_thickness_nm: float | list | None = None,
                 ):

        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        if isinstance(edges, np.ndarray):
            edges = edges.tolist()
        if isinstance(n_per_edge, np.ndarray):
            n_per_edge = n_per_edge.tolist()
        if isinstance(edge_thickness_nm, np.ndarray):
            edge_thickness_nm = edge_thickness_nm.tolist()

        if n_per_edge is None and edge_thickness_nm is None:
            raise TypeError("Geometry requires either n_per_edge or edge_thickness_nm")
        if n_per_edge is not None and edge_thickness_nm is not None:
            raise ValueError("Specify either n_per_edge or edge_thickness_nm, not both")

        self.vertices = vertices
        self.edges = edges
        self.edge_thickness_nm = None
        self.edge_thickness_actual_nm = None
        if edge_thickness_nm is None:
            self.n_per_edge = self._standardize_n_per_edge(n_per_edge)
        else:
            self.edge_thickness_nm = self._standardize_edge_thickness_nm(edge_thickness_nm)
            self.n_per_edge, self.edge_thickness_actual_nm = self._n_per_edge_from_edge_thickness(
                self.edge_thickness_nm
            )
        self.edge_lengths_nm = self._validate_input()

    def to_dict(self):
        """Return constructor-compatible geometry data for JSON serialization."""
        data = {
            "vertices": self.vertices,
            "edges": self.edges,
        }
        if self.edge_thickness_nm is None:
            data["n_per_edge"] = self.n_per_edge
        else:
            data["edge_thickness_nm"] = self.edge_thickness_nm
        return data

    def get_vertex_position(self, idx: int):
        return np.array(self.vertices[idx])

    def _validate_input(self) -> list:
        """Validate geometry shape, edge references, connectivity, and edge widths."""
        if len(self.vertices) == 0:
            raise ValueError("No vertices provided")
        if len(self.edges) == 0:
            raise ValueError("No edges provided")
        if not all(len(v) == 3 for v in self.vertices):
            raise ValueError("ERROR: Each vertex must be a 3D coordinate (x, y, z)")

        edge_lengths = []
        for e in self.edges:
            if len(e) != 2:
                raise ValueError("ERROR: Each edge must connect exactly two vertices")
            i, j = int(e[0]), int(e[1])
            if i < 0 or i >= len(self.vertices) or j < 0 or j >= len(self.vertices):
                raise ValueError("ERROR: Edge references invalid vertex index")
            el = np.linalg.norm(np.array(self.vertices[i]) - np.array(self.vertices[j]))
            edge_lengths.append(float(el))

        if not self._is_single_connected_component():
            raise ValueError("ERROR: Input geometry must be a single connected component")

        for n in self.n_per_edge:
            if not isinstance(n, int) or n <= 1:
                raise ValueError("ERROR: n_per_edge must be an integer of least 2 for hado")

        return edge_lengths

    @staticmethod
    def _is_number(value) -> bool:
        return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)

    def _standardize_n_per_edge(self, n_per_edge: int | list) -> list:
        """Normalize scalar or per-edge helix counts to a list of integers."""
        if isinstance(n_per_edge, np.ndarray):
            n_per_edge = n_per_edge.tolist()

        if isinstance(n_per_edge, (int, np.integer)) and not isinstance(n_per_edge, bool):
            return [int(n_per_edge)] * len(self.edges)

        if not isinstance(n_per_edge, (list, tuple)):
            raise TypeError("n_per_edge must be an int or a list")

        if len(n_per_edge) != len(self.edges):
            raise ValueError(
                f"n_per_edge length ({len(n_per_edge)}) must match number of edges ({len(self.edges)})"
            )

        return [int(i) for i in list(n_per_edge)]

    def _standardize_edge_thickness_nm(self, edge_thickness_nm: float | list) -> list:
        """Normalize scalar or per-edge thickness values to a list of floats."""
        if isinstance(edge_thickness_nm, np.ndarray):
            edge_thickness_nm = edge_thickness_nm.tolist()

        if self._is_number(edge_thickness_nm):
            values = [float(edge_thickness_nm)] * len(self.edges)
        else:
            if not isinstance(edge_thickness_nm, (list, tuple)):
                raise TypeError("edge_thickness_nm must be a number or a list")
            if len(edge_thickness_nm) != len(self.edges):
                raise ValueError(
                    f"edge_thickness_nm length ({len(edge_thickness_nm)}) must match number of edges "
                    f"({len(self.edges)})"
                )
            values = [float(i) for i in list(edge_thickness_nm)]

        if any(i <= 0 for i in values):
            raise ValueError("edge_thickness_nm must contain positive values")
        return values

    def _n_per_edge_from_edge_thickness(self, edge_thickness_nm: list) -> tuple[list, list]:
        """Resolve requested honeycomb diameters to the closest even helix counts."""
        from hado.core.automation.routing.honeycomb_xsect import select_honeycomb_ring_by_diameter
        from hado.core.automation.routing.lattice import get_lattice_config

        lattice = get_lattice_config("dna_honeycomb")
        selections = {}
        n_per_edge = []
        actual_thickness = []
        for thickness in edge_thickness_nm:
            key = round(float(thickness), 6)
            if key not in selections:
                selection = select_honeycomb_ring_by_diameter(
                    key,
                    lattice.helix_spacing,
                    helix_diameter=lattice.diameter,
                )
                selections[key] = (selection["n_per_edge"], selection["actual_diameter"])
            selected_n, selected_thickness = selections[key]
            n_per_edge.append(selected_n)
            actual_thickness.append(selected_thickness)
        return n_per_edge, actual_thickness

    def scale(self, scale: float):
        """ Simple scaling of vertices in the design """
        assert scale > 0, "ERROR: Scale must be larger than 0"
        verts = np.array(self.vertices, dtype=float)
        centroid = np.mean(verts, axis=0)
        new_vertices = (verts - centroid) * scale + centroid
        self.vertices = new_vertices.tolist()
        self.edge_lengths_nm = self._validate_input()

    def set_n_per_edge(self, n_per_edge: int | list) -> None:
        """Update helix counts and re-run geometry validation."""
        self.edge_thickness_nm = None
        self.edge_thickness_actual_nm = None
        self.n_per_edge = self._standardize_n_per_edge(n_per_edge)
        self.edge_lengths_nm = self._validate_input()

    def set_edge_thickness_nm(self, edge_thickness_nm: float | list) -> None:
        """Update target honeycomb edge thicknesses and re-run geometry validation."""
        self.edge_thickness_nm = self._standardize_edge_thickness_nm(edge_thickness_nm)
        self.n_per_edge, self.edge_thickness_actual_nm = self._n_per_edge_from_edge_thickness(
            self.edge_thickness_nm
        )
        self.edge_lengths_nm = self._validate_input()

    @classmethod
    def read_in_mesh(cls, filepath: str | Path, n_per_edge: int = 2, edge_thickness_nm: float | list | None = None):
        """Reads in a mesh file (ply or obj currently) and creates the Geometry object."""
        suffix = Path(filepath).suffix.lower()
        if suffix == '.ply':
            v, e = cls._parse_ply(filepath)
        elif suffix == '.obj':
            v, e = cls._parse_obj(filepath)
        elif suffix == '.lm':
            # lm is a simple format of vertices written as (v1, v2, v3) and edges as (vi, vj) per line
            v, e = cls._parse_lm(filepath)
        else:
            raise ValueError(f"Unsupported filetype: {suffix}. Valid files are `.ply`, `.obj`, or '.lm' files.")

        if edge_thickness_nm is not None:
            return cls(v, e, edge_thickness_nm=edge_thickness_nm)
        return cls(v, e, n_per_edge=n_per_edge)

    @staticmethod
    def _parse_obj(filepath: str | Path):
        """Parse vertices and edges from obj file (i.e., the v, f, and l records)."""
        vertices = []
        edges = set()

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                prefix = parts[0].lower()

                if prefix == 'v':
                    vertices.append(tuple(map(float, parts[1:4])))

                elif prefix == 'f':
                    face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]

                    for i in range(len(face_indices)):
                        v1 = face_indices[i]
                        v2 = face_indices[(i + 1) % len(face_indices)]
                        edges.add(tuple(sorted((v1, v2))))

                elif prefix == 'l':
                    line_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                    for i in range(len(line_indices) - 1):
                        edges.add(tuple(sorted((line_indices[i], line_indices[i + 1]))))

        return vertices, list(edges)

    @staticmethod
    def _parse_ply(filepath: str | Path):
        """Parse vertices and face-derived edges from an ASCII PLY file."""
        def _convert_face_to_edges(face) -> list:
            # Strip the first element which just denotes number of vertices in face:
            nf = face[1:]
            _edges = []
            for _i in range(0, len(nf)):
                if _i == len(nf) - 1:
                    _edges.append((min(nf[_i], nf[0]), max(nf[0], nf[_i])))
                else:
                    _edges.append((min(nf[_i], nf[_i + 1]), max(nf[_i], nf[_i + 1])))
            return _edges

        vertices, edges = [], []
        head_info = True
        ct = 0
        with open(filepath, 'r') as f:
            for row in f.readlines():
                if not head_info:
                    row_data = tuple(map(float, row.split()))
                    if len(row_data) == 3:
                        vertices.append(row_data)
                        ct += 1
                    else:
                        row_data = tuple(map(float, row.split()))
                        unique_edges = _convert_face_to_edges(row_data)
                        for i in unique_edges:
                            if i not in edges:
                                edges.append((int(i[0]), int(i[1])))
                if 'end_header' in row:
                    head_info = False

        return vertices, edges

    @staticmethod
    def _parse_lm(filepath: str | Path):
        """ Read in very simply files of vertices and edges instead of relying on polyhedral / convex meshes """
        vertices = []
        edges = []
        ct = 0
        with open(filepath, 'r') as f:
            for row in f.readlines():
                try:
                    row_data = tuple(map(float, row.strip().split(' ')))
                except ValueError:
                    continue

                if len(row_data) == 3:
                    vertices.append(tuple(map(float, row.strip().split(' '))))
                    ct += 1
                else:
                    edge = tuple(map(int, row.strip().split(' ')))  # We want integers here
                    if edge not in edges:
                        edges.append(edge)
        return vertices, edges

    def _is_single_connected_component(self) -> bool:
        """Return True only when every vertex belongs to one connected component."""
        parent = list(range(len(self.vertices)))
        all_vertices_in_edges = set([v for edge in self.edges for v in edge])
        if len(all_vertices_in_edges) != len(self.vertices):
            return False

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x, y):
            root_x = _find(x)
            root_y = _find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        for vi, vj in self.edges:
            _union(int(vi), int(vj))

        root_set = set(_find(i) for i in range(len(self.vertices)))
        return len(root_set) == 1

def get_color_palette(n_colors):
    """Return n_colors color-blind-friendly hex colors for staple exports."""
    default_palette = [
        "#CC79A7",
        "#E69F00",
        "#009E73",
        "#56B4E9",
        "#000000",
        "#F0E442",
        "#D55E00",
        "#0072B2",
    ]

    if n_colors <= len(default_palette):
        return default_palette[:n_colors]

    # Otherwise, genereate spaced HSL colors
    colors = []
    for i in range(n_colors):
        h = i / n_colors  # evenly spaced hues
        s = 0.65  # moderate saturation
        l = 0.55  # safe mid-lightness
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        colors.append("#{0:02X}{1:02X}{2:02X}".format(
            int(r * 255), int(g * 255), int(b * 255)
        ))

    return colors
