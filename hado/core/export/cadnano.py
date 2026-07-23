import json
from pathlib import Path

import numpy as np

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.automation.pipeline.types import emit_runtime_message
from hado.core.automation.routing.honeycomb_xsect import get_honeycomb_mapping_cadnano
from hado.core.automation.routing.square_xsect import get_square_mapping_cadnano


def write_cadnano(filename_no_extension: str, design: HadoNucleotideModel,
                  verbose: bool = False, filepath: str | Path = ".", diagnostics=None) -> None:
    """
    Writes the caDNAno JSON file using the caDNAno writer object. Note that currently only honeycomb and square grids
    are supported for export.

    :param filepath: Filepath to write files to (does not include filename / extension)
    :type filepath: str | Path

    :param filename_no_extension: Name of the .json file that will be created
    :type filename_no_extension: str

    :param design: Resultant design model from the hado automation pipeline.
    :type design: HadoNucleotideModel

    :param verbose: If True, prints detailed logs during execution. Defaults to False.
    :type verbose: bool, optional

    :rtype: None
    """
    temp = CaDNAnoWriter(
        filename_no_extension,
        design,
        verbose,
        diagnostics=diagnostics,
    )
    temp.write(filename_no_extension, directory=filepath)


def get_cadnano_json(filename_no_extension: str, design: HadoNucleotideModel, verbose: bool = False,
                     diagnostics=None) -> dict:
    """Return caDNAno v2 JSON data for a HADO nucleotide model."""
    temp = CaDNAnoWriter(
        filename_no_extension,
        design,
        verbose,
        diagnostics=diagnostics,
    )
    return temp.get_json_data()


def _transform_3D_positions_to_grid_locations(design: HadoNucleotideModel, lattice_type, precision,
                                             allow_custom_cross_section: bool = False):
    """ Transforms a list of (X, Y) locations into the proper 2D grid elements (row, col) when exporting to caDNAno. """
    # This is super hacky and I need to fix it later on if I expand to different grid types but
    # simply speaking this code is from an old project and it was easier to just fit in here than refractor it properly
    all_locations = design.get_helix_bundle_grid_locations()
    helix_to_bundle = design.get_helix_to_bundle()
    split_indices = np.where(np.diff(helix_to_bundle))[0] + 1
    locations = np.split(all_locations, split_indices)
    spacing_length = design.get_spacing_distance()
    if _design_uses_custom_cross_section(design):
        if not allow_custom_cross_section:
            raise ValueError('ERROR: caDNAno export is disabled for custom cross-sections.')
        return _get_abstract_mapping_cadnano(locations, precision)

    lattice_key = lattice_type.lower()
    if lattice_key == 'dna_honeycomb':
        return get_honeycomb_mapping_cadnano(spacing_length, locations, precision)
    if lattice_key == 'dna_square':
        return get_square_mapping_cadnano(spacing_length, locations, precision)
    raise ValueError('ERROR: Only dna_honeycomb and dna_square exports are currently configured.')


def _design_uses_custom_cross_section(design: HadoNucleotideModel) -> bool:
    scaffold_args = getattr(design, 'scaffold_args', None)
    return bool(scaffold_args is not None and scaffold_args.has_custom_cross_section())


def _get_abstract_mapping_cadnano(grids_to_fill, precision, col_buffer=3, row_buffer=2):
    max_column = 30
    cur_column, cur_row, local_max_height = 0, 0, 0
    grid_data = {}

    for bundle_index, grid in enumerate(grids_to_fill):
        helix_count = len(grid)
        width = max(1, min(max_column, int(np.ceil(np.sqrt(helix_count)))))
        height = int(np.ceil(helix_count / width))
        width_needed = width + col_buffer
        height_needed = height + row_buffer

        if cur_column + width_needed <= max_column:
            lower_bound = (cur_column, cur_row)
            cur_column += width_needed + 1
            local_max_height = max(local_max_height, height_needed)
        else:
            cur_row += local_max_height + row_buffer
            local_max_height = height_needed
            lower_bound = (0, cur_row)
            cur_column = width_needed + 1

        bundle_mapping = {}
        for helix_index, physical_point in enumerate(grid):
            local_col = helix_index % width
            local_row = helix_index // width
            key = tuple(np.round(physical_point, decimals=precision))
            bundle_mapping[key] = (lower_bound[0] + local_col, lower_bound[1] + local_row)
        grid_data[bundle_index] = bundle_mapping

    return grid_data

class CaDNAnoWriter:
    def __init__(self, design_name: str, design: HadoNucleotideModel, verbose: bool, ignore_breaks: bool = False,
                 convert_scaf_break: bool = False, diagnostics=None, allow_custom_cross_section: bool = False):
        self.design = design
        self._cadnano_export_precision = 5   # Just for writing tuples in dicts,
        self.even_idx, self.odd_idx = 0, 1   # For labelling helix indexes when doing cadnano export
        self._ignore_breaks = ignore_breaks  # Mainly for debugging export
        self._convert_scaf_break = convert_scaf_break
        self.staple_colors = design.get_staple_color_palette()
        self._allow_custom_cross_section = allow_custom_cross_section

        # Write the json that will be used in the write function
        vstrands, design_to_cadnano_num, cadnano_to_design_num = self._get_vstrands()
        self.global_json = {
            'name': design_name,
            'vstrands': vstrands,
        }
        self.verbose = verbose
        self.diagnostics = diagnostics
        self.design_to_cadnano_num = design_to_cadnano_num
        self.cadnano_to_design_num = cadnano_to_design_num

    def write(self, filename_no_extension: str, directory: str | Path = ".") -> None:
        """Write the generated caDNAno JSON file to `directory`."""
        directory = Path(directory) if directory else Path.cwd()
        try:
            write_path = (Path(directory) / f"{filename_no_extension}.json").resolve()
            write_path.parent.mkdir(parents=True, exist_ok=True)
            with open(write_path, 'w') as f:
                json.dump(self.global_json, f, indent=4)
            emit_runtime_message(
                f"Successfully wrote file to: {write_path}",
                diagnostics=self.diagnostics,
                verbose=self.verbose,
            )
        except Exception as e:
            emit_runtime_message(
                f"Failed to write JSON file: {e}",
                diagnostics=self.diagnostics,
                verbose=self.verbose,
                warning=True,
            )

    def get_json_data(self) -> dict:
        """Return the generated caDNAno JSON dictionary."""
        return self.global_json

    @staticmethod
    def _connect_3to5(vh1, three_prime, vh2, five_prime, sender_direction: bool, receiver_direction: bool):
        """Connect a sender 3-prime nucleotide to a receiver 5-prime nucleotide."""
        vh1_num, vh2_num = vh1['num'], vh2['num']
        sender_previous = three_prime - 1 if sender_direction else three_prime + 1
        receiver_next = five_prime + 1 if receiver_direction else five_prime - 1
        return (
            [vh1_num, sender_previous, vh2_num, five_prime],
            [vh1_num, three_prime, vh2_num, receiver_next],
        )

    def _get_vstrands(self):
        """ Converts HadoNucleotideModel -> honeycomb grid places + scaffolds """
        # vstrand = helix_i : {"stap_colors: [], "num": n, "scafLoop": [], "stap": Staples, "skip": [0 ....],
        #                      "scaf": Scaffold, "stapLoop": [], "col": c, "row": r, "loop": [0 ...]}
        # length of lists determines if square (multiple of 32) or honeycomb (multiple of 21)
        # Overall, skip, loop,  ... are [0] * N whre N is the max_offset (multiple of 32 / 21)
        # scafLoop and stapLoop are both unused / empty lists
        helix_mapping, offset = self._get_mapping()
        if (offset % 21 != 0) and (offset % 32 != 0):
            raise ValueError('ERROR: HadoNucleotideModel scaffold nucleotides should be either a multiple of 21 or 32')

        vstrands = []
        cadnano_to_design, design_to_cadnano = {}, {}  # For mapping of helix_num to actual row in self.design

        grid_locations = self.design.get_helix_bundle_grid_locations()
        helix_to_bundle = self.design.get_helix_to_bundle()
        scaffold_dirs = self.design.get_scaffold_directions()
        scaffold_nucleotides = self.design.get_scaffold_nucleotides()
        staple_nucleotides = self.design.get_staple_nucleotides()
        scaffold_crossovers = self.design.get_scaffold_crossovers()
        staple_crossovers = self.design.get_staple_crossovers()
        staple_breaks = self.design.get_staple_break_points()
        scaffold_start_point = self.design.get_scaffold_start_point()
        staple_dirs = self.design.get_staple_directions()

        for i, (grid_loc, h2b, scaf_dir, scaf_nt, stap_nt) in enumerate(zip(grid_locations, helix_to_bundle,
                                                                            scaffold_dirs, scaffold_nucleotides,
                                                                            staple_nucleotides)):
            # Grab the proper row / column from above mapping
            grids_for_bundle = helix_mapping[h2b]
            temp = tuple(np.round(grid_loc, decimals=self._cadnano_export_precision))
            col, row = grids_for_bundle[temp]
            num = self._get_helix_num(scaf_dir)
            stap_dir = not scaf_dir
            next_vstrand = {
                "stap_colors": [],
                "num": num,
                "scafLoop": [],  # Unused
                "stapLoop": [],  # Unused
                "skip": [0] * offset,
                "loop": [0] * offset,
                "col": int(col),
                "row": int(row),
                "scaf": self._draw_initial_strand(num, scaf_nt, scaf_dir),
                "stap": self._draw_initial_strand(num, stap_nt, stap_dir),
            }
            vstrands.append(next_vstrand)
            cadnano_to_design[num] = i
            design_to_cadnano[i] = num

        # After populating all vstrands, loop over & apply the crossovers to apply them easily:
        for row in scaffold_crossovers:
            h1, h2 = row[0], row[2]
            vh1, vh2 = vstrands[h1], vstrands[h2]

            # After getting new sender / receiver values, update the idx:
            if row[-1] == -1:
                three_prime = five_prime = int(row[1])  # Same nt position for internal xovers
            else:
                three_prime, five_prime = int(row[1]), int(row[3])
            new_sender, new_receiver = self._connect_3to5(
                vh1,
                three_prime,
                vh2,
                five_prime,
                bool(scaffold_dirs[h1]),
                bool(scaffold_dirs[h2]),
            )

            vstrands[h1]["scaf"][three_prime] = new_sender
            vstrands[h2]["scaf"][five_prime] = new_receiver

        # Repeat above but for staples:
        for row in staple_crossovers:
            h1, h2 = row[0], row[2]
            vh1, vh2 = vstrands[h1], vstrands[h2]

            if row[-1] == -1:
                three_prime = five_prime = int(row[1])  # Same nt position for internal xovers
            else:
                three_prime, five_prime = int(row[1]), int(row[3])
            new_sender, new_receiver = self._connect_3to5(
                vh1,
                three_prime,
                vh2,
                five_prime,
                bool(staple_dirs[h1]),
                bool(staple_dirs[h2]),
            )

            vstrands[h1]["stap"][three_prime] = new_sender
            vstrands[h2]["stap"][five_prime] = new_receiver


        # Add in the break points if specified:
        # 5' format: [-1, -1, helix_num, nct+1]
        # 3' format: [helix_num, nct-1, -1, -1]
        if not self._ignore_breaks:
            for staple_break in staple_breaks:
                h1, nt1, nt2 = int(staple_break[0]), int(staple_break[1]), int(staple_break[2])
                if nt2 == -1:
                    nt1 = nt1
                else:
                    nt1, nt2 = min(nt1, nt2), max(nt1, nt2)
                cadnano_helix_num = vstrands[h1]['num']

                if staple_dirs[h1]:
                    if nt2 == -1:
                        # When nt2 is -1, we want to loko at the +-1 values of nt1 to determine if this is a 3' or
                        # 5' end (otherwise nt2 would have been specified)
                        ntm1 = vstrands[h1]["stap"][nt1-1]

                        # If the nt to the left of our break point on an odd-running staple is part of a crossover,
                        # then this is a 3' end
                        if ntm1[0] != cadnano_helix_num or ntm1[2] != cadnano_helix_num:
                            vstrands[h1]["stap"][nt1] = [-1, -1, cadnano_helix_num, nt1 + 1]
                        else:
                            # Otherwise this is a 3' end
                            vstrands[h1]["stap"][nt1] = [cadnano_helix_num, nt1 - 1, -1, -1]
                    else:
                        vstrands[h1]["stap"][nt1] = [cadnano_helix_num, nt1 - 1, -1, -1]
                        vstrands[h1]["stap"][nt2] = [-1, -1, cadnano_helix_num, nt2 + 1]
                else:
                    # nt1 is 3' and nt2 is 5'
                    if nt2 == -1:
                        ntm1 = vstrands[h1]["stap"][nt1 - 1]
                        # If the nt to the left of our break point on an odd-running staple is part of a crossover,
                        # then this is a 3' end
                        if ntm1[0] != cadnano_helix_num or ntm1[2] != cadnano_helix_num:
                            vstrands[h1]["stap"][nt1] = [cadnano_helix_num, nt1 + 1, -1, -1]
                        else:
                            # Otherwise this is a 5' end
                            vstrands[h1]["stap"][nt1] = [-1, -1, cadnano_helix_num, nt1 - 1]
                    else:
                        vstrands[h1]["stap"][nt1] = [-1, -1, cadnano_helix_num, nt1 - 1]
                        vstrands[h1]["stap"][nt2] = [cadnano_helix_num, nt2 + 1, -1, -1]


            scaf_start_point = scaffold_start_point
            if len(scaf_start_point) > 0:
                design_h1, nt1 = int(scaf_start_point[0]), int(scaf_start_point[1])
                h1 = design_to_cadnano[design_h1]
                nt2 = nt1 + 1

                if self._convert_scaf_break:
                    cor_vstrand_idx = h1
                    cadnano_helix_num = vstrands[h1]['num']
                else:
                    cor_vstrand_idx = None
                    cadnano_helix_num = h1
                    for ict, i in enumerate(vstrands):
                        if i['num'] == h1:
                            cor_vstrand_idx = ict
                            break
                    if cor_vstrand_idx is None:
                        raise RuntimeError('ERROR: Unable to locate proper vstrand')

                # cadnano_helix_num = vstrands[h1]['num']
                # NOTE: Do not update the cadnano_helix_num here because the sequence_design already verifies this
                if scaffold_dirs[design_h1]:
                    vstrands[cor_vstrand_idx]["scaf"][nt1] = [cadnano_helix_num, nt1-1, -1, -1]
                    vstrands[cor_vstrand_idx]["scaf"][nt2] = [-1, -1, cadnano_helix_num, nt2+1]

                else:
                    vstrands[cor_vstrand_idx]["scaf"][nt1] = [-1, -1, cadnano_helix_num, nt1-1]
                    vstrands[cor_vstrand_idx]["scaf"][nt2] = [cadnano_helix_num, nt2+1, -1, -1]

        # Finally, add in staple colors
        def _hex_to_int(hex_color):
            hex_color = hex_color.lstrip('#')
            return int(hex_color, 16)

        if self.staple_colors is not None:
            for vs, strand in enumerate(vstrands):
                bundle_color = self.staple_colors[helix_to_bundle[vs]]
                new_stap_colors = []
                for nt, s in enumerate(strand['stap']):
                    if s == [-1, -1, -1, -1]:
                        continue
                    elif s[0] == -1 and s[1] == -1:
                        new_stap_colors.append([int(nt), _hex_to_int(bundle_color)])
                strand['stap_colors'] = new_stap_colors

        return vstrands, design_to_cadnano, cadnano_to_design

    def _get_mapping(self):
        """ Gets the mapping for honeycomb / square grid. This step is needed because currently, each grid_position
        is a local coordinate specific to its HelixGroup. Thus, we must allocate these helices in 2D grid space for
        a proper caDNAno export.
        """
        scaffold_nts = self.design.get_scaffold_nucleotides()
        num_helices, length = scaffold_nts.shape

        # Create the scaffold geometry (currently using just 1 constant geometry for all HelixBundle groups
        transformed_helix_groups = _transform_3D_positions_to_grid_locations(self.design,
                                                                             self.design.scaffold_args.lattice_type,
                                                                             self._cadnano_export_precision,
                                                                             self._allow_custom_cross_section,
                                                                             )
        return transformed_helix_groups, length

    def _get_helix_num(self, runs_53):
        """ Uses the provided direction to return the cadnano export Helix number """
        if runs_53:
            temp = self.even_idx
            self.even_idx += 2
        else:
            temp = self.odd_idx
            self.odd_idx += 2
        return temp

    @staticmethod
    def _draw_even(helix_num, nts):
        """ Draws a helix that runs 5 -> 3 using the provided boolean of nts """
        helix_nts = []
        first_nt_found, last_nt_found = False, False
        for nct, n in enumerate(nts):
            if n:
                # 5' for even-run
                if not first_nt_found:
                    # First nt for even running helix will look like [-1, -1, helix_num, nct+1]
                    nt = [-1, -1, helix_num, nct+1]
                    helix_nts.append(nt)
                    first_nt_found = True

                # 3' for even-run
                elif not nts[nct+1] and not last_nt_found:
                    # If the next scaf_nt is a False value, we have found the last_nt
                    # The will look like [helix_num, nct-1, -1, -1] for the nct'th nucleotide
                    nt = [helix_num, nct-1, -1, -1]
                    helix_nts.append(nt)
                    last_nt_found = True  # Set this flag to prevent more writing

                # Otherwise we are connecting along the same helix, this:
                else:
                    helix_nts.append([helix_num, nct-1, helix_num, nct+1])

            else:
                helix_nts.append([-1, -1, -1, -1])

        return helix_nts

    @staticmethod
    def _draw_odd(helix_num, nts):
        """ Draws a helix that runs 3 -> 5 using the provided boolean of nts """
        helix_nts = []
        first_nt_found, last_nt_found = False, False
        for nct, n in enumerate(nts):
            if n:
                # 3' for odd-run is first nt found
                if not first_nt_found:
                    nt = [helix_num, nct + 1, -1, -1]
                    helix_nts.append(nt)
                    first_nt_found = True

                # 5' is the last nt found
                elif not nts[nct + 1] and not last_nt_found:
                    nt = [-1, -1, helix_num, nct-1]
                    helix_nts.append(nt)
                    last_nt_found = True  # Set this flag to prevent more writing

                # Otherwise we are connecting along the same helix:
                else:
                    helix_nts.append([helix_num, nct + 1, helix_num, nct - 1])
            else:
                helix_nts.append([-1, -1, -1, -1])

        return helix_nts

    def _draw_initial_strand(self, helix_num, nts, direction,):
        """ Draws an initial strand that has no crossovers for a helix of offset_length """
        if direction:
            helix_no_connections = self._draw_even(helix_num, nts)
        else:
            helix_no_connections = self._draw_odd(helix_num, nts)
        return helix_no_connections
