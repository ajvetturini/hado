from pathlib import Path
import csv
import numpy as np

from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.utils import ScaffoldArgs
from hado.core.export import CaDNAnoWriter
from hado.core.automation.pipeline.types import emit_runtime_message

NUMERIC_TO_SEQUENCE = {
    0: 'A',
    1: 'C',
    2: 'G',
    3: 'T',
}
SEQUENCE_TO_NUMERIC = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}
SEQUENCE_PAIRS_NUMERIC = np.array([3, 2, 1, 0])



def sequence_design(design: HadoNucleotideModel, scaffold_args: ScaffoldArgs, unpaired_sequence: str,
                    filepath: str | Path, filename_no_extension: str, verbose: bool = False,
                    write_output_sequences: bool = True, diagnostics=None):
    """Assign scaffold/staple sequences and optionally write the staple CSV export."""
    temp = CaDNAnoWriter(filename_no_extension, design, verbose, diagnostics=diagnostics)
    json_data = temp.get_json_data()

    start_pt = design.get_scaffold_start_point()
    if len(start_pt) == 0:
        return False, None
    start_pt = [temp.design_to_cadnano_num[start_pt[0]], start_pt[1]]  # Correct for cadnano indexing of odd / even
    scaf_seq, staple_seqs, staple_groups = _sequence(json_data, start_pt, scaffold_args, unpaired_sequence)
    vstrands = json_data['vstrands']

    if scaf_seq is not None and write_output_sequences:
        try:
            _export_staple_sequences(filename_no_extension, scaf_seq, start_pt, staple_seqs, staple_groups,
                                     filepath, vstrands)
        except Exception as e:
            # E.g., If a user already has the CSV file open then the export can't write
            emit_runtime_message(
                f'ERROR: Unable to export sequences due to error {e}',
                diagnostics=diagnostics,
                verbose=verbose,
                warning=True,
            )

        all_strands = _format_sequences_for_export(filename_no_extension, scaf_seq, start_pt,
                                                   staple_seqs, staple_groups, vstrands)
        return True, all_strands

    elif scaf_seq is not None:
        all_strands = _format_sequences_for_export(filename_no_extension, scaf_seq, start_pt,
                                                   staple_seqs, staple_groups, vstrands)
        return True, all_strands

    else:
        # scaf_seq is None if there are too many scaffold nucleotides required for user-provided sequence
        # thus simply return False but still return a tuple of metadata for printing out results in GUI:
        # NOTE: Here staple_seqs is just a list of the length of the staple (since scaffold was too short)
        return False, staple_seqs


def _preprocess_input_data(data: dict):
    """ Converts a cadnano to an array of the sequences and also stores the scaffold routing path so that we know how
    the scaffold rotation will update the resultant staple sequences.
    """
    # This is from a previous project, so I use the cadnano file to script the strand export
    vstrands = data.get('vstrands', None)
    vstrands = sorted(vstrands, key=lambda x: x['num'])

    all_scaf_nts, all_stap_nts = [], []
    total_scaf_nts = 0
    for v in vstrands:
        scaf = np.array(v.get('scaf'))
        stap = np.array(v.get('stap'))
        scaf_nts = np.where(np.any(scaf != -1, axis=1), 1, -1)

        # We use a "4" label for staples on purpose because there may be un-paired staple nucleotides during sequencing
        # from which we will use a standard nucleotide
        stap_nts = np.where(np.any(stap != -1, axis=1), 4, -1)
        all_scaf_nts.append(scaf_nts)
        all_stap_nts.append(stap_nts)
        scaf_nts_in_vstrand = (scaf_nts == 1).sum()
        total_scaf_nts += scaf_nts_in_vstrand

    all_scaf_nts = np.array(all_scaf_nts)
    all_stap_nts = np.array(all_stap_nts)

    # Because we can have even / odd mix, make map of vstrands
    vstrand_map = {}
    for vct, v in enumerate(vstrands):
        vstrand_map[v['num']] = vct  # Usually this is a 1:1 mapping (v['num'] == vct), but prevents IndexErrors

    scaf_path, scaf_map = _get_scaffold_connection(vstrands, vstrand_map, total_scaf_nts)
    all_staples, all_staple_maps = _get_staple_routes(vstrands, vstrand_map, all_stap_nts)
    if len(scaf_path) != total_scaf_nts:
        raise ValueError('ERROR: Scaffold path is ill-defined, are you sure there is only one scaffold in the input file?')

    # Return arrays of the scaffold / staple nts and the scaffold routing path such that we can easily "update"
    # the assigned sequence
    return (all_scaf_nts, all_stap_nts), scaf_map, all_staples, all_staple_maps, total_scaf_nts, vstrand_map, scaf_path


def _sequence(json_data, start_point, scaffold_args: ScaffoldArgs, unpaired_sequence):
    """Apply a scaffold sequence to caDNAno-derived scaffold and staple routes."""
    # Performs a simple sequencing task using the passed in scaffold sequence in the kwargs
    read_in_nts, scaf_map, all_staples, staple_maps, total_scaf_nts, vstrand_map, _ = _preprocess_input_data(json_data)

    # Make jsure total_scaf_nts does NOT exceed the sequence args:
    if total_scaf_nts > len(scaffold_args.scaffold_sequence):
        return None, all_staples, total_scaf_nts

    scaf_nts, stap_nts = read_in_nts
    scaffold_sequence_numeric = np.array([SEQUENCE_TO_NUMERIC[c] for c in scaffold_args.scaffold_sequence],
                                         dtype=np.int8)
    start_point = np.array(start_point)

    # Finally, sequence the scaffold from the start (0th rotation):
    sliced_sequence = _slice_sequence(0, total_scaf_nts, scaffold_sequence_numeric,
                                      len(scaffold_sequence_numeric))

    unpaired_nt_numeric = SEQUENCE_TO_NUMERIC[unpaired_sequence.upper()]

    # Convert routing map dict to numpy array
    num_rows, num_cols = scaf_nts.shape
    scaf_routing_map_arr = np.zeros((num_rows, num_cols, 2), dtype=np.int16)
    for k, v in scaf_map.items():
        scaf_routing_map_arr[vstrand_map[k[0]], k[1], 0] = vstrand_map[v[0]]
        scaf_routing_map_arr[vstrand_map[k[0]], k[1], 1] = v[1]

    # Convert list of lists of tuples
    N_max = max(len(s) for s in all_staples)
    staple_groups = []
    for staple in all_staples:
        padded = np.full((N_max, 2), -1, dtype=np.int16)

        # Convert staple using vstrand map for proper assignment:
        converted_staple = []
        for s in staple:
            converted_staple.append((vstrand_map[s[0]], s[1]))
        padded[:len(staple)] = converted_staple
        staple_groups.append(np.array(padded, dtype=np.int32))

    staple_groups = np.array(staple_groups)

    # Next, find indices where staple nt is unpaired
    for r in range(stap_nts.shape[0]):
        for c in range(stap_nts.shape[1]):
            # if a 4 remains in the stap_nt_copy then it is an unpaired nucleotide
            if stap_nts[r, c] == 4 and scaf_nts[r, c] == -1:
                stap_nts[r, c] = unpaired_nt_numeric  # Replace with unpaired nucleotide value

    # Apply the sequence to the design
    scaf_nt_local, stap_nt_local = _sequence_arrays(
        sliced_sequence, scaf_nts, stap_nts, scaf_routing_map_arr, start_point
    )
    final_staple_set = _convert_staple_array_to_groups(staple_groups, stap_nt_local)

    return sliced_sequence, final_staple_set, staple_groups


def _slice_sequence(cur_rotation: int, num_scaf_nts: int, scaffold_sequence_numeric: np.ndarray, sequence_length: int):
    """ Slices a numeric sequence based on rotation of the scaffold """
    if cur_rotation + num_scaf_nts <= sequence_length:
        sliced_sequence = scaffold_sequence_numeric[cur_rotation: cur_rotation + num_scaf_nts]
    else:
        part1 = scaffold_sequence_numeric[cur_rotation:]
        part2 = scaffold_sequence_numeric[:(cur_rotation + num_scaf_nts) % sequence_length]
        sliced_sequence = np.concatenate((part1, part2))
    return sliced_sequence


def _sequence_arrays(sliced_sequence: np.ndarray, scaf_nts: np.ndarray, stap_nts: np.ndarray,
                     scaf_route_map: np.ndarray, start_pt: np.ndarray):
    """ Sequences the scaffold / staple array using the sliced scaffold """
    current_pos_row, current_pos_col = start_pt[0], start_pt[1]

    # Start point should be incremented by 1 when using an even helix start:
    if current_pos_row % 2 == 0:
        current_pos_col += 1

    for nt_encoded in sliced_sequence:
        scaf_nts[current_pos_row, current_pos_col] = nt_encoded
        if stap_nts[current_pos_row, current_pos_col] != -1:
            stap_nts[current_pos_row, current_pos_col] = SEQUENCE_PAIRS_NUMERIC[nt_encoded]

        # Advance along the scaffold route using the pre-computed array
        next_pos = scaf_route_map[current_pos_row, current_pos_col]
        current_pos_row, current_pos_col = next_pos[0], next_pos[1]

    return scaf_nts, stap_nts


def _convert_staple_array_to_groups(staple_groups: np.ndarray, staple_nucleotide_array: np.ndarray):
    """ Converts the 2D array to the proper staple sequences """
    converted_staples = []
    for staple in staple_groups:
        staple_seq = []
        for l in range(len(staple)):
            row, col = staple[l, 0], staple[l, 1]
            if row != -1:
                sequence_numeric = staple_nucleotide_array[row, col]
                staple_seq.append(NUMERIC_TO_SEQUENCE[sequence_numeric])
        converted_staples.append(staple_seq)
    return converted_staples


def _find_path(start_helix_num: int, start_idx: int, vstrands: list, scaf_or_stap: str, max_num_nts: int,
               is_circular: bool, vstrand_map: dict):
    """ Finds that directed path for a given start point and helps find the scaffold_path and all staple_paths """
    cur_point = (start_helix_num, int(start_idx))
    path_found = [cur_point]

    # If a circular strand is being analyzed, we need to manually set the end_point which uses the even/odd notation
    # of cadnano:
    start_pos = vstrands[vstrand_map[start_helix_num]][scaf_or_stap][start_idx]
    if is_circular:
        end_point = (start_pos[0], start_pos[1])
    else:
        # otherwise, if not circular, end_point is simply None because the 3' end will terminate the while loop:
        end_point = None

    scaf_or_stap = scaf_or_stap.lower()
    if scaf_or_stap not in ['scaf', 'stap']:
        raise ValueError('ERROR: scaf_or_stap should be either scaf or stap')
    while_tracker = 0
    next_nt_map = {}

    while True:
        helix, idx = cur_point
        cadnano_position = vstrands[vstrand_map[helix]][scaf_or_stap][idx]

        # Look at the next position which is determined by the final two values:
        next_helix, next_idx = cadnano_position[2], cadnano_position[3]
        next_point = (next_helix, next_idx)

        if next_point == cur_point:
            break
        elif next_point == (-1, -1):
            break

        path_found.append(next_point)
        next_nt_map[cur_point] = next_point

        # This condition below is for when there is_circular strand
        if next_point == end_point and is_circular:
            break

        # Update and iterate while counter to prevent infinite loop:
        cur_point = next_point
        while_tracker += 1
        if while_tracker > (max_num_nts + 1):
            raise Exception(f'ERROR reading in {scaf_or_stap}')

    return path_found, next_nt_map

def _find_5prime_end(helix_num: int, idx: int, vstrands: list, scaf_or_stap: str, vstrand_map: dict):
    """Walk backward from a nucleotide until the strand's 5-prime endpoint is found."""
    id_from, base_from, id_to, base_to = vstrands[vstrand_map[helix_num]][scaf_or_stap][idx]
    id_from_before = helix_num

    circular_seen = {}
    is_circular = False

    while not (id_from == -1 and base_from == -1):
        if (id_from, base_from) in circular_seen:
            is_circular = True
            break
        circular_seen[(id_from, base_from)] = True
        id_from_before = id_from
        idx = base_from

        id_from, base_from, check1, check2 = vstrands[vstrand_map[id_from]][scaf_or_stap][base_from]

        if check1 == -1 and check2 == -1:
            if id_from == -1 and base_from == -1:
                raise ValueError(f'ERROR: {scaf_or_stap} path is ill-defined, DFS ended up at an inactive nucleotide.')
            # 5' end reached
            break
    return id_from_before, idx, is_circular

def _get_scaffold_connection(vstrands: list, vstrand_map: dict, num_scaf_nts: int):
    """ Reads in the vstrands scaf elements to define out a full scaffold path """
    # Determine if scaffold is circular or not
    start_num, start_idx = None, None
    end_num, end_idx = None, None
    for num, v in enumerate(vstrands):
        if v.get('num') % 2 != 0:
            continue
        scaf = np.array(v.get('scaf'))
        matching_indices = np.where((scaf[:, 0] == scaf[:, 2]) & (scaf[:, 0] != -1))[0]
        start_idx = int(matching_indices[0]) + 1
        start_num = num

        end_num = num
        end_idx = int(matching_indices[0])
        break
    if start_num is None:
        raise ValueError('ERROR: can not find even-running helix, json file is likely incorrectly formatted.')

    # The below function will keep start_helix / start_idx the same IF the strand is_circular, and if not,
    # then the start_helix and start_idx will be set to the 5' end such that the _find_path function works
    start_helix, start_idx, is_circular = _find_5prime_end(start_num, start_idx, vstrands, 'scaf',
                                                           vstrand_map)
    scaf_path, scaf_map = _find_path(start_helix, start_idx, vstrands, 'scaf', num_scaf_nts, is_circular,
                                     vstrand_map)

    # Check final value in scaf_map:
    if (end_num, end_idx) not in scaf_map:
        scaf_map[(end_num, end_idx)] = (start_num, start_idx)
    return scaf_path, scaf_map

def _get_staple_routes(vstrands: dict, vstrand_map: dict, staple_nts: np.ndarray):
    """ Creates a list of lists for all staples in the cadnano design """
    stap_copy = staple_nts.copy()
    all_staples = []
    all_staple_maps = []

    while_loop_counter = 0
    while np.any(stap_copy != -1):
        # Grab any (row, col) in stap_copy that contains a 4 value (which signals that nt is present)
        row, col = np.argwhere(stap_copy == 4)[0]
        start_helix, start_idx, is_circular = _find_5prime_end(row, col, vstrands, 'stap', vstrand_map)

        # Find the staple path, I should do a better job of setting a max boudn (e.g., the 100000 number), but the logic
        # works
        stap_path, stap_map = _find_path(start_helix, start_idx, vstrands, 'stap', 100000,
                                         is_circular, vstrand_map)

        # Store the staple_path and set all (row, col) values in the stap_copy to -1 to continue the loop
        all_staples.append(stap_path)
        all_staple_maps.append(stap_map)
        for row, col in stap_path:
            stap_copy[vstrand_map[row], col] = -1

        # Prevent infinite loop just in case (i should set a better bound eventually, but this works for now)
        while_loop_counter += 1
        if while_loop_counter > 1e6:
            raise Exception('ERROR: Can not read in staples (or number of staple > 1 million) leading to '
                            'infinite loop in sequence.py.')

    return all_staples, all_staple_maps

def _export_staple_sequences(filename_no_extension: str, scaf_seq_numeric: np.ndarray, start_point: np.ndarray,
                             sequenced_staples: list, staple_groups: np.ndarray, filepath: str | Path, vstrands: list):
    """ Exports the best staple set found to a simple CSV """
    data = _format_sequences_for_export(filename_no_extension, scaf_seq_numeric, start_point,
                                        sequenced_staples, staple_groups, vstrands)
    filepath = Path(filepath)
    base = Path(filepath).expanduser() if filepath else Path.cwd()
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    output_path = base / f"{filename_no_extension}.csv"
    
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def _format_sequences_for_export(filename_no_extension: str, scaf_seq_numeric: np.ndarray, start_point: np.ndarray,
                                 sequenced_staples: list, staple_groups: np.ndarray, vstrands: list):
    """ Exports the best staple set found to a simple CSV """
    scaf_seq_str = ''.join([NUMERIC_TO_SEQUENCE[nt] for nt in scaf_seq_numeric])
    if len(sequenced_staples) != len(staple_groups):
        raise RuntimeError('ERROR: Staple sequence list length not equal to staple groups')
    core_staple_count = 1  # Start indexing from 1
    binding_staple_count = 1

    rows_to_write = [['Start (5p)', 'End (3p)', 'Name', 'Sequence']]

    scaffold_seq = ''.join(scaf_seq_str)
    scaffold_start_string = f'{start_point[0]}[{start_point[1]}]'

    if start_point[0] % 2 == 0:
        scaffold_end_nt = start_point[1] + 1
        scaffold_end_string = f'{start_point[0]}[{scaffold_end_nt}]'
        rows_to_write.append([scaffold_end_string, scaffold_start_string, 'Scaffold Sequence', scaffold_seq])
    else:
        scaffold_end_nt = start_point[1] - 1
        scaffold_end_string = f'{start_point[0]}[{scaffold_end_nt}]'
        rows_to_write.append([scaffold_start_string, scaffold_end_string, 'Scaffold Sequence', scaffold_seq])

    for (staple_seq, staple) in zip(sequenced_staples, staple_groups):
        clean_staple = staple[~np.all(staple == -1, axis=1)]
        start_string = f'{clean_staple[0][0]}[{clean_staple[0][1]}]'
        end_string = f'{clean_staple[-1][0]}[{clean_staple[-1][1]}]'
        seq_connected = ''.join(staple_seq)

        # Name the staple based on the vstrands data:
        staple_i_name = _convert_staple_name(clean_staple, vstrands, filename_no_extension,
                                             binding_staple_count, core_staple_count)
        if 'Core' in staple_i_name:
            core_staple_count += 1
        else:
            binding_staple_count += 1

        rows_to_write.append([start_string, end_string, staple_i_name, seq_connected])
    return rows_to_write

def _convert_staple_name(staple, vstrands, fname, bind_count, core_count):
    """ Labels a staple as core / binding based on the location of the 3' and 5' ends """
    start = staple[0]
    end = staple[-1]
    start_strand = vstrands[start[0]]
    end_strand = vstrands[end[0]]

    # Use the nt position of each strand to check if scaf / stap exists at that point:
    scaf_start = start_strand['scaf'][start[1]]
    scaf_end = end_strand['scaf'][end[1]]

    if scaf_start == [-1, -1, -1, -1] or scaf_end == [-1, -1, -1, -1]:
        # If either scaf start / scaf end are empty nucleotides, then we know that this staple is a binding staple
        # Otherwise it would have it's 3' or 5' end matched
        stap_name = f'{fname}_Potential_Binding_Staple_{bind_count}'
    else:
        stap_name = f'{fname}_Core_Staple_{core_count}'
    return stap_name
