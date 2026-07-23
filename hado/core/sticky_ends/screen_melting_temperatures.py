from pathlib import Path
from math import comb
from itertools import combinations, product
import numpy as np
from hado.core.sticky_ends.utils import get_reverse_complement, StickyEndArgs, get_dna_model
import plotly.graph_objs as go
import csv

valid_nupack = True
try:
    from nupack import *
except ImportError:
    print("NUPACK package is required but not installed. Please install NUPACK 4.0 to use these tools.")
    valid_nupack = False

def screen_args_into_sequences(sticky_end_args: StickyEndArgs):
    """Return sticky-end sequences that satisfy composition and melting-temperature filters."""
    all_unique_sequences = _get_all_sequences(sticky_end_args)
    validated_seqs = []
    for seq in all_unique_sequences:
        if seq not in validated_seqs:
            validated_seqs.append(get_reverse_complement(seq))
    final_validated_seqs = [seq for seq in validated_seqs if _is_valid_seq(sticky_end_args, seq)]

    all_candidate_sequences = _screen_melting_temperatures(final_validated_seqs, sticky_end_args)
    return all_candidate_sequences

def generate_sequences(sticky_end_args: StickyEndArgs, filename_no_extension: str, filepath: str | Path = None):
    """Generate screened sticky-end sequences and write the candidate list to csv."""
    if not valid_nupack:
        print('ERROR: Unable to use this function without NUPACK installed.')
        return
    output_path = Path(filepath) if filepath is not None else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)

    all_unique_sequences = _get_all_sequences(sticky_end_args)
    validated_seqs = []
    for seq in all_unique_sequences:
        if seq not in validated_seqs:
            validated_seqs.append(get_reverse_complement(seq))
    final_validated_seqs = [seq for seq in validated_seqs if _is_valid_seq(sticky_end_args, seq)]

    all_candidate_sequences = _screen_melting_temperatures(final_validated_seqs, sticky_end_args)

    tm = f"{sticky_end_args.melting_temp_celsius:.1f}".replace(".", "p")
    tol = f"{sticky_end_args.melting_temp_tolerance:.1f}".replace(".", "p")
    fname = f"{filename_no_extension}_{sticky_end_args.total_nts}nts_{sticky_end_args.num_gc_nts_in_total}GC_" \
            f"{tm}_C_Tm_{tol}_C_tol.csv"
    output_file = output_path / fname
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        for seq in all_candidate_sequences:
            writer.writerow([seq])

    return all_candidate_sequences

def get_melting_curves(sequences: list[str], sticky_end_args: StickyEndArgs,
                       sequence_labels: dict = None):
    """Plot NUPACK-derived melting curves for candidate sticky-end sequences."""
    # Sweep thru a temperature range to plot melting curve and don't filter based on fraction
    # because just checking curves
    if not valid_nupack:
        print('ERROR: Unable to use this function without NUPACK installed.')
        return
    temperature_range = np.linspace(sticky_end_args.melting_curve_lower_bound_C,
                                    sticky_end_args.melting_curve_upper_bound_C,
                                    sticky_end_args.melting_curve_num_samples)
    melting_temps, unbound_fractions = _compute_melting_temp(sequences, temperature_range, sticky_end_args)

    fig = go.Figure()
    for seq, unbound_fraction in unbound_fractions.items():
        labelname = sequence_labels[seq] if sequence_labels is not None else f'{seq}: {melting_temps[seq]:.1f} C'
        trace = go.Scatter(x=temperature_range, y=unbound_fraction, name=labelname)
        fig.add_trace(trace)

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text="Melting Curves",
            font=dict(family="Arial", size=28, color="black")
        ),
        xaxis=dict(
            title=dict(
                text="Temperature (C)",
                font=dict(family="Arial", size=24, color="black")
            ),
            tickfont=dict(family="Arial", size=20, color="black"),
            showgrid=True,
            zeroline=True
        ),
        yaxis=dict(
            title=dict(
                text="Unbound Fraction",
                font=dict(family="Arial", size=24, color="black")
            ),
            tickfont=dict(family="Arial", size=20, color="black"),
            showgrid=True,
            zeroline=True
        )
    )
    return fig

def _screen_melting_temperatures(sequences: list[str], sticky_end_args: StickyEndArgs):
    """ Screens the melting temperature using a user-specified loewr / upper bound (should be bounding 50%) """
    unbound_fractions = _calc_unbound_fraction(sequences, sticky_end_args)

    near_50percent_melting = {}
    for seq, frac in zip(sequences, unbound_fractions):
        value = frac
        if sticky_end_args.unbound_fraction_lower_bound < value < sticky_end_args.unbound_fraction_upper_bound:
            near_50percent_melting[seq] = value

    temperature_range = np.linspace(sticky_end_args.melting_curve_lower_bound_C,
                                    sticky_end_args.melting_curve_upper_bound_C,
                                    sticky_end_args.melting_curve_num_samples)
    all_seqs = list(near_50percent_melting.keys())
    melting_temperatures, unbound_fractions = _compute_melting_temp(all_seqs, temperature_range, sticky_end_args)

    screened_candidates = {}
    temperature = sticky_end_args.melting_temp_celsius
    tol = sticky_end_args.melting_temp_tolerance
    for seq, tp in melting_temperatures.items():
        if (tp > temperature - tol) and (tp < temperature + tol):
            screened_candidates[seq] = tp

    return screened_candidates


def _calc_unbound_fraction(all_seqs, sticky_end_args: StickyEndArgs, temp_override=None):
    """ Uses NUPACK 4.0 to compute unbound fraction at given temperature """
    dna_model = get_dna_model(sticky_end_args, temp_override)

    all_tubes = [_create_tube(seq, sticky_end_args) for seq in all_seqs]
    tube_results = tube_analysis(tubes=all_tubes, model=dna_model)  # tube_analysis from NUPACK

    unbound_fractions = []

    for tube in all_tubes:
        result = tube_results[tube]

        total_strand_molarity = 0.0
        unbound_strand_molarity = 0.0

        for complex_obj, conc in result.complex_concentrations.items():
            # Get the number of strands in this specific complex (e.g., ssDNA is 1, dsDNA is 2)
            num_strands_in_complex = len(complex_obj.strands)
            total_strand_molarity += (conc * num_strands_in_complex)

            if num_strands_in_complex == 1:  # single sstrands are unbounded
                unbound_strand_molarity += conc

        # Calculate the fraction of the total strands that are not hybridized
        if total_strand_molarity > 0:
            unbound_fractions.append(unbound_strand_molarity / total_strand_molarity)
        else:
            unbound_fractions.append(1.0)

    return unbound_fractions

def _create_tube(seq, sticky_end_args: StickyEndArgs):
    seqs = [seq, get_reverse_complement(seq)]
    strands = [Strand(seq, name=seq) for seq in seqs]  # Strand from NUPACK

    c1 = Complex([strands[0]])  # Complex from NUPACK
    c2 = Complex([strands[1]])
    c3 = Complex([strands[0], strands[1]])

    origami_concentration_nm = sticky_end_args.dna_concentration_nM
    origami_concentration_M = origami_concentration_nm / 1e9
    strands = {strands[0]: origami_concentration_M, strands[1]: origami_concentration_M}
    t1 = Tube(strands=strands, complexes=SetSpec(include=[c1, c2, c3]), name=f'{seq}')  # Tube / SetSpec from NUPACK

    return t1


def _compute_melting_temp(all_seqs, temp_range, sticky_end_args: StickyEndArgs):
    unbound_fractions = {seq: [] for seq in all_seqs}

    for t in temp_range:
        seq_wise_fractions = _calc_unbound_fraction(all_seqs, sticky_end_args, temp_override=t)
        for seq, frac in zip(all_seqs, seq_wise_fractions):
            unbound_fractions[seq].append(frac)

    melting_temperatures = {}
    for seq, all_fracs in unbound_fractions.items():
        melt = temp_range[np.abs(np.array(all_fracs) - 0.5).argmin()]
        melting_temperatures[seq] = melt

    return melting_temperatures, unbound_fractions


def _is_valid_seq(sticky_end_args: StickyEndArgs, seq: str) -> bool:
    # First check if there is a run of 3 or more consecutive identical nucleotides (e.g., AAAA, TTT, GGGGG, ...)
    nucleobases = "ACGT"
    max_val = max(sticky_end_args.num_gc_nts_in_total, sticky_end_args.num_at_nts_in_total)
    for i in range(3, max_val):
        if any(base * i in seq for base in nucleobases):
            return False

    # All nucleobases should be in the string
    if any(base not in seq for base in nucleobases):
        return False

    return True


def _get_all_sequences(sticky_end_args: StickyEndArgs):
    total_sequences = _count_total_sequences(sticky_end_args)
    generated_sequences = set(_generate_all_unique_sequences(sticky_end_args))
    assert len(generated_sequences) == total_sequences, 'ERROR: Mismatch in expected vs generated unique sequences.'
    return generated_sequences


def _generate_all_unique_sequences(sticky_end_args: StickyEndArgs):
    # Deterministically generate all possible sequences with the given number of GC and AT nucleotides
    total_len = sticky_end_args.total_nts
    num_gc = sticky_end_args.num_gc_nts_in_total
    num_at = sticky_end_args.num_at_nts_in_total

    all_indices = list(range(total_len))

    for gc_positions in combinations(all_indices, num_gc):
        gc_positions_set = set(gc_positions)
        at_positions = [i for i in all_indices if i not in gc_positions_set]

        for gc_values in product(['G', 'C'], repeat=num_gc):

            for at_values in product(['A', 'T'], repeat=num_at):
                sequence = [''] * total_len
                for pos, val in zip(gc_positions, gc_values):
                    sequence[pos] = val

                for pos, val in zip(at_positions, at_values):
                    sequence[pos] = val

                yield "".join(sequence)


def _count_total_sequences(sticky_end_args: StickyEndArgs):
    positions_for_gc = comb(sticky_end_args.total_nts, sticky_end_args.num_gc_nts_in_total)
    variations = (2 ** sticky_end_args.num_gc_nts_in_total) * (2 ** sticky_end_args.num_at_nts_in_total)
    return positions_for_gc * variations
