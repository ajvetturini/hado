import csv
from pathlib import Path
valid_nupack = True
try:
    from nupack import Model
except ImportError:
    print("NUPACK package is required but not installed. Please install NUPACK 4.0 to use these tools.")
    valid_nupack = False

class StickyEndArgs:
    """Validated parameters for sticky-end melting screens and sequence selection."""

    def __init__(self,
                 total_nts: int,
                 num_gc_nts_in_total: int,
                 melting_temp_celsius: float,
                 melting_temp_tolerance: float = 0.5,
                 dna_concentration_nM: float = 10.0,
                 NaCl_concentration_mM: float = 0.0,     # 0.00 mM (hado origami does not use NaCl currently)
                 MgCl2_concentration_mM: float = 12.5,   # 12.5 mM

                 # Parameters used in initial melting temperature screen
                 unbound_fraction_lower_bound: float = 0.45,
                 unbound_fraction_upper_bound: float = 0.55,
                 melting_curve_lower_bound_C: float = 0,
                 melting_curve_upper_bound_C: float = 50,
                 melting_curve_num_samples: int = 100,

                 # Parameters used in sequence selection / simulated annealing
                 number_of_optimal_sticky_end_sequences: int = 4,
                 ):

        assert num_gc_nts_in_total <= total_nts, "Number of GC nucleotides cannot exceed total nucleotides."
        assert 0 < total_nts <= 15, "Sticky end overhangs should not be overly-long (8 is default)."
        assert 0 < melting_temp_celsius <= 100, "Melting temperature must be positive and reasonably low for " \
                                                "sticky end overhangs. Currently upper-bound set at 100 C but " \
                                                "DNA origami will be melted well before that"
        assert 0 < melting_temp_tolerance <= 5, "Melting temperature tolerance must be positive and reasonably low."

        assert dna_concentration_nM > 0, "DNA concentration must be positive."
        assert NaCl_concentration_mM >= 0, "NaCl concentration must be positive."
        assert MgCl2_concentration_mM >= 0, "MgCl2 concentration must be positive."
        assert 0 <= unbound_fraction_lower_bound < unbound_fraction_upper_bound <= 1, \
            "Unbound fraction bounds must be between 0 and 1, with lower bound less than upper bound."
        assert 0 <= melting_curve_lower_bound_C < melting_curve_upper_bound_C, \
            'Melting curve bounds must be larger than 0 and lower_bound must be less than upper_bound'
        assert melting_curve_num_samples > 0, 'Range must be positive integer'
        assert number_of_optimal_sticky_end_sequences > 0, 'Number of sequences stored must be positive integer'
        self.total_nts = total_nts
        self.num_gc_nts_in_total = num_gc_nts_in_total
        self.num_at_nts_in_total = total_nts - num_gc_nts_in_total
        self.melting_temp_celsius = melting_temp_celsius
        self.melting_temp_tolerance = melting_temp_tolerance
        self.dna_concentration_nM = dna_concentration_nM
        self.NaCl_concentration_mM = NaCl_concentration_mM
        self.MgCl2_concentration_mM = MgCl2_concentration_mM
        self.unbound_fraction_lower_bound = unbound_fraction_lower_bound
        self.unbound_fraction_upper_bound = unbound_fraction_upper_bound
        self.melting_curve_lower_bound_C = melting_curve_lower_bound_C
        self.melting_curve_upper_bound_C = melting_curve_upper_bound_C
        self.melting_curve_num_samples = melting_curve_num_samples
        self.number_of_optimal_sticky_end_sequences = number_of_optimal_sticky_end_sequences


complement_wc = str.maketrans('ACGT', 'TGCA')
def get_reverse_complement(seq: str) -> str:
    """Return the Watson-Crick-Franklin reverse complement of a DNA sequence."""
    return seq.upper().translate(complement_wc)[::-1]


def read_in_sequences_and_complements(filepath: Path):
    """Read one DNA sequence per CSV row, preserving first occurrence order."""
    sequences = []
    with filepath.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) != 1:
                raise ValueError(f"Expected 1 columns (sequence), but got {len(row)}: {row}")
            seq = row[0].strip().upper()
            if seq not in sequences:  # Make sure no duplicates
                sequences.append(seq)
    return sequences


def get_dna_model(sticky_end_args: StickyEndArgs, temperature_override: float = None) -> 'Model':
    """Create the NUPACK DNA model for the requested salt and temperature settings."""
    if not valid_nupack:
        print('ERROR: Unable to find NUPACK install, returning...')
        return
    nacl_M = sticky_end_args.NaCl_concentration_mM / 1000.0
    nupack_na = max(nacl_M, 0.05)  # Can't use 0.0 in NUPACK, so set a minimum of 50 uM
    mgcl_M = sticky_end_args.MgCl2_concentration_mM / 1000.0
    temperature = sticky_end_args.melting_temp_celsius if temperature_override is None else temperature_override

    return Model(material='dna', celsius=temperature, sodium=nupack_na, magnesium=mgcl_M)
