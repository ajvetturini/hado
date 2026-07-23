from pathlib import Path

import pandas as pd

from hado.core.sticky_ends.utils import read_in_sequences_and_complements, StickyEndArgs, get_dna_model, \
    get_reverse_complement
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import plotly.graph_objs as go

valid_nupack = True
try:
    from nupack import *
except ImportError:
    print("NUPACK package is required but not installed. Please install NUPACK 4.0 to use these tools.")
    valid_nupack = False

def get_optimal_from_sequences_list(sequences: list, sticky_end_args: StickyEndArgs, hyperparameters: dict = None,
                                    dG_matrix_path: Path | str | None = None,
                                    filter_secondary_structures: bool = True):
    """Choose an orthogonal sticky-end subset from an in-memory candidate list."""
    needs_nupack = filter_secondary_structures or dG_matrix_path is None
    if needs_nupack and not valid_nupack:
        print('ERROR: Unable to use this function without NUPACK installed.')
        return
    sequences = list(sequences)

    if dG_matrix_path:
        dGs_filled = np.load(dG_matrix_path)
        if filter_secondary_structures:
            dna_model = get_dna_model(sticky_end_args)
            sequences = _filter_out_secondary_structures(sequences, dna_model, sticky_end_args)
    else:
        dna_model = get_dna_model(sticky_end_args)
        sequences = _filter_out_secondary_structures(sequences, dna_model, sticky_end_args)
        dGs_filled = _get_all_dGs(sequences, dna_model)

    optimal_sequence_indices, best_distance, all_histories, initial_indices = _optimize_selection(
        sequences, dGs_filled, sticky_end_args, hyperparameters
    )

    optimal_sequences = [sequences[int(i)] for i in optimal_sequence_indices]
    optimal_reverse_complements = [get_reverse_complement(i) for i in optimal_sequences]
    return optimal_sequences, optimal_reverse_complements, all_histories, (initial_indices, optimal_sequence_indices), \
           dGs_filled

def choose_optimal_sequences(sequences_filepath: Path,
                             sticky_end_args: StickyEndArgs,
                             filename_no_extension: str,
                             filepath: str | Path = None,
                             hyperparameters: dict = None,
                             ):
    """Read screened sequences, optimize a subset, and write the selected sticky ends."""
    if not valid_nupack:
        print('ERROR: Unable to use this function without NUPACK installed.')
        return

    output_path = Path(filepath) if filepath is not None else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)

    base_sequences = read_in_sequences_and_complements(sequences_filepath)
    dna_model = get_dna_model(sticky_end_args)
    sequences = _filter_out_secondary_structures(base_sequences, dna_model, sticky_end_args)

    prev_dG_filepath = output_path / f'{filename_no_extension}_dG.npy'
    if prev_dG_filepath.is_file():
        dGs_filled = np.load(prev_dG_filepath)
    else:
        dGs_filled = _get_all_dGs(sequences, dna_model)
        np.save(prev_dG_filepath, dGs_filled)

    optimal_sequence_indices, best_distance, all_histories, initial_indices = _optimize_selection(
        sequences, dGs_filled, sticky_end_args, hyperparameters
    )
    optimal_sequences = [sequences[int(i)] for i in optimal_sequence_indices]
    optimal_reverse_complements = [get_reverse_complement(i) for i in optimal_sequences]
    
    fname = f"{filename_no_extension}_OptimallyFoundStaples.txt"
    output_file = output_path / fname
    with open(output_file, 'w') as f:
        f.write('Sequence, Reverse Complement\n')
        for seq, rev_comp in zip(optimal_sequences, optimal_reverse_complements):
            f.write(f'{seq}, {rev_comp}\n')
    return optimal_sequences, optimal_reverse_complements, all_histories, (initial_indices, optimal_sequence_indices)


def _estimate_white_temp(num_white_steps: int, rng_seed: int, n_total_seqs: int,
                         num_sequences_needed: int, seq_reverse, seq_seq) -> float:
    """ Estimates initial temperature for a simulated annealing instance by taking standard deviation of a
    random walk """
    key = jax.random.PRNGKey(rng_seed)
    key, subkey = jax.random.split(key)
    initial_indices = jax.random.choice(
        subkey,
        jnp.arange(n_total_seqs),
        shape=(num_sequences_needed,),
        replace=False
    )

    def scan_step(carry, _):
        key, indices = carry
        key, key_pos, key_cand = jax.random.split(key, 3)

        # Randomly replace
        replace_pos = jax.random.randint(key_pos, shape=(), minval=0, maxval=num_sequences_needed)

        # Use logits as a little hack to circumvent re-selecting indices
        logits = jnp.zeros(n_total_seqs)
        logits = logits.at[indices].set(-jnp.inf)

        candidate = jax.random.categorical(key_cand, logits)
        new_indices = indices.at[replace_pos].set(candidate)

        score = _score(new_indices, seq_reverse, seq_seq)
        return (key, new_indices), score

    initial_carry = (key, initial_indices)
    _, scores = jax.lax.scan(scan_step, initial_carry, None, length=num_white_steps)

    std_dev = jnp.std(scores, ddof=1)
    final_temp = jnp.maximum(std_dev, 1e-6)
    return float(final_temp)

def _get_hamming_indices(sequences, n_to_choose):
    """Greedy max-min Hamming initialization """
    # Convert sequences to ordinals for faster comparison
    seq_array = np.array([list(s) for s in sequences])
    selected_idx = [0]
    remaining_idx = list(range(1, len(sequences)))

    while len(selected_idx) < n_to_choose:
        # Distance of all remaining to all selected
        current_selected = seq_array[selected_idx]
        current_remaining = seq_array[remaining_idx]

        diffs = current_remaining[:, None, :] != current_selected[None, :, :]
        dists = np.sum(diffs, axis=2)
        min_dists = np.min(dists, axis=1)

        best_in_rem = np.argmax(min_dists)
        selected_idx.append(remaining_idx.pop(best_in_rem))
    return selected_idx

@jit
def _score(indices, seq_reverse, seq_seq):
    """
    Calculates the Margin of the given selection of sequences. The crosstalk term considers binding energies between
    the N selected sequences and the N reverse complements. This N is the desired number of sequences the user needs.
    The stacking term considers the energies between the N selected sequences and themselves. The ideal system will
    produce sequences that do not strongly interact with themselves (stacking) or with other sequences' reverse
    complements (cross-talk).

    Overall, we assume that any combination that was screened in the previous step is equally as likely to be used
    in the final design (i.e., we don't consider the on-target binding strength ). Overall, this whole sticky end
    optimization pipeline has a lot of assumptions and is meant moreso as a heuristic to quickly narrow down a large
    candidate space to a set of quality sequences that do not have strong off-target interactions for colloidal assembly
    """
    sub_sr = seq_reverse[indices, :][:, indices]  # Subset of seqs vs reverse complements
    sub_ss = seq_seq[indices, :][:, indices]  # Subset of seqs vs themselves
    on_target_energies = jnp.diag(sub_sr)  # Diagnol of seqs vs targets is the on-target valuation

    # Crosstalk (Seq i binding reverse complement j)
    # Adding the large positive value makes sure on-target interactions aren't considered
    st_off_diag = sub_sr + jnp.eye(sub_sr.shape[0]) * 100.0
    strongest_crosstalk = jnp.min(st_off_diag, axis=1)

    # Self-interacting block (called stacking here, might be a loose usage of stack tho) (Seq i binding Seq j)
    strongest_stacking = jnp.min(sub_ss, axis=1)

    # Calculate Margins
    margin_crosstalk = strongest_crosstalk - on_target_energies
    margin_stacking = strongest_stacking - on_target_energies

    # Use worst-case scenario of either type as score
    min_margin = jnp.min(jnp.minimum(margin_crosstalk, margin_stacking))
    return min_margin

def _optimize_selection(sequences: list[str], dGs_filled: np.array, sticky_end_args: StickyEndArgs,
                        hyperparameters: dict):
    """ Optimizes the selection of N staple sequences from the dG matrix that maximizes the binding energy to
    it's target while minimizing any off-target binding to other strands in the set. """
    dg_matrix = jnp.array(dGs_filled)
    num_sequences_needed = sticky_end_args.number_of_optimal_sticky_end_sequences
    n_total_seqs = len(sequences)
    assert dg_matrix.shape == (2 * n_total_seqs, 2 * n_total_seqs), "Matrix dimension mismatch!"
    assert num_sequences_needed <= n_total_seqs, 'ERROR: Can not optimize selection, fewer sequences than ' \
                                                 '`number_of_optimal_sticky_end_sequences` specified.'

    # Get top 2 quadrants of interaction matrix
    # seq_reverse: interaction energies between sequences and reverse complements of sequences
    # seq_seq: interaction energies between sequences and themselves
    seq_reverse = dg_matrix[0:n_total_seqs, n_total_seqs:2 * n_total_seqs]
    seq_seq = dg_matrix[0:n_total_seqs, 0:n_total_seqs]

    # Base parameters below set via ablation study (see Figure S24)
    base_hp = {'num_chains': 4096,
               'steps': 50000,
               'temp_init': {'name': 'white', 'num_white_steps': 1000},
               'cooling_rate': 0.99,
               'rng_seed': 8,
               'init_strategy': "hamming",
               }
    if hyperparameters:
        base_hp.update(hyperparameters)

    base_hp['num_chains'] = int(base_hp['num_chains'])
    base_hp['steps'] = int(base_hp['steps'])
    base_hp['rng_seed'] = int(base_hp['rng_seed'])

    if isinstance(base_hp['temp_init'], dict):
        temp = base_hp['temp_init']
        if temp['name'] == 'white':
            base_hp['temp_init'] = _estimate_white_temp(temp['num_white_steps'], base_hp['rng_seed'],
                                                        n_total_seqs, num_sequences_needed, seq_reverse, seq_seq)
        else:
            raise ValueError('ERROR: Temperature initialization scheme not supported, valid options are: `white`')
    else:
        base_hp['temp_init'] = float(base_hp['temp_init'])

    base_hp['cooling_rate'] = float(base_hp['cooling_rate'])
    rng_key = jax.random.PRNGKey(base_hp['rng_seed'])

    if base_hp['init_strategy'].lower() == "hamming":
        hamming_indices = _get_hamming_indices(sequences, num_sequences_needed)
        initial_indices_base = jnp.array(hamming_indices)
    else:
        initial_indices_base = None  # Random init will be used

    @vmap
    def run_chain(chain_key):
        def step_fn(state, _):
            curr_indices, curr_score, k, t = state

            k1, k2, k3 = jax.random.split(k, 3)
            idx_to_replace = jax.random.randint(k1, (), 0, num_sequences_needed)

            # n_total_seqs used because only choosing from sequences, reverse complement is implied
            # as a reminder, n_total_seqs here is len(sequencse) not len(dg_matrix)
            nv = jax.random.randint(k2, (), 0, n_total_seqs)

            proposed_indices = curr_indices.at[idx_to_replace].set(nv)
            proposed_score = _score(proposed_indices, seq_reverse, seq_seq)
            is_unique_subset = jnp.sum(proposed_indices[:, None] == proposed_indices[None, :]) == num_sequences_needed

            # Metropolis
            delta = proposed_score - curr_score
            accept_prob = jnp.exp(delta / t)
            accept = (jax.random.uniform(k3) < accept_prob) & is_unique_subset

            new_indices = jnp.where(accept, proposed_indices, curr_indices)
            new_score = jnp.where(accept, proposed_score, curr_score)
            return (new_indices, new_score, k3, t * base_hp['cooling_rate']), new_score

        if initial_indices_base is not None:
            init_idx = initial_indices_base
        else:
            # Otherwise just use completely random initialization
            init_idx = jax.random.choice(chain_key, n_total_seqs, shape=(num_sequences_needed,), replace=False)

        initial_score = _score(init_idx, seq_reverse, seq_seq)
        final_state, score_history = jax.lax.scan(
            step_fn, (init_idx, initial_score, chain_key, base_hp['temp_init']), jnp.arange(base_hp['steps'])
        )
        return final_state[0], score_history, init_idx

    keys = jax.random.split(rng_key, base_hp['num_chains'])
    all_final_indices, all_histories, all_initial_indices = run_chain(keys)
    best_chain_idx = jnp.argmax(all_histories[:, -1])
    return np.array(all_final_indices[best_chain_idx]), np.array(all_histories[best_chain_idx, -1]), \
           np.array(all_histories), np.array(all_initial_indices[best_chain_idx])

def _get_all_dGs(sequences: list[str], dna_model: 'Model'):
    """ Filters the DNA sticky end sequences (which have already been screened for similar melting temperatures
    when bound to their complements) by considering secondary structures using NUPACKs mean free energy (mfe) evaluator.
    Then evaluates the dG between sequences to later use in an optimization loop
    """
    targets = [get_reverse_complement(s) for s in sequences]
    full_library = sequences + targets
    n = len(full_library)

    strands = [Strand(seq, name=f"S_{i}") for i, seq in enumerate(full_library)]

    complexes_flat = []
    indices_flat = []
    for i in range(n):
        for j in range(i + 1):  # Range 0 to i == fill in lower triangle of symmetric matrix
            c = Complex([strands[i], strands[j]])
            complexes_flat.append(c)
            indices_flat.append((i, j))

    dg_matrix = np.zeros((n, n))
    BATCH_SIZE = 50000

    for k in range(0, len(complexes_flat), BATCH_SIZE):
        batch_complexes = complexes_flat[k: k + BATCH_SIZE]
        batch_indices = indices_flat[k: k + BATCH_SIZE]

        results = complex_analysis(complexes=batch_complexes, model=dna_model, compute=['pfunc'])

        for c, (row, col) in zip(batch_complexes, batch_indices):
            gfe = results[c].free_energy
            dg_matrix[row, col] = gfe
            dg_matrix[col, row] = gfe  # symmetry

    return dg_matrix


def _filter_out_secondary_structures(sequences: list[str], dna_model: 'Model', sticky_end_args: StickyEndArgs) -> list[str]:
    # If the residual energy is smaller than this threshold, then thermal noise will likely break apart
    # the structure (this is used to somewhat limit the matrix size for dG optimization)
    R = .0019872159  # kcal / mol * K
    T_kelvin = sticky_end_args.melting_temp_celsius + 273.15
    energy_threshold = -R * T_kelvin

    filtered_sequences = []
    for seq in sequences:
        res = mfe(strands=[seq], model=dna_model)[0]  # min free energy
        if res.energy > energy_threshold:
            # Only keep sequences if free energy is more-negative (i.e, more stable) than a thermal noise threshold
            # of the environment (-RT)
            filtered_sequences.append(seq)
    return filtered_sequences


def get_optimization_results_plot(histories, **kwargs):
    """Plot simulated-annealing score histories for sticky-end selection."""
    steps = np.arange(histories.shape[1])
    mean_curve = np.mean(histories, axis=0)
    min_curve = np.min(histories, axis=0)
    max_curve = np.max(histories, axis=0)

    fig = go.Figure()
    width = kwargs.get('width', 600)
    height = kwargs.get('height', 400)
    title = kwargs.get('title', "")
    titlesize = kwargs.get('titlesize', 28)
    fontcolor = kwargs.get('fontcolor', 'black')
    fontfamily = kwargs.get('fontfamily', 'Arial')
    axissize = kwargs.get('axissize', 28)
    ticksize = kwargs.get('ticksize', 24)
    bgcolor = kwargs.get('bgcolor', 'white')
    fillcolor = kwargs.get('fillcolor', 'rgba(0, 0, 255, 0.2)')
    tracecolor = kwargs.get('tracecolor', 'blue')
    tracewidth = kwargs.get('tracewidth', 4)
    max_num_trajectories = kwargs.get('max_num_trajectories', 5)
    trajectory_color = kwargs.get('trajectory_color', 'lightblue')

    # Min–max shaded region
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=max_curve,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=min_curve,
            mode="lines",
            fill="tonexty",
            fillcolor=fillcolor,
            line=dict(width=0),
            name="Min–Max Range",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for i in range(min(max_num_trajectories, histories.shape[0])):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=histories[i],
                mode="lines",
                line=dict(color=trajectory_color, width=2),
                opacity=0.3,
                showlegend=False
            )
        )

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=mean_curve,
            mode="lines",
            line=dict(color=tracecolor, width=tracewidth),
            name="Mean Score",
            showlegend=False,
        )
    )

    fig.update_layout(
        width=width,
        height=height,

        title=dict(
            text=title,
            font=dict(size=titlesize, color=fontcolor, family=fontfamily),
            x=0.5
        ),

        xaxis=dict(
            title=dict(
                text="Iteration",
                font=dict(size=axissize, color=fontcolor, family=fontfamily)
            ),
            tickfont=dict(size=ticksize, color=fontcolor, family=fontfamily)
        ),

        yaxis=dict(
            title=dict(
                text="Margin (kcal/mol)",
                font=dict(size=axissize, color=fontcolor, family=fontfamily)
            ),
            tickfont=dict(size=ticksize, color=fontcolor, family=fontfamily)
        ),

        plot_bgcolor=bgcolor,
        paper_bgcolor=bgcolor
    )

    return fig


def get_interaction_matrix_plot(dG_filepath: str | Path = None, dG_matrix: np.ndarray = None, indices: list[int] = None,
                                df: pd.DataFrame = None, colorbar_range: tuple[float, float] = None):
    """Plot the sticky-end interaction matrix, optionally restricted to selected indices."""
    assert dG_filepath is None or dG_matrix is None, 'ERROR: One of dG_filepath or dG_matrix must be left as None'
    if dG_filepath is not None:
        matrix = np.load(dG_filepath)
    else:
        matrix = dG_matrix

    # Ensure matrix is treated as a numpy array
    matrix = np.asarray(matrix)

    mid_point = matrix.shape[0] // 2
    top_half = matrix[:mid_point, :]
    top_left = top_half[:mid_point, :mid_point]
    top_right = top_half[:mid_point, mid_point:]

    x_labels = None
    y_labels = None

    if indices is not None:
        # Ensure df is provided if indices are specified
        if df is None:
            raise ValueError("df (DataFrame) must be provided when 'indices' is specified.")

        top_left = top_left[np.ix_(indices, indices)]
        top_right = top_right[np.ix_(indices, indices)]
        matrix = np.hstack((top_left, top_right))

        # Process sequences to use as replacement for x / y axis labels
        sub_df = df.iloc[indices]
        sticky_seqs = sub_df['Sticky End Sequences'].tolist()

        # Explicitly assuming your get_reverse_complement function outputs a standard 5'->3' string
        rcs = [get_reverse_complement(s) for s in sticky_seqs]

        # Use monospace font so characters line up perfectly, and add explicit 5' and 3' directions
        # Left side x-axis (Sticky Ends) -> Orange
        # Right side x-axis (Reverse Complements) -> Light Purple
        x_labels = [f'<span style="color:orange; font-family:Courier New, monospace;">5\'-{seq}-3\'</span>' for seq in
                    sticky_seqs] + \
                   [f'<span style="color:mediumpurple; font-family:Courier New, monospace;">5\'-{seq}-3\'</span>' for
                    seq in rcs]

        # Y-axis (Sticky Ends) -> Orange
        y_labels = [f'<span style="color:orange; font-family:Courier New, monospace;">5\'-{seq}-3\'</span>' for seq in
                    sticky_seqs]

        mid_line_pos = len(indices) - 0.5

    else:
        matrix = np.asarray(matrix)
        m, n = matrix.shape
        mid_line_pos = (m // 2) - 0.5

    heatmap_kwargs = dict(
        z=matrix,
        colorscale="Viridis",
        colorbar=dict(title="dG Value"),
    )

    if colorbar_range is not None:
        vmin, vmax = colorbar_range
        heatmap_kwargs.update(zmin=vmin, zmax=vmax)

    # Map explicit coordinate values if we are tracking custom sequence labels
    if indices is not None:
        heatmap_kwargs.update(
            x=list(range(len(x_labels))),
            y=list(range(len(y_labels)))
        )

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

    fig.update_layout(
        title="Interaction Matrix",
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),
        width=1050,  # Slightly widened to account for the 5' and 3' text additions
        height=550,
    )

    # Apply specialized layouts, custom text, and line annotations if indices are active
    if indices is not None:
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(x_labels))),
                ticktext=x_labels,
                tickangle=90,  # Rotates labels 90 degrees (5' will be at the top, touching the plot)
                scaleanchor="y",
                constrain="domain",
                # Added an explanatory title to the axis to ground the visualization setup
                title=dict(
                    text='<span style="color:orange; font-weight:bold;">Sticky Ends</span> vs '
                         '<span style="color:mediumpurple; font-weight:bold;">Reverse Complements</span> (Both written 5\'→3\')',
                    font=dict(size=13)
                )
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(y_labels))),
                ticktext=y_labels,
                autorange="reversed"
            )
        )

        # Add a thick black vertical line at the exact center boundary
        fig.add_vline(x=mid_line_pos, line_width=4, line_color="black")
    else:
        fig.add_vline(x=mid_line_pos, line_width=4, line_color="black")
        fig.add_hline(y=mid_line_pos, line_width=4, line_color="black")

    return fig
