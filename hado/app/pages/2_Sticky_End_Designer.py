import streamlit as st
from math import comb
import numpy as np
import pandas as pd
from hado.core.sticky_ends.utils import StickyEndArgs, read_in_sequences_and_complements
from hado.core.sticky_ends.screen_melting_temperatures import screen_args_into_sequences, get_melting_curves
from hado.core.sticky_ends.optimize_selection import get_optimal_from_sequences_list, \
    get_optimization_results_plot, get_interaction_matrix_plot

from hado.app.utils import apply_page_width
from pathlib import Path
import plotly.io as pio

valid_nupack = True
try:
    from nupack import *
except ImportError:
    valid_nupack = False


APP_DIR = Path(__file__).parent
INPUT_DIR = APP_DIR / "default_inputs"
DEFAULT_SCREENED_SEQUENCES = INPUT_DIR / 'hado_default_screened_seqs.csv'
DEFAULT_DG_MATRIX = INPUT_DIR / 'hado_default_ddG.npy'
DEFAULT_MELTING_CURVES = INPUT_DIR / 'hado_default_melting_curves.json'


def _read_default_sequences_for_matrix():
    """Return the default sequence list whose ordering matches the precomputed matrix."""
    sequences = read_in_sequences_and_complements(DEFAULT_SCREENED_SEQUENCES)
    matrix = np.load(DEFAULT_DG_MATRIX)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2 != 0:
        raise ValueError("Default dG matrix must be square with an even number of rows.")

    num_matrix_sequences = matrix.shape[0] // 2
    if len(sequences) < num_matrix_sequences:
        raise ValueError("Default sequence file has fewer rows than the default dG matrix expects.")

    return sequences[:num_matrix_sequences]


def _read_default_melting_curves():
    if not DEFAULT_MELTING_CURVES.is_file():
        return None
    return pio.from_json(DEFAULT_MELTING_CURVES.read_text())

def _render_input_parameters(preloaded):
    if 'opt_results' not in st.session_state:
        st.session_state.opt_results = None

    tab1, tab2 = st.tabs(["Sticky End Parameters", "Simulated Annealing Hyperparameters"])

    disabled = True if preloaded else False

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("Physical Properties")
                total_nts = st.number_input("Total NTs", min_value=4, max_value=14, value=8, disabled=disabled)
                num_gc_nts = st.number_input("GC NTs in Total", min_value=1, max_value=total_nts, value=total_nts // 2,
                                             disabled=disabled)
                melting_temp = st.slider("Melting Temp (°C)", min_value=0.0, max_value=100.0, value=15.0, step=0.5,
                                         disabled=disabled)
                melt_tol = st.number_input("Melting Temp Tolerance", min_value=0.5, max_value=5.0, value=0.5, step=0.1,
                                           disabled=disabled)
                dna_conc = st.number_input("DNA Concentration (nM)", min_value=1.0, max_value=100.0, value=10.0,
                                           disabled=disabled)
                nacl_conc = st.number_input("NaCl Concentration (mM)", min_value=0.0, max_value=500.0, value=0.0,
                                            disabled=disabled)
                mgcl2_conc = st.number_input("MgCl2 Concentration (mM)", min_value=0.0, max_value=100.0, value=12.5,
                                             step=0.1, disabled=disabled)

        with col2:
            with st.container(border=True):
                st.subheader("Melting Temperature Screen Parameters")
                unbound_low = st.slider("Unbound Fraction (Lower)", 0.4, 0.49, 0.45, disabled=disabled)
                unbound_high = st.slider("Unbound Fraction (Upper)", 0.51, 0.6, 0.55, disabled=disabled)
                curve_high = st.number_input("Melting Curve Upper Bound (°C)", value=75, min_value=0, max_value=100,
                                             step=1, disabled=disabled)
                curve_low = st.number_input("Melting Curve Lower Bound (°C)", value=0, min_value=0,
                                            max_value=curve_high, disabled=disabled)
                num_samples = st.number_input("Melting Curve Samples", min_value=1, max_value=1000,
                                              value=100, disabled=disabled)

                st.divider()
                st.subheader("Sequence Selection")
                num_sequences = st.number_input("Optimal Sticky End Sequences", min_value=2, max_value=32, value=8)

    with tab2:
        with st.container(border=True):
            st.subheader('Simulated Annealing Hyperparameters')
            col1, col2 = st.columns(2)

            with col1:
                num_chains = st.number_input("Number of Chains", min_value=1, max_value=128, value=64)
                steps = st.number_input("Steps", min_value=100, max_value=25000, value=5000)
                temp_init = st.number_input("Initial Temperature", min_value=0.001, max_value=10.0, value=1.0)

            with col2:
                cooling_rate = st.number_input("Cooling Rate", min_value=0.90000, max_value=0.99999, value=0.99500,
                                               format="%.5f")
                rng_seed = st.number_input("RNG Seed", value=8)
                init_strategy = st.selectbox("Initialization Strategy", ["hamming", "random"])

    st.divider()
    st.subheader('Run')
    if st.button('Start optimization'):
        st.session_state.opt_results = None
        if preloaded:
            args = StickyEndArgs(8, 4, 15.0, number_of_optimal_sticky_end_sequences=num_sequences)

        else:
            args = StickyEndArgs(total_nts, num_gc_nts, melting_temp, melt_tol, dna_conc, nacl_conc, mgcl2_conc,
                                 unbound_low, unbound_high, curve_low, curve_high, num_samples, num_sequences)

        base_hp = {'num_chains': num_chains, 'steps': steps, 'temp_init': temp_init,
                   'cooling_rate': cooling_rate, 'rng_seed': rng_seed, 'init_strategy': init_strategy}
        _run_custom(args, base_hp, num_sequences, preloaded)


    if st.session_state.opt_results is not None:
        res = st.session_state.opt_results
        if res != -1:
            df = res['all_seqs']
            st.divider()
            with st.container(border=True):
                st.subheader("Melting Temp Screen Results")
                st.metric(label="Candidate Sequences Found", value=res['num_found'])
                if res['fig_melting'] is not None:
                    st.plotly_chart(res['fig_melting'], width="stretch")
                else:
                    st.caption("Precomputed default melting curves were not found; optimizer-only tuning is still available.")

            st.write(f"Search space explored: {res['search_space_text']}")

            st.divider()
            st.subheader("Optimization Results")
            st.plotly_chart(res['fig_sim'], width="stretch")

            st.write("### Interaction Matrix Comparison (dG)")
            st.caption("**Goal:** Maximize the margin between the worst off-target ensemble free energy and its "
                       "on-target, reverse complement ensemble free energy. Conceptually, this should produce a set "
                       "of sequences with minimal cross-talk between sticky ends and the reverse complements.")

            m_col1, m_col2 = st.columns(2)
            color_range = (float(np.ceil(np.min(res['dG_matrix']))), float(np.max(res['dG_matrix'])))

            st.write("**Full Matrix**")
            st.plotly_chart(get_interaction_matrix_plot(dG_matrix=res['dG_matrix'], colorbar_range=color_range),
                            width="stretch")

            with m_col1:
                st.write("**Initial Set**")
                st.plotly_chart(get_interaction_matrix_plot(dG_matrix=res['dG_matrix'], indices=res['init_idx'],
                                                        colorbar_range=color_range, df=df),
                                width="stretch")

            with m_col2:
                st.write("**Optimized Set**")
                st.plotly_chart(get_interaction_matrix_plot(dG_matrix=res['dG_matrix'], indices=res['opt_idx'],
                                                        colorbar_range=color_range, df=df),
                                width="stretch")

            st.divider()
            st.subheader("Export Final Sequences")

            df_results = res['df_opt']
            st.dataframe(df_results, hide_index=True)

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Sequences as CSV",
                data=csv,
                file_name="optimal_sticky_ends.csv",
                mime="text/csv",
            )
        if st.button("Reset and Clear Results"):
            st.session_state.opt_results = None
            st.rerun()

def _run_custom(args, base_hp, num_sequences, preloaded):
    """ This runs the full sequence generation (melting temperature screen) to sequence set selection (optimizing for
    least interacting set of sticky ends). Overall, if the preloaded flag is set true, then a base matrix is used
    to skip the most computationally-expensive step of this procedure.
    """
    with st.status("Performing sticky end screening and combination optimization...", expanded=True) as status:
        if preloaded:
            status.update(label="Loading precomputed default sticky-end inputs...", state="running")
            sequences = _read_default_sequences_for_matrix()
            fig_melting = _read_default_melting_curves()
        else:
            status.update(label="Screening for melting temperature...", state="running")
            sequences = screen_args_into_sequences(args)
            fig_melting = get_melting_curves(sequences, args, None)

        num_found = len(sequences) if sequences else 0
        if num_found < args.number_of_optimal_sticky_end_sequences:
            status.update(label="Screening failed!", state="error")
            st.error("Not enough sequences found from the melting temperature screen. Your melting temperature "
                     "may be too high / low for your set sticky end parameters.")
            results = -1
            st.session_state.opt_results = results

        if st.session_state.opt_results is None:
            status.update(label="Creating dG matrix and running simulated annealing...", state="running")

            if preloaded:
                carry = get_optimal_from_sequences_list(
                    sequences,
                    args,
                    base_hp,
                    DEFAULT_DG_MATRIX,
                    filter_secondary_structures=False,
                )
            else:
                carry = get_optimal_from_sequences_list(sequences, args, base_hp)

            opt_seqs, opt_revs, histories, indices, dG_matrix = carry
            init_idx, opt_idx = indices

            results = {
                'num_found': num_found,
                'fig_melting': fig_melting,
                'fig_sim': get_optimization_results_plot(histories),
                'dG_matrix': dG_matrix,
                'init_idx': init_idx,
                'opt_idx': opt_idx,
                'all_seqs': pd.DataFrame({
                    "Sticky End Sequences": sequences,
                }),
                'df_opt': pd.DataFrame({
                    "Sticky End Sequence": opt_seqs,
                    "Reverse Complement": opt_revs,
                    "Index": opt_idx
                }),
                'search_space_text': f"{len(sequences)}c{num_sequences} ({comb(num_found, num_sequences):,d})"
            }
            st.session_state.opt_results = results
            status.update(label="Finished!", state="complete", expanded=False)


if __name__ == '__main__':
    apply_page_width()
    st.title("Sticky End Sequence Optimization")
    if valid_nupack:
        st.markdown(
             '<u>**NOTE**</u>\n\n Building the matrices takes some time (~5-10 minutes or more depending on your '
             'choice of parameters), so please be patient! \n\n'
             ''
             'Alternatively, if you are OK with using the default values that were used in the journal article '
             '(8-nt sticky ends, 4 of which are GCs), then check the box below to pre-load the dG matrix. You '
             'can still change the simulated annealing hyperparameters and the desired number of sticky ends but you '
             'can not change the thermodynamic values.',
             unsafe_allow_html=True,
        )
        preload_ddg = st.checkbox('Preload dG matrix?', help='If checked, then the algorithm skips building a '
                                                              'custom dG matrix which saves on a majority of this '
                                                              'functions computational cost. Mainly useful if you '
                                                              'want to explore this tool.')
        _render_input_parameters(preload_ddg)
    else:
        st.markdown(
            '<u>**NOTE**</u>\n\n NUPACK is not installed, so custom thermodynamic screens are unavailable. '
            'This is expected if using the web application as NUPACK requires your own unique license and to run '
            'hado locally (see [FAQs](./FAQs) page for how to run locally).',
            unsafe_allow_html=True,
        )

        # preload_ddg = st.checkbox('Preload dG matrix?', value=True, disabled=True,
        #                           help='If checked, then the algorithm skips building a custom dG matrix which '
        #                                'saves on a majority of this functions computational cost. Mainly useful if '
        #                                'you want to explore this tool.')
        # _render_input_parameters(preloaded=True)

