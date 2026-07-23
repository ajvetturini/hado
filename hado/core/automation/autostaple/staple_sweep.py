from __future__ import annotations

from copy import copy
from itertools import product

from hado.core.automation.autostaple.hollowframe import autostaple_hollowframe
from hado.core.automation.model.nucleotide_model import HadoNucleotideModel
from hado.core.utils import (
    MAX_DIFFERENCE,
    MAX_DIFFERENCE_TARGET,
    MAX_DIST_BETWEEN,
    MAX_POST_BUNDLE,
    MAX_RUN_POST_XOVER,
    MAX_TARGET,
    MIN_DIST_BETWEEN,
    MIN_POST_BUNDLE,
    MIN_RUN_POST_XOVER,
    MIN_TARGET,
)


def _verify_sweep_constraints(target, run_post_xover, run_post_bundle, min_dist_between, **kwargs):
    """ Verifies that these constraints are within physically realistic bounds """
    def _verify(bounded_range, prescribed_min_val, prescribed_max_val, name, lk, uk):
        if len(bounded_range) != 2:
            raise ValueError(f'ERROR: Range for {name} should be specified in format (min_val, max_val)')
        min_val, max_val = min(bounded_range), max(bounded_range)

        if min_val < prescribed_min_val:
            raise ValueError(f'ERROR: Unable to sweep StapleArgs parameter: {name} as the min value specified in '
                             f'it\'s range is less than the allowable kwarg default `{lk}` of {prescribed_min_val}')
        if max_val > prescribed_max_val:
            raise ValueError(f'ERROR: Unable to sweep StapleArgs parameter: {name} as the max value specified in '
                             f'it\'s range is larger than the allowable kwarg default `{uk}` of {prescribed_max_val}')

        if name == 'target range':
            if (max_val - min_val) >= MAX_DIFFERENCE_TARGET:
                return ValueError(
                    f'ERROR: Unable to sweep StapleArgs parameter: {name} as the range between it\'s min and '
                    f'max values is too high which would lead to a very long sweep computation. Manually '
                    f'run this tool and update the MAX_DIFFERENCE_TARGET value to a larger value if you choose.')
        else:
            if (max_val - min_val) >= MAX_DIFFERENCE:
                return ValueError(
                    f'ERROR: Unable to sweep StapleArgs parameter: {name} as the range between it\'s min and '
                    f'max values is too high which would lead to a very long sweep computation. Manually '
                    f'run this tool and update the MAX_DIFFERENCE value to a larger value if you choose.')


    a, aa = 'min_target', 'max_target'
    b, bb = 'min_run_post_xover', 'max_run_post_xover'
    c, cc = 'min_post_bundle', 'max_post_bundle'
    d, dd = 'min_dist_between', 'max_dist_between'

    min_target, max_target = kwargs.get(a, MIN_TARGET), kwargs.get(aa, MAX_TARGET)
    min_post_xo, max_post_xo = kwargs.get(b, MIN_RUN_POST_XOVER), kwargs.get(bb, MAX_RUN_POST_XOVER)
    min_bundle, max_bundle = kwargs.get(c, MIN_POST_BUNDLE), kwargs.get(cc, MAX_POST_BUNDLE)
    min_dist, max_dist = kwargs.get(d, MIN_DIST_BETWEEN), kwargs.get(dd, MAX_DIST_BETWEEN)

    _verify(target,  min_target, max_target, 'target range', a, aa)
    _verify(run_post_xover,  min_post_xo, max_post_xo, 'Run post crossover range', b, bb)
    _verify(run_post_bundle,  min_bundle, max_bundle, 'Run post bundle connection range', c, cc)
    _verify(min_dist_between,  min_dist, max_dist, 'Min distance between crossover types range', d, dd)

def sweep_staple_args(design: HadoNucleotideModel,
                      target_range: tuple = (40, 50),
                      run_post_xover_range: tuple = (3, 5),
                      run_post_bundle_range: tuple = (3, 5),
                      min_dist_between_xover_range: tuple = (3, 5),
                      **kwargs
                      ):
    """ Computationally expensive function (expect runtimes in the minutes-hours) that sweeps over all combinations of
    the constraint ranges for the hado autostaple functionality. This function is mainly used to collect /
    generate massive amounts of staple combinations and was mainly used to set the default values in the stapling
    algorithm.

    ** BEWARE ** This function can get memory-intensive depending on how many models you are generating. I added some
                 safety measures in the _verify_sweep_constraints function to limit total sizes. This function is
                 primarily meant to be used for generating data, ** NOT FOR GENERAL USE!! **

    Returns a dictionary mapping the constraint combination to a tuple of (HadoNucleotideModel, Target, MSE) where
    the constraint combination is of the specified (run_post_xover, run_post_bundle, mind_dist_between), target is i'th
    target_range, and MSE is the autobreak score
    """
    _verify_sweep_constraints(target_range, run_post_xover_range,
                              run_post_bundle_range, min_dist_between_xover_range, **kwargs)

    staple_map = design.get_connected_helices()
    final_positions = design.get_rotated_helix_positions()
    if staple_map is None or final_positions is None:
        raise ValueError('ERROR: Unable to sweep staple args as connected_helices or rotated_positions is not set'
                         'at the HadoNucleotideModel level (see set_connected_helices or set_rotated_helix_positions)')

    all_target_to_output = {}
    for i in range(target_range[0], target_range[1]+1):
        constraint_to_output = {}
        for run_post_xover, run_post_bundle, min_dist_between_xover in product(
                range(run_post_xover_range[0], run_post_xover_range[1] + 1),
                range(run_post_bundle_range[0], run_post_bundle_range[1] + 1),
                range(min_dist_between_xover_range[0], min_dist_between_xover_range[1] + 1),
        ):
            intermediate_design = copy(design)
            intermediate_design.staple_args.target_staple_length = i
            intermediate_design.staple_args.min_run_post_xover = run_post_xover
            intermediate_design.staple_args.min_run_post_bundle_connection = run_post_bundle
            intermediate_design.staple_args.min_dist_between_xovers = min_dist_between_xover

            final_intermediate, cost = autostaple_hollowframe(intermediate_design, staple_map, final_positions)
            constraint_to_output[(run_post_xover, run_post_bundle,
                                  min_dist_between_xover)] = (final_intermediate, cost)

        all_target_to_output[i] = constraint_to_output
    return all_target_to_output
