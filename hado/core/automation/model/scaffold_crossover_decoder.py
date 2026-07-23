from __future__ import annotations

import numpy as np


class ScaffoldCrossoverDecoder:
    """Decode graph-level scaffold traversal into nucleotide-level scaffold crossovers."""

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def populate_scaffold_crossovers(self, scaffold_path: list, **kwargs):
        """ Populates the scaffold crossovers using the caDNAno-style JSON-format """
        visit_counts, visits = self._count_connections(scaffold_path)
        scaffold_mapping_to_nts, final_nucleotides_to_flip = self._decode_visits(visits)
        self._flip_scaffold_and_staple_nucleotides_from_dict(final_nucleotides_to_flip)

        not_verified = True
        while_loop_counter, max_counter = 0, kwargs.get('max_scaffold_decoding_iterations', 1000)
        _prev_moves = []
        decoded_xovers = []

        while not_verified:
            decoded_xovers = []
            for s in scaffold_path:
                decoded_xovers.append(scaffold_mapping_to_nts[s])
            not_verified, update = self._verify_crossovers(decoded_xovers)
            if not_verified:
                scaffold_mapping_to_nts = self._update_scaffold_mapping_during_verification(scaffold_mapping_to_nts,
                                                                                            update, visits)
            else:
                break

            while_loop_counter += 1
            _prev_moves.append(update)
            if while_loop_counter > max_counter:
                raise ValueError("ERROR: Unable to decode scaffold routing into nucleotide positions. This is likely "
                                 "due to a short edge post-mitering, try modifying the cross-section, decreasing the "
                                 "target_miter_threshold or making the edge a bit longer.")

        # Since decoding can slightly add / remove nts from scaffold to place crossovers in the right place,
        # we need to re-specify the scaffold / staple nucleotides prior to stapling
        min_max_check = {i: [float('inf'), float('-inf')] for i in range(len(self._helix_to_bundle))}
        for x in decoded_xovers:
            h1, nt1, h2, nt2 = x
            min_max_check[h1][0] = min(min_max_check[h1][0], nt1)
            min_max_check[h1][1] = max(min_max_check[h1][1], nt1)

            if nt2 != -1:
                min_max_check[h2][0] = min(min_max_check[h2][0], nt2)
                min_max_check[h2][1] = max(min_max_check[h2][1], nt2)
            else:
                min_max_check[h2][0] = min(min_max_check[h2][0], nt1)
                min_max_check[h2][1] = max(min_max_check[h2][1], nt1)

        nts = []
        _, num_nts = self._scaffold_nucleotides.shape
        for h, (min_nt, max_nt) in min_max_check.items():
            temp = np.full(num_nts, False)
            temp[min_nt:max_nt+1] = True
            nts.append(temp)

        self.set_scaffold_nucleotides(nts)
        self.set_staple_nucleotides(nts)
        self.set_scaffold_crossovers(decoded_xovers)

    def _count_connections(self, scaffold_connections):
        """ Counts # of instances of each use of a helix in a design for ease of decoding """
        counts = {i: 0 for i in range(len(self._helix_to_bundle))}
        visits = {i: [] for i in range(len(self._helix_to_bundle))}
        for s in scaffold_connections:
            for j in list(s):
                if not isinstance(j, str):
                    counts[j] += 1
                    visits[j].append(s)
        return counts, visits

    @staticmethod
    def _sort_crossovers(crossovers_in_helix: list):
        """ Sorts the crossovers in a helix based on the tags (either length 2, 'INTERNAL', or 'INTERNAL_MIDDLE') """
        data = {'INTERNAL_MIDDLE': [], 'INTERNAL': [], 'END': []}
        visited = {}
        for c in crossovers_in_helix:
            if len(c) == 2:
                data['END'].append(c)
            elif len(c) == 3 and c[2] == 'INTERNAL':
                data['INTERNAL'].append(c)
            elif len(c) == 3 and c[2] == 'INTERNAL_MIDDLE':
                h1, h2, _ = c
                if (h1, h2) not in visited:
                    visited[(h2, h1)] = (h1, h2)
                else:
                    data['INTERNAL_MIDDLE'].append(((h2, h1, 'INTERNAL_MIDDLE'), (h1, h2, 'INTERNAL_MIDDLE')))
            else:
                raise Exception('ERROR: Invalid tag for sorting')
        return data

    @staticmethod
    def _get_middle_most_nt(h1, h2, threes_and_fives):
        """ Calculates a middle-most point """
        h1_middle = (threes_and_fives[h1]['3p'] + threes_and_fives[h1]['5p']) / 2
        h2_middle = (threes_and_fives[h2]['3p'] + threes_and_fives[h2]['5p']) / 2
        overall_middle = (h1_middle + h2_middle) // 2
        return int(overall_middle)

    def _process_sorted_crossover_data(self, helix_num: int, sorted_data: dict):
        """ Maps the step in the scaffold_path to a nucleotide-level mapping """
        threes_and_fives = self._get_three_and_five_ends_per_helix()
        mapping = {}
        nucleotides_to_flip = {'add': [], 'remove': []}
        for k, v in sorted_data.items():
            if k == 'INTERNAL':
                # This means we want the end-most connection between h1, h2
                for step in v:
                    h1, h2, _ = step
                    first_sender = threes_and_fives[h1]['3p']
                    cor_nt_position = self.get_nearest_scaffold_crossover_index(h1, h2, first_sender)
                    mapping[step] = np.array([h1, cor_nt_position, h2, -1])

                    if cor_nt_position > first_sender:
                        for i in range(first_sender, cor_nt_position+1):
                            nucleotides_to_flip['add'].append((h1, i))
                            nucleotides_to_flip['add'].append((h2, i))
                    elif cor_nt_position < first_sender:
                        for i in range(cor_nt_position, first_sender+1):
                            nucleotides_to_flip['remove'].append((h1, i))
                            nucleotides_to_flip['remove'].append((h2, i))

            elif k == 'INTERNAL_MIDDLE':
                # If length of v is just 1, then we can safely use the middle-most point as the crossover
                if len(v) == 1:
                    a, b = v[0]
                    h1a, h2a, _ = a
                    h1b, h2b, _ = b
                    middle_most_nt = self._get_middle_most_nt(h1a, h2a, threes_and_fives)
                    cor_nt_position_a = self.get_nearest_scaffold_crossover_index(h1a, h2a, middle_most_nt)
                    cor_nt_position_b = self.get_nearest_scaffold_crossover_index(h1b, h2b, cor_nt_position_a)
                    mapping[a] = np.array([h1a, cor_nt_position_a, h2a, -1])
                    mapping[b] = np.array([h1b, cor_nt_position_b, h2b, -1])
                    continue

                # Otherwise, we want to spread out in len(v)+1 "strips:
                num_strips = len(v) + 1
                cur_helix_true_nts = np.where(self._scaffold_nucleotides[helix_num])[0]
                strip_width = len(cur_helix_true_nts) // num_strips
                selected_positions = [
                    cur_helix_true_nts[i * strip_width]
                    for i in range(1, num_strips)
                ]

                loop_corrected = selected_positions if self._scaffold_dirs[helix_num] else selected_positions[::-1]
                if len(loop_corrected) != len(v):
                    raise RuntimeError('ERROR: Number of crossover positions does not match.')

                for i, step in enumerate(v):
                    a, b = step
                    h1a, h2a, _ = a
                    h1b, h2b, _ = b
                    p = loop_corrected[i]
                    cor_nt_position_a = self.get_nearest_scaffold_crossover_index(h1a, h2a, p)
                    cor_nt_position_b = self.get_nearest_scaffold_crossover_index(h1b, h2b, cor_nt_position_a)
                    mapping[a] = np.array([h1a, cor_nt_position_a, h2a, -1])
                    mapping[b] = np.array([h1b, cor_nt_position_b, h2b, -1])

            elif k == 'END':
                for step in v:
                    mapping[step] = self._decode_end_crossover(step, threes_and_fives)
            else:
                raise ValueError('ERROR: Invalid key for processing')

        return mapping, nucleotides_to_flip

    def _decode_visits_per_bundle(self, edge_i: dict):
        bundle_mapping = {}
        bundle_nucleotides_to_flip = {'add': [], 'remove': []}
        for helix_num, crossovers_in_helix in edge_i.items():
            data = self._sort_crossovers(crossovers_in_helix)
            helix_mapping, nucleotides_to_flip = self._process_sorted_crossover_data(helix_num, data)
            for k, v in helix_mapping.items():
                if k not in bundle_mapping:
                    bundle_mapping[k] = v
            for k, v in nucleotides_to_flip.items():
                bundle_nucleotides_to_flip[k].extend(v)
        return bundle_mapping, bundle_nucleotides_to_flip

    def _decode_visits(self, visits: dict):
        """ Decodes the scaffold path into nucleotide-level representation and ensures crossovers are properly
        placed.
        """
        final_mapping = {}
        final_nucleotides_to_flip = {'add': [], 'remove': []}
        helix_to_bundle = self.get_helix_to_bundle()
        for i in range(len(self.edge_xsect_definitions)):
            indices = np.where(helix_to_bundle == i)[0]
            visits_bundle_i = {x: y for x, y in visits.items() if x in indices}
            bundle_mapping, bundle_nucleotides_to_flip = self._decode_visits_per_bundle(visits_bundle_i)
            for k, v in bundle_mapping.items():
                if k not in final_mapping:
                    final_mapping[k] = v
            for k, v in bundle_nucleotides_to_flip.items():
                final_nucleotides_to_flip[k].extend(v)
        return final_mapping, final_nucleotides_to_flip

    def _update_scaffold_mapping_during_verification(self, scaffold_mapping: dict, new_update, visits):
        """ Updates the scaffold mapping to ensure INTERNAL_MIDDLE scaffold crossovers are properly placed """
        xo_to_update, add_to_position, is_internal, og_val = new_update

        if is_internal:
            period_third = int(self._period // 3)
            h1, h2 = xo_to_update[0], xo_to_update[2]
            # If add_to_position is True, we want to increase the position
            if add_to_position:
                cur_pos = xo_to_update[1] + period_third
                while True:
                    cur_pos = self.get_nearest_scaffold_crossover_index(h1, h2, cur_pos)
                    if cur_pos >= (og_val + self.staple_args.min_run_post_xover):
                        cur_pos2 = self.get_nearest_scaffold_crossover_index(h2, h1, cur_pos)
                        scaffold_mapping[(h1, h2, 'INTERNAL_MIDDLE')] = np.array([h1, cur_pos, h2, -1])
                        scaffold_mapping[(h2, h1, 'INTERNAL_MIDDLE')] = np.array([h2, cur_pos2, h1, -1])
                        break
                    else:
                        cur_pos += period_third
            else:
                cur_pos = xo_to_update[1] - period_third
                while True:
                    cur_pos = self.get_nearest_scaffold_crossover_index(h1, h2, cur_pos)
                    if cur_pos <= (og_val - self.staple_args.min_run_post_xover):
                        cur_pos2 = self.get_nearest_scaffold_crossover_index(h2, h1, cur_pos)
                        scaffold_mapping[(h1, h2, 'INTERNAL_MIDDLE')] = np.array([h1, cur_pos, h2, -1])
                        scaffold_mapping[(h2, h1, 'INTERNAL_MIDDLE')] = np.array([h2, cur_pos2, h1, -1])
                        break
                    else:
                        cur_pos -= period_third

        else:
            # Otherwise, we must shift down / up the recieving nucleotide based on is_addition
            h1, nt1, h2, nt2 = xo_to_update

            if add_to_position:
                new_nt1 = nt2 + (nt2 + og_val - self.staple_args.min_run_post_bundle_connection)
                scaffold_mapping[(h1, h2)] = np.array([h1, new_nt1, h2, nt2])
            else:
                new_nt1 = nt2 - (nt2 - og_val + self.staple_args.min_run_post_bundle_connection)
                scaffold_mapping[(h1, h2)] = np.array([h1, new_nt1, h2, nt2])
        return scaffold_mapping

    def _flip_scaffold_and_staple_nucleotides_from_dict(self, nucleotides_to_flip):
        cur_scaf_nts = self.get_scaffold_nucleotides()
        for set_true in nucleotides_to_flip['add']:
            row, col = set_true
            cur_scaf_nts[(row, col)] = True
        for set_false in nucleotides_to_flip['remove']:
            row, col = set_false
            cur_scaf_nts[(row, col)] = False
        self.set_scaffold_nucleotides(cur_scaf_nts)
        self.set_staple_nucleotides(cur_scaf_nts)

    def _decode_end_crossover(self, crossover_tuple, threes_and_fives):
        sender, receiver = crossover_tuple
        hbs, hbr = self._helix_to_bundle[sender], self._helix_to_bundle[receiver],
        if hbs == hbr:
            raise ValueError('ERROR: Can not add edge crossover to helices belonging to same helix bundle.')

        nt_sender = threes_and_fives[sender]['3p']
        nt_receiver = threes_and_fives[receiver]['5p']

        return np.array([sender, nt_sender, receiver, nt_receiver])

    def _verify_crossovers(self, decoded_xovers):
        """ Verifies the path from the decoded crossovers visits all nucleotides """
        scaf_nts = self.get_scaffold_nucleotides()
        scaf_dirs = self.get_scaffold_directions()
        threes_and_fives = self._get_three_and_five_ends_per_helix()
        nts = np.sum(scaf_nts)

        visited = set()

        cur_nt = decoded_xovers[0][1] if decoded_xovers[0][-1] == -1 else decoded_xovers[0][3]
        for i, xo in enumerate(decoded_xovers[1:]):
            h1, nt1, h2, nt2 = xo
            dir_h1 = scaf_dirs[h1]
            is_internal = True if nt2 == -1 else False

            if dir_h1:
                if nt1 < cur_nt:
                    return True, (xo, True, is_internal, cur_nt)
                for n in range(cur_nt, nt1 + 1):
                    visited.add((int(h1), int(n)))
            else:
                if nt1 > cur_nt:
                    return True, (xo, False, is_internal, cur_nt)
                for n in range(nt1, cur_nt + 1):
                    visited.add((int(h1), int(n)))
            cur_nt = nt1 if nt2 == -1 else nt2

        # Finally, verify the last segment to check all visited nts in terms of (row, col)
        last_h = decoded_xovers[-1][2]
        last_3p = threes_and_fives[last_h]['3p']
        dir_last = scaf_dirs[last_h]
        if dir_last:
            for n in range(cur_nt, last_3p + 1):
                visited.add((int(last_h), int(n)))
        else:
            for n in range(last_3p, cur_nt + 1):
                visited.add((int(last_h), int(n)))

        if len(visited) == nts:
            return False, None
        else:
            all_required = set(map(tuple, np.argwhere(scaf_nts)))
            need_to_set_false = all_required - visited
            if len(need_to_set_false) > 0:
                for i in need_to_set_false:
                    scaf_nts[i[0], i[1]] = False

            need_to_set_true = visited - all_required
            if len(need_to_set_true) > 0:
                for i in need_to_set_true:
                    scaf_nts[i[0], i[1]] = True
            self.set_scaffold_nucleotides(scaf_nts)
            self.set_staple_nucleotides(scaf_nts)
            return False, None

    def _get_three_and_five_ends_per_helix(self):
        """ Returns a list of tuples of the (3', 5') index in nucleotides signalling the ends of each helix """
        three_and_five = {}
        nucleotides = self.get_scaffold_nucleotides()
        scaf_dirs = self.get_scaffold_directions()

        for i, nt_helix in enumerate(nucleotides):
            nts_indices = np.where(nt_helix)[0]
            max_end = nts_indices[-1] if len(nts_indices) > 0 else -1
            min_end = nts_indices[0] if len(nts_indices) > 0 else -1

            if scaf_dirs[i]:
                three_and_five[i] = {'3p': max_end, '5p': min_end}
            else:
                three_and_five[i] = {'3p': min_end, '5p': max_end}
        return three_and_five
