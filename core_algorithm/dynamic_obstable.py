"""
dynamic_obstable.py — Universal Observation Table for the L*-SHA Learner
========================================================================

The Observation Table is the data structure at the heart of L*.  It is
indexed along two axes:

    * Rows  — words from the prefix-closed *S* set (upper rows) plus their
              one-step extensions in *low_S* (lower rows).
    * Cols  — distinguishing-suffix words from the *E* set.

Each cell ``(s, e)`` stores the (model, distribution) pair the Teacher
returns for the word ``s · e``.  The Learner asks whether the table is
*closed* (every lower row is equivalent to some upper row) and *consistent*
(equivalent rows behave the same when extended), and refines S/E whenever
either property is broken.

This module is **case-study agnostic** — it never inspects the actual
event symbols or signal values.  The semantic decisions (strict vs weak
equality, which flow fits which segment) live in ``teacher.py``; the
table only manages the bookkeeping and the final SHA reconstruction.
"""

import os
from typing import List, Dict

from core_algorithm.lsha.sha_learning.domain.lshafeatures import Trace, State, EMPTY_STRING
from core_algorithm.lsha.sha_learning.domain.shafeatures import StochasticHybridAutomaton, Location, Edge
from core_algorithm.lsha.sha_learning.learning_setup.logger import Logger

LOGGER = Logger('Obs.Table Handler')

class Row:
    """
    A single row of the observation table — one filled (model, distribution)
    State per column in E.

    Notes:
      * ``is_populated`` is the "do I know anything at all about this row"
        predicate: a row whose every cell is still ``(None, None)`` is
        treated as a placeholder and skipped by closedness checks.
      * Equality is *strict* element-wise — the Teacher's ``eqr_query``
        is the place to implement weak equality (wildcard for unobserved).
    """

    def __init__(self, state: List[State]):
        self.state = state

    def is_populated(self):
        return any([s.observed() for s in self.state])

    def __str__(self):
        return '\t|\t'.join([str(s) for s in self.state])

    def __eq__(self, other):
        return all([s == other.state[i] for i, s in enumerate(self.state)])

    def __hash__(self):
        return hash(str(self))


class ObsTable:
    """
    The observation table itself.

    Internal structure:
        __S       : upper-row words (the "agreed-upon prefixes" of the SHA)
        __low_S   : lower-row words (one-step extensions of S, candidates
                    for promotion to S when L* finds them inequivalent
                    to anything in S)
        __E       : column words (the distinguishing suffixes)
        __upp_obs : List[Row], one per S word, each Row has len(E) States
        __low_obs : List[Row], one per low_S word, each Row has len(E) States

    A subtle Python pitfall: the initial ``upp_obs``/``low_obs`` lists are
    built by ``[Row(...)] * n``, which creates n references to the SAME
    Row.  This works for the empty-template case here because the cells
    are never mutated in-place — they are replaced via ``set_*``.  If
    someone ever changes the Learner to mutate cells, the construction
    would need to switch to a list comprehension.
    """

    def __init__(self, s: List[Trace], e: List[Trace], low_s: List[Trace]):
        self.__S: List[Trace] = s
        self.__low_S: List[Trace] = low_s
        self.__E: List[Trace] = e
        self.__upp_obs: List[Row] = [Row([State([(None, None)])] * len(e))] * len(s)
        self.__low_obs: List[Row] = [Row([State([(None, None)])] * len(e))] * len(low_s)

    def get_S(self):
        return self.__S

    def add_S(self, word: Trace):
        self.__S.append(word)

    def get_E(self):
        return self.__E

    def add_E(self, word: Trace):
        self.__E.append(word)

    def get_low_S(self):
        return self.__low_S

    def add_low_S(self, word: Trace):
        self.__low_S.append(word)

    def del_low_S(self, index: int):
        self.get_low_S().pop(index)

    def get_upper_observations(self):
        return self.__upp_obs

    def set_upper_observations(self, obs_table: List[Row]):
        self.__upp_obs = obs_table

    def get_lower_observations(self):
        return self.__low_obs

    def set_lower_observations(self, obs_table: List[Row]):
        self.__low_obs = obs_table

    def __str__(self, filter_empty=False):
        result = ''

        rows = self.get_upper_observations() + self.get_lower_observations()
        populated_rows = [i for i, row in enumerate(rows) if row.is_populated()]

        # max_tabs = max(
        #     [len(str(word)) for i, word in enumerate(self.get_S() + self.get_low_S()) if i in populated_rows])

        # --- FIX: Safe max() for empty tables ---
        valid_lens = [len(str(word)) for i, word in enumerate(self.get_S() + self.get_low_S()) if i in populated_rows]
        max_tabs = max(valid_lens) if valid_lens else 8
        
        HEADER = ' ' * max_tabs + '|'

        len_row_cells = [[len(s.label) for s in r.state] for r in rows]
        col_width = [max([l for r in len_row_cells for j_2, l in enumerate(r) if j_2 == j]) for j, e in
                     enumerate(self.get_E())]

        # column (E set) labels
        HEADER += '|'.join([str(e) + ' ' * (col_width[j] - len(str(e))) for j, e in enumerate(self.get_E())])
        result += HEADER + '\n'

        SEPARATOR = '-' * max_tabs + '+' + '+'.join(['-' * c for c in col_width])
        result += SEPARATOR + '\n'

        # print short words row labels
        for i, s_word in enumerate(self.get_S() + self.get_low_S()):
            if i == len(self.get_S()):
                result += SEPARATOR + '\n'
            row = rows[i]
            if filter_empty and not row.is_populated():
                pass
            else:
                ROW = str(s_word)
                ROW += ' ' * (max_tabs - len(str(s_word))) + '|'
                ROW += '|'.join([s.label + ' ' * (col_width[j] - len(s.label)) for j, s in enumerate(row.state)])
                result += ROW + '\n'

        result += SEPARATOR + '\n'
        return result

    def print(self, filter_empty=False):
        LOGGER.warn(self.__str__(filter_empty))

    def get_loc_from_word(self, word: Trace, locations: List[Location], seq_to_loc: Dict[Trace, str], teacher):
        candidate_dest_locs = []
        
        # Pull EQ_CONDITION dynamically from the UI parameters
        eq_condition = teacher.config.get('eq_condition', 's').lower()

        if word in seq_to_loc.keys():
            loc = [l for l in locations if l.name == seq_to_loc[word]][0]
            candidate_dest_locs.append(loc)
        else:
            curr_row = None
            if word in self.get_S():
                curr_row = self.get_upper_observations()[self.get_S().index(word)]
            elif word in self.get_low_S():
                curr_row = self.get_lower_observations()[self.get_low_S().index(word)]
            elif len(word) > 0 and Trace(word[:-1]) in self.get_S() \
                    and Trace([word[-1]]) in self.get_E():
                row = self.get_S().index(Trace(word[:-1]))
                column = self.get_E().index(Trace([word[-1]]))
                needed_state = self.get_upper_observations()[row].state[column]
                curr_row = Row([needed_state] + [State([(None, None)])] * (len(self.get_E()) - 1))
            elif len(word) > 0 and Trace(word[:-1]) in self.get_low_S() \
                    and Trace([word[-1]]) in self.get_E():
                row = self.get_low_S().index(Trace(word[:-1]))
                column = self.get_E().index(Trace([word[-1]]))
                needed_state = self.get_lower_observations()[row].state[column]
                curr_row = Row([needed_state] + [State([(None, None)])] * (len(self.get_E()) - 1))

            # When the word doesn't match any of the bookkeeping cases above
            # (e.g. a deep counterexample whose 1-step prefix isn't in S/low_S
            # or whose final symbol isn't in E) we have no row to compare;
            # return [] so the caller treats it as an unobservable transition.
            if curr_row is None or not curr_row.is_populated():
                return []

            for i, row in enumerate(self.get_upper_observations()):
                if eq_condition == 's':
                    if self.get_S()[i] in seq_to_loc.keys() and teacher.eqr_query(curr_row, row, strict=True):
                        loc = [l for l in locations if l.name == seq_to_loc[self.get_S()[i]]][0]
                        candidate_dest_locs.append(loc)
                else:
                    if self.get_S()[i] in seq_to_loc.keys() and teacher.eqr_query(curr_row, row, strict=False):
                        loc = [l for l in locations if l.name == seq_to_loc[self.get_S()[i]]][0]
                        candidate_dest_locs.append(loc)

        return candidate_dest_locs

    def add_init_edges(self, locations: List[Location], edges: List[Edge], seq_to_loc: Dict[Trace, str], teacher):
        init_loc = Location('__init__', None)
        locations.append(init_loc)

        one_word_upper = [word for word in self.get_S() if len(word) == 1]
        one_word_lower = [word for word in self.get_low_S() if len(word) == 1]

        for word in one_word_upper + one_word_lower:
            dest_locs = self.get_loc_from_word(word, locations, seq_to_loc, teacher)
            for dest_loc in dest_locs:
                if dest_loc is not None:
                    edges.append(Edge(init_loc, dest_loc, sync=str(word)))

        return locations, edges

    def to_sha(self, teacher):
        locations: List[Location] = []
        upp_obs: List[Row] = self.get_upper_observations()
        low_obs: List[Row] = self.get_lower_observations()
        
        # Pull EQ_CONDITION dynamically from the UI parameters
        eq_condition = teacher.config.get('eq_condition', 's').lower()
        
        unique_sequences: List[Trace] = []
        unique_sequences_dict: Dict[Trace, str] = {}
        for i, row in enumerate(upp_obs):
            row_already_present = False
            for seq in unique_sequences:
                row_2 = upp_obs[self.get_S().index(seq)]
                if eq_condition == 's':
                    if teacher.eqr_query(row, row_2, strict=True):
                        row_already_present = True
                        break
                else:
                    if teacher.eqr_query(row, row_2, strict=False):
                        row_already_present = True
                        break
            if not row_already_present:
                unique_sequences.append(self.get_S()[i])
                
        # Create a new location for each unique sequence
        for index, seq in enumerate(unique_sequences):
            seq_index = self.get_S().index(seq)
            row = upp_obs[seq_index]
            new_name = StochasticHybridAutomaton.LOCATION_FORMATTER.format(len(locations))
            
            # --- DYNAMIC FLOW LABEL FIX: Handles 1D (Energy) and 2D (HRI) variables ---
            try:
                if hasattr(row.state[0], 'vars') and isinstance(row.state[0].vars, (list, tuple)):
                    flat_vars = []
                    for v in row.state[0].vars:
                        if isinstance(v, (list, tuple)):
                            flat_vars.extend(v)
                        else:
                            flat_vars.append(v)
                    new_flow = ', '.join([getattr(v, 'label', str(v)) for v in flat_vars])
                else:
                    new_flow = str(row.state[0])
            except Exception:
                new_flow = str(row.state[0])
                
            locations.append(Location(new_name, new_flow))
            unique_sequences_dict[seq] = new_name

        # start building edges list for upper part of the table
        edges: List[Edge] = []
        for s_i, s_word in enumerate(self.get_S()):
            for t_i, t_word in enumerate(self.get_E()):
                if upp_obs[s_i].state[t_i].observed():
                    word: Trace = s_word + t_word
                    entry_word = Trace(word[:len(word) - 1])
                    if len(entry_word) == 0:
                        continue

                    start_loc = self.get_loc_from_word(entry_word, locations, unique_sequences_dict, teacher)[0]
                    dest_locs = self.get_loc_from_word(word, locations, unique_sequences_dict, teacher)

                    labels = str(Trace(word[-1:]))
                    for dest_loc in dest_locs:
                        new_edge = Edge(start_loc, dest_loc, sync=labels)
                        if start_loc is not None and dest_loc is not None and new_edge not in edges:
                            edges.append(new_edge)

        # start building edges list for lower part of the table
        for s_i, s_word in enumerate(self.get_low_S()):
            for t_i, t_word in enumerate(self.get_E()):
                if low_obs[s_i].state[t_i].observed():
                    word: Trace = s_word + t_word
                    entry_word = Trace(word[:len(word) - 1])
                    if len(entry_word) == 0:
                        continue

                    start_loc = self.get_loc_from_word(entry_word, locations, unique_sequences_dict, teacher)[0]
                    dest_locs = self.get_loc_from_word(word, locations, unique_sequences_dict, teacher)

                    if word != '':
                        labels = str(word.sub_prefix(entry_word))
                    else:
                        labels = EMPTY_STRING
                    for dest_loc in dest_locs:
                        new_edge = Edge(start_loc, dest_loc, sync=labels)
                        if start_loc is not None and dest_loc is not None and new_edge not in edges:
                            edges.append(new_edge)

        locations, edges = self.add_init_edges(locations, edges, unique_sequences_dict, teacher)
        learned_sha = StochasticHybridAutomaton(locations, edges)

        learned_sha.sanity_check(unique_sequences_dict)

        return learned_sha