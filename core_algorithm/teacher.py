import os
from typing import List, Dict

import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from core_algorithm.lsha.sha_learning.domain.lshafeatures import TimedTrace, FlowCondition, ProbDistribution, Trace
# from core_algorithm.lsha.sha_learning.domain.obstable import ObsTable, Row, State
from core_algorithm.lsha.sha_learning.domain.sigfeatures import SampledSignal, Timestamp
from core_algorithm.lsha.sha_learning.domain.sulfeatures import SystemUnderLearning
from core_algorithm.lsha.sha_learning.learning_setup.fastddtw import fast_ddtw, plot_aligned_signals
from core_algorithm.lsha.sha_learning.learning_setup.logger import Logger
# from core_algorithm.lsha.sha_learning.learning_setup.trace_gen import TraceGenerator
from .dynamic_tracegenerator import CustomTraceGenerator as TraceGenerator
# from core_algorithm.lsha.sha_learning.domain.obstable import ObsTable, Row, State
from .dynamic_obstable import ObsTable, Row, State
# from .dynamic_sul import parse_trace_to_signals, is_chg_pt_dynamic, label_event_dynamic

LOGGER = Logger('TEACHER')

# config = configparser.ConfigParser()
# config.read(
#     os.path.dirname(os.path.abspath(__file__)).split('sha_learning')[0] + 'sha_learning/resources/config/config.ini')
# config.sections()

# CS = config['SUL CONFIGURATION']['CASE_STUDY']
# NOISE = float(config['LSHA PARAMETERS']['DELTA'])
# P_VALUE = 0.0
# MI_QUERY = config['LSHA PARAMETERS']['MI_QUERY'] == 'True'
# PLOT_DDTW = config['LSHA PARAMETERS']['PLOT_DDTW'] == 'True'
# HT_QUERY = config['LSHA PARAMETERS']['HT_QUERY'] == 'True'
# HT_QUERY_TYPE = config['LSHA PARAMETERS']['HT_QUERY_TYPE']
# EQ_CONDITION = config['LSHA PARAMETERS']['EQ_CONDITION'].lower()

    # noise = models.FloatField(default=0.0)
    # p_value = models.FloatField(default=0.05)
    # mi_query = models.BooleanField(default=False)
    # plot_ddtw = models.BooleanField(default=False)
    # ht_query = models.BooleanField(default=False)
    # ht_query_type = models.CharField(max_length=100, null=True, blank=True)
    # eq_condition = models.CharField(max_length=100, null=True, blank=True)
    # is_stochastic = models.BooleanField(default=False)


# pov: str = None, start_dt: str = None, end_dt: str = None, start_ts: int = None, end_ts: int = None,

class CustomTeacher:
    """
    LSHA Teacher (the "oracle" half of L*) wired to a configurable SUL.

    Roles:
      * Answers Model-Identification queries (mi_query) — given a word,
        which FlowCondition fits the observed signal segment best?
      * Answers Hypothesis-Testing queries (ht_query) — given a flow,
        which probability distribution best matches the observed parameter
        statistics?  Has both a deterministic (D) and stochastic (S) variant.
      * Answers row-equality queries (eqr_query) — strict or weak equality
        depending on the ``eq_condition`` knob.
      * Performs Refinement (ref_query) — when the observation table is
        ambiguous, asks the TraceGenerator for more traces and folds them in.
      * Searches for Counterexamples (get_counterexample) — looks for words
        whose row would violate the table's closedness or consistency.

    Hyperparameters arrive as a plain dict from ``tasks.py`` and override the
    legacy config.ini values; nothing in this module hard-codes a case study.
    """

    def __init__(self, sul: SystemUnderLearning, trace_generator=None, config_data: Dict = None):

        self.sul = sul

        # --- DYNAMIC CONFIGURATION LOADING ---
        self.config = config_data if config_data else {}

        # Load Hyperparameters from the passed dictionary (populated by Django UI).
        # Strategy is normalised to {'CSV','UPPAAL'}: tasks.py and the trace
        # generator already collapse 'SIM'/'CSV'/'UPP' aliases, but we accept
        # any spelling here defensively in case the Teacher is ever wired up
        # by a different caller (CLI, tests, etc.).
        raw_strategy = (self.config.get('resample_strategy', 'SIM') or 'SIM').strip().upper()
        if raw_strategy in ('SIM', 'CSV', 'STATIC'):
            self.resample_strategy = 'CSV'
        elif raw_strategy in ('UPPAAL', 'UPP', 'VERIFYTA'):
            self.resample_strategy = 'UPPAAL'
        else:
            self.resample_strategy = raw_strategy
        self.noise = float(self.config.get('noise', 0.0))
        self.p_value = float(self.config.get('p_value', 0.05))
        self.mi_query_flag = self.config.get('mi_query', False)
        self.plot_ddtw = self.config.get('plot_ddtw', False)
        self.ht_query_flag = self.config.get('ht_query', False)
        self.ht_query_type = self.config.get('ht_query_type', 'D')
        self.eq_condition = str(self.config.get('eq_condition', 's')).lower()
        self.n_min = int(self.config.get('n_min', 10))
        self.is_aggregation = self.config.get('is_aggregation', False)

        # --- LOGGING TO VERIFY DATABASE VALUES ---
        LOGGER.info("========================================")
        LOGGER.info("TEACHER HYPERPARAMETERS LOADED:")
        LOGGER.info(f"  Noise (Delta)   : {self.noise}")
        LOGGER.info(f"  P-Value         : {self.p_value}")
        LOGGER.info(f"  MI Query        : {self.mi_query_flag}")
        LOGGER.info(f"  Plot DDTW       : {self.plot_ddtw}")
        LOGGER.info(f"  HT Query        : {self.ht_query_flag}")
        LOGGER.info(f"  HT Query Type   : {self.ht_query_type}")
        LOGGER.info(f"  Eq Condition    : {self.eq_condition}")
        LOGGER.info(f"  N Min (Refine)  : {self.n_min}")
        LOGGER.info("========================================")

        # System-Dependent Attributes
        self.symbols = sul.symbols
        self.flows = sul.flows
        self.distributions = [v.distr for v in sul.vars]

        # Trace-Dependent Attributes
        self.timed_traces: List[TimedTrace] = sul.timed_traces
        self.signals: List[List[SampledSignal]] = sul.signals

        self.TG = trace_generator
        self.hist = {}

    def add_distribution(self, d: ProbDistribution, f: FlowCondition):
        self.sul.add_distribution(d, f)


    # THIS PART REMAINS AS IT WAS IN THE ORIGINAL CODE
    # QUERIES
    @staticmethod
    def derivative(t: List[Timestamp], values: List[float]):
        # returns point-to-point increments for a given time-series
        # (derivative approximation)
        t = [x.to_secs() for x in t]
        increments = []
        try:
            increments = [(v - values[i - 1]) / (t[i] - t[i - 1]) for (i, v) in enumerate(values) if i > 0]
        except ZeroDivisionError:
            avg_dt = sum([x - t[i - 1] for (i, x) in enumerate(t) if i > 0]) / (len(t) - 1)
            increments = [(v - values[i - 1]) / avg_dt for (i, v) in enumerate(values) if i > 0]
        finally:
            LOGGER.info("========================================")
            LOGGER.info(f"INCREMENTS: {increments}")
            LOGGER.info("========================================")
            return increments

    # #############################################
    # # MODEL IDENTIFICATION QUERY:
    # # for a given prefix (word), gets all corresponding segments
    # # and returns the flow condition that best fits such segments
    # # If not enough data are available to draw a conclusion, returns None
    # #############################################

    def mi_query(self, word: Trace):
        """
        MODEL-IDENTIFICATION query.

        For the prefix ``word`` collect every signal segment in the
        trace bank that follows that prefix; for each segment compute its
        DDTW distance against every candidate FlowCondition's ideal curve
        and pick the closest match.  The flow that wins ≥75 % of the
        segments is reported as the model for that prefix; otherwise we
        report ``None`` (the table cell stays empty, forcing more refinement).

        When ``mi_query`` is disabled in the UI the answer collapses to
        the SUL's ``default_m`` flow — useful for case studies (ENERGY,
        GREEN) where the user only cares about discrete event sequences.
        """
        if not self.mi_query_flag or word == '':
    #       return self.flows[0][self.sul.default_m]
            return self.flows[0][self.sul.default_m]
        else:
            segments = self.sul.get_segments(word)
            if len(segments) > 0:
                if len(self.flows[0]) == 1:
                    return self.flows[0][0]

                fits = []
                for segment in segments:
                    if len(segment) < 3:
                        continue
                    interval = [pt.timestamp for pt in segment]
                    # observed values and (approximate) derivative
                    real_behavior = [pt.value for pt in segment]
                    min_distance = 10000
                    best_fit = None

                    # for each model from the given input set
                    for flow in self.flows[0]:
                        ideal_model = flow.f(interval, segment[0].value)
                        # applies DDTW
                        res = fast_ddtw(real_behavior, ideal_model)

                        if self.plot_ddtw:
                            plot_aligned_signals(real_behavior, ideal_model, res[1])

                        if res[0] < min_distance:
                            min_distance = res[0]
                            best_fit = flow
                    else:
                        fits.append(best_fit)

                unique_fits = set(fits)
                freq = -1
                best_fit = None
                for f in unique_fits:
                    matches = sum([x == f for x in fits]) / len(fits)
                    if matches > freq:
                        freq = matches
                        best_fit = f
                if freq > 0.75:
                    return best_fit
                else:
                    LOGGER.info("!! INCONSISTENT PHYSICAL BEHAVIOR !!")
                    return None
            else:
                return None

    # THIS FUNCTION REMAINS AS IT WAS IN THE ORIGINAL CODE

    # #############################################
    # # HYPOTHESIS TESTING QUERY:
    # # for a given prefix (word), gets all corresponding segments
    # # and returns the random variable that best fits the randomly
    # # generated model parameters.
    # # If none of the available rand. variables fits the set of segments,
    # # a new one is added
    # # If available data are not enough to draw a conclusion, returns None
    # #############################################
    def to_hist(self, values: List[float], d_id: int, update=False):
        try:
            if d_id in self.hist:
                distr = [d for d in self.distributions[0] if d.d_id == d_id][0]
                old_avg = distr.params['avg']
                old_v = len(self.hist[d_id])
                self.hist[d_id].extend(values)
            else:
                old_avg = 0.0
                old_v = 0
                self.hist[d_id] = values
            distr = [d for d in self.distributions[0] if d.d_id == d_id][0]
            distr.params['avg'] = (old_avg * old_v + sum(values)) / (old_v + len(values))
        except AttributeError:
            self.hist: Dict[int, List[float]] = {d.d_id: [] for d in self.distributions[0]}
            self.hist[d_id] = values


    # ROUTER: Sends the query to Deterministic or Stochastic based on the UI settings
    def ht_query(self, word: Trace, flow: FlowCondition, save=True):
        """
        HYPOTHESIS-TESTING query — pick the probability distribution that
        best fits the parameter samples observed for ``word`` under flow
        ``flow``.  Two backends:

          * **D** (deterministic): exact-match on the scalar parameter.
            Used when the case study has noiseless metrics (THERMO/HRI on
            UPPAAL traces — the K and R values come back numerically clean).
          * **S** (stochastic): two-sample Kolmogorov–Smirnov test against
            every existing distribution.  Used when sensors are noisy or
            when the user explicitly enables aggregation
            (ENERGY mean-power across many bins, GREEN absorption across
            many decanter cycles).

        Returns ``None`` when ``flow`` is None (model unknown), or the
        SUL's ``default_d`` distribution when ht_query is disabled.
        """
        if flow is None:
            return None

        if not self.ht_query_flag or word == '':
            return self.distributions[self.sul.default_d]

        if self.ht_query_type == 'D':
            return self.ht_d_query(word, flow, save)
        else:
            return self.ht_s_query(word, flow, save)




    # DETERMINISTIC: Exact parameter matching
    # 

    def ht_d_query(self, word: Trace, flow: FlowCondition, save=True):
        segments = self.sul.get_segments(word)
        if len(segments) > 0:
            eligible_distributions = self.sul.vars[0].get_distr_for_flow(flow.f_id)

            metrics = [self.sul.get_ht_params(segment, flow) for segment in segments]
            metrics = [met for met in metrics if met is not None]
            unique_metrics = list(set(metrics))

            if len(unique_metrics) > 1:
                LOGGER.error('INCONSISTENT PHYSICAL BEHAVIOR')
                raise RuntimeError

            best_fit: ProbDistribution = None
            try:
                for distr in self.hist:
                    value = self.hist[distr][0]
                    fits = [e_d for e_d in eligible_distributions if e_d.d_id == distr]
                    if value == unique_metrics[0] and len(fits) > 0:
                        best_fit = fits[0]
                        break
            except AttributeError:
                pass

            if best_fit is None:
                new_distr = ProbDistribution(len(self.distributions[0]), {'avg': unique_metrics[0]})
                if save:
                    self.add_distribution(new_distr, flow)
                    self.to_hist(metrics, new_distr.d_id)
                    self._log_new_distribution(new_distr, unique_metrics[0], 'D')
                return new_distr
            else:
                self.to_hist(metrics, best_fit.d_id, update=True)
                return best_fit


    # ------------------------------------------------------------------
    # Sampled diagnostic — fires once per N new distributions to track
    # whether the HT query is converging or producing a fresh distribution
    # for every segment (the classic "infinite chain" failure mode).
    # ------------------------------------------------------------------
    _DISTR_LOG_EVERY = 25
    _distr_log_count = 0

    def _log_new_distribution(self, distr, metric, qtype):
        CustomTeacher._distr_log_count += 1
        n = CustomTeacher._distr_log_count
        if n <= 5 or n % CustomTeacher._DISTR_LOG_EVERY == 0:
            LOGGER.info(
                f'[HT-{qtype}] allocated new distribution id={distr.d_id} '
                f'avg={metric:.4f} (total_new={n})'
            )

    def ht_s_query(self, word: Trace, flow: FlowCondition, save=True):
        segments = self.sul.get_segments(word)
        if len(segments) > 0:
            # distr associated with selected flow
            eligible_distributions = self.sul.vars[0].get_distr_for_flow(flow.f_id)

            # randomly distributed metrics for each segment
            metrics = [self.sul.get_ht_params(segment, flow) for segment in segments]
            metrics = [met for met in metrics if met is not None]
            avg_metrics = sum(metrics) / len(metrics)

            min_dist, best_fit = 1000, None

            try:
                for distr in self.hist:
                    if len(self.hist[distr]) == 0 or len(metrics) == 0:
                        continue
                    # change is aggregation to need to fill out
                    if not self.is_aggregation:
                        v1 = metrics
                        noise1 = [0] * len(v1)
                    else:
                        v1 = [avg_metrics] * 50
                        # When noise=0 skip the RNG entirely to guarantee
                        # deterministic results for the same input data.
                        noise1 = (np.random.normal(0.0, self.noise, size=len(v1))
                                  if self.noise > 0 else np.zeros(len(v1)))

                    v1 = [x + noise1[i] for i, x in enumerate(v1)]

                    v2 = []

                    if self.is_aggregation:
                        v2 = self.hist[distr]
                        noise2 = [0] * len(v2)
                    else:
                        for m in self.hist[distr]:
                            v2 += [m] * 10
                        noise2 = (np.random.normal(0.0, self.noise, size=len(v2))
                                  if self.noise > 0 else np.zeros(len(v2)))
                    v2 = [x + noise2[i] for i, x in enumerate(v2)]

                    # P_VALUE => self.p_value
                    statistic, pvalue = stats.ks_2samp(v1, v2)
                    fits = [d for d in eligible_distributions if d.d_id == distr]
                    if statistic <= min_dist and pvalue >= self.p_value and len(fits) > 0:
                        min_dist = statistic
                        best_fit = fits[0]
            except AttributeError:
                pass

            if best_fit is not None and min_dist < 1.0:
                self.to_hist(metrics, best_fit.d_id, update=True)
                return best_fit
            else:
                new_avg = sum(metrics) / len(metrics)
                new_distr = ProbDistribution(len(self.distributions[0]), {'avg': new_avg})
                if save:
                    self.add_distribution(new_distr, flow)
                    self.to_hist(metrics, new_distr.d_id)
                    self._log_new_distribution(new_distr, new_avg, 'S')
                return new_distr

    #############################################
    # ROW EQUALITY QUERY:
    # checks if two rows (row(s1), row(s2)) are weakly equal
    # returns true/false
    #############################################
    def eqr_query(self, row1: Row, row2: Row, strict=False):
        """
        ROW-EQUALITY query.

        ``strict=True`` (the ``s`` UI setting) requires the two rows to be
        identical cell-for-cell — including unobserved (None,None) cells.
        Strict equality matches the published L*-SHA semantics and is the
        right choice when the trace data is dense enough to fill every
        cell (THERMO, HRI under UPPAAL).

        ``strict=False`` (the ``w`` UI setting) is the *weak* variant:
        two rows are considered equal as long as the cells that ARE
        observed agree.  Cells that one row has not yet observed are
        treated as wildcards.  This is the right choice for sparse
        real-world data (ENERGY, GREEN) where a long tail of
        rarely-seen prefixes would otherwise stall the L* table from
        ever closing.
        """
        if strict:
            return row1 == row2

        for i, state in enumerate(row1.state):
            # if both rows have filled cells which differ from each other,
            # weak equality is violated
            if state.observed() and row2.state[i].observed() and state != row2.state[i]:
                return False
        else:
            return True

    #############################################
    # KNOWLEDGE REFINEMENT QUERY:
    # checks if there are ambiguous words in the observation table
    # if so, it samples new traces (through the TraceGenerator)
    # to gain more knowledge about the system under learning
    #############################################
    def ref_query(self, table: ObsTable):
        
        """
        REFINEMENT query.

        Walks every row of the observation table looking for *ambiguous
        words* — those whose row is consistent with multiple existing
        rows, or those that simply have not yet collected ``n_min``
        observations.  For each ambiguous word the Teacher asks the
        TraceGenerator for fresh evidence and feeds it into the SUL.

        The CSV/UPPAAL split is important here:

          * In **UPPAAL mode** every call to ``self.TG.get_traces(n)``
            launches a fresh verifyta simulation, so we can keep asking
            for more.  The trace is processed file-by-file.
          * In **CSV mode** the data is static.  The TraceGenerator yields
            the file list ONCE and then returns ``[]`` on every subsequent
            call (see ``CustomTraceGenerator.get_traces_csv``).  We forward
            the entire file list to ``sul.process_data`` in a single call
            so the SUL's CSV parser can concatenate them, and the empty
            return on subsequent iterations gracefully terminates the loop.
        """
        n_resample = int(self.n_min)
        S = table.get_S()
        upp_obs: List[Row] = table.get_upper_observations()
        lS = table.get_low_S()
        low_obs: List[Row] = table.get_lower_observations()

        LOGGER.info(
            f'[REF] start n_min={n_resample} '
            f'|S|={len(S)} |low_S|={len(lS)} |E|={len(table.get_E())}'
        )

        # find all words which are ambiguous
        # (equivalent to multiple rows)
        amb_words: List[Trace] = []
        # Plain enumerate (no tqdm) to keep logs grep-friendly; the loop is
        # quadratic in |S|+|low_S| so on small tables it's fast.
        for i, row in enumerate(upp_obs + low_obs):
            # if there are not enough observations of a word,
            # it needs a refinement query
            s = S[i] if i < len(upp_obs) else lS[i - len(upp_obs)]
            for e_i, e in enumerate(table.get_E()):
                if len(self.sul.get_segments(s + e)) < n_resample:
                    amb_words.append(s + e)

            if not row.is_populated():
                continue

            # find equivalent rows
            eq_rows: List[Row] = []
            for (j, row_2) in enumerate(upp_obs):
                if row_2.is_populated() and i != j and self.eqr_query(row, row_2):
                    eq_rows.append(row_2)
            if len(set(eq_rows)) > 1:
                amb_words.append(s)

        # sample new traces only for ambiguous words which
        # are not prefixes of another ambiguous word
        uq = amb_words
        # for i, w in tqdm(enumerate(amb_words)):
        #     suffixes = [w2 for w2 in amb_words if w2 != w and w2.startswith(w)]
        #     if len(suffixes) == 0:
        #         uq.append(w)

        LOGGER.info(f'[REF] ambiguous_words={len(uq)} (unique={len(set(uq))})')

        # Per-word logs are noisy but we DO want to know the count of distinct
        # words we asked the trace generator to refine.  Cap individual logs
        # at the first 5 to keep the worker log readable on long runs.
        AMBIG_LOG_BUDGET = 5
        for w_i, word in enumerate(uq):
            if w_i < AMBIG_LOG_BUDGET:
                LOGGER.info(f'[REF] requesting traces #{w_i + 1}/{len(uq)} word="{word}"')
            elif w_i == AMBIG_LOG_BUDGET:
                LOGGER.info(f'[REF] (further per-word logs suppressed; total={len(uq)})')

            for e in table.get_E():
                self.TG.set_word(word + e)
                path = self.TG.get_traces(n_resample)

                # ----------------------------------------------------------
                # CSV mode: process each file as its OWN trace.
                #
                # Why per-file (not concatenated): the LSHA library's
                # ``get_segments`` only matches traces whose **prefix** is
                # the queried word (``t.startswith(word)``).  If we
                # concatenated all CSVs into one giant trace we'd only ever
                # have one prefix (whatever event fires first in the day —
                # almost always ``i_0``) and queries for any other event
                # symbol return [], so those rows never get filled and the
                # learner collapses to a single state.
                #
                # Per-file processing gives one trace per CSV; different
                # operating windows naturally start with different events
                # (e.g. part2 of the W7 dataset starts with m_1), which
                # restores observability for every event symbol.
                # ----------------------------------------------------------
                if path is not None and len(path) > 0:
                    for sim in path:
                        # Each call appends one new trace to sul.traces;
                        # works identically for CSV files and UPPAAL .txts.
                        self.sul.process_data(sim)
                elif path is None:
                    LOGGER.debug('!! An error occurred while generating traces !!')
                # If path is [] (CSV strategy after first call), silently no-op.

        LOGGER.info(f'[REF] done — sul.traces total={len(self.sul.traces)}')

    #############################################
    # COUNTEREXAMPLE QUERIES:
    #############################################
    def not_closed(self, table, new_row):
        # Dynamically checks DB for 's' (strong) or 'w' (weak)
        is_strict = (self.eq_condition == 's')
        eq_rows = [row for row in table.get_upper_observations() if
                   self.eqr_query(new_row, row, strict=is_strict)]

        not_ambiguous = len(set(eq_rows)) <= 1
        return len(eq_rows) == 0, not_ambiguous

    def not_consistent(self, table, S, low_S, new_row, prefix):
        # Dynamically checks DB for 's' (strong) or 'w' (weak)
        is_strict = (self.eq_condition == 's')
        
        for s_i, s_word in enumerate(S):
            old_row = table.get_upper_observations()[s_i] if s_i < len(S) else \
                table.get_lower_observations()[s_i - len(S)]

            equal = self.eqr_query(old_row, new_row, strict=is_strict)

            if equal:
                for event in self.sul.events:
                    discr_is_prefix = False
                    for e in table.get_E():
                        if str(e).startswith(event.symbol):
                            continue
                    if s_word + Trace([event]) in S:
                        old_row_a: Row = table.get_upper_observations()[
                            S.index(s_word + Trace([event]))]
                    elif s_word + Trace([event]) in low_S:
                        old_row_a: Row = table.get_lower_observations()[
                            low_S.index(s_word + Trace([event]))]
                    else:
                        continue
                        
                    row_1_filled = old_row_a.state[0].observed()
                    row_2 = Row([])
                    
                    for e in table.get_E():
                        id_model_2 = self.mi_query(prefix + Trace([event]) + e)
                        id_distr_2 = self.ht_query(prefix + Trace([event]) + e, id_model_2, save=False)
                        if id_model_2 is None or id_distr_2 is None:
                            row_2.state.append(State([(None, None)]))
                        else:
                            row_2.state.append(State([(id_model_2, id_distr_2)]))
                            
                    row_2_filled = row_2.state[0].observed()
                    
                    if row_1_filled and row_2_filled and not discr_is_prefix and \
                       not self.eqr_query(row_2, old_row_a, strict=is_strict):
                        return True, event, s_word
        return False, None, None

    def get_counterexample(self, table: ObsTable):
        """
        COUNTEREXAMPLE search — the third L* oracle question.

        Iterates every prefix of every observed trace; for any prefix not
        already in S ∪ low_S, materialises its hypothetical row and
        checks two failure modes against the current observation table:

          * **non-closedness** — the new row is not equivalent to any
            existing upper row.  Returning the prefix forces the Learner
            to add it to S and rebalance.
          * **non-consistency** — there exists an event ``a`` and a row
            ``s_word`` already in S such that row(s_word) ≡ row(prefix)
            but row(s_word·a) ≢ row(prefix·a).  Reporting the prefix
            forces the Learner to extend the suffix set E.

        In CSV mode (static data) we additionally look at whether the
        last unconsumed prefix introduces an event symbol the table has
        not yet committed to S; if so we return that prefix to seed one
        more L* iteration before the loop exits.
        """
        LOGGER.info('Looking for counterexample...')

        S = table.get_S()
        low_S = table.get_low_S()

        traces: List[Trace] = self.sul.traces
        not_counter: List[Trace] = []
        for i, trace in tqdm(enumerate(traces), total=len(traces)):
            for prefix in trace.get_prefixes():
                LOGGER.debug('Checking {}'.format(str(prefix)))
                if prefix not in S and prefix not in low_S and prefix not in not_counter:
                    new_row = Row([])
                    for e_i, e_word in enumerate(table.get_E()):
                        word = prefix + e_word
                        id_model = self.mi_query(word)
                        id_distr = self.ht_query(word, id_model, save=False)
                        if id_model is not None and id_distr is not None:
                            new_row.state.append(State([(id_model, id_distr)]))
                        else:
                            new_row.state.append(State([(None, None)]))
                            
                    if new_row.is_populated():
                        not_closed, not_ambiguous = self.not_closed(table, new_row)
                        if not_closed:
                            LOGGER.warn("!! MISSED NON-CLOSEDNESS !!")
                            return prefix

                        elif not_ambiguous:
                            not_consistent, event, s_word = self.not_consistent(table, S, low_S, new_row, prefix)
                            if not_consistent:
                                LOGGER.warn("!! MISSED NON-CONSISTENCY ({}, {}) !!".format(Trace([event]), s_word))
                                return prefix
                            else:
                                not_counter.append(prefix)
                        else:
                            not_counter.append(prefix)
        else:
            # IS IT TRUE OR NOT????
            if self.resample_strategy == "SIM" and len(not_counter) > 0:
                new_events = set([e.symbol for x in not_counter for e in x.events]) - \
                             set([e.symbol for t in S for e in t.events])
                LOGGER.info(f"NEW EVENTS (SUL): {new_events}")
                if len(new_events) > 0:  # or not_counter[-1] not in S:
                    return not_counter[-1]
                else:
                    return None
            else:
                return None