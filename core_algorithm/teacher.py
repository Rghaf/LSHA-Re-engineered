import os
from typing import List, Dict

import numpy as np
import scipy.stats as stats
from tqdm import tqdm

from core_algorithm.lsha.sha_learning.domain.lshafeatures import TimedTrace, FlowCondition, ProbDistribution, Trace
from core_algorithm.lsha.sha_learning.domain.obstable import ObsTable, Row, State
from core_algorithm.lsha.sha_learning.domain.sigfeatures import SampledSignal, Timestamp
from core_algorithm.lsha.sha_learning.domain.sulfeatures import SystemUnderLearning
from core_algorithm.lsha.sha_learning.learning_setup.fastddtw import fast_ddtw, plot_aligned_signals
from core_algorithm.lsha.sha_learning.learning_setup.logger import Logger
from core_algorithm.lsha.sha_learning.learning_setup.trace_gen import TraceGenerator

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



class CustomTeacher:
    def __init__(self, sul: SystemUnderLearning, pov: str = None,
                 start_dt: str = None, end_dt: str = None, start_ts: int = None, end_ts: int = None,
                 config_data: Dict = None):
        self.sul = sul

        # --- DYNAMIC CONFIGURATION LOADING ---
        self.config = config_data if config_data else {}
        
        # Load Hyperparameters from the passed dictionary (populated by Django UI)
        self.noise = float(self.config.get('noise', 0.0))
        self.p_value = float(self.config.get('p_value', 0.05))
        self.mi_query_flag = self.config.get('mi_query', False)
        self.plot_ddtw = self.config.get('plot_ddtw', False)
        self.ht_query_flag = self.config.get('ht_query', False)
        self.ht_query_type = self.config.get('ht_query_type', 'D')
        self.eq_condition = str(self.config.get('eq_condition', 's')).lower()
        self.is_stochastic = self.config.get('is_stochastic', False)
        self.n_min = int(self.config.get('n_min', 10))
        self.allow_unobserved_events = self.config.get('allowUnobservedEvents', False)

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
        LOGGER.info(f"  Is Stochastic   : {self.is_stochastic}")
        LOGGER.info(f"  N Min (Refine)  : {self.n_min}")
        LOGGER.info("========================================")

        # System-Dependent Attributes
        self.symbols = sul.symbols
        self.flows = sul.flows
        self.distributions = [v.distr for v in sul.vars]

        # Trace-Dependent Attributes
        self.timed_traces: List[TimedTrace] = sul.timed_traces
        self.signals: List[List[SampledSignal]] = sul.signals

        # self.TG = TraceGenerator(pov=pov, start_dt=start_dt, end_dt=end_dt, start_ts=start_ts, end_ts=end_ts, config_data=self.config)
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

    # def mi_query(self, word: Trace):
    def mi_query(self, word: Trace):
        # In all cases we have MI_QUERY, from django. it can be ture or false.
        # The value is defined in the config.ini: MI_QUERY = False
        # Considering that we defined the variable in django models: mi_query = models.BooleanField(default=False)
    #     if not MI_QUERY or word == '':
        if not self.mi_query_flag or word == '':
    #       return self.flows[0][self.sul.default_m]
            return self.flows[0][self.sul.default_m]
        else:
            segments = self.sul.get_segments(word)

            if len(segments) > 0:
                if len(self.flows[0]) == 1:
                    return self.flows[0][0]
                else:
                    # Since we don't know why there is a specified value for THERMO case study
                    # for now we just use the general statement and ignore the line which is for THERMO
                    pass

                fits = []
                for s in segments:
                    if len(s) < 3:
                        continue
                    interval = [pt.timestamp for pt in s]
                    real_behavior = [pt.value for pt in s]

                    # QUESTION
                    # Should we get this parameters from the user?
                    min_distance = 1000 
                    best_fit = None
    #     else:
    #         segments = self.sul.get_segments(word)
    #         ## THESE LINES HAVE TO BE CHANGED TO BE 
    #         ## ************   CS NAME  ************* ##
    #         ### ************************************* ###
    #         if len(segments) > 0:
    #             if len(self.flows[0]) == 1:
    #                 return self.flows[0][0]
    #             if CS == 'THERMO' and word[-1].symbol == 'h_0':
    #                 return self.flows[0][2]
    #             fits = []
    #             for segment in segments:
    #                 if len(segment) < 3:
    #                     continue
    #                 interval = [pt.timestamp for pt in segment]
    #                 # observed values and (approximate) derivative
    #                 real_behavior = [pt.value for pt in segment]
    #                 min_distance = 10000
    #                 best_fit = None
                    for flow in self.flows[0]:
                        ideal_model = flow.f(interval, s[0].value)
                        res = fast_ddtw(real_behavior, ideal_model)

                        if self.plot_ddtw:
                            plot_aligned_signals(real_behavior, ideal_model, res[1])

                        if res[0] < min_distance:
                            min_distance = res[0]
                            best_fit = flow
                    else:
                        fits.append(best_fit)
    #                 # for each model from the given input set
    #                 for flow in self.flows[0]:
    #                     ideal_model = flow.f(interval, segment[0].value)
    #                     # applies DDTW
    #                     res = fast_ddtw(real_behavior, ideal_model)

    #                     if PLOT_DDTW:
    #                         plot_aligned_signals(real_behavior, ideal_model, res[1])

    #                     if res[0] < min_distance:
    #                         min_distance = res[0]
    #                         best_fit = flow
    #                 else:
    #                     fits.append(best_fit)
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
    #             unique_fits = set(fits)
    #             freq = -1
    #             best_fit = None
    #             for f in unique_fits:
    #                 matches = sum([x == f for x in fits]) / len(fits)
    #                 if matches > freq:
    #                     freq = matches
    #                     best_fit = f
    #             if freq > 0.75:
    #                 return best_fit
    #             else:
    #                 LOGGER.info("!! INCONSISTENT PHYSICAL BEHAVIOR !!")
    #                 return None
    #         else:
    #             return None

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
        if flow is None:
            return None

        # Uses dynamic ht_query_flag from DB
        if not self.ht_query_flag or word == '':
            return self.distributions[self.sul.default_d]

        # Uses dynamic ht_query_type from DB
        if self.ht_query_type == 'D':
            return self.ht_d_query(word, flow, save)
        else:
            return self.ht_s_query(word, flow, save)

    # DETERMINISTIC: Exact parameter matching
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
                return new_distr
            else:
                self.to_hist(metrics, best_fit.d_id, update=True)
                return best_fit

    # STOCHASTIC: Uses KS-Tests to match Gaussian distributions
    def ht_s_query(self, word: Trace, flow: FlowCondition, save=True):
        segments = self.sul.get_segments(word)
        if len(segments) > 0:
            eligible_distributions = self.sul.vars[0].get_distr_for_flow(flow.f_id)

            metrics = [self.sul.get_ht_params(segment, flow) for segment in segments]
            metrics = [met for met in metrics if met is not None]
            
            min_dist, best_fit = 1000, None

            try:
                for distr in self.hist:
                    if len(self.hist.get(distr, [])) == 0 or len(metrics) == 0:
                        continue

                    # Clean, universal noise injection based strictly on DB parameters
                    v1 = list(metrics)
                    if self.noise > 0.0:
                        noise1 = np.random.normal(0.0, self.noise, size=len(v1))
                        v1 = [x + noise1[i] for i, x in enumerate(v1)]

                    v2 = list(self.hist[distr])
                    if self.noise > 0.0:
                        noise2 = np.random.normal(0.0, self.noise, size=len(v2))
                        v2 = [x + noise2[i] for i, x in enumerate(v2)]

                    # Uses dynamic p_value from DB
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
                avg = sum(metrics) / len(metrics) if len(metrics) > 0 else 0.0
                new_distr = ProbDistribution(len(self.distributions[0]), {'avg': avg})
                if save:
                    self.add_distribution(new_distr, flow)
                    self.to_hist(metrics, new_distr.d_id)
                return new_distr

    #############################################
    # ROW EQUALITY QUERY:
    # checks if two rows (row(s1), row(s2)) are weakly or strongly equal
    #############################################
    def eqr_query(self, row1: Row, row2: Row, strict=False):
        if strict:
            return row1 == row2

        for i, state in enumerate(row1.state):
            if state.observed() and row2.state[i].observed() and state != row2.state[i]:
                return False
        else:
            return True

    #############################################
    # KNOWLEDGE REFINEMENT QUERY:
    # checks if there are ambiguous words in the observation table
    # if so, it samples new traces (through the TraceGenerator)
    #############################################
    def ref_query(self, table: ObsTable):
        LOGGER.info('Performing ref query...')

        # Replaced hardcoded config call with dynamic UI property
        n_resample = self.n_min
        S = table.get_S()
        upp_obs: List[Row] = table.get_upper_observations()
        lS = table.get_low_S()
        low_obs: List[Row] = table.get_lower_observations()

        amb_words: List[Trace] = []
        for i, row in tqdm(enumerate(upp_obs + low_obs)):
            s = S[i] if i < len(upp_obs) else lS[i - len(upp_obs)]
            for e_i, e in enumerate(table.get_E()):
                if len(self.sul.get_segments(s + e)) < n_resample:
                    amb_words.append(s + e)

            if not row.is_populated():
                continue

            eq_rows: List[Row] = []
            for (j, row_2) in enumerate(upp_obs):
                if row_2.is_populated() and i != j and self.eqr_query(row, row_2):
                    eq_rows.append(row_2)
            if len(set(eq_rows)) > 1:
                amb_words.append(s)

        uq = amb_words
        for word in tqdm(uq, total=len(uq)):
            LOGGER.info('Requesting new traces for {}'.format(str(word)))
            for e in table.get_E():
                self.TG.set_word(word + e)
                path = self.TG.get_traces(n_resample)
                if path is not None:
                    for sim in path:
                        self.sul.process_data(sim)
                else:
                    LOGGER.debug('!! An error occurred while generating traces !!')

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
            # Replaced the hardcoded 'ENERGY' / 'AUTO_TWIN' hack with the dynamic UI flag!
            if self.allow_unobserved_events and len(not_counter) > 0:
                new_events = set([e.symbol for x in not_counter for e in x.events]) - \
                             set([e.symbol for t in S for e in t.events])
                if len(new_events) > 0:
                    return not_counter[-1]
                else:
                    return None
            else:
                return None

   

    # def ht_query(self, word: Trace, flow: FlowCondition, save=True):
    #     if flow is None:
    #         return None

    #     if not HT_QUERY or word == '':
    #         return self.distributions[self.sul.default_d]

    #     if HT_QUERY_TYPE == 'D':
    #         return self.ht_d_query(word, flow, save)
    #     else:
    #         return self.ht_s_query(word, flow, save)

    # def ht_d_query(self, word: Trace, flow: FlowCondition, save=True):
    #     segments = self.sul.get_segments(word)
    #     if len(segments) > 0:
    #         eligible_distributions = self.sul.vars[0].get_distr_for_flow(flow.f_id)

    #         metrics = [self.sul.get_ht_params(segment, flow) for segment in segments]
    #         metrics = [met for met in metrics if met is not None]
    #         unique_metrics = list(set(metrics))
    #         if len(unique_metrics) > 1:
    #             LOGGER.error('INCONSISTENT PHYSICAL BEHAVIOR')
    #             raise RuntimeError

    #         best_fit: ProbDistribution = None

    #         try:
    #             for distr in self.hist:
    #                 value = self.hist[distr][0]
    #                 fits = [e_d for e_d in eligible_distributions if e_d.d_id == distr]
    #                 if value == unique_metrics[0] and len(fits) > 0:
    #                     best_fit = fits[0]
    #                     break
    #         except AttributeError:
    #             pass

    #         if best_fit is None:
    #             new_distr = ProbDistribution(len(self.distributions[0]), {'avg': unique_metrics[0]})
    #             if save:
    #                 self.add_distribution(new_distr, flow)
    #                 self.to_hist(metrics, new_distr.d_id)
    #             return new_distr
    #         else:
    #             self.to_hist(metrics, best_fit.d_id, update=True)
    #             return best_fit

    # def ht_s_query(self, word: Trace, flow: FlowCondition, save=True):
    #     segments = self.sul.get_segments(word)
    #     if len(segments) > 0:
    #         # distr associated with selected flow
    #         eligible_distributions = self.sul.vars[0].get_distr_for_flow(flow.f_id)

    #         # randomly distributed metrics for each segment
    #         metrics = [self.sul.get_ht_params(segment, flow) for segment in segments]
    #         metrics = [met for met in metrics if met is not None]
    #         avg_metrics = sum(metrics) / len(metrics)

    #         min_dist, best_fit = 1000, None

    #         try:
    #             for distr in self.hist:
    #                 if len(self.hist[distr]) == 0 or len(metrics) == 0:
    #                     continue
    #         ## THESE LINES HAVE TO BE CHANGED TO BE 
    #         ## ************   CS NAME  ************* ##
    #         ### ************************************* ###
    #                 if CS == 'THERMO':
    #                     v1 = metrics
    #                     noise1 = [0] * len(v1)
    #                 else:
    #                     v1 = [avg_metrics] * 50
    #                     noise1 = np.random.normal(0.0, NOISE, size=len(v1))

    #                 v1 = [x + noise1[i] for i, x in enumerate(v1)]

    #                 v2 = []
    #                 if CS == 'THERMO':
    #                     v2 = self.hist[distr]
    #                     noise2 = [0] * len(v2)
    #                 else:
    #                     for m in self.hist[distr]:
    #                         v2 += [m] * 10
    #                     noise2 = np.random.normal(0.0, NOISE, size=len(v2))
    #                 v2 = [x + noise2[i] for i, x in enumerate(v2)]

    #                 statistic, pvalue = stats.ks_2samp(v1, v2)
    #                 fits = [d for d in eligible_distributions if d.d_id == distr]
    #                 if statistic <= min_dist and pvalue >= P_VALUE and len(fits) > 0:
    #                     min_dist = statistic
    #                     best_fit = fits[0]
    #         except AttributeError:
    #             pass

    #         if best_fit is not None and min_dist < 1.0:
    #             self.to_hist(metrics, best_fit.d_id, update=True)
    #             return best_fit
    #         else:
    #             new_distr = ProbDistribution(len(self.distributions[0]), {'avg': sum(metrics) / len(metrics)})
    #             if save:
    #                 self.add_distribution(new_distr, flow)
    #                 self.to_hist(metrics, new_distr.d_id)
    #             return new_distr

    # #############################################
    # # ROW EQUALITY QUERY:
    # # checks if two rows (row(s1), row(s2)) are weakly equal
    # # returns true/false
    # #############################################
    # def eqr_query(self, row1: Row, row2: Row, strict=False):
    #     if strict:
    #         return row1 == row2

    #     for i, state in enumerate(row1.state):
    #         # if both rows have filled cells which differ from each other,
    #         # weak equality is violated
    #         if state.observed() and row2.state[i].observed() and state != row2.state[i]:
    #             return False
    #     else:
    #         return True

    # #############################################
    # # KNOWLEDGE REFINEMENT QUERY:
    # # checks if there are ambiguous words in the observation table
    # # if so, it samples new traces (through the TraceGenerator)
    # # to gain more knowledge about the system under learning
    # #############################################
    # def ref_query(self, table: ObsTable):
    #     LOGGER.info('Performing ref query...')

    #     n_resample = int(config['LSHA PARAMETERS']['N_min'])
    #     S = table.get_S()
    #     upp_obs: List[Row] = table.get_upper_observations()
    #     lS = table.get_low_S()
    #     low_obs: List[Row] = table.get_lower_observations()

    #     # find all words which are ambiguous
    #     # (equivalent to multiple rows)
    #     amb_words: List[Trace] = []
    #     for i, row in tqdm(enumerate(upp_obs + low_obs)):
    #         # if there are not enough observations of a word,
    #         # it needs a refinement query
    #         s = S[i] if i < len(upp_obs) else lS[i - len(upp_obs)]
    #         for e_i, e in enumerate(table.get_E()):
    #             if len(self.sul.get_segments(s + e)) < n_resample:
    #                 amb_words.append(s + e)

    #         if not row.is_populated():
    #             continue

    #         # find equivalent rows
    #         eq_rows: List[Row] = []
    #         for (j, row_2) in enumerate(upp_obs):
    #             if row_2.is_populated() and i != j and self.eqr_query(row, row_2):
    #                 eq_rows.append(row_2)
    #         if len(set(eq_rows)) > 1:
    #             amb_words.append(s)

    #     # sample new traces only for ambiguous words which
    #     # are not prefixes of another ambiguous word
    #     uq = amb_words
    #     # for i, w in tqdm(enumerate(amb_words)):
    #     #     suffixes = [w2 for w2 in amb_words if w2 != w and w2.startswith(w)]
    #     #     if len(suffixes) == 0:
    #     #         uq.append(w)

    #     for word in tqdm(uq, total=len(uq)):
    #         LOGGER.info('Requesting new traces for {}'.format(str(word)))
    #         for e in table.get_E():
    #             self.TG.set_word(word + e)
    #             path = self.TG.get_traces(n_resample)
    #             if path is not None:
    #                 for sim in path:
    #                     self.sul.process_data(sim)
    #             else:
    #                 LOGGER.debug('!! An error occurred while generating traces !!')

    # #############################################
    # # COUNTEREXAMPLE QUERY:
    # # looks for a counterexample to current obs. table
    # # returns counterexample t if:
    # # -> t highlights non-closedness
    # # -> t highlights non-consistency
    # #############################################
    # def not_closed(self, table, new_row):
    #     if EQ_CONDITION == 's':
    #         eq_rows = [row for row in table.get_upper_observations() if
    #                    self.eqr_query(new_row, row, strict=True)]
    #     else:
    #         eq_rows = [row for row in table.get_upper_observations() if
    #                    self.eqr_query(new_row, row, strict=False)]

    #     not_ambiguous = len(set(eq_rows)) <= 1
    #     diff_rows = [row for row in eq_rows if
    #                  not self.eqr_query(new_row, row, strict=True)]

    #     # FIXME: len(eq_rows)==0 does not work when rows have misaligned /bot in the signature
    #     # thus missing actual counterexample.
    #     # len(diff_rows) > 0 only works if all rows share at least one cell (as in the energy
    #     # CS, where all traces start with 'l').
    #     # A more accurate condition (TBD) should be a combination of the two.
    #     return len(eq_rows) == 0, not_ambiguous

    # def not_consistent(self, table, S, low_S, new_row, prefix):
    #     for s_i, s_word in enumerate(S):
    #         old_row = table.get_upper_observations()[s_i] if s_i < len(S) else \
    #             table.get_lower_observations()[s_i - len(S)]

    #         # finds equal rows in S
    #         if EQ_CONDITION == 's':
    #             equal = self.eqr_query(old_row, new_row, strict=True)
    #         else:
    #             equal = self.eqr_query(old_row, new_row, strict=False)

    #         if equal:
    #             for event in self.sul.events:
    #                 # if the hypothetical discriminating event is already in E
    #                 discr_is_prefix = False
    #                 for e in table.get_E():
    #                     if str(e).startswith(event.symbol):
    #                         continue
    #                 # else checks all 1-step distant rows
    #                 if s_word + Trace([event]) in S:
    #                     old_row_a: Row = table.get_upper_observations()[
    #                         S.index(s_word + Trace([event]))]
    #                 elif s_word + Trace([event]) in low_S:
    #                     old_row_a: Row = table.get_lower_observations()[
    #                         low_S.index(s_word + Trace([event]))]
    #                 else:
    #                     continue
    #                 row_1_filled = old_row_a.state[0].observed()
    #                 row_2 = Row([])
    #                 for e in table.get_E():
    #                     id_model_2 = self.mi_query(prefix + Trace([event]) + e)
    #                     id_distr_2 = self.ht_query(prefix + Trace([event]) + e, id_model_2,
    #                                                save=False)
    #                     if id_model_2 is None or id_distr_2 is None:
    #                         row_2.state.append(State([(None, None)]))
    #                     else:
    #                         row_2.state.append(State([(id_model_2, id_distr_2)]))
    #                 row_2_filled = row_2.state[0].observed()
    #                 if EQ_CONDITION == 's' and (row_1_filled and row_2_filled and not discr_is_prefix and
    #                                             not self.eqr_query(row_2, old_row_a, strict=True)):
    #                     return True, event, s_word
    #                 elif EQ_CONDITION == 'w' and (row_1_filled and row_2_filled and not discr_is_prefix and
    #                                               not self.eqr_query(row_2, old_row_a, strict=False)):
    #                     return True, event, s_word
    #     return False, None, None

    # def get_counterexample(self, table: ObsTable):
    #     LOGGER.info('Looking for counterexample...')

    #     S = table.get_S()
    #     low_S = table.get_low_S()

    #     # FIXME: Uncomment just for debugging purposes (i.e., to terminate the learning with a smaller table)
    #     # if CS == 'ENERGY' and max([len(t) for t in low_S]) >= 7:
    #     #    return None

    #     traces: List[Trace] = self.sul.traces
    #     not_counter: List[Trace] = []
    #     for i, trace in tqdm(enumerate(traces), total=len(traces)):
    #         for prefix in trace.get_prefixes():
    #             LOGGER.debug('Checking {}'.format(str(prefix)))
    #             # prefix is still not in table
    #             if prefix not in S and prefix not in low_S and prefix not in not_counter:
    #                 # fills hypothetical new row
    #                 new_row = Row([])
    #                 for e_i, e_word in enumerate(table.get_E()):
    #                     word = prefix + e_word
    #                     id_model = self.mi_query(word)
    #                     id_distr = self.ht_query(word, id_model, save=False)
    #                     if id_model is not None and id_distr is not None:
    #                         new_row.state.append(State([(id_model, id_distr)]))
    #                     else:
    #                         new_row.state.append(State([(None, None)]))
    #                 # if there are sufficient data to fill the new row
    #                 if new_row.is_populated():

    #                     not_closed, not_ambiguous = self.not_closed(table, new_row)
    #                     if not_closed:
    #                         # found non-closedness
    #                         LOGGER.warn("!! MISSED NON-CLOSEDNESS !!")
    #                         return prefix

    #                     # checks non-consistency only for rows that are not ambiguous
    #                     elif not_ambiguous:

    #                         not_consistent, event, s_word = self.not_consistent(table, S, low_S, new_row, prefix)
    #                         if not_consistent:
    #                             LOGGER.warn(
    #                                 "!! MISSED NON-CONSISTENCY ({}, {}) !!".format(Trace([event]), s_word))
    #                             return prefix
    #                         else:
    #                             not_counter.append(prefix)

    #                     else:
    #                         not_counter.append(prefix)
    #     else:
    #         # TODO: workaround to treat as a counterexample traces with events
    #         # that are missing from the current obs. table (which is NOT the case in L*).
    #         # If we want to treat as a counterexample traces that are not captured by the obs. table
    #         # (again, difforming from L*) including all potential combinations,
    #         # a more sophisticated procedure than this must be developed.


    #         ## THESE LINES HAVE TO BE CHANGED TO BE 
    #         ## ************   CS NAME  ************* ##
    #         ### ************************************* ###
    #         if CS in ['ENERGY', 'AUTO_TWIN'] and len(not_counter) > 0:
    #             new_events = set([e.symbol for x in not_counter for e in x.events]) - \
    #                          set([e.symbol for t in S for e in t.events])
    #             if len(new_events) > 0:  # or not_counter[-1] not in S:
    #                 return not_counter[-1]
    #             else:
    #                 return None
    #         else:
    #             return None
