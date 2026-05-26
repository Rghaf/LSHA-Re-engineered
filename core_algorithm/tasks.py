import os
import sys
import logging
import json
import numpy as np
from functools import partial
from celery import shared_task
from django.conf import settings
from django.core.files import File
from datetime import datetime

# --- Path Setup ---
if str(settings.LSHA_ROOT) not in sys.path:
    sys.path.insert(0, str(settings.LSHA_ROOT))

# --- Imports from your project ---
from core_algorithm.lsha.sha_learning.domain.lshafeatures import Trace, TimedTrace, FlowCondition
from core_algorithm.lsha.sha_learning.domain.sigfeatures import Event, Timestamp, SampledSignal
# from core_algorithm.lsha.sha_learning.domain.obstable import Row, State, ObsTable
from core_algorithm.lsha.sha_learning.learning_setup.logger import Logger
from core_algorithm.lsha.sha_learning.learning_setup.learner import Learner

# --- Plotting & Reporting Imports ---
import core_algorithm.lsha.sha_learning.pltr.sha_pltr as ha_pltr
import core_algorithm.lsha.sha_learning.pltr.lsha_report as report

from rest_api.models import CaseStudy, CsvFile
from core_algorithm.lsha.sha_learning.domain.sulfeatures import SystemUnderLearning, RealValuedVar

# --- NEW IMPORTS: The Extracted Modules ---
from .dynamic_tracegenerator import CustomTraceGenerator
from .dynamic_sul import (
    parse_data_dynamic, 
    is_chg_pt_dynamic, 
    label_event_dynamic, 
    get_physics_param_dynamic
)
from .teacher import CustomTeacher
from .dynamic_obstable import ObsTable, Row, State

@shared_task
def run_lsha_learning_task(case_study_id):
    try:
        LOGGER = Logger(f'LSHA_TASK_{case_study_id}')
    except NameError:
        logging.basicConfig(level=logging.INFO)
        LOGGER = logging.getLogger(f'LSHA_TASK_{case_study_id}')

    try:
        try:
            cs_instance = CaseStudy.objects.get(id=case_study_id)
        except CaseStudy.DoesNotExist:
            return {"status": "Error", "message": f"CaseStudy {case_study_id} not found."}

        LOGGER.info(f"Starting Task: {cs_instance.name} (ID: {case_study_id})")

        # ===============================================
        # Define Paths
        # ===============================================
        UPPAAL_BIN = "/opt/uppaal/lib/app/bin/verifyta"
        if not os.path.exists(UPPAAL_BIN):
             UPPAAL_BIN = "/usr/bin/verifyta" 
        
        OUTPUT_DIR = "/home/rghaf/Projects/lsha_web/results/upp_results"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Normalise the strategy spelling once.  Historically the DB has
        # held "SIM", the new web UI emits "CSV", and older configs spelt
        # it "UPPAAL" / "UPP".  Centralising the alias here means every
        # downstream comparison ("if RESAMPLE_STRATEGY == X") sees a
        # canonical value and stays in sync with dynamic_tracegenerator
        # and dynamic_sul.
        RAW_STRATEGY = (cs_instance.resample_strategy or 'UPPAAL').strip().upper()
        if RAW_STRATEGY in ('CSV', 'SIM', 'STATIC'):
            RESAMPLE_STRATEGY = 'CSV'
        else:
            RESAMPLE_STRATEGY = 'UPPAAL'
        LOGGER.info(f"RESAMPLE STRATEGY: {RESAMPLE_STRATEGY}")
        
        # Boolean shortcut used in several places below; "are we in
        # CSV-mode regardless of which spelling the DB used?".
        IS_CSV_MODE = (RESAMPLE_STRATEGY == 'CSV')

        # ----------------------------------------------------------
        # Translate English logical keywords ("or", "and") into the
        # UPPAAL operator forms ("||", "&&") for every column-name field
        # the user can type in the Django form.  The web UI accepts the
        # friendly spelling because it is easier to read; the parser
        # later strips whitespace and matches the section header
        # "amy.busy || amy.p_2", which would NOT match
        # "amy.busy or amy.p_2" with whitespace stripped.
        #
        # Apply to driver_signal, main_variable, and context_variables
        # uniformly — historically only driver_signal was converted,
        # which silently broke HRI when the user typed "or" in the
        # main_variable or context_variables fields.
        # ----------------------------------------------------------
        def _to_uppaal(s):
            """Recursively normalise English logical keywords to UPPAAL operators."""
            if isinstance(s, str):
                return (s
                        .replace(" or ",  " || ")
                        .replace(" OR ",  " || ")
                        .replace(" and ", " && ")
                        .replace(" AND ", " && "))
            if isinstance(s, list):
                return [_to_uppaal(x) for x in s]
            return s

        DRIVER_SIGNAL     = _to_uppaal(cs_instance.driver_signal)
        MAIN_VARIABLE     = _to_uppaal(cs_instance.main_variable)
        CONTEXT_VARIABLES = _to_uppaal(cs_instance.context_variables)

        if CONTEXT_VARIABLES is None:
            CONTEXT_VARIABLES = []

        # Django's JSONField stores the value as a Python dict and returns
        # it that way; some legacy CaseStudy rows have a literal JSON string
        # because they were inserted via the Django admin or shell.  Accept
        # both shapes so the REST API and the admin keep working.
        raw_user_json = cs_instance.user_json
        if isinstance(raw_user_json, (dict, list)):
            data_dict = raw_user_json
        elif isinstance(raw_user_json, (str, bytes, bytearray)):
            data_dict = json.loads(raw_user_json)
        else:
            data_dict = {}

        # When the UI sends the full outer JSON document as the user_json field
        # value (instead of only the inner user_json block), the stored string
        # parses into a dict whose top-level keys are 'name', 'email', …,
        # 'user_json'.  In that case data_dict.get('events') returns [] because
        # events live inside data_dict['user_json'].  Unwrap and also recover
        # any outer-level fields that the DB may not have stored separately.
        if (isinstance(data_dict, dict)
                and 'user_json' in data_dict
                and isinstance(data_dict.get('user_json'), dict)
                and not any(k in data_dict for k in ('events', 'models', 'variables'))):
            outer_data = data_dict
            data_dict = data_dict['user_json']
            if not CONTEXT_VARIABLES:
                CONTEXT_VARIABLES = _to_uppaal(outer_data.get('context_variables', []) or [])
            if not DRIVER_SIGNAL:
                DRIVER_SIGNAL = _to_uppaal(outer_data.get('driver_signal', []) or [])
            if not MAIN_VARIABLE:
                MAIN_VARIABLE = _to_uppaal(outer_data.get('main_variable', '') or '')

        events = data_dict.get('events', [])
        real_events = []
        for e in events:
            real_events.append(Event(e.get('channel', ''), e.get('guard', ''), e.get('symbol', '')))
            if 'trigger_value' in e:
                real_events[-1].trigger_value = e['trigger_value']
            
        trace_gen_config = data_dict.get('trace_generation', {})

        LOGGER = Logger(f'RESAMPLE STRATEGY:{RESAMPLE_STRATEGY}')

        
        uppaal_model_path = cs_instance.uppaal_model_file.path if cs_instance.uppaal_model_file else None
        uppaal_query_path = cs_instance.uppaal_query_file.path if cs_instance.uppaal_query_file else None
        csv_paths = []

        
        
        
        if IS_CSV_MODE:
            # Directly query the database table to guarantee we get the latest data.
            # Works for either DB spelling ("SIM" or "CSV") thanks to the
            # IS_CSV_MODE normalisation done above.
            csv_objects = CsvFile.objects.filter(case_study_id=case_study_id)

            if csv_objects.exists():
                for csv in csv_objects:
                    if csv.file:
                        abs_path = csv.file.path
                        LOGGER.info(f"FOUND CSV: {abs_path}")
                        csv_paths.append(abs_path)
            else:
                LOGGER.info(f"CRITICAL WARNING: Database reports 0 CSV files for Case Study {case_study_id}!")

            trace_gen_config['csv_files'] = csv_paths

            LOGGER.info(f"CSV PATHS: {csv_paths}")

        try:
            custom_tg = CustomTraceGenerator(
                cs_name=cs_instance.name,
                resample_strategy=RESAMPLE_STRATEGY,
                uppaal_bin_path=UPPAAL_BIN,
                uppaal_model_path=uppaal_model_path,
                uppaal_query_path=uppaal_query_path,
                csv_files=csv_paths,
                output_dir=OUTPUT_DIR,
                trace_gen_config=trace_gen_config
            )
            LOGGER.info("Generator Initialized.")
            LOGGER.info(f"Generator: {custom_tg}")
        except Exception as e:
            LOGGER.error(f"FAILED to initialize generator: {e}")
            raise e

        # ---------------------------------------------------------
        # PHASE 1: TRACE GENERATOR TEST
        # ---------------------------------------------------------
        LOGGER.info("="*40)
        LOGGER.info("PHASE 1: CUSTOM TRACE GENERATOR TEST")
        LOGGER.info("="*40)

        generated_files = []
        
        if IS_CSV_MODE or len(csv_paths) > 0:
            LOGGER.info("CSV Strategy detected. Bypassing trace generation to feed raw CSVs to Learner.")
            generated_files = csv_paths

            # Force the strategy to CSV for the SUL Engine
            # RESAMPLE_STRATEGY = 'SIM'

            # --- THE SILVER BULLET (ONE-SHOT) ---
            #
            # The Teacher's ref_query calls TG.get_traces() once per ambiguous
            # word; in CSV mode we have a fixed bundle of files and there is
            # no benefit to re-processing them each time.  Returning the same
            # paths on every call caused process_data to be invoked O(words *
            # files) times, blowing sul.traces up to 10k+ identical entries
            # and making get_counterexample sweep over all of them — a single
            # iteration was taking 30 seconds.
            #
            # One-shot semantics: yield the CSV bundle on the FIRST call, an
            # empty list on every subsequent call.  ref_query treats `[]` as
            # "no fresh data available" and silently no-ops, so the loop
            # converges instead of accumulating duplicates.
            #
            # bundle_all_csv=true (in trace_generation.csv) treats ALL uploaded
            # files as a single merged trace rather than individual per-file
            # traces.  Use this when files cover the same time window but split
            # signals across them (e.g. GREEN: pump-speed, temperature, and
            # decanter data in separate CSVs that must be joined before parsing).
            _csv_sub = trace_gen_config.get('csv', {}) or {}
            _bundle  = bool(_csv_sub.get('bundle_all_csv', False)
                            or trace_gen_config.get('bundle_all_csv', False))

            _csv_yielded = {'done': False}
            def _csv_paths_once(*_args, **_kwargs):
                if _csv_yielded['done']:
                    return []
                _csv_yielded['done'] = True
                # Bundle mode: one "trace" = all files merged.
                # Per-file mode: one trace per file (default, preserves
                # per-operating-window observability for Energy/W7 datasets).
                return [csv_paths] if _bundle else csv_paths
            custom_tg.get_traces = _csv_paths_once
            
        else:
            if len(real_events) > 0:
                test_trace_len = min(3, len(real_events))
                test_trace_obj = Trace(real_events[:test_trace_len])
                custom_tg.set_word(test_trace_obj)
            else:
                custom_tg.set_word(Trace([]))

            LOGGER.info("Calling get_traces(1)...")
            generated_files = custom_tg.get_traces(1)
            
        LOGGER.info(f"Generated Files Available for SUL: {generated_files}")
        # ---------------------------------------------------------
        # PHASE 2: SUL FUNCTIONS TEST
        # ---------------------------------------------------------
        sul_args = {
            'name': cs_instance.name,
            'resample_strategy': RESAMPLE_STRATEGY,          # <--- THE FIX: Explicitly pass the strategy
            'trace_generation': trace_gen_config,            # <--- THE FIX: Pass the JSON instructions
            'default_m': 0,
            'default_d': 0,
            'driver': DRIVER_SIGNAL,
            'main_var': MAIN_VARIABLE,
            'context_variables': CONTEXT_VARIABLES,
            'events': data_dict.get('events', []),
            'models': data_dict.get('models', []),
            # New web-UI fields — the dynamic SUL knows how to consume each:
            #   variables[]  — declarative role/source/type, replaces the
            #                  legacy main_var/driver_signal/context split
            #   constants    — symbolic thresholds (p_min, MIN_SPEED, …)
            #                  injected into every guard's eval context
            #   aliases      — top-level friendly→physical column mapping,
            #                  in addition to per-variable "source" fields
            'variables': data_dict.get('variables', []),
            'constants': data_dict.get('constants', {}),
            'aliases':   data_dict.get('aliases', {}),
        }

        LOGGER.info(f"[SUL Args] Initial: main='{MAIN_VARIABLE}', driver={DRIVER_SIGNAL}, context={CONTEXT_VARIABLES}")

        # ----------------------------------------------------------
        # Resolve the new variables[] / aliases JSON shape NOW so the
        # downstream SystemUnderLearning gets correct main_var / driver
        # labels even when the DB-level main_variable / driver_signal
        # fields were left blank.  Without this, RealValuedVar(label=None)
        # would prevent the SUL from finding the main signal in the
        # parsed dict, and the L* loop would silently produce zero events.
        # ----------------------------------------------------------
        try:
            from .dynamic_sul import _resolve_variable_roles, _build_target_vars
            _build_target_vars(sul_args)   # also calls _resolve_variable_roles
            # ----------------------------------------------------------
            # Re-read the FRIENDLY main / driver labels UNCONDITIONALLY.
            #
            # _build_target_vars now writes the friendly names back into
            # sul_args (translating any physical-name spelling the user
            # may have typed in the Django form).  We MUST overwrite the
            # local MAIN_VARIABLE / DRIVER_SIGNAL with those friendly
            # names so:
            #   (a) RealValuedVar(label=MAIN_VARIABLE) below uses the same
            #       label the parse adapter will give the main signal, and
            #   (b) SystemUnderLearning's self.driver list contains the
            #       friendly labels that find_chg_pts will filter against.
            #
            # Without this, find_chg_pts crashes with
            # ``IndexError: list index out of range`` because driver_sig
            # is empty (the parse adapter labelled the SampledSignal as
            # 'RPM' but self.driver still held 'HEADSTOCK__…__RPM').
            # ----------------------------------------------------------
            if sul_args.get('main_var'):
                MAIN_VARIABLE = sul_args['main_var']
            if sul_args.get('driver'):
                DRIVER_SIGNAL = sul_args['driver']
            if sul_args.get('context_variables') is not None:
                CONTEXT_VARIABLES = sul_args['context_variables']
            LOGGER.info(
                f"[Args] Resolved main='{MAIN_VARIABLE}', driver={DRIVER_SIGNAL}, "
                f"alias_map={sul_args.get('alias_map') or {}}"
            )
        except Exception as exc:
            LOGGER.warn(f"[Args] Could not pre-resolve variables[]: {exc}")

        # ---------------------------------------------------------
        # PHASE 2 — End-to-end smoke test of the dynamic SUL helpers
        #
        # Runs the parser ONCE on the full trace bundle, scans the
        # entire trace for change-points, and logs the resulting
        # event-symbol histogram.  This is the single most useful
        # diagnostic for spotting "only one event ever fires"
        # failure modes BEFORE the L* loop spirals.
        # ---------------------------------------------------------
        if generated_files and len(generated_files) > 0 and os.path.exists(generated_files[0]):
            LOGGER.info("=" * 60)
            LOGGER.info("[PHASE 2] Dynamic SUL smoke test")
            LOGGER.info("=" * 60)

            signals = parse_data_dynamic(generated_files, args=sul_args)
            n_time = len(signals.get('time', []))
            LOGGER.info(f"[PHASE 2] parse_data_dynamic → {n_time} samples, "
                        f"signals={list(signals.keys())}")

            if n_time > 0:
                # Walk the full trace, collect (index, symbol) for every
                # change-point.  Capped at 100k to avoid runaway logs.
                cap = min(n_time, 100_000)
                chg_indices = []
                event_hist  = {}
                for i in range(1, cap):
                    if is_chg_pt_dynamic(signals, i, args=sul_args):
                        chg_indices.append(i)
                        sym = label_event_dynamic(signals, i, args=sul_args)
                        event_hist[sym] = event_hist.get(sym, 0) + 1

                LOGGER.info(f"[PHASE 2] change-points: {len(chg_indices)} "
                            f"(scanned {cap} samples)")
                if event_hist:
                    hist_str = ', '.join(f"{k}={v}" for k, v in
                                         sorted(event_hist.items(), key=lambda x: -x[1]))
                    LOGGER.info(f"[PHASE 2] event histogram: {hist_str}")
                    if len(event_hist) == 1:
                        LOGGER.warn(
                            "[PHASE 2] Only one event symbol fired across "
                            "the trace — guards likely failing or data range "
                            "doesn't trigger other events. Inspect '[SUL] Guard "
                            "eval failed' / 'No guard matched' lines above."
                        )
                else:
                    LOGGER.warn(
                        "[PHASE 2] No change-points found in the first "
                        f"{cap} samples — check tolerances and round_columns."
                    )

                # Sample physics-param so we can see if the metric is stable
                if chg_indices:
                    s_i = chg_indices[0]
                    e_i = chg_indices[1] if len(chg_indices) > 1 else min(s_i + 20, n_time - 1)
                    params = get_physics_param_dynamic(signals, s_i, e_i, args=sul_args)
                    LOGGER.info(f"[PHASE 2] sample segment params {s_i}->{e_i}: {params}")
        else:
            LOGGER.warn("[PHASE 2] SKIPPED — no input files available.")

        # ---------------------------------------------------------
        # PHASE 3: BUILD LSHA SUL OBJECTS
        # ---------------------------------------------------------
        flows = []
        m2d = {}
        for m in data_dict.get('models', []):
            m_id = m['id']
            m_type = m['type']
            
            def make_ideal_flow(model_type):
                def ideal_flow(interval, initial_val):
                    t_floats = []
                    for t in interval:
                        if hasattr(t, 'to_secs'):
                            t_floats.append(t.to_secs())
                        else:
                            t_floats.append(float(t))
                            
                    t_norm = np.array(t_floats) - t_floats[0]
                    
                    if 'DECAY' in model_type and 'LINEAR' not in model_type:
                        return initial_val * np.exp(-0.01 * t_norm)
                    elif 'GROWTH' in model_type and 'LINEAR' not in model_type:
                        return initial_val + (100.0 - initial_val) * (1 - np.exp(-0.01 * t_norm))
                    elif 'LINEAR_GROWTH' in model_type:
                        return initial_val + 0.1 * t_norm
                    elif 'LINEAR_DECAY' in model_type:
                        return initial_val - 0.1 * t_norm
                    return np.full_like(t_norm, initial_val)
                return ideal_flow

            flows.append(FlowCondition(m_id, make_ideal_flow(m_type)))
            m2d[m_id] = [m_id]

        rv_vars = [RealValuedVar(flows=flows, distr=[], m2d=m2d, label=MAIN_VARIABLE)]
        
        # SUL ADAPTERS
        class CustomPoint:
            def __init__(self, t_obj, val):
                self.timestamp = t_obj
                self.t = t_obj
                self.value = val

        class CustomSignal:
            def __init__(self, label, points):
                self.label = label
                self.points = points
                self.t = [pt.t for pt in points]
                self.values = [pt.value for pt in points]
                # Pre-sorted seconds array for O(log n) binary search in get_segments.
                # Points are built from a time-sorted DataFrame so _secs is non-decreasing.
                self._secs = [pt.timestamp.to_secs() for pt in points]

        def parse_adapter(sim, *, args):
            sig_dict = parse_data_dynamic(sim, args=args)
            args['__current_trace_cache__'] = sig_dict
            # Reset the change-point iteration counter so is_chg_pt_adapter
            # can map its (curr, prev) call sequence to a positional index
            # in sig_dict['time'].  The legacy SystemUnderLearning.find_chg_pts
            # iterates samples in order but passes only value lists (no
            # timestamps) to is_chg_pt, so we cannot recover the index any
            # other way.  See is_chg_pt_adapter for the consumer side.
            args['__chg_iter_counter__'] = -1

            res = []
            t_floats = sig_dict.get('time', [])
            legacy_driver = args.get('driver')
            if isinstance(legacy_driver, list) and len(legacy_driver) > 0:
                legacy_driver = legacy_driver[0]
                args['driver'] = legacy_driver

            # IMPORTANT: keep the Timestamp.to_secs() value identical to the
            # raw float in ``sig_dict['time']`` so label_event_adapter's
            #   np.where(sig_dict['time'] == t.to_secs())
            # lookup actually finds the row.  Wrapping with year=2026 / day=1
            # made to_secs() return ~6.4e10 (years×days×86400 + sec), which
            # never matched any element of the seconds-from-start array — so
            # label_event_adapter silently fell through to events_list[0]
            # (= ``i_0``) and the learner saw a single-symbol trace.
            #
            # Timestamp(0, 0, 0, 0, 0, sec).to_secs() == sec, which is exactly
            # what we need.
            for k, v in sig_dict.items():
                if k == 'time': continue
                label = k
                if k == 'main': label = args['main_var']
                if isinstance(args.get('driver_signals'), list) and k in args.get('driver_signals'):
                    label = legacy_driver

                points = []
                for i in range(len(t_floats)):
                    t_obj = Timestamp(0, 0, 0, 0, 0, float(t_floats[i]))
                    points.append(CustomPoint(t_obj, v[i]))
                res.append(CustomSignal(label, points))
            return res

        def is_chg_pt_adapter(curr, prev, *, args):
            """
            Bridge the legacy ``SystemUnderLearning.find_chg_pts`` to the
            dynamic ``is_chg_pt_dynamic``.

            The legacy iterator walks the sample axis in order and hands us
            ``(curr, prev)`` as **value lists** — one driver value per slot,
            with NO timestamp.  A prior implementation tried to recover the
            timestamp from ``curr[0]``, but ``curr[0]`` is just a driver
            value (often 0/1 for boolean drivers); the lookup
            ``np.where(time == 0 or 1)`` matched at most one row, the
            exception fell through, and the adapter degenerated into
            ``curr[1] != prev[1]`` — i.e., change-points fired only on
            transitions of the *second* driver.  For GREEN's all-boolean
            drivers that meant only ``MarciaEspulsore`` transitions ever
            reached the L\\* learner, producing the 51-state ``esp_on``
            chain instead of the full 17-symbol alphabet.

            The fix uses a per-trace iteration counter that
            ``parse_adapter`` resets, so each call maps to the next sample
            index in the dynamic engine's signals dict.
            """
            sig_dict = args.get('__current_trace_cache__')

            # Bump the counter regardless of return path so call N always
            # corresponds to sample index N for the current trace.
            counter = args.get('__chg_iter_counter__', -1) + 1
            args['__chg_iter_counter__'] = counter

            if sig_dict and 'time' in sig_dict and counter < len(sig_dict['time']):
                return is_chg_pt_dynamic(sig_dict, counter, args=args)

            # Fallback: no parsed signals cached — best-effort comparison
            # using whatever shape the legacy iterator passed in.
            try:
                if hasattr(curr, 'value'):
                    return curr.value != prev.value
                if isinstance(curr, (list, tuple)) and len(curr) >= 2 and isinstance(prev, (list, tuple)):
                    # Treat curr/prev as value lists; if any slot differs it's a change.
                    return any(c != p for c, p in zip(curr, prev))
                return curr != prev
            except Exception:
                return False

        def label_event_adapter(events_list, new_signals, t_val, *, args):
            sig_dict = args.get('__current_trace_cache__')
            if not sig_dict:
                return events_list[0] if events_list else None
                
            try:
                t_sec = t_val.to_secs() if hasattr(t_val, 'to_secs') else float(t_val)
                idx = np.where(sig_dict['time'] == t_sec)[0][0]
                
                symbol = label_event_dynamic(sig_dict, idx, args=args)
                for e_obj in events_list:
                    if e_obj.symbol == symbol:
                        return e_obj
            except Exception:
                pass
            return events_list[0] if events_list else None

        # ----------------------------------------------------------
        # Models whose discriminating quantity is a SLOPE rather than a
        # raw value.  When the active flow at a segment's start belongs
        # to one of these families, the adapter must hand the Teacher
        # the per-step ``rate`` rather than the segment ``mean``.
        #
        # Why: cumulative counters (e.g. the W7 dataset's
        # HEADSTOCK__SPINDLE_DRIVE___1___ENERGY column) are monotonic.
        # Two adjacent segments of the same physical mode will produce
        # *different* segment means (the counter has just kept on
        # accumulating), but the same per-step slope.  Reporting the
        # mean to the deterministic HT-query allocates a brand-new
        # ProbDistribution for every segment — exactly the
        # ``ALARM ALARM ALARM`` infinite-distribution pathology shown
        # in the obs-table dump.  Using the rate keeps the metric
        # stable across cumulative segments and lets the table close.
        # ----------------------------------------------------------
        _RATE_BASED_FLOWS = {
            'LINEAR', 'LINEAR_GROWTH', 'LINEAR_DECAY',
            'EXP_GROWTH', 'EXP_DECAY',
        }

        def _flow_type_for(segment, args):
            """Look up the model TYPE (string) for the event active at the
            segment's start, by reusing the dynamic SUL's helpers.
            Returns ``MEAN`` when nothing can be determined safely."""
            sig_dict = args.get('__current_trace_cache__') or {}
            if not sig_dict:
                return 'MEAN'
            try:
                from .dynamic_sul import (
                    label_event_dynamic as _lbl,
                    _get_model_type_for_segment as _typ,
                )
                pt_start = segment[0]
                start_t = pt_start.t.to_secs() if hasattr(pt_start, 't') else (
                    pt_start[0].to_secs() if hasattr(pt_start[0], 'to_secs')
                    else float(pt_start[0])
                )
                start_idx = int(np.where(sig_dict['time'] == start_t)[0][0])
                return (_typ(sig_dict, start_idx, args) or 'MEAN').upper()
            except Exception:
                return 'MEAN'

        def get_physics_param_adapter(segment, flow, *, args):
            sig_dict = args.get('__current_trace_cache__')
            if not sig_dict or len(segment) < 2:
                return 0.0

            try:
                pt_start, pt_end = segment[0], segment[-1]
                start_t = pt_start.t.to_secs() if hasattr(pt_start, 't') else (pt_start[0].to_secs() if hasattr(pt_start[0], 'to_secs') else float(pt_start[0]))
                end_t = pt_end.t.to_secs() if hasattr(pt_end, 't') else (pt_end[0].to_secs() if hasattr(pt_end[0], 'to_secs') else float(pt_end[0]))

                start_idx = np.where(sig_dict['time'] == start_t)[0][0]
                end_idx = np.where(sig_dict['time'] == end_t)[0][0]

                params = get_physics_param_dynamic(sig_dict, start_idx, end_idx, args=args)

                # Decide which of {mean, rate} to report based on the model
                # type of the event that fired at the segment start.  See
                # _RATE_BASED_FLOWS comment above for the rationale.
                flow_type = _flow_type_for(segment, args)

                val = 0.0
                if flow_type in _RATE_BASED_FLOWS and 'rate' in params:
                    val = float(params['rate'])
                elif 'mean' in params:
                    val = float(params['mean'])
                elif 'rate' in params:
                    val = float(params['rate'])

                # THE MAGIC FIX: Prevent infinite loops in D mode!
                return round(val, 4)

            except Exception as e:
                return 0.0

        sul = SystemUnderLearning(
            rv_vars=rv_vars,
            events=real_events,
            parse_f=partial(parse_adapter, args=sul_args),
            label_f=partial(label_event_adapter, args=sul_args),
            param_f=partial(get_physics_param_adapter, args=sul_args),
            is_chg_pt=partial(is_chg_pt_adapter, args=sul_args),
            args=sul_args
        )

        # ---------------------------------------------------------
        # TRACE SPLITTING (max_trace_events)
        #
        # With a single long trace (e.g. 271 events for GREEN), L*'s
        # get_segments(word) returns at most ONE segment per word prefix,
        # which is below any n_min > 1.  The teacher then issues a
        # counterexample whose length grows with the trace, driving
        # make_closed to iterate O(n_events) times and producing an
        # O(n³) hang.
        #
        # Setting max_trace_events=K in trace_generation.csv causes
        # process_data to split the single long trace into ceil(N/K)
        # sub-traces of ≤K events each (all sharing the same raw signal
        # array).  get_segments then finds multiple matches per prefix,
        # satisfying n_min and bounding counterexample length to K.
        # ---------------------------------------------------------
        _csv_sub_mt = trace_gen_config.get('csv', {}) or {}
        _max_trace_events = int(_csv_sub_mt.get('max_trace_events', 0) or 0)
        if _max_trace_events > 0:
            _orig_pd = sul.process_data

            def _split_pd(path, _orig=_orig_pd, _max=_max_trace_events):
                _orig(path)
                if not sul.traces or len(sul.traces[-1]) <= _max:
                    return
                tt   = sul.timed_traces.pop()
                sigs = sul.signals.pop()
                sul.traces.pop()
                n_ev = len(tt)
                for start in range(0, n_ev, _max):
                    end = min(start + _max, n_ev)
                    sub_tt = TimedTrace(list(tt.t[start:end]), list(tt.e[start:end]))
                    sul.timed_traces.append(sub_tt)
                    sul.signals.append(sigs)
                    sul.traces.append(Trace(tt=sub_tt))

            sul.process_data = _split_pd
            LOGGER.info(
                f"[PHASE 3] max_trace_events={_max_trace_events}: "
                "process_data monkey-patched — long traces will be split into sub-traces."
            )

        # ---------------------------------------------------------
        # PHASE 4: CUSTOM TEACHER & LEARNER
        # ---------------------------------------------------------
        teacher_config = {
            'noise': getattr(cs_instance, 'noise', 0.0),
            'p_value': getattr(cs_instance, 'p_value', 0.05),
            'mi_query': getattr(cs_instance, 'mi_query', False),
            'plot_ddtw': getattr(cs_instance, 'plot_ddtw', False),
            'ht_query': getattr(cs_instance, 'ht_query', False),
            'ht_query_type': getattr(cs_instance, 'ht_query_type', 'D'),
            'eq_condition': getattr(cs_instance, 'eq_condition', 's'),
            'n_min': getattr(cs_instance, 'n_min', 10),
            'is_aggregation': getattr(cs_instance, 'is_aggregation', False)
        }

        teacher = CustomTeacher(
            sul=sul,
            config_data=teacher_config,
            trace_generator=custom_tg
        )

        print("\n" + "="*40)
        print("PHASE 5: RUNNING L* LEARNING ALGORITHM")
        print("="*40)

        startTime = datetime.now()

        try:
            long_traces = [Trace(events=[e]) for e in sul.events]
            obs_table = ObsTable([], [Trace(events=[])], long_traces)
            learner = Learner(teacher, obs_table)

            LOGGER.info("[PHASE 5] Running run_lsha() …")
            # debug_print=False — emits |S|/|E|/|low_S| summaries instead of
            # dumping the full obs table on every make_closed / make_consistent
            # / counterexample iteration (the workingworking… wall of text).
            learned_ha = learner.run_lsha(debug_print=False, filter_empty=True)
            LOGGER.info("[PHASE 5] Learning complete — automaton generated.")

            # ---------------------------------------------------------
            # PHASE 6: SAVING THE RESULTS
            # ---------------------------------------------------------
            FINAL_OUT_DIR = os.path.join(settings.BASE_DIR, "results", "final_results")
            os.makedirs(FINAL_OUT_DIR, exist_ok=True)

            clean_name = cs_instance.name.replace(" ", "_")
            SHA_NAME = f"{clean_name}_{cs_instance.resample_strategy}"

            # Build the graph object, then render explicitly so PDF is created
            # even when view=False (without opening a viewer window).
            graphviz_sha = ha_pltr.to_graphviz(learned_ha, SHA_NAME, FINAL_OUT_DIR + "/", view=False)

            rendered_pdf_path = graphviz_sha.render(view=False)

            pdf_path = rendered_pdf_path if os.path.exists(rendered_pdf_path) else os.path.join(FINAL_OUT_DIR, f"{SHA_NAME}.pdf")
            if not os.path.exists(pdf_path):
                pdf_path = os.path.join(FINAL_OUT_DIR, f"{SHA_NAME}.gv.pdf") 

            txt_path = os.path.join(FINAL_OUT_DIR, f"{SHA_NAME}_source.txt")

            with open(txt_path, 'w') as f:
                f.write(graphviz_sha.source)

            cs_instance = CaseStudy.objects.get(id=case_study_id) 

            if os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    cs_instance.final_result_pdf.save(f"{SHA_NAME}_automaton.pdf", File(f), save=False)
            
            if os.path.exists(txt_path):
                with open(txt_path, 'rb') as f:
                    cs_instance.final_result_txt.save(f"{SHA_NAME}_source.txt", File(f), save=False)

            cs_instance.status = 'COMPLETED'
            cs_instance.save()

            events_labels_dict = {e.get('symbol', ''): e.get('symbol', '') for e in events}

            report.save_data(
                teacher.symbols, 
                teacher.distributions, 
                learner.obs_table,
                len(teacher.signals), 
                datetime.now() - startTime, 
                SHA_NAME, 
                events_labels_dict,
                FINAL_OUT_DIR + "/"
            )

            LOGGER.info(f"----> EXPERIMENTAL RESULTS SAVED IN: {FINAL_OUT_DIR}")

        except Exception as e:
            LOGGER.error(f"Learning Algorithm Failed: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "Error", "message": f"Learning Error: {e}"}

        return {"status": "Success", "message": "LSHA Algorithm finished successfully!"}

    except Exception as e:
        LOGGER.error(f"Task Execution Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "Error", "message": f"Task Error: {e}"}