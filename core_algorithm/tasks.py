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
from core_algorithm.lsha.sha_learning.domain.lshafeatures import Trace, FlowCondition
from core_algorithm.lsha.sha_learning.domain.sigfeatures import Event, Timestamp, SampledSignal
from core_algorithm.lsha.sha_learning.domain.obstable import Row, State, ObsTable
from core_algorithm.lsha.sha_learning.learning_setup.logger import Logger
from core_algorithm.lsha.sha_learning.learning_setup.learner import Learner

# --- Plotting & Reporting Imports ---
import core_algorithm.lsha.sha_learning.pltr.sha_pltr as ha_pltr
import core_algorithm.lsha.sha_learning.pltr.lsha_report as report

from rest_api.models import CaseStudy
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

        RESAMPLE_STRATEGY = cs_instance.resample_strategy
        DRIVER_SIGNAL = cs_instance.driver_signal
        MAIN_VARIABLE = cs_instance.main_variable
        CONTEXT_VARIABLES = cs_instance.context_variables

        if CONTEXT_VARIABLES is None:
            CONTEXT_VARIABLES = []
        
        json_data_str = cs_instance.user_json
        data_dict = json.loads(json_data_str)

        events = data_dict.get('events', [])
        real_events = []
        for e in events:
            real_events.append(Event(e.get('channel', ''), e.get('guard', ''), e.get('symbol', '')))
            if 'trigger_value' in e:
                real_events[-1].trigger_value = e['trigger_value']
            
        trace_gen_config = data_dict.get('trace_generation', {})

        # ===============================================
        # Safely extract file paths (Prevents Django .path crashes!)
        # ===============================================
        uppaal_model_path = cs_instance.uppaal_model_file.path if cs_instance.uppaal_model_file else None
        uppaal_query_path = cs_instance.uppaal_query_file.path if cs_instance.uppaal_query_file else None
        # csv_file_path = cs_instance.csv_file.path if cs_instance.csv_file else None

        try:
            custom_tg = CustomTraceGenerator(
                cs_name=cs_instance.name,
                resample_strategy=RESAMPLE_STRATEGY,
                uppaal_bin_path=UPPAAL_BIN,
                uppaal_model_path=uppaal_model_path,
                uppaal_query_path=uppaal_query_path,
                # csv_file=csv_file_path,
                output_dir=OUTPUT_DIR,
                trace_gen_config=trace_gen_config
            )
            print("Generator Initialized.")
        except Exception as e:
            print(f"FAILED to initialize generator: {e}")
            raise e

        # ---------------------------------------------------------
        # PHASE 1: TRACE GENERATOR TEST
        # ---------------------------------------------------------
        print("\n" + "="*40)
        print("PHASE 1: CUSTOM TRACE GENERATOR TEST")
        print("="*40)

        generated_files = []
        
        if RESAMPLE_STRATEGY == 'CSV':
            print("CSV Strategy detected. Bypassing test to preserve the file flag for the Learner.")
            # if csv_file_path:
            #     generated_files = [csv_file_path]
        else:
            if len(real_events) > 0:
                test_trace_len = min(3, len(real_events))
                test_trace_obj = Trace(real_events[:test_trace_len])
                custom_tg.set_word(test_trace_obj)
            else:
                custom_tg.set_word(Trace([]))

            print("Calling get_traces(1)...")
            generated_files = custom_tg.get_traces(1)
            
        print(f"Generated Files Available for SUL: {generated_files}")
        
        # ---------------------------------------------------------
        # PHASE 2: SUL FUNCTIONS TEST
        # ---------------------------------------------------------
        sul_args = {
            'name': cs_instance.name,
            'default_m': 0, 
            'default_d': 0, 
            'driver': DRIVER_SIGNAL,
            'main_var': MAIN_VARIABLE,
            'context_variables': CONTEXT_VARIABLES,
            'events': data_dict.get('events', []),
            'models': data_dict.get('models', [])
        }

        if generated_files and os.path.exists(generated_files[0]):
            print("\n" + "="*40)
            print("PHASE 2: DYNAMIC SUL FUNCTIONS TEST")
            print("="*40)
            
            trace_file_path = generated_files[0]
            print(">>> Testing parse_data_dynamic...")
            signals = parse_data_dynamic(trace_file_path, args=sul_args)
            print(f"    Signals Extracted: {list(signals.keys())}")
            
            print("\n>>> Testing is_chg_pt_dynamic...")
            chg_indices = []
            if 'time' in signals:
                for i in range(1, min(100, len(signals['time']))):
                    if is_chg_pt_dynamic(signals, i, args=sul_args):
                        chg_indices.append(i)
                print(f"    Change points found at indices: {chg_indices}")

                print("\n>>> Testing label_event_dynamic...")
                if chg_indices:
                    idx = chg_indices[0]
                    lbl = label_event_dynamic(signals, idx, args=sul_args)
                    print(f"    Event at index {idx}: {lbl}")

                print("\n>>> Testing get_physics_param_dynamic...")
                if chg_indices:
                    start_i = chg_indices[0]
                    end_i = start_i + 20
                    if end_i < len(signals['time']):
                        params = get_physics_param_dynamic(signals, start_i, end_i, args=sul_args)
                        print(f"    Fitted Params: {params}")
            print("\n" + "="*40)
        else:
            print("\n[SKIP] Phase 2 Skipped (No File).")

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

        def parse_adapter(sim, *, args):
            sig_dict = parse_data_dynamic(sim, args=args)
            args['__current_trace_cache__'] = sig_dict
            
            res = []
            t_floats = sig_dict.get('time', [])
            legacy_driver = args.get('driver')
            if isinstance(legacy_driver, list) and len(legacy_driver) > 0:
                legacy_driver = legacy_driver[0]
                args['driver'] = legacy_driver
            
            for k, v in sig_dict.items():
                if k == 'time': continue
                label = k
                if k == 'main': label = args['main_var']
                if isinstance(args.get('driver_signals'), list) and k in args.get('driver_signals'):
                    label = legacy_driver
                
                points = []
                for i in range(len(t_floats)):
                    t_obj = Timestamp(2026, 1, 1, 0, 0, int(t_floats[i]))
                    points.append(CustomPoint(t_obj, v[i]))
                res.append(CustomSignal(label, points))
            return res

        def is_chg_pt_adapter(curr, prev, *, args):
            sig_dict = args.get('__current_trace_cache__')
            t_sec, curr_val, prev_val = None, None, None
            
            if hasattr(curr, 't'):
                t_sec = curr.t.to_secs()
                curr_val, prev_val = curr.value, prev.value
            elif isinstance(curr, (list, tuple)) and len(curr) >= 2:
                t_sec = curr[0].to_secs() if hasattr(curr[0], 'to_secs') else float(curr[0])
                curr_val, prev_val = curr[1], prev[1] if isinstance(prev, (list, tuple)) else None
            else:
                curr_val, prev_val = curr, prev
                
            if not sig_dict or t_sec is None:
                return curr_val != prev_val
            
            try:
                idx = np.where(sig_dict['time'] == t_sec)[0][0]
                return is_chg_pt_dynamic(sig_dict, idx, args=args)
            except Exception:
                return curr_val != prev_val

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
                
                val = 0.0
                if 'mean' in params: val = float(params['mean'])
                elif 'rate' in params: val = float(params['rate'])
                
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

            LOGGER.info("Running run_lsha()...")
            learned_ha = learner.run_lsha(filter_empty=True)
            LOGGER.info("Learning Complete! Automaton Generated.")

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