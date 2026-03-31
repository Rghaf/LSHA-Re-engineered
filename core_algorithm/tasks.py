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
# from core_algorithm.lsha.sha_learning.learning_setup.plotter import distr_hist

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

        # ==============================================
        # 1. Fetch Data
        # In this section we get the Case Study from the django model and
        # make the data ready to use in the code
        # ==============================================
    try:
        try:
            cs_instance = CaseStudy.objects.get(id=case_study_id)
        except CaseStudy.DoesNotExist:
            return {"status": "Error", "message": f"CaseStudy {case_study_id} not found."}

        LOGGER.info(f"Starting Task: {cs_instance.name} (ID: {case_study_id})")

        # ===============================================
        # Define Paths
        # This paths should be based on the actual paths where you have installed
        # UPPAAL, and where you want save the results come from the UPPAAL on the
        # machine which is going to run the Code.
        # ===============================================
        UPPAAL_BIN = "/opt/uppaal/lib/app/bin/verifyta"
        if not os.path.exists(UPPAAL_BIN):
             UPPAAL_BIN = "/usr/bin/verifyta" 
        
        OUTPUT_DIR = "/home/rghaf/Projects/lsha_web/results/upp_results"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        
        DRIVER_SIGNAL = cs_instance.driver_signal
        MAIN_VARIABLE = cs_instance.main_variable
        CONTEXT_VARIABLES = cs_instance.context_variables

        if CONTEXT_VARIABLES is None:
            CONTEXT_VARIABLES = []
        
        json_data_str = cs_instance.user_json
        data_dict = json.loads(json_data_str)

        # ===============================================
        # Parse JSON & Variables
        # Here we extrace the variables from the JSON field which is filled by the user.
        # ===============================================
        events = data_dict.get('events', [])
        real_events = []
        for e in events:
            real_events.append(Event(e.get('channel', ''), e.get('guard', ''), e.get('symbol', '')))
            if 'trigger_value' in e:
                real_events[-1].trigger_value = e['trigger_value']
            
        trace_gen_config = data_dict.get('trace_generation', {})

        # ---------------------------------------------------------
        # PHASE 1: TRACE GENERATOR TEST
        # ---------------------------------------------------------
        print("\n" + "="*40)
        print("PHASE 1: CUSTOM TRACE GENERATOR TEST")
        print("="*40)


        # ===============================================
        # Run the Custom Trace Generator
        # In this section we initialize the CustomTraceGenerator that implemented instead of
        # the original TraceGenerator by sending the required parameters.
        # ===============================================
        try:
            custom_tg = CustomTraceGenerator(
                cs_name=cs_instance.name,
                resample_strategy=cs_instance.resample_strategy,
                uppaal_bin_path=UPPAAL_BIN,
                uppaal_model_path=cs_instance.uppaal_model_file.path,
                uppaal_query_path=cs_instance.uppaal_query_file.path,
                output_dir=OUTPUT_DIR,
                trace_gen_config=trace_gen_config
            )
            print("Generator Initialized.")
        except Exception as e:
            print(f"FAILED to initialize generator: {e}")
            raise e

        if len(real_events) > 0:
            test_trace_len = min(3, len(real_events))
            test_trace_obj = Trace(real_events[:test_trace_len])
            custom_tg.set_word(test_trace_obj)
        else:
            custom_tg.set_word(Trace([]))

        print("Calling get_traces(1)...")
        generated_files = custom_tg.get_traces(1)
        print(f"Generated Files: {generated_files}")
        
        # ---------------------------------------------------------
        # PHASE 2: SUL FUNCTIONS TEST
        # ---------------------------------------------------------

        # ===============================================
        # Test SUL functions
        # In this section, we define the parameters required for the SUL functions
        # and we test if data is correct and all functions of our new dynamic sul
        # which is a dynamic version of the old sul_functions is working correctly
        # or not.
        # ===============================================
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

            # 1. Test Parse Data
            print(">>> Testing parse_data_dynamic...")
            signals = parse_data_dynamic(trace_file_path, args=sul_args)
            print(f"    Signals Extracted: {list(signals.keys())}")
            
            # 2. Test Change Point
            print("\n>>> Testing is_chg_pt_dynamic...")
            chg_indices = []
            if 'time' in signals:
                for i in range(1, min(100, len(signals['time']))):
                    if is_chg_pt_dynamic(signals, i, args=sul_args):
                        chg_indices.append(i)
                print(f"    Change points found at indices: {chg_indices}")

                # 3. Test Label Event
                print("\n>>> Testing label_event_dynamic...")
                if chg_indices:
                    idx = chg_indices[0]
                    lbl = label_event_dynamic(signals, idx, args=sul_args)
                    print(f"    Event at index {idx}: {lbl}")
                else:
                    print("    No events detected to label.")

                # 4. Test Physics Params
                print("\n>>> Testing get_physics_param_dynamic...")
                if chg_indices:
                    start_i = chg_indices[0]
                    end_i = start_i + 20
                    if end_i < len(signals['time']):
                        params = get_physics_param_dynamic(signals, start_i, end_i, args=sul_args)
                        print(f"    Fitted Params (idx {start_i}-{end_i}): {params}")
            else:
                print("    [Error] No time signal found. Parsing failed.")

            print("\n" + "="*40)
            print("ALL TESTS COMPLETE")
            print("="*40)
        else:
            print("\n[SKIP] Phase 2 Skipped (No File).")

       # ---------------------------------------------------------
        # PHASE 3: BUILD LSHA SUL OBJECTS
        # ---------------------------------------------------------
                # ===============================================
        # Test SUL functions
        # In this section, based on the user input in the JSON field, we build the
        # SUL objects which are required to run the algorithm.
        # ===============================================
        flows = []
        m2d = {}
        
        for m in data_dict.get('models', []):
            m_id = m['id']
            m_type = m['type']
            
            def make_ideal_flow(model_type):
                def ideal_flow(interval, initial_val):
                    # Safely convert Timestamp objects to raw seconds!
                    t_floats = []
                    for t in interval:
                        if hasattr(t, 'to_secs'):
                            t_floats.append(t.to_secs())
                        else:
                            t_floats.append(float(t)) # Fallback if it's already a number
                            
                    # Now do the math on the raw floats
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
        
      # =========================================================
        # SUL ADAPTERS: Translating Dicts <-> SampledSignal objects
        # =========================================================

        # 1. Custom Data Structures to perfectly spoof the legacy SUL
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
            res = []
            
            t_floats = sig_dict.get('time', [])
            
            for k, v in sig_dict.items():
                if k == 'time': continue
                label = args['driver'] if k == 'driver' else (args['main_var'] if k == 'main' else k)
                
                # Build perfect custom points
                points = []
                for i in range(len(t_floats)):
                    t_obj = Timestamp(2026, 1, 1, 0, 0, int(t_floats[i]))
                    points.append(CustomPoint(t_obj, v[i]))
                
                # Build a perfect custom signal
                res.append(CustomSignal(label, points))
                
            return res

        def is_chg_pt_adapter(curr, prev, *, args):
            return curr != prev

        def label_event_adapter(events_list, new_signals, t_val, *, args):
            val = None
            # Find the driver signal's value at this exact timestamp
            for sig in new_signals:
                if sig.label == args['driver']:
                    for pt in sig.points:
                        if pt.t.to_secs() == t_val.to_secs():
                            val = pt.value
                            break
                    break

            # Match the value dynamically to the actual Event OBJECT and return it
            if val is not None:
                for e_obj in events_list:
                    if hasattr(e_obj, 'trigger_value'):
                        try:
                            if float(e_obj.trigger_value) == float(val):
                                return e_obj
                        except (ValueError, TypeError):
                            pass
            
            # Absolute dynamic fallback to prevent crashes if a value goes rogue
            return events_list[0] if events_list else None

        def get_physics_param_adapter(segment, flow, *, args):
            if not segment or len(segment) < 2:
                return 0.0
            
            pt_start, pt_end = segment[0], segment[-1]
            dt = pt_end.t.to_secs() - pt_start.t.to_secs()
            dv = pt_end.value - pt_start.value
            
            return (dv / dt) if dt > 0 else 0.0

        # =========================================================
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
        # PHASE 3: CUSTOM TEACHER INITIALIZATION
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

        LOGGER.info("Passing config to CustomTeacher...")
        
        teacher = CustomTeacher(
            sul=sul,
            config_data=teacher_config,
            trace_generator=custom_tg
        )

        # ---------------------------------------------------------
        # PHASE 4: RUNNING THE L* LEARNING ALGORITHM
        # ---------------------------------------------------------
        print("\n" + "="*40)
        print("PHASE 4: RUNNING L* LEARNING ALGORITHM")
        print("="*40)

        startTime = datetime.now()

        try:
            # 1. Initialize the Observation Table with initial queries
            LOGGER.info("Initializing Observation Table...")
            long_traces = [Trace(events=[e]) for e in sul.events]
            obs_table = ObsTable([], [Trace(events=[])], long_traces)
            
            # 2. Initialize the Learner
            learner = Learner(teacher, obs_table)

            # 3. RUN LEARNING ALGORITHM
            LOGGER.info("Running run_lsha()... This will take some time depending on trace generation!")
            learned_ha = learner.run_lsha(filter_empty=True)
            LOGGER.info("Learning Complete! Automaton Generated.")

            # ---------------------------------------------------------
            # PHASE 5: SAVING THE RESULTS
            # ---------------------------------------------------------
            LOGGER.info("Saving results and generating Graphviz plots...")
            
            # Create a dedicated directory for the final outputs
            FINAL_OUT_DIR = os.path.join(settings.BASE_DIR, "results", "final_results")
            os.makedirs(FINAL_OUT_DIR, exist_ok=True)

            clean_name = cs_instance.name.replace(" ", "_")
            SHA_NAME = f"{clean_name}_{cs_instance.resample_strategy}"

            # Generate Graphviz plot (Set view=False so it doesn't try to open a PDF viewer on your server!)
            graphviz_sha = ha_pltr.to_graphviz(learned_ha, SHA_NAME, FINAL_OUT_DIR + "/", view=True)

            # 1. Define the EXACT string paths on the hard drive
            pdf_path = os.path.join(FINAL_OUT_DIR, f"{SHA_NAME}.pdf")
            
            # (Graphviz sometimes appends .gv.pdf depending on the version, this fallback catches both!)
            if not os.path.exists(pdf_path):
                pdf_path = os.path.join(FINAL_OUT_DIR, f"{SHA_NAME}.gv.pdf") 

            txt_path = os.path.join(FINAL_OUT_DIR, f"{SHA_NAME}_source.txt")

            # Save SHA source to .txt file
            with open(txt_path, 'w') as f:
                f.write(graphviz_sha.source)
            LOGGER.info(f"Graphviz source saved to: {txt_path}")

            # 2. Refresh the DB instance so we don't overwrite fresh data!
            cs_instance = CaseStudy.objects.get(id=case_study_id) 

            # 3. Save the Graphviz PDF using the STRING PATH
            if os.path.exists(pdf_path):
                with open(pdf_path, 'rb') as f:
                    cs_instance.final_result_pdf.save(f"{SHA_NAME}_automaton.pdf", File(f), save=False)
            else:
                LOGGER.error(f"CRITICAL: PDF not found at {pdf_path}")

            # 4. Save the Text Source file
            if os.path.exists(txt_path):
                with open(txt_path, 'rb') as f:
                    cs_instance.final_result_txt.save(f"{SHA_NAME}_source.txt", File(f), save=False)
            else:
                LOGGER.error(f"CRITICAL: TXT not found at {txt_path}")

            # 5. Mark as completed and permanently save the database row!
            cs_instance.status = 'COMPLETED'
            cs_instance.save()

            # Plot distribution history if stochastic
            if teacher.ht_query_type == 'S':
                try:
                    distr_hist(teacher.hist, SHA_NAME) 
                except Exception as e:
                    LOGGER.warn(f"Could not plot distribution history: {e}")

            # Prepare the events_labels_dict for the report
            events_labels_dict = {e.get('symbol', ''): e.get('symbol', '') for e in events}

            # Save the final experimental data report
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