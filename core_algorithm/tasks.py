import os
import sys
import logging
import shutil
import glob
import configparser
from datetime import datetime
from celery import shared_task
from django.conf import settings # Use Django settings

# --- CRITICAL: Add all module paths to the worker's sys.path ---
# This uses the paths you just defined in settings.py

# --- Now, your other imports (like 'import sha_learning...') will work ---
# import sha_learning.pltr.lsha_report as report
# ... (rest of your imports)
# --- IMPORTANT: Add project paths to sys.path for Celery worker ---
# Resolve root paths from Django settings with safe defaults
try:
    LSHA_ROOT = getattr(settings, 'LSHA_ROOT', os.path.join(settings.BASE_DIR, 'LSHA_WEB/core_algorithm', 'lsha'))
    AUTOTWIN_ROOT = getattr(settings, 'AUTOTWIN_ROOT', os.path.join(settings.BASE_DIR, 'core_algorithm', 'autotwin_automata_learning'))
    SKG_ROOT = getattr(settings, 'SKG_ROOT', os.path.join(settings.BASE_DIR, 'core_algorithm', 'skg_connector'))
    MAPPER_ROOT = getattr(settings, 'MAPPER_ROOT', os.path.join(settings.BASE_DIR, 'core_algorithm', 'sha2dt_semantic_mapper'))
except Exception:
    # If settings or BASE_DIR are not available for some reason, fall back to None
    LSHA_ROOT = None
    AUTOTWIN_ROOT = None
    SKG_ROOT = None
    MAPPER_ROOT = None

# Ensure the worker can import the LSHA packages
if LSHA_ROOT:
    sys.path.insert(0, str(LSHA_ROOT))
if AUTOTWIN_ROOT:
    sys.path.insert(0, str(AUTOTWIN_ROOT))
if SKG_ROOT:
    sys.path.insert(0, str(SKG_ROOT))
if MAPPER_ROOT:
    sys.path.insert(0, str(MAPPER_ROOT))
# --- End Path Setup ---

# --- Now import your project modules ---
try:
    import sha_learning.pltr.lsha_report as report
    import sha_learning.pltr.sha_pltr as ha_pltr
    # ... (Import ALL other necessary modules from learn_model.py) ...
    from sha_learning.case_studies.auto_twin.sul_definition import getSUL
    # ... (Import specific SULs) ...
    from sha_learning.domain.lshafeatures import Trace
    from sha_learning.domain.obstable import ObsTable
    from sha_learning.domain.sulfeatures import SystemUnderLearning
    from sha_learning.learning_setup.learner import Learner
    from sha_learning.learning_setup.logger import Logger
    from sha_learning.learning_setup.teacher import Teacher
    from sha_learning.pltr.energy_pltr import distr_hist
    # Handle potential import errors for graphics if worker has no display
    try:
        import matplotlib
        matplotlib.use('Agg') # Force non-interactive backend
    except ImportError:
        matplotlib = None # Allow task to run even if plotting fails
except ImportError as e:
    # Log or handle import errors - Celery workers might have path issues
    print(f"Celery Task Import Error: {e}")
    # You might need more robust path handling here
# --- End Imports ---


@shared_task
def run_lsha_learning_task(case_study_name, pov_arg, start_dt_arg, end_dt_arg):
    """
    Celery task to run the LSHA learning algorithm.
    """

    # Initialize logger early (use provided Logger if available, otherwise fall back)
    try:
        LOGGER = Logger(f'LSHA_TASK_{case_study_name}')
    except Exception:
        LOGGER = logging.getLogger(f'LSHA_TASK_{case_study_name}')

    try:
        startTime = datetime.now()

        # --- Configuration Loading ---
        # compute a local LSHA root fallbacking to Django settings if needed
        lsha_root = LSHA_ROOT or getattr(settings, 'LSHA_ROOT', os.path.join(settings.BASE_DIR, 'core_algorithm', 'lsha'))

        config = configparser.ConfigParser()
        config_file_path = os.path.join(lsha_root, 'sha_learning', 'resources', 'config', 'config.ini')
        if not os.path.exists(config_file_path):
             return f"Error: Config file not found at {config_file_path}"
        config.read(config_file_path)

        # --- Get SUL based on input ---
        CS = case_study_name
        RESAMPLE_STRATEGY = config['SUL CONFIGURATION']['RESAMPLE_STRATEGY'] # Still read from config
        LOGGER.info(f"Starting LSHA task for {CS} with POV={pov_arg}")  # now safe to use LOGGER
        SUL: SystemUnderLearning = None
        events_labels_dict = None
        # (Copy the SUL selection logic from learn_model.py here)
        if CS == 'THERMO':
            from sha_learning.case_studies.thermostat.sul_definition import thermostat_cs
            SUL = thermostat_cs
        elif CS == 'HRI':
             from sha_learning.case_studies.hri.sul_definition import hri_cs
             SUL = hri_cs
        # ... (Add other elif conditions for ENERGY, AUTO_TWIN, GR3N) ...
        elif CS == 'AUTO_TWIN':
             from sha_learning.case_studies.auto_twin.sul_definition import getSUL
             SUL, events_labels_dict = getSUL()
        else:
            raise ValueError(f"Unknown Case Study: {CS}")

        # --- Initialize Teacher & Learner ---
        TEACHER = Teacher(SUL, pov=pov_arg, start_dt=start_dt_arg, end_dt=end_dt_arg)

        long_traces = [Trace(events=[e]) for e in SUL.events]
        obs_table = ObsTable([], [Trace(events=[])], long_traces)
        LEARNER = Learner(TEACHER, obs_table)

        # --- Run Learning Algorithm ---
        LEARNED_HA = LEARNER.run_lsha(filter_empty=True)

        # --- Save Results ---
        # (Copy result saving logic: paths, plotting, report generation)
        # Ensure paths are correct within the Django project context
        HA_SAVE_PATH = config['SUL CONFIGURATION']['SHA_SAVE_PATH'].format(
             os.path.join(LSHA_ROOT, 'sha_learning', '') # Adjust format path if needed
        )
        REPORT_SAVE_PATH = config['SUL CONFIGURATION']['REPORT_SAVE_PATH'] # Assuming absolute or needs formatting

        # Create directories if they don't exist
        os.makedirs(HA_SAVE_PATH, exist_ok=True)
        # Make sure the base path for report exists too if REPORT_SAVE_PATH isn't absolute

        SHA_NAME = '{}_{}_{}'.format(CS, RESAMPLE_STRATEGY, config['SUL CONFIGURATION']['CS_VERSION'])

        if matplotlib and ha_pltr: # Check if plotting available
             graphviz_sha = ha_pltr.to_graphviz(LEARNED_HA, SHA_NAME, HA_SAVE_PATH, view=False) # view=False for server

             # Save source
             sha_source = graphviz_sha.source
             source_path = os.path.join(HA_SAVE_PATH, f"{SHA_NAME}_source.txt")
             with open(source_path, 'w') as f:
                 f.write(sha_source)

             # Save plot if configured
             if config['DEFAULT']['PLOT_DISTR'] == 'True' and config['LSHA PARAMETERS']['HT_QUERY_TYPE'] == 'S':
                 distr_hist(TEACHER.hist, SHA_NAME, HA_SAVE_PATH) # Pass save path

        # Save report data
        report_full_path_prefix = os.path.join(REPORT_SAVE_PATH, SHA_NAME) # Combine path and name prefix

        if report:
            report.save_data(TEACHER.symbols, TEACHER.distributions, LEARNER.obs_table,
                            len(TEACHER.signals), datetime.now() - startTime, SHA_NAME, events_labels_dict,
                            report_full_path_prefix) # Pass the prefix instead of cwd

        result_message = f'----> EXPERIMENTAL RESULTS SAVED IN: {report_full_path_prefix}.txt'
        LOGGER.info(result_message)
        return result_message # Return success message

    except Exception as e:
        LOGGER.error(f"LSHA Task Failed: {e}")
        # Optionally: Use Celery's state update to report failure
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        return f"Error during LSHA task execution: {e}"

# --- Helper function for distr_hist if needed ---
def distr_hist(hist, name, save_path):
    # Placeholder: Implement or adapt the histogram plotting to save to save_path
    print(f"Plotting histograms for {name} to {save_path}")
    pass