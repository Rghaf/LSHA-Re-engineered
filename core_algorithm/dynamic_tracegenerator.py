"""
dynamic_tracegenerator.py — Universal Trace Generator
=====================================================

The Trace Generator is the bridge between the L* learner (which periodically
asks "give me more traces") and the underlying data source (UPPAAL simulation
or static CSV files uploaded by the user).

Two strategies are supported:

  * **UPPAAL**  (``resample_strategy = "UPPAAL"``)
        Patches the user's `.xml` model file in place — injecting the next
        sequence of forced events into the UPPAAL `force_act[]` array — then
        invokes the `verifyta` simulator to produce a fresh trace text file.
        Each call to ``get_traces(n)`` runs `n` independent simulations.

  * **CSV** (``resample_strategy = "CSV"`` or legacy ``"SIM"``)
        The data source is a fixed bundle of CSV files uploaded once.  Since
        these are static — re-asking for new traces would just hand back the
        same data — we yield the file list ONCE and then return an empty
        list on every subsequent call.  This intentionally short-circuits
        the L* refinement loop so the Teacher does not spin forever.

The same generator instance also handles the multi-driver case (THERMO
and HRI use only one force variable; future case studies can use several
force variables in parallel).  The ``build_event_strings`` method emits
one UPPAAL array literal per force variable.

JSON config consumed (all live under ``trace_generation`` in the user's
JSON; the new UI may also nest CSV-only options under ``csv: {...}``):

    xml_force_variable   : str   — single force-array name (THERMO/HRI legacy)
    xml_force_variables  : list  — multiple force-array names (multi-driver)
    xml_action_variable  : str   — boolean flag in the XML to toggle
    max_length           : int   — pad event arrays to this length with -1
    csv.max_length       : int   — same, but inside the new nested csv block
"""

import os
import random
import subprocess


# ---------------------------------------------------------------------------
# Strategy normalisation
# ---------------------------------------------------------------------------
# The web UI / DB / JSON have used at least three names for the CSV strategy
# over time: "SIM", "CSV", and "sim".  Centralise the alias so every dispatch
# below ("if strategy == X") agrees on the canonical spelling.
_CSV_ALIASES    = {'CSV', 'SIM', 'CSV_FILES', 'STATIC'}
_UPPAAL_ALIASES = {'UPPAAL', 'UPP', 'VERIFYTA'}


def _normalise_strategy(raw):
    """Collapse any UI/JSON spelling of the strategy to either ``CSV`` or ``UPPAAL``."""
    s = (raw or 'UPPAAL').strip().upper()
    if s in _CSV_ALIASES:
        return 'CSV'
    if s in _UPPAAL_ALIASES:
        return 'UPPAAL'
    # Anything unknown defaults to UPPAAL — historically the safer default
    # since the CSV path makes assumptions about file uploads being present.
    return 'UPPAAL'


class CustomTraceGenerator:
    """
    Trace Generator with a unified API across UPPAAL and CSV data sources.

    Construction parameters (all keyword-only by convention):
      cs_name           — case study display name (used in output filenames)
      resample_strategy — "UPPAAL" / "CSV" / legacy "SIM" (case-insensitive)
      output_dir        — where verifyta should drop newly generated traces
      trace_gen_config  — the ``trace_generation`` block from the user's JSON
      uppaal_bin_path   — absolute path to the verifyta executable
      uppaal_model_path — absolute path to the user's .xml model file
      uppaal_query_path — absolute path to the .q query file
      csv_files         — list of absolute paths to user-uploaded CSV files
    """

    def __init__(self, cs_name, resample_strategy, output_dir, trace_gen_config,
                 uppaal_bin_path=None, uppaal_model_path=None, uppaal_query_path=None,
                 csv_files=None):
        # The case study name is used in generated filenames; spaces are
        # replaced with underscores so the file is shell-friendly.
        self.cs_name = cs_name.replace(" ", "_")

        # Centralised normalisation so every later branch agrees on whether
        # we are CSV-mode or UPPAAL-mode regardless of how it was spelt.
        self.resample_strategy = _normalise_strategy(resample_strategy)

        # UPPAAL specific paths
        self.uppaal_bin_path = uppaal_bin_path
        self.uppaal_model_path = uppaal_model_path
        self.uppaal_query_path = uppaal_query_path

        # CSV specific paths (Now safely expects a LIST of file paths)
        self.csv_files = list(csv_files) if csv_files else []

        # Augment the CSV file list with any extra groups declared in the
        # JSON.  GREEN, for example, splits boolean signals and real-valued
        # signals across different files; we merge them into a single list
        # so the SUL parser sees them all.  Duplicates are skipped.
        cfg = trace_gen_config or {}
        nested_csv = cfg.get('csv') if isinstance(cfg.get('csv'), dict) else {}
        for key in ('boolean_files', 'real_files', 'extra_files', 'files'):
            for src in (cfg.get(key), nested_csv.get(key)):
                if isinstance(src, list):
                    for p in src:
                        # Only merge absolute paths that actually exist on
                        # disk; "files" inside the JSON is sometimes just
                        # a hint about original filenames, not real paths.
                        if isinstance(p, str) and os.path.exists(p) and p not in self.csv_files:
                            self.csv_files.append(p)

        self.output_dir = output_dir

        # Keep both the raw config and the flattened nested-csv config
        # available so ``max_length`` and friends can be looked up from
        # either location without spreading the lookup logic everywhere.
        self.config = cfg
        self._csv_cfg = nested_csv

        # Padding length for the UPPAAL force_act[] arrays.  Read from the
        # nested csv block first (new UI) then the flat block (legacy).
        self.max_e = int(self._csv_cfg.get('max_length',
                          self.config.get('max_length', 15)))

        # XML field names for the boolean execution flag and the single
        # force array — kept for backward compatibility with single-driver
        # case studies.  Multi-driver case studies use ``xml_force_variables``.
        self.xml_force_var = self.config.get('xml_force_variable', 'force_open')
        self.xml_action_var = self.config.get('xml_action_variable', 'force_exe')

        # Set later by ``set_word`` — the prefix word the Teacher wants the
        # next batch of UPPAAL simulations to begin with.
        self.word = None
        self.events = []

        # Flag to prevent infinite loops when processing static CSV files.
        # See ``get_traces_csv`` for the rationale.
        self.csv_yielded = False

    # ------------------------------------------------------------------
    # Word / event handoff from the Teacher
    # ------------------------------------------------------------------

    def set_word(self, w):
        """
        Store the word the Teacher wants the next UPPAAL simulation to start
        with.  In CSV mode this is a no-op because the trace file is fixed;
        we still record it so the Teacher's polymorphic API works.
        """
        self.word = w
        self.events = w.events

    def build_event_strings(self):
        """
        Dynamically build the UPPAAL `int force_act[MAX_E] = {...}` array
        literals from the current word's events.  Returns a dictionary
        mapping XML variable name → padded literal string, plus the count
        of real events (used by ``fix_model`` to size the TAU horizon).

        Multi-driver: when ``xml_force_variables`` is a list, one array is
        emitted per variable, and each event's per-driver value is taken
        from its ``trigger_values`` dict.  Single-driver: the legacy
        ``trigger_value`` scalar is broadcast to whichever single variable
        is in use.  Missing or untriggered events are encoded as ``-1``.
        """
        arrays = {}

        # 1. Determine the force variables (Supports array for multi-driver, or string for single)
        force_vars = self.config.get('xml_force_variables')
        if not force_vars:
            force_vars = [self.xml_force_var]

        # 2. Build the array for each variable
        for fv in force_vars:
            ints = []
            for e in self.events:
                # Handle Multi-Driver (Dictionary)
                if hasattr(e, 'trigger_values') and isinstance(e.trigger_values, dict):
                    ints.append(int(e.trigger_values.get(fv, -1)))

                # Handle Single-Driver (Legacy)
                elif hasattr(e, 'trigger_value') and e.trigger_value is not None:
                    ints.append(int(e.trigger_value))
                else:
                    ints.append(-1)

            # 3. Pad the array to MAX_E length with -1
            padded_ints = ints + [-1] * (self.max_e - len(ints))
            padded_ints = padded_ints[:self.max_e]

            # 4. Format as UPPAAL array string
            arrays[fv] = "{" + ", ".join(map(str, padded_ints)) + "};\n"

        return arrays, len(self.events)

    # ------------------------------------------------------------------
    # UPPAAL model patching
    # ------------------------------------------------------------------

    def fix_model(self):
        """
        Patch the UPPAAL model file *in place* so the next `verifyta` call
        executes the events the Teacher just requested.

        Three lines are rewritten on each invocation:

          ``int <force_var>[MAX_E] = {...};``
              The trigger sequence built by ``build_event_strings``.

          ``bool <action_var> = ...;``
              Forced to ``true`` so the model honours the forced events
              instead of running its native non-deterministic logic.

          ``const int TAU = ...;``
              The simulation horizon, scaled to be at least 200 time units
              and proportional to the number of forced events (50 each), so
              short prefixes don't truncate the trace prematurely.
        """
        print(f"[TraceGen] Fixing model at: {self.uppaal_model_path}")
        if not os.path.exists(self.uppaal_model_path):
            print(f"Error: Model file not found at {self.uppaal_model_path}")
            return

        with open(self.uppaal_model_path, 'r') as f:
            lines = f.readlines()

        # Get our multi-driver array strings
        arrays, event_count = self.build_event_strings()
        print(f"[TraceGen] Injecting Values into: {list(arrays.keys())}")

        tau_val = max(event_count * 50, 200)

        target_action_key = f"bool {self.xml_action_var} ="
        target_tau_key = "const int TAU ="

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Loop through all our driver arrays and patch them if we find them
            for fv, values_str in arrays.items():
                target_force_key = f"int {fv}[MAX_E] ="
                if stripped.startswith(target_force_key):
                    lines[i] = f"int {fv}[MAX_E] = {values_str}"

            if stripped.startswith(target_action_key):
                lines[i] = f"bool {self.xml_action_var} = true;\n"
            elif stripped.startswith(target_tau_key):
                lines[i] = f"const int TAU = {tau_val};\n"

        with open(self.uppaal_model_path, 'w') as f:
            f.writelines(lines)
        print("[TraceGen] Model patched successfully.")

    # ------------------------------------------------------------------
    # Public dispatcher
    # ------------------------------------------------------------------

    def get_traces(self, n: int = 1):
        """
        The Teacher calls this whenever it needs more data.

        Returns:
            list[str]  — absolute paths to the trace files the Teacher
                         should hand to ``sul.process_data``.  Empty list
                         if no fresh data could be produced (this also
                         signals "stop refining" to the Teacher).
        """
        # --- THE ROUTER ---
        if self.resample_strategy == 'CSV':
            print(f"[TraceGen] CSV Strategy Detected.")
            return self.get_traces_csv(n)
        elif self.resample_strategy == 'UPPAAL':
            return self.get_traces_uppaal(n)

        return []

    # ------------------------------------------------------------------
    # UPPAAL implementation
    # ------------------------------------------------------------------

    def get_traces_uppaal(self, n: int):
        """
        Run ``verifyta`` ``n`` times, each producing one fresh trace text
        file in ``self.output_dir``.  Each invocation re-seeds Python's
        RNG so the filename and any randomised UPPAAL choices vary, and
        every file is verified non-empty before it is returned.
        """
        self.fix_model()
        new_traces = []

        print(f"[TraceGen] Ensuring output dir: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        for i in range(n):
            random.seed()
            n_rand = random.randint(0, 2 ** 32)

            trace_filename = f"{self.cs_name}_trace_{n_rand}.txt"
            final_path = os.path.join(self.output_dir, trace_filename)

            cmd = [
                self.uppaal_bin_path,
                '-t0',
                self.uppaal_model_path,
                self.uppaal_query_path
            ]

            print(f"[TraceGen] Executing Command: {' '.join(cmd)}")

            try:
                with open(final_path, 'w') as outfile:
                    result = subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, text=True)

                if result.returncode != 0:
                    print(f"[TraceGen Error] UPPAAL Failed (Code {result.returncode})")
                    print(f"STDERR: {result.stderr}")
                    continue

                if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    print(f"[TraceGen] SUCCESS: File created ({os.path.getsize(final_path)} bytes)")
                    new_traces.append(final_path)
                else:
                    print("[TraceGen] FAILURE: File created but empty.")
            except Exception as e:
                print(f"[TraceGen] CRITICAL ERROR during subprocess: {e}")

        return new_traces

    # ------------------------------------------------------------------
    # CSV implementation
    # ------------------------------------------------------------------

    def get_traces_csv(self, n: int = 1):
        """
        Hand the Teacher the static list of CSV files — once.

        L* normally re-queries the trace generator inside ``ref_query``
        looking for fresh evidence.  CSV data is FIXED, so calling
        ``get_traces_csv`` a second time would either (a) re-feed the same
        rows (causing the Teacher to oscillate forever) or (b) waste work
        re-parsing identical data.  We solve both by returning the file
        list on the first call and an empty list on every subsequent one.
        The Teacher treats an empty list as "no further refinement
        possible" and gracefully exits the refinement loop.
        """
        # 1. Validate that we actually have CSV files in our list
        if not self.csv_files or len(self.csv_files) == 0:
            print("[TraceGen Error] No CSV files were found or provided to the Generator!!")
            return []

        # 2. Prevent L* infinite loops!
        # A CSV is static data. We only hand the list of file paths to the SUL engine ONCE per learning run.
        if not self.csv_yielded:
            self.csv_yielded = True
            print(f"[TraceGen] Yielding {len(self.csv_files)} CSV file(s) to SUL Engine: {self.csv_files}")
            return self.csv_files
        else:
            # Silently return empty arrays for all subsequent requests to stop the Teacher from looping
            return []
