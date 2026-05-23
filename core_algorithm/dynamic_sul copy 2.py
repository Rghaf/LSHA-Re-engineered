"""
dynamic_sul.py  –  Dynamic System Under Learning (SUL) Functions
=================================================================

This module is the **data-processing core** of the LSHA learning engine.
It replaces the hard-coded, case-study-specific ``sul_functions.py`` files
(thermostat, HRI, energy, gr3n, …) with a single, configuration-driven
implementation that works for *any* custom system the user defines through
the React web UI.

All behaviour is controlled by the ``args`` dictionary assembled in
``tasks.py`` from the Django ``CaseStudy`` model and the user's JSON field.

---------------------------------------------------------------------------
SIGNALS DICTIONARY (used internally throughout this module)
---------------------------------------------------------------------------
Every function that receives or returns signal data uses this format::

    {
        'time'          : np.ndarray   # absolute time in seconds from trace start
        'main'          : np.ndarray   # the main observable variable (e.g. temperature)
        '<driver_name>' : np.ndarray   # one entry per driver, keyed by original name
        '<ctx_name>'    : np.ndarray   # one entry per context variable, same scheme
    }

---------------------------------------------------------------------------
EXPECTED USER JSON STRUCTURE
---------------------------------------------------------------------------
Below is the canonical JSON the user fills in the web UI.  Every field
except ``events`` is optional; sensible defaults are applied when omitted.

For UPPAAL-based case studies (thermostat, HRI):

.. code-block:: json

    {
      "models": [
        { "id": 0, "name": "Cooling",  "type": "EXP_DECAY"  },
        { "id": 1, "name": "Heating",  "type": "EXP_GROWTH" }
      ],
      "events": [
        { "symbol": "h_0", "trigger_value": 0.0, "model_id": 0, "guard": "main < 20" },
        { "symbol": "h_1", "trigger_value": 1.0, "model_id": 1, "guard": "main >= 20" }
      ],
      "trace_generation": {
        "xml_force_variable":  "force_act",
        "xml_action_variable": "force_exe",
        "max_length": 15,
        "step_size":  1.0
      }
    }

For CSV-based case studies (energy, gr3n, custom):

.. code-block:: json

    {
      "models": [
        { "id": 0, "name": "Idle",      "type": "MEAN" },
        { "id": 1, "name": "Low Speed", "type": "MEAN" },
        { "id": 2, "name": "High Speed","type": "MEAN" }
      ],
      "events": [
        {
          "symbol": "e_idle",
          "model_id": 0,
          "guard": "RPM < MIN_SPEED"
        },
        {
          "symbol": "e_low",
          "model_id": 1,
          "guard": "MIN_SPEED <= RPM and RPM < MID_SPEED"
        },
        {
          "symbol": "e_high",
          "model_id": 2,
          "guard": "RPM >= MID_SPEED"
        }
      ],
      "trace_generation": {
        "strategy":           "SIM",
        "time_column":        "_time",
        "time_format":        "datetime",
        "csv_format":         "horizontal",
        "interpolate_method": "linear",
        "replace_values":     { "OFF": 0, "ON": 1 },
        "round_columns":      { "RPM": 100 },
        "physics_constants":  {
          "MIN_SPEED":  500,
          "MID_SPEED": 1200,
          "MAX_SPEED": 2000,
          "SPEED_RANGE": 100,
          "tolerances": {
            "RPM": 100
          }
        }
      }
    }

---------------------------------------------------------------------------
EXTENDED JSON STRUCTURE (THERMO / HRI / ENERGY / GREEN, V2 web-UI shape)
---------------------------------------------------------------------------
The dynamic SUL also accepts a richer JSON shape used by the new web UI.
Every legacy field above keeps working; the additions below are optional
overlays that simplify configuration:

* **variables[]** — declares each signal once with a ``role`` field, so the
  UI does not have to populate the legacy ``main_var`` / ``driver_signal``
  / ``context_variables`` lists separately.  Roles understood:
      "estimated"          → mapped to internal key ``main``
      "driver"             → kept under the original column name
      "context" / other    → kept under the original column name
  Each entry may also carry an optional ``source`` field — the *physical*
  name of the column inside the trace file.  That separation lets the
  user write guards in friendly names (``F``, ``h.busy``, ``T``) while
  the parser reads the actual UPPAAL/CSV column (``humanFatigue[currH-1]``,
  ``amy.busy || amy.p_2``, ``T_r``).  A top-level ``aliases`` map provides
  the same capability without touching the variables block::
      "aliases": { "F": "humanFatigue[currH - 1]",
                   "h.busy": "amy.busy || amy.p_2",
                   "T": "T_r" }
* **constants** at top level — shorthand for
  ``trace_generation.physics_constants``.  Both are merged into the guard
  evaluation context.
* **trace_generation.strategy = "CSV"** — equivalent to legacy ``"SIM"``;
  routes parsing through :func:`_parse_csv`.
* **trace_generation.csv** — a nested config block.  Anything inside is
  merged with (and overrides) the flat trace_generation keys, so the new
  UI can ship a single ``csv: {...}`` object instead of ten flat fields::

    "csv": {
        "files":              [...],   # informational, files come from the DB
        "boolean_files":      [...],   # GREEN: extra CSV file group, all merged
        "real_files":         [...],   # GREEN: extra CSV file group, all merged
        "timestamp_column":   "TimeStamp",
        "wide_or_long":       "long",  # equivalent to csv_format=vertical
        "object_name_column": "DataObjectName",
        "field_column":       "DataObjectField",
        "value_column":       "Value",
        "filter_object":      "DecanterMB",
        "max_length":          50
    }

* **Guards may use ``prev.X``** — translated to ``prev_X`` before
  evaluation, so a guard like
  ``"MarciaDecanter == 1 and prev.MarciaDecanter == 0"`` works as written.
  Variable names containing a dot (``t.ON``, ``r.open``, ``h.busy``) are
  also supported: the placeholder substitution treats them as one token.

---------------------------------------------------------------------------
MODEL TYPES
---------------------------------------------------------------------------
  MEAN
  CONSTANT       (alias of MEAN: returns 0 rate, mean is computed normally)
  LINEAR         (linear slope estimate via consecutive increments)
  EXP_DECAY
  EXP_GROWTH
  LINEAR_DECAY
  LINEAR_GROWTH
"""

import math
import re
import numpy as np
import pandas as pd


# ===========================================================================
# UTILITY HELPERS
# ===========================================================================


def _uppaal_normalize(name):
    """
    Translate English logical keywords into UPPAAL operators.
    Required for HRI headers like 'amy.busy || amy.p_2'.
    """
    if not isinstance(name, str):
        return name
    return (name
            .replace(' or ',  ' || ')
            .replace(' OR ',  ' || ')
            .replace(' and ', ' && ')
            .replace(' AND ', ' && '))

def build_robust_pattern(var_name):
    """
    Immune to whitespace differences in UPPAAL headers.
    """
    v = _uppaal_normalize(str(var_name))
    v = v.replace(' ', '')
    # Interleave whitespace regex between every character
    flexible_var = r"\s*".join([re.escape(c) for c in v])
    # Match the variable name followed by the [0]: data line
    pattern = flexible_var + r"\s*:\s*\[0\]:\s*((?:\([0-9\.eE\-+]+\s*,\s*[0-9\.eE\-+]+\)\s*)+)"
    return pattern

# Handle scientific notation and negative numbers in UPPAAL traces
_TV_PAIR_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)"
)


# Pre-compiled regex for the (timestamp, value) tuples inside a UPPAAL "[0]:"
# data line.  Designed to be MORE permissive than build_robust_pattern's
# positive-only payload capture: this regex extracts the individual tuples
# from that captured payload AND additionally accepts:
#   * negative numbers           (-1.5)
#   * scientific notation        (1e-3, -2.5E+10)
#   * arbitrary whitespace inside the parentheses
# UPPAAL emits some derived signals (clocks, fatigue, position offsets)
# in scientific notation when their absolute value is very small, so without
# this the fallback regex would silently drop those tuples.

def flatten_vars(raw_data):
    """
    Helper to ensure we always get a flat list of strings, 
    even if Django sends stringified JSON.
    """
    if not raw_data: 
        return []
    if isinstance(raw_data, str):
        try:
            # Handle stringified JSON lists from Django Admin
            raw_data = json.loads(raw_data)
        except Exception:
            return [raw_data]
    if not isinstance(raw_data, list):
        return [raw_data]
        
    flat = []
    for item in raw_data:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


# Regex: rewrite "prev.<word>" → "prev_<word>" so guards can use natural
# attribute-style syntax for "previous-step" lookups (e.g. prev.MarciaDecanter).
# We rewrite ONLY the leading "prev." prefix; any further dots (as in
# "prev.t.ON") are left alone so the placeholder substitution can match the
# original column name "t.ON" verbatim.
_PREV_DOT_RE = re.compile(r'\bprev\.')


# def _uppaal_normalize(name):
#     """
#     Translate English logical keywords (``or``, ``and``) into the UPPAAL
#     operators (``||``, ``&&``) inside a *column / source* name.

#     Why this exists:  the React UI lets a user type a friendly UPPAAL
#     section header such as ``"amy.busy or amy.p_2"``, but UPPAAL itself
#     emits the trace section with the symbolic operator
#     ``"amy.busy || amy.p_2"``.  Without this normalisation the parser
#     would strip whitespace from the user-supplied form ("amy.busyoramy.p_2")
#     and never match the actual UPPAAL header ("amy.busy||amy.p_2").

#     Applied wherever a *physical* column name lives in the JSON:
#       * ``aliases`` values
#       * per-variable ``source`` fields
#     """
#     if not isinstance(name, str):
#         return name
#     # Pad the patterns with spaces so we don't accidentally rewrite
#     # tokens like "operator" or "android".
#     return (name
#             .replace(' or ',  ' || ')
#             .replace(' OR ',  ' || ')
#             .replace(' and ', ' && ')
#             .replace(' AND ', ' && '))


_GUARD_FAILURE_LOG_BUDGET = 5   # at most N guard-eval errors per process
_guard_failure_log_count = 0


def safe_eval(expr, context):
    """
    Evaluate a guard expression inside a restricted Python sandbox.

    Logs the FIRST few eval errors with the offending expression and the
    set of variable names that were available, so a user who mistyped a
    constant ("S_THRESHOLD" vs "S_OFF") learns immediately instead of
    silently getting only the fallback event.
    """
    global _guard_failure_log_count

    if not expr or str(expr).strip() == "":
        # An empty guard acts as an unconditional match (catch-all branch)
        return True

    try:
        # Sandbox: no built-ins, only 'math' module allowed
        safe_ctx = {"__builtins__": None, "math": math}
        safe_expr = str(expr)

        # Translate UI-friendly "prev.X" syntax → internal "prev_X" key.
        # Done BEFORE placeholder substitution so the rewritten token can
        # match the context key written by label_event_dynamic.
        safe_expr = _PREV_DOT_RE.sub('prev_', safe_expr)

        # Replace each variable with a positional placeholder.
        # Sorting longest-first prevents "RPM" from clobbering "prev_RPM".
        for idx, key in enumerate(sorted(context.keys(), key=len, reverse=True)):
            placeholder = f"__VAR_{idx}__"
            safe_expr = safe_expr.replace(key, placeholder)
            safe_ctx[placeholder] = context[key]

        return bool(eval(safe_expr, safe_ctx))

    except Exception as exc:
        if _guard_failure_log_count < _GUARD_FAILURE_LOG_BUDGET:
            _guard_failure_log_count += 1
            print(
                f"[SUL] Guard eval failed: expr={expr!r} err={exc!r} "
                f"available_vars={sorted(context.keys())}"
            )
            if _guard_failure_log_count == _GUARD_FAILURE_LOG_BUDGET:
                print("[SUL] (Suppressing further guard-eval error logs.)")
        return False


# ===========================================================================
# INTERNAL HELPER – derive variable roles from the new JSON shape
# ===========================================================================

def _resolve_variable_roles(args):
    """
    Convert the new ``variables[]`` JSON block into the legacy fields
    (``main_var``, ``driver_signal``, ``context_variables``) when the
    legacy fields are missing or empty, AND build the ``alias_map`` that
    pairs each user-friendly name with its physical column name.

    Each entry in ``variables`` has the form::

        { "name": "<friendly>", "source": "<physical>",
          "type": "REAL|INT|BOOL", "role": "estimated|driver|context" }

    ``source`` is optional — when omitted the friendly name IS the
    physical column name.  The mapping is stored back on ``args`` under
    the key ``alias_map`` so the parsers can translate friendly → physical
    when reading a trace file (UPPAAL section header, CSV column).

    Roles map as follows:
      * ``estimated``   → main observable (single column)
      * ``driver``      → driver signal (multiple allowed)
      * anything else   → context variable

    The function never overwrites an already-populated legacy field; the
    UI's explicit driver/main settings always win.
    """
    if not isinstance(args, dict):
        return

    # Seed the alias map with the optional top-level ``aliases`` block so
    # the user can supply friendly→physical mappings WITHOUT having to
    # restructure the variables[] array.  Normalise English-keyword forms
    # (``"amy.busy or amy.p_2"``) into UPPAAL operator forms
    # (``"amy.busy || amy.p_2"``) so the parser can match the section header.
    alias_map = {k: _uppaal_normalize(v) for k, v in (args.get('aliases') or {}).items()}

    variables = args.get('variables') or []
    if not isinstance(variables, list) or not variables:
        # No variables block — still publish whatever aliases the user
        # gave at the top level so _build_target_vars can pick them up.
        if alias_map:
            args['alias_map'] = alias_map
        return

    main_var     = args.get('main_var') or args.get('main_variable')
    drivers      = flatten_vars(
        args.get('driver_signal')
        or args.get('driver_signals')
        or args.get('driver', [])
    )
    context_vars = flatten_vars(args.get('context_variables', []))

    derived_main = None
    derived_drv  = []
    derived_ctx  = []
    for v in variables:
        if not isinstance(v, dict):
            continue
        name = v.get('name')
        if not name:
            continue
        # Per-variable physical-column override (the "source" field).
        # If present and different from the friendly name, register the
        # mapping in the alias map so parsers can find the actual column.
        # Normalise English-keyword forms (``"amy.busy or amy.p_2"``) to
        # the UPPAAL operator form (``"amy.busy || amy.p_2"``) so the
        # whitespace-stripped header lookup in _parse_uppaal succeeds.
        source = _uppaal_normalize(v.get('source'))
        if source and source != name:
            alias_map[name] = source
        role = (v.get('role') or '').strip().lower()
        if role == 'estimated':
            if derived_main is None:
                derived_main = name
        elif role == 'driver':
            derived_drv.append(name)
        else:
            derived_ctx.append(name)

    # Only fill in fields that the UI did not already set
    if not main_var and derived_main:
        args['main_var'] = derived_main
    if not drivers and derived_drv:
        args['driver'] = derived_drv
    if not context_vars and derived_ctx:
        args['context_variables'] = derived_ctx

    # Publish the aggregated alias map back onto args so that _build_target_vars
    # (and any downstream code) can read it without re-parsing variables[].
    if alias_map:
        args['alias_map'] = alias_map


# ===========================================================================
# INTERNAL HELPER – flatten the optional "csv" sub-block
# ===========================================================================

def _resolve_csv_config(trace_config):
    """
    The new JSON nests CSV options under ``trace_generation.csv``.
    Flatten that into the same dict the legacy code expects, with the
    nested keys taking priority over any flat duplicates.

    Also normalises the alternate key names used by the new UI:
        wide_or_long       ↔ csv_format ("wide"="horizontal", "long"="vertical")
        timestamp_column   ↔ time_column
        field_column       ↔ key_column
        object_name_column → adds an implicit "filter_object" filter
    """
    if not isinstance(trace_config, dict):
        return trace_config or {}

    flat = dict(trace_config)
    nested = trace_config.get('csv')
    if isinstance(nested, dict):
        # Nested values take priority over the flat duplicates
        for k, v in nested.items():
            flat[k] = v

    # Alias normalisation – keep both names available so downstream code
    # that already uses the legacy keys keeps working.
    if 'wide_or_long' in flat and 'csv_format' not in flat:
        wol = str(flat['wide_or_long']).strip().lower()
        flat['csv_format'] = 'vertical' if wol == 'long' else 'horizontal'

    if 'timestamp_column' in flat and 'time_column' not in flat:
        flat['time_column'] = flat['timestamp_column']

    if 'field_column' in flat and 'key_column' not in flat:
        flat['key_column'] = flat['field_column']

    return flat


# ===========================================================================
# INTERNAL HELPER – build target-variable map
# ===========================================================================

def _build_target_vars(args):
    """
    Build the mapping from *physical column names* (as they appear in the
    UPPAAL trace section header or the CSV column header) to the *internal
    signal keys* that the rest of the engine uses (``main`` for the
    estimated variable, friendly name for everything else).

    The mapping table is consumed by both ``_parse_csv`` and
    ``_parse_uppaal`` to look up which raw column to read for each
    requested signal.

    Workflow:
      1. _resolve_variable_roles populates main_var / driver / context_vars
         from the new ``variables[]`` block if the legacy fields are empty,
         and builds an ``alias_map`` of friendly→physical names.
      2. We iterate the (possibly updated) main / driver / context lists
         and, for each friendly name, register
            target_vars[ physical ] = internal_key
         where ``physical`` is the column to actually read from the file
         (taken from alias_map or defaulting to the friendly name itself).
    """
    # First, give the new "variables[]" block a chance to populate the
    # legacy fields if the UI did not set them.  This may also publish
    # ``args['alias_map']``.
    _resolve_variable_roles(args)

    # Friendly→physical lookup.  Built by _resolve_variable_roles from
    # both the per-variable ``source`` fields and the top-level
    # ``aliases`` map.  Falls back to an empty dict so the lookup below
    # is always safe.
    alias_map = args.get('alias_map') or {}

    # Reverse lookup (physical → friendly).  Needed because the Django
    # CaseStudy form may have been populated with PHYSICAL column names
    # (e.g. driver_signal=['HEADSTOCK__SPINDLE_MOTOR___1___RPM']) even
    # though the JSON variables[] block declared the FRIENDLY name
    # (e.g. RPM).  Without this reverse step the signals dict would key
    # the driver by its physical name and guards written in friendly
    # names ("RPM >= S_MID") would never find the value, silently making
    # every guard False and producing zero change points.
    reverse_alias = {phys: friend for friend, phys in alias_map.items()}

    def _phys(name):
        """Translate a friendly variable name to its physical column name."""
        return alias_map.get(name, name)

    def _to_friendly(name):
        """Translate a physical column name to its friendly key, if known.
        Identity for names that are already friendly (or have no alias)."""
        return reverse_alias.get(name, name)

    main_var = args.get('main_var') or args.get('main_variable', '')

    # Start with the main variable mapped to the special key 'main'.
    # We key by the PHYSICAL column name so the parser can locate it in
    # the trace; the internal key 'main' is what the rest of the SUL
    # (segments, params, plotting) refers to it by.
    target_vars = {}
    if main_var:
        # Accept either spelling: friendly (RPM) or physical
        # (HEADSTOCK__...).  Either one collapses to the same target.
        friendly_main = _to_friendly(main_var)
        target_vars[_phys(friendly_main)] = 'main'

    # Each driver signal is stored under its FRIENDLY name so guards
    # written in friendly names (e.g. "h.busy == 1") can find it.
    drivers = flatten_vars(
        args.get('driver_signal')
        or args.get('driver_signals')
        or args.get('driver', [])
    )
    friendly_drivers = []
    for driver_name in drivers:
        if driver_name:                     # guard against None/empty entries
            # Translate any physical name from the DB field to its
            # friendly counterpart so the guards can reference it.
            friendly = _to_friendly(driver_name)
            target_vars[_phys(friendly)] = friendly
            friendly_drivers.append(friendly)

    # Context variables (extra signals that influence guards but are not
    # the primary observable) are stored under their own friendly names
    # as well, with the same physical→friendly translation.
    context_vars = flatten_vars(args.get('context_variables', []))
    friendly_contexts = []
    for ctx_name in context_vars:
        if ctx_name:
            friendly_ctx = _to_friendly(ctx_name)
            target_vars[_phys(friendly_ctx)] = friendly_ctx
            friendly_contexts.append(friendly_ctx)

    # ----------------------------------------------------------------
    # Normalise args itself to the FRIENDLY spellings.
    #
    # The downstream SystemUnderLearning constructor reads
    # ``args['driver']`` and stores it on ``self.driver``; the parse
    # adapter then labels every driver SampledSignal with its FRIENDLY
    # name (because target_vars maps the physical column → friendly key).
    # If we leave ``args['driver']`` holding the user's original PHYSICAL
    # spelling, the SUL's filter
    #   driver_sig = [sig for sig in new_signals if sig.label in self.driver]
    # returns an empty list — and ``find_chg_pts`` crashes with
    # ``IndexError: list index out of range`` at ``driver[0]``.
    #
    # Writing the friendly spellings back here keeps every layer in sync
    # regardless of whether the user typed friendly or physical names in
    # the DB form.
    # ----------------------------------------------------------------
    if main_var:
        args['main_var'] = _to_friendly(main_var)
    if friendly_drivers:
        args['driver'] = friendly_drivers
    if friendly_contexts:
        args['context_variables'] = friendly_contexts

    return target_vars


# ===========================================================================
# PARSE – main entry point
# ===========================================================================

def parse_data_dynamic(file_paths, args=None):
    """
    Parse one or more trace files and return a uniform signals dictionary.

    This is the **single entry point** for both data sources:

    * **SIM strategy** – one or more CSV files produced by a real or simulated
      physical system.  The ``trace_generation.strategy`` field in the user's
      JSON must equal ``"SIM"`` (or the top-level ``resample_strategy`` field
      on the Django model must equal ``"SIM"``).

    * **UPPAAL strategy** – a plain-text file produced by the UPPAAL
      ``verifyta`` simulator.  Any strategy value other than ``"SIM"``
      routes here (``"UPPAAL"`` is the conventional value).

    The newer UI emits ``"CSV"`` instead of ``"SIM"`` and may nest CSV
    options under ``trace_generation.csv``; both shapes are handled here.

    """
    if not args:
        print("[SUL] parse_data_dynamic called without args – returning empty.")
        return {}

    # Normalise to list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    print(f"[SUL] Parsing {len(file_paths)} trace file(s)...")

    # Determine strategy: prefer the JSON-level flag, fall back to the model field.
    # Accept "CSV" and "SIM" as synonyms; anything else routes to UPPAAL.
    raw_trace_config = args.get('trace_generation', {}) or {}
    trace_config     = _resolve_csv_config(raw_trace_config)
    strategy = (trace_config.get('strategy') or args.get('resample_strategy', 'UPPAAL') or '').strip().upper()

    target_vars = _build_target_vars(args)

    if strategy in ('SIM', 'CSV'):
        signals = _parse_csv(file_paths, target_vars, trace_config)
    else:
        # UPPAAL only uses the first file; multiple files are not supported
        signals = _parse_uppaal(file_paths[0], target_vars)

    # ------------------------------------------------------------------
    # Cache the set of signal keys whose values are boolean (strictly
    # in {0, 1}) so is_chg_pt_dynamic can apply strict-equality to them
    # regardless of any per-driver tolerance.  Mutating ``args`` here
    # mirrors the existing pattern in tasks.parse_adapter, which sets
    # ``args['__current_trace_cache__'] = signals``.  Detection is
    # purely data-driven — no hard-coded field names.
    # ------------------------------------------------------------------
    if signals:
        args['__boolean_keys__'] = _detect_boolean_signal_keys(signals)

    return signals


def _detect_boolean_signal_keys(signals):
    """
    Return the set of signal keys whose non-NaN values are strictly in {0, 1}.

    Called once per parsed trace and stashed on ``args`` so subsequent
    is_chg_pt_dynamic calls can short-circuit to strict-equality for
    boolean drivers — a 0→1 (or 1→0) transition then always fires even
    when the L* event label happens to fall through to the same
    catch-all bucket on adjacent samples.
    """
    out = set()
    for key, arr in signals.items():
        if key == 'time':
            continue
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue
        try:
            clean = arr[~np.isnan(arr)] if arr.dtype.kind == 'f' else arr
        except TypeError:
            continue
        if clean.size == 0:
            continue
        uniq = set(np.unique(clean).tolist())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            out.add(key)
    return out


# ===========================================================================
# PARSE – CSV route (SIM strategy)
# ===========================================================================

# Default replace map: anywhere in a CSV that boolean literals or strings
# appear (Python True/False, "True"/"False"/"true"/"false") we coerce them
# to numeric 1/0 so they survive ``pd.to_numeric`` and pivot aggregation.
_BOOL_REPLACE = {
    True: 1, False: 0,
    'True': 1, 'False': 0,
    'true': 1, 'false': 0,
    'TRUE': 1, 'FALSE': 0,
}


def _parse_csv(file_paths, target_vars, trace_config):
    """
    Parse one or more CSV files and return a uniform signals dictionary.

    Supported CSV layouts
    ~~~~~~~~~~~~~~~~~~~~~
    **horizontal** (default):
        One row per time-step, one column per variable.  The time column is
        identified by ``trace_config['time_column']`` (defaults to the first
        column in the file).

    **vertical**:
        Each row is ``(timestamp, variable_name, value)`` – the format used
        by many industrial SCADA systems and the gr3n case study.  Set
        ``trace_config['csv_format'] = "vertical"`` and supply
        ``key_column``, ``value_column``, ``time_column`` to configure the
        pivot.

    Preprocessing pipeline (applied in this fixed order)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Load and concatenate all provided CSVs into one ``DataFrame``.
    2. *If vertical format*: pivot into horizontal layout.
    3. Replace token values supplied in ``replace_values``
       (e.g. ``{"OFF": 0, "ON": 1}``).
    4. Parse and normalise the time axis to **seconds from the first sample**.
    5. Interpolate NaN gaps with ``interpolate_method`` (default ``"linear"``).
    6. Round selected columns to discrete buckets via ``round_columns``
       (e.g. ``{"RPM": 100}`` rounds RPM to the nearest 100).

    Configuration keys (all inside ``trace_generation``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    strategy           : "SIM"  (required to reach this route)
    csv_format         : "horizontal" | "vertical"  (default: "horizontal")
    key_column         : column containing variable names when vertical
    value_column       : column containing values when vertical
    time_column        : column to use as the time axis
    time_format        : "datetime" | "numeric"  (default: "datetime")
    replace_values     : dict of token → numeric substitutions
    interpolate_method : pandas interpolation method name  (default: "linear")
    round_columns      : { column_name: bucket_size }

    Extra keys honoured by the new web UI (see _resolve_csv_config):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    wide_or_long       : "wide" | "long"        (alias of csv_format)
    timestamp_column   : alias of time_column
    field_column       : alias of key_column
    object_name_column : column whose value is matched against ``filter_object``
    filter_object      : keep only rows where object_name_column == filter_object
    boolean_files      : list of additional CSV paths whose Value column holds
                         True/False booleans – concatenated with file_paths
    real_files         : list of additional CSV paths whose Value column holds
                         numeric REALs – concatenated with file_paths

    derive_columns     : { source_col: { factor, clip_min, target } | factor }
                         Replace ``source_col`` with its discrete derivative
                         multiplied by ``factor``.  Mirrors the legacy
                         ``energy/sul_functions.py`` transform
                         ``power = 60 * (curr_energy - prev_energy)`` so a
                         cumulative kWh totalizer can be turned into an
                         instantaneous-power signal without editing the CSV.
                         Examples::

                             # short form: column is replaced in-place
                             "derive_columns": { "ENERGY": 60 }

                             # long form: write into a new column instead
                             "derive_columns": {
                                 "HEADSTOCK__SPINDLE_DRIVE___1___ENERGY": {
                                     "factor": 60,
                                     "clip_min": 0,
                                     "target": "POWER_kW"
                                 }
                             }

    """
    try:
        # ---------------------------------------------------------------
        # Step 0 – Splice in any extra file groups declared by the UI
        # (GREEN-style "boolean_files" + "real_files").  These are added
        # to the file_paths list so the existing concat path picks them up.
        # ---------------------------------------------------------------
        merged_paths = list(file_paths or [])
        for extra_key in ('boolean_files', 'real_files', 'extra_files'):
            extra = trace_config.get(extra_key)
            if isinstance(extra, list):
                for p in extra:
                    if isinstance(p, str) and p not in merged_paths:
                        merged_paths.append(p)
        file_paths = merged_paths

        # ---------------------------------------------------------------
        # Step 1 – Load all CSVs and concatenate into one DataFrame
        # ---------------------------------------------------------------
        dfs = []
        for fp in file_paths:
            if str(fp).lower().endswith('.csv'):
                dfs.append(pd.read_csv(fp))
            else:
                print(f"[SUL] Skipping non-CSV file: {fp}")

        if not dfs:
            print("[SUL] No .csv files found among the provided paths.")
            return {}

        df = pd.concat(dfs, ignore_index=True)
        print(f"[SUL] CSV loaded: {len(df)} rows × {len(df.columns)} columns")

        # ---------------------------------------------------------------
        # Step 1b – Object-name filter (long-format SCADA dumps).
        #
        # GREEN's CSV looks like:
        #   "DataObjectName","DataObjectField","Value","TimeStamp"
        #   "DecanterMB", ...
        #   "PumpController", ...
        #
        # The user can pin a single object by setting ``filter_object``
        # (and optionally ``object_name_column``).  If the filter removes
        # *every* row we fall back to the unfiltered df with a loud
        # warning rather than returning empty.
        # ---------------------------------------------------------------
        filter_object = trace_config.get('filter_object')
        obj_col       = trace_config.get('object_name_column', 'DataObjectName')
        if filter_object and obj_col in df.columns:
            mask = df[obj_col].astype(str).str.strip() == str(filter_object).strip()
            kept = mask.sum()
            if kept == 0:
                print(f"[SUL] Warning: filter_object='{filter_object}' matched 0 rows in '{obj_col}' – ignoring filter.")
            else:
                df = df[mask].copy()
                print(f"[SUL] Object filter '{obj_col}=={filter_object}' kept {kept} rows.")

        # ---------------------------------------------------------------
        # Step 1c – Always coerce boolean literals BEFORE pivot so that
        # aggfunc='mean' does not choke on True/False objects.
        # ---------------------------------------------------------------
        try:
            df = df.replace(_BOOL_REPLACE)
        except Exception as exc:
            # Some pandas builds dislike replace() with mixed dtypes;
            # fall back to a per-column safe pass.
            print(f"[SUL] Warning: bulk boolean replace failed ({exc}); doing per-column fallback.")
            for c in df.columns:
                try:
                    df[c] = df[c].replace(_BOOL_REPLACE)
                except Exception:
                    pass

        # ---------------------------------------------------------------
        # Step 1d – Deduplicate (timestamp, field) for vertical CSVs.
        #
        # SCADA dumps sometimes emit the same field twice at the exact
        # same timestamp (sub-microsecond bursts, replayed buffers,
        # device-side double-writes).  Without this step, pivot_table's
        # aggfunc='mean' would average the duplicates; for booleans
        # coerced to 0/1 that yields 0.5, which silently breaks the
        # downstream boolean change-point logic and the
        # ``prev.<field> == 0`` style guards in label_event_dynamic.
        # Keeping the LAST entry preserves the chronologically latest
        # state, and is a no-op when there are no duplicates.
        # ---------------------------------------------------------------
        if trace_config.get('csv_format') == 'vertical':
            time_col_raw = trace_config.get('time_column', 'TimeStamp')
            key_col_raw  = trace_config.get('key_column',  'DataObjectField')
            if time_col_raw in df.columns and key_col_raw in df.columns:
                before = len(df)
                df = df.drop_duplicates(subset=[time_col_raw, key_col_raw], keep='last')
                if before != len(df):
                    print(f"[SUL] Dropped {before - len(df)} duplicate "
                          f"({time_col_raw}, {key_col_raw}) rows before pivot.")

        # ---------------------------------------------------------------
        # Step 2 – Pivot vertical-format CSVs to horizontal
        #
        # Vertical format:  TimeStamp | DataObjectField | Value
        # Horizontal result: TimeStamp | <field1> | <field2> | …
        # ---------------------------------------------------------------
        if trace_config.get('csv_format') == 'vertical':
            key_col      = trace_config.get('key_column',   'DataObjectField')
            val_col      = trace_config.get('value_column', 'Value')
            time_col_raw = trace_config.get('time_column',  'TimeStamp')

            # Guarantee Value is numeric for the mean aggregation; non-coercible
            # entries become NaN and will be cleaned by the later interpolate.
            df[val_col] = pd.to_numeric(df[val_col], errors='coerce')

            df = (df.pivot_table(index=time_col_raw,
                                 columns=key_col,
                                 values=val_col,
                                 aggfunc='mean')   # mean handles duplicate entries
                    .reset_index())
            print(f"[SUL] Pivoted vertical CSV to {len(df.columns)} columns.")

        # ---------------------------------------------------------------
        # Step 3 – Replace token values with numerics
        #
        # Example: replace_values = {"OFF": 0.0, "ON": 1.0, "LOAD": 1.0}
        # This handles "dirty" sensor data that uses strings instead of numbers.
        # ---------------------------------------------------------------
        replacements = trace_config.get('replace_values', {})
        if replacements:
            df = df.replace(replacements)
            print(f"[SUL] Applied {len(replacements)} value replacement(s).")

        # ---------------------------------------------------------------
        # Step 4 – Normalise the time axis to seconds from the first sample
        #
        # time_format = "datetime"  →  parse as ISO datetime string and convert
        # time_format = "numeric"   →  treat as a plain number and subtract t0
        # ---------------------------------------------------------------
        time_col    = trace_config.get('time_column', df.columns[0])
        time_format = trace_config.get('time_format', 'datetime')

        if time_format == 'datetime':
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            # Sort chronologically – a long-format pivot or multi-file concat
            # may otherwise leave the rows in arbitrary order.
            df = df.sort_values(by=time_col).reset_index(drop=True)
            # Convert to elapsed seconds (float)
            df['time'] = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds()
        else:
            # Already numeric – just zero-reference it
            df['time'] = pd.to_numeric(df[time_col], errors='coerce')
            df = df.sort_values(by='time').reset_index(drop=True)
            df['time'] = df['time'] - df['time'].iloc[0]

        # ---------------------------------------------------------------
        # Step 5 – Interpolate NaN gaps (dtype-aware, per-column)
        #
        # After a vertical pivot most rows carry NaN in every column
        # except the one that changed.  We partition the numeric columns:
        #
        #   * boolean columns (non-NaN values strictly in {0, 1}) are
        #     held with forward-fill regardless of ``interpolate_method``.
        #     Linear interpolation between a False (0) and a True (1)
        #     row would synthesise 0.5 samples that silently break the
        #     boolean change-point logic in is_chg_pt_dynamic and the
        #     ``prev.X == 0`` guards in label_event_dynamic.
        #
        #   * real-valued columns follow the user's ``interpolate_method``
        #     (default linear).  Trailing NaNs are cleaned by bfill/ffill.
        #
        # Detection runs on the pivoted DataFrame so it is purely
        # data-driven — no hard-coded field names or per-case-study lists.
        # ---------------------------------------------------------------
        interp_method = trace_config.get('interpolate_method', 'linear')
        if interp_method and interp_method.lower() != 'none':
            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != 'time']

            # Partition numeric columns into boolean-like vs real-valued.
            # Boolean detection runs on the raw (pre-interpolation) data
            # so a column that legitimately holds only 0s and 1s is not
            # masked by interpolation artefacts.
            boolean_cols = []
            real_cols    = []
            for c in num_cols:
                uniq = set(pd.unique(df[c].dropna()).tolist())
                if uniq and uniq.issubset({0, 1, 0.0, 1.0}):
                    boolean_cols.append(c)
                else:
                    real_cols.append(c)

            if boolean_cols:
                df[boolean_cols] = df[boolean_cols].ffill().bfill()
                print(f"[SUL] Forward-filled {len(boolean_cols)} boolean column(s): "
                      f"{boolean_cols}")

            if real_cols:
                m = interp_method.strip().lower()
                # ffill / bfill / pad / backfill cannot be passed to
                # ``DataFrame.interpolate`` from pandas 2.2 onwards –
                # pandas wants ``df.ffill()`` / ``df.bfill()`` directly.
                if m in ('ffill', 'pad'):
                    df[real_cols] = df[real_cols].ffill().bfill()
                elif m in ('bfill', 'backfill'):
                    df[real_cols] = df[real_cols].bfill().ffill()
                else:
                    df[real_cols] = df[real_cols].interpolate(method=interp_method).bfill().ffill()

        # ---------------------------------------------------------------
        # Step 5b – Derive new columns from cumulative ones
        #
        # Mirrors the legacy ``energy/sul_functions.py`` transform that
        # converts a kWh totalizer into instantaneous power
        # (``power = 60 * (curr_energy - prev_energy)``).  Without this,
        # every segment's mean of the cumulative column grows
        # monotonically and the L* HT-query allocates a fresh
        # distribution for every segment.
        #
        # Two shapes are accepted:
        #     "derive_columns": { "ENERGY": 60 }                       # short
        #     "derive_columns": { "ENERGY": { "factor": 60,
        #                                     "clip_min": 0,
        #                                     "target": "POWER_kW" } } # long
        #
        # Behaviour:
        #   * ``factor``    : scalar multiplier (defaults to 1.0)
        #   * ``clip_min``  : if set, values below this are clipped (e.g.
        #                     a totalizer reset would otherwise produce a
        #                     huge negative spike)
        #   * ``target``    : output column name; when omitted the source
        #                     column is overwritten in-place so existing
        #                     ``round_columns`` / variable mappings still work.
        # ---------------------------------------------------------------
        derive_cfg = trace_config.get('derive_columns') or {}
        for src_col, spec in derive_cfg.items():
            if src_col not in df.columns:
                print(f"[SUL] derive_columns: source column '{src_col}' not in CSV — skipped.")
                continue
            if isinstance(spec, dict):
                factor   = float(spec.get('factor', 1.0))
                clip_min = spec.get('clip_min')
                tgt_col  = spec.get('target') or src_col
            else:
                factor   = float(spec)
                clip_min = None
                tgt_col  = src_col
            series = pd.to_numeric(df[src_col], errors='coerce').ffill().bfill()
            derivative = series.diff().fillna(0.0) * factor
            if clip_min is not None:
                derivative = derivative.clip(lower=float(clip_min))
            df[tgt_col] = derivative
            print(
                f"[SUL] Derived '{tgt_col}' = {factor} * d/dt('{src_col}') "
                f"(samples={len(derivative)}, "
                f"min={derivative.min():.3f}, max={derivative.max():.3f}, "
                f"mean={derivative.mean():.3f})"
            )

        # ---------------------------------------------------------------
        # Step 6 – Round selected columns to discrete buckets
        #
        # This is critical for driver signals like RPM that are continuous
        # in reality but need to be discretised for change-point detection.
        # e.g. round_columns = { "RPM": 100 } rounds every RPM value to the
        # nearest 100.
        # ---------------------------------------------------------------
        for col_name, bucket_size in (trace_config.get('round_columns') or {}).items():
            if col_name in df.columns:
                df[col_name] = (
                    pd.to_numeric(df[col_name], errors='coerce') / bucket_size
                ).round() * bucket_size
                print(f"[SUL] Rounded column '{col_name}' to nearest {bucket_size}.")
            else:
                print(f"[SUL] round_columns: column '{col_name}' not in CSV — skipped.")

        # ---------------------------------------------------------------
        # Build the final signals dictionary
        # ---------------------------------------------------------------
        signals = {'time': df['time'].to_numpy()}
        n_rows  = len(df)

        for orig_name, internal_key in target_vars.items():
            if orig_name in df.columns:
                signals[internal_key] = (
                    pd.to_numeric(df[orig_name], errors='coerce')
                      .fillna(0.0)
                      .to_numpy()
                )
            else:
                print(f"[SUL] Warning: column '{orig_name}' not found – filling with zeros.")
                signals[internal_key] = np.zeros(n_rows)

        # ---------------------------------------------------------------
        # Diagnostic summary — gives operators a one-glance health-check
        # of the parsed signals before they hit the L* loop.
        # ---------------------------------------------------------------
        try:
            t = signals['time']
            print(f"[SUL] CSV parse complete:")
            print(f"[SUL]   rows         = {n_rows}")
            if len(t) > 0:
                print(f"[SUL]   time         = [{t[0]:.3f}, {t[-1]:.3f}] s "
                      f"(span={t[-1] - t[0]:.1f} s)")
            print(f"[SUL]   signal keys  = {list(signals.keys())}")
            for key in signals:
                if key == 'time':
                    continue
                arr = signals[key]
                if len(arr) == 0:
                    continue
                uniq = np.unique(arr)
                print(
                    f"[SUL]   '{key}' min={float(np.min(arr)):.3f} "
                    f"max={float(np.max(arr)):.3f} "
                    f"mean={float(np.mean(arr)):.3f} "
                    f"unique={len(uniq)}"
                )
        except Exception as diag_exc:
            print(f"[SUL] Diagnostics error: {diag_exc}")

        return signals

    except Exception as exc:
        print(f"[SUL] CSV parse error: {exc}")
        import traceback
        traceback.print_exc()
        return {}


# ===========================================================================
# PARSE – UPPAAL route
# ===========================================================================

def _parse_uppaal(file_path, target_vars):
    try:
        with open(file_path, 'r') as fh:
            content = fh.read()

        raw_data = {}
        all_times = set()

        for physical_name, internal_key in target_vars.items():
            if not physical_name:
                continue

            pattern = build_robust_pattern(physical_name)
            match = re.search(pattern, content)
            
            if not match:
                print(f"[SUL] Warning: UPPAAL variable '{physical_name}' not found.")
                continue

            payload = match.group(1)
            pairs = _TV_PAIR_RE.findall(payload)

            data_points = []
            for t_str, v_str in pairs:
                try:
                    t_f, v_f = float(t_str), float(v_str)
                    data_points.append((t_f, v_f))
                    all_times.add(t_f)
                except: continue

            raw_data[internal_key] = data_points

        if not all_times:
            return {}

        sorted_times = sorted(list(all_times))
        signals = {'time': np.array(sorted_times)}

        for internal_key, points in raw_data.items():
            arr = np.zeros(len(sorted_times))
            if points:
                # Same forward-fill logic as your working version
                pts = sorted(points, key=lambda x: x[0])
                val_map = {t: v for t, v in pts}
                curr_val = pts[0][1]
                for i, t in enumerate(sorted_times):
                    if t in val_map:
                        curr_val = val_map[t]
                    arr[i] = curr_val
            signals[internal_key] = arr

        return signals
    except Exception as e:
        print(f"[SUL] UPPAAL Error: {e}")
        return {}


# ===========================================================================
# CHANGE-POINT DETECTION
# ===========================================================================

def is_chg_pt_dynamic(signals, index, args=None):

    if index == 0:
        return False

    # ------------------------------------------------------------------
    # CHECK 1 – Discrete event label change
    # ------------------------------------------------------------------
    curr_label = label_event_dynamic(signals, index,     args)
    prev_label = label_event_dynamic(signals, index - 1, args)
    if curr_label != prev_label:
        return True

    # ------------------------------------------------------------------
    # CHECK 2 – Continuous driver signal step (noise-filtered)
    # ------------------------------------------------------------------
    drivers = flatten_vars(
        (args or {}).get('driver_signal')
        or (args or {}).get('driver_signals')
        or (args or {}).get('driver', [])
    )

    trace_config = (args or {}).get('trace_generation', {}) or {}
    constants    = trace_config.get('physics_constants', {}) or {}

    # Per-variable tolerance map.  Missing entries fall back to default_tol.
    tolerances  = constants.get('tolerances', {}) or {}
    # Legacy single-value fallback (older configs used "SPEED_RANGE" key);
    # if neither is present we default to None which means "no CHECK-2 firing
    # for this driver" – we trust CHECK 1 (label change) to catch anything
    # important and avoid a change-point on every micro-jitter of e.g. RPM.
    default_tol = constants.get('SPEED_RANGE')

    # Signal keys the parser detected as boolean (values strictly in {0, 1}).
    # Cached on args by parse_data_dynamic.  For these drivers we use strict
    # equality regardless of tolerance, so a 0→1 transition always fires
    # even when the L* event label happens to stay in the same catch-all
    # bucket across adjacent samples.
    boolean_keys = (args or {}).get('__boolean_keys__', set()) or set()

    for driver_name in drivers:
        # The driver is stored under its original name in the signals dict
        if driver_name not in signals:
            continue

        curr_val = signals[driver_name][index]
        prev_val = signals[driver_name][index - 1]

        # Boolean driver: any value change is a change point.  Skips the
        # numeric tolerance branch entirely so the user does not need to
        # configure a per-driver tolerance for every boolean flag.
        if driver_name in boolean_keys:
            if curr_val != prev_val:
                return True
            continue

        tol = tolerances.get(driver_name, default_tol)

        try:
            curr_f = float(curr_val)
            prev_f = float(prev_val)
            if tol is None:
                # No tolerance configured for this numeric driver — rely on
                # CHECK 1 (label change) instead of firing on every numeric
                # jitter.  This is what keeps ENERGY sane when the user has
                # not supplied a per-driver tolerance.
                continue
            if abs(curr_f - prev_f) > float(tol):
                return True
        except (ValueError, TypeError):
            # Non-numeric driver (string or bool): strict equality check
            if curr_val != prev_val:
                return True

    return False


# ===========================================================================
# EVENT LABELING
# ===========================================================================

def label_event_dynamic(signals, index, args=None):
    """
    Identify which event symbol applies at the given time index.

    Guards are evaluated with a context that contains:

      * Every signal's current value (key = original column name; the
        estimated variable is ALSO available under the literal key ``main``).
      * Every signal's previous value, prefixed with ``prev_`` (e.g.
        ``prev_RPM``).  Guards may write either ``prev_RPM`` or the more
        natural ``prev.RPM`` syntax.
      * Every entry in ``trace_generation.physics_constants`` and the
        top-level ``constants`` block, so guards can reference symbolic
        thresholds (``MIN_SPEED``, ``p_min``, …) instead of magic numbers.
    """
    event_defs = (args or {}).get('events', [])
    if not event_defs:
        return "default_event"

    # ------------------------------------------------------------------
    # Build evaluation context
    # ------------------------------------------------------------------
    context = {}

    # Current and previous values for every signal
    for key, arr in signals.items():
        if key == 'time':
            continue
        context[key]            = arr[index]
        context[f"prev_{key}"]  = arr[index - 1] if index > 0 else arr[index]

    # Physics constants (e.g. MIN_SPEED, SPEED_RANGE) so guards can use
    # symbolic names rather than hard-coded magic numbers.
    # We accept both the legacy ``trace_generation.physics_constants`` AND
    # the newer top-level ``constants`` block.
    trace_config = (args or {}).get('trace_generation', {}) or {}
    physics_consts = trace_config.get('physics_constants', {}) or {}
    context.update(physics_consts)

    top_constants = (args or {}).get('constants', {}) or {}
    if isinstance(top_constants, dict):
        context.update(top_constants)

    # ------------------------------------------------------------------
    # Evaluate each event guard in definition order
    # ------------------------------------------------------------------
    for event_def in event_defs:
        guard = event_def.get('guard', '')
        try:
            if safe_eval(guard, context):
                return event_def.get('symbol', 'unknown')
        except Exception as exc:
            print(f"[SUL] Warning: error evaluating guard '{guard}': {exc}")

    # No guard matched – return the first event symbol as a safe fallback.
    # This is a HARD-TO-NOTICE failure mode: if every guard silently
    # eval-errors (e.g. an undefined constant) we'd return the first
    # symbol every time and the learner would see only one event.  We
    # log the first few occurrences so the user sees it.
    global _fallback_log_count
    try:
        _fallback_log_count
    except NameError:
        _fallback_log_count = 0
    if _fallback_log_count < 3:
        _fallback_log_count += 1
        sample = {
            k: context[k] for k in list(context.keys())[:8]
        }
        print(
            f"[SUL] No guard matched at index={index}; falling back to "
            f"'{event_defs[0].get('symbol', 'unknown')}'. "
            f"Context sample={sample}"
        )
    return event_defs[0].get('symbol', 'unknown')


# ===========================================================================
# PHYSICS PARAMETER EXTRACTION
# ===========================================================================

def get_physics_param_dynamic(signals, start_i, end_i, args=None):
    """
    Extract a scalar physics parameter that characterises a signal segment.

    """
    default = {'mean': 0.0, 'rate': 0.0}

    if 'main' not in signals or start_i >= end_i:
        return default

    segment = signals['main'][start_i:end_i]
    if len(segment) == 0:
        return default

    # Arithmetic mean is always computed and always returned
    mean_val = float(np.mean(segment))

    # Determine the flow model type for this segment
    model_type = _get_model_type_for_segment(signals, start_i, args)

    # Compute the model-specific rate parameter
    rate_val = _compute_rate(segment, model_type)

    return {'mean': mean_val, 'rate': rate_val}


# ===========================================================================
# INTERNAL HELPERS for parameter extraction
# ===========================================================================

def _get_model_type_for_segment(signals, start_i, args):
    """
    Look up the flow model type for the event that fired at ``start_i``.

    """
    if args is None:
        return 'MEAN'

    # Identify which event is active at the start of this segment
    symbol = label_event_dynamic(signals, start_i, args)

    # Resolve event → model_id
    model_id = None
    for event_def in args.get('events', []):
        if event_def.get('symbol') == symbol:
            model_id = event_def.get('model_id')
            break

    if model_id is None:
        return 'MEAN'

    # Resolve model_id → type
    for model_def in args.get('models', []):
        if model_def.get('id') == model_id:
            return (model_def.get('type', 'MEAN') or 'MEAN').upper()

    return 'MEAN'


def _compute_rate(segment, model_type):
    """
    Compute the rate scalar for a 1-D signal segment given its flow model type.

    Parameters
    ----------
    segment    : np.ndarray  – 1-D array of signal values for the segment.
    model_type : str         – One of ``MEAN``, ``CONSTANT``, ``LINEAR``,
                               ``EXP_DECAY``, ``EXP_GROWTH``,
                               ``LINEAR_DECAY``, ``LINEAR_GROWTH``.

    Returns
    -------
    float
        The estimated rate/slope parameter.  Returns ``0.0`` when the
        segment is too short, the model type is ``MEAN``/``CONSTANT``, or a
        numerical error occurs (e.g. log of a non-positive number).
    """
    try:
        vals = list(segment)
        if len(vals) < 2:
            return 0.0

        if model_type == 'EXP_DECAY':
            # x(t) = x₀ · exp(−R · t)
            # Consecutive-ratio estimate:  R ≈ −log(vₜ / vₜ₋₁)
            # Only use steps where both values are positive and the value changed
            rates = [
                -math.log(vals[i] / vals[i - 1])
                for i in range(1, len(vals))
                if vals[i] > 0 and vals[i - 1] > 0 and vals[i] != vals[i - 1]
            ]
            return sum(rates) / len(rates) if rates else 0.0

        elif model_type == 'EXP_GROWTH':
            # x(t) = x_ss − (x_ss − x₀) · exp(−K · t)
            # For small time steps: Δx ≈ K · (x_ss − xₜ); approximate K from increments
            increments = [
                vals[i] - vals[i - 1]
                for i in range(1, len(vals))
                if vals[i] != vals[i - 1]
            ]
            return sum(increments) / len(increments) if increments else 0.0

        elif model_type == 'LINEAR_DECAY':
            # x(t) = x₀ − α · t  →  slope α is the mean decrease per step
            decrements = [
                vals[i - 1] - vals[i]   # positive when the signal is decreasing
                for i in range(1, len(vals))
                if vals[i] != vals[i - 1]
            ]
            return sum(decrements) / len(decrements) if decrements else 0.0

        elif model_type == 'LINEAR_GROWTH':
            # x(t) = x₀ + α · t  →  slope α is the mean increase per step
            increments = [
                vals[i] - vals[i - 1]
                for i in range(1, len(vals))
                if vals[i] != vals[i - 1]
            ]
            return sum(increments) / len(increments) if increments else 0.0

        elif model_type == 'LINEAR':
            # Generic linear segment without a forced direction:
            # estimate the signed slope from consecutive increments.
            increments = [
                vals[i] - vals[i - 1]
                for i in range(1, len(vals))
                if vals[i] != vals[i - 1]
            ]
            return sum(increments) / len(increments) if increments else 0.0

        else:
            # MEAN / CONSTANT / unrecognised type – no meaningful rate
            return 0.0

    except (ValueError, ZeroDivisionError):
        return 0.0
