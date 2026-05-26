"""
post_all_case_studies.py
========================
Create and run all 6 reference case studies via the Django REST API.

Case studies covered:
  1. THERMO V1  — UPPAAL thermostat, 3 events (heater on/off, window)
  2. THERMO V5  — UPPAAL thermostat, 4 events (2 drivers: t.ON + r.open)
  3. HRI V1     — UPPAAL HRI fatigue, 2 events, driver = amy.busy||amy.p_2
  4. HRI V5     — UPPAAL HRI fatigue, 2 events, driver = 4-state combined
  5. ENERGY     — CSV W7 CNC machine, 4 speed-band events
  6. GREEN      — CSV GR3N decanter, 5 speed×temperature events

Run from the project root with the lsha-web conda env active and the
Django+Celery stack running:
    python post_all_case_studies.py [--only thermo_v1,hri_v5] [--no-trigger]

Options:
    --only <ids>      Comma-separated list of case IDs to run (default: all).
                      Valid IDs: thermo_v1 thermo_v5 hri_v1 hri_v5 energy green
    --no-trigger      Post case studies but do NOT trigger the algorithm.

Requires: requests  (pip install requests)
Server must be running at http://localhost:8000
Celery worker must be running (needed when --no-trigger is NOT set)
"""

import argparse
import json
import os
import sys
import time
import requests

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
_CASES   = os.path.join(_HERE, "case_files")
_CSV     = os.path.join(_HERE, "media", "uploads", "csv", "files")
_GREEN   = os.path.join(_HERE, "uploads", "GREEN")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_case_study(json_path, uppaal_model=None, uppaal_query=None):
    """Create a CaseStudy via multipart form-data, return the new id."""
    with open(json_path) as fh:
        cfg = json.load(fh)

    # Convert JSON payload to form-data (Django ModelViewSet expects form fields
    # for FileField-containing models; JSON fields go as JSON-encoded strings).
    data = {}
    for k, v in cfg.items():
        if k in ("uppaal_model_file", "uppaal_query_file"):
            continue           # handled via files= below
        if isinstance(v, (dict, list)):
            data[k] = json.dumps(v)
        elif isinstance(v, bool):
            data[k] = "true" if v else "false"
        else:
            data[k] = str(v)

    files = {}
    if uppaal_model and os.path.exists(uppaal_model):
        files["uppaal_model_file"] = (os.path.basename(uppaal_model),
                                      open(uppaal_model, "rb"), "application/xml")
    if uppaal_query and os.path.exists(uppaal_query):
        files["uppaal_query_file"] = (os.path.basename(uppaal_query),
                                      open(uppaal_query, "rb"), "text/plain")

    resp = requests.post(f"{BASE_URL}/api/case-study/", data=data, files=files, timeout=60)
    if resp.status_code not in (200, 201):
        print(f"  ERROR {resp.status_code}: {resp.text}")
        return None
    body = resp.json()
    cs_id = body["id"]
    print(f"  Created case study id={cs_id}  name={body['name']!r}  "
          f"strategy={body['resample_strategy']}")
    return cs_id


def _post_case_study_inline(payload, uppaal_model=None, uppaal_query=None):
    """Create a CaseStudy from a dict rather than a file."""
    data = {}
    for k, v in payload.items():
        if k in ("uppaal_model_file", "uppaal_query_file"):
            continue
        if isinstance(v, (dict, list)):
            data[k] = json.dumps(v)
        elif isinstance(v, bool):
            data[k] = "true" if v else "false"
        else:
            data[k] = str(v)

    files = {}
    if uppaal_model and os.path.exists(uppaal_model):
        files["uppaal_model_file"] = (os.path.basename(uppaal_model),
                                      open(uppaal_model, "rb"), "application/xml")
    if uppaal_query and os.path.exists(uppaal_query):
        files["uppaal_query_file"] = (os.path.basename(uppaal_query),
                                      open(uppaal_query, "rb"), "text/plain")

    resp = requests.post(f"{BASE_URL}/api/case-study/", data=data, files=files, timeout=60)
    if resp.status_code not in (200, 201):
        print(f"  ERROR {resp.status_code}: {resp.text}")
        return None
    body = resp.json()
    cs_id = body["id"]
    print(f"  Created case study id={cs_id}  name={body['name']!r}  "
          f"strategy={body['resample_strategy']}")
    return cs_id


def _upload_csv(cs_id, csv_path):
    """Upload a CSV file and attach it to the given case study."""
    fname = os.path.basename(csv_path)
    with open(csv_path, "rb") as fh:
        resp = requests.post(
            f"{BASE_URL}/api/csv-files/",
            data={"case_study": cs_id},
            files={"file": (fname, fh, "text/csv")},
            timeout=60,
        )
    if resp.status_code not in (200, 201):
        print(f"    CSV ERROR {resp.status_code}: {resp.text}")
        return None
    fid = resp.json()["id"]
    print(f"    Uploaded {fname}  → csv_file id={fid}")
    return fid


def _trigger(cs_id):
    """POST to /run/ to start the Celery algorithm task."""
    resp = requests.post(f"{BASE_URL}/run/", json={"case_study_id": cs_id}, timeout=30)
    if resp.status_code not in (200, 201, 202):
        print(f"    Trigger ERROR {resp.status_code}: {resp.text}")
        return None
    task_id = resp.json().get("task_id", "?")
    print(f"    Algorithm triggered: task_id={task_id}")
    return task_id


def _poll_result(cs_id, timeout=300, interval=15):
    """Poll GET /api/case-study/<id>/ until final_result_txt is set or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(f"{BASE_URL}/api/case-study/{cs_id}/", timeout=15)
        if resp.status_code != 200:
            break
        body = resp.json()
        if body.get("final_result_txt"):
            return body["final_result_txt"]
        time.sleep(interval)
    return None


# ---------------------------------------------------------------------------
# Case study definitions
# ---------------------------------------------------------------------------

# ── GREEN Decanter (inline, uses max_trace_events splitting) ─────────────────
GREEN_PAYLOAD = {
    "name":               "GREEN Decanter",
    "resample_strategy":  "SIM",
    "driver_signal":      ["sp", "tmp"],
    "main_variable":      "Ta",
    "context_variables":  [],
    "noise":              0.0,
    "n_min":              2,
    "p_value":            0.05,
    "mi_query":           False,
    "ht_query":           True,
    "ht_query_type":      "S",
    "eq_condition":       "W",
    "user_json": {
        "variables": [
            {"name": "sp",  "source": "SpeedSP",                     "role": "driver",    "type": "REAL"},
            {"name": "tmp", "source": "Value",                       "role": "driver",    "type": "REAL"},
            {"name": "Ta",  "source": "TCuscinettiAlimentazione",    "role": "estimated", "type": "REAL"},
        ],
        "constants": {
            "SP_STEP": 10, "TMP_LOW": 30, "TMP_HIGH": 55,
            "tolerances": {"sp": 1, "tmp": 5},
        },
        "models": [{"id": 0, "name": "Running", "type": "MEAN"}],
        "events": [
            {"symbol": "sp_low",       "model_id": 0, "guard": "sp > 0 and sp <= 20"},
            {"symbol": "sp_mid",       "model_id": 0, "guard": "sp > 20 and sp <= 40"},
            {"symbol": "sp_high",      "model_id": 0, "guard": "sp > 40"},
            {"symbol": "sp_stop_cold", "model_id": 0, "guard": "sp == 0 and tmp <= 40"},
            {"symbol": "sp_stop_hot",  "model_id": 0, "guard": "sp == 0 and tmp > 40"},
            {"symbol": "s",            "model_id": 0, "guard": ""},
        ],
        "trace_generation": {
            "strategy": "SIM",
            "csv": {
                "wide_or_long":       "long",
                "timestamp_column":   "TimeStamp",
                "field_column":       "DataObjectField",
                "value_column":       "Value",
                "column_remap":       {"time": "TimeStamp", "value": "Value"},
                "round_columns":      {"SpeedSP": 5},
                "interpolate_method": "linear",
                "bundle_all_csv":     True,
                "max_trace_events":   10,
            },
        },
    },
}

GREEN_CSV_FILES = [
    os.path.join(_GREEN, "20250514_PumpSpeedData_JanToMay_REAL.csv"),
    os.path.join(_GREEN, "20250514_TemperatureData_JanToMay_REAL.csv"),
    os.path.join(_GREEN, "20250514_DecanterData_JanToMay_REAL.csv"),
]

# ── ENERGY CSV files ─────────────────────────────────────────────────────────
ENERGY_CSV_FILES = [
    os.path.join(_CSV, "W7_2019-10-14_part0.csv"),
    os.path.join(_CSV, "W7_2019-10-14_part2.csv"),
    os.path.join(_CSV, "W7_2019-10-14_part3_0.csv"),
]

# ---------------------------------------------------------------------------
# Registry: all runnable case studies
# ---------------------------------------------------------------------------

CASE_STUDIES = {
    "thermo_v1": {
        "label":       "THERMO V1",
        "json":        os.path.join(_CASES, "thermo_v1.json"),
        "uppaal_xml":  os.path.join(_CASES, "thermostat-v1.xml"),
        "uppaal_q":    os.path.join(_CASES, "thermostat.q"),
    },
    "thermo_v5": {
        "label":       "THERMO V5",
        "json":        os.path.join(_CASES, "thermo_v5.json"),
        "uppaal_xml":  os.path.join(_CASES, "thermostat-v5.xml"),
        "uppaal_q":    os.path.join(_CASES, "thermostat.q"),
    },
    "hri_v1": {
        "label":       "HRI V1",
        "json":        os.path.join(_CASES, "hri_v1.json"),
        "uppaal_xml":  os.path.join(_CASES, "hri-w_ref-V1.xml"),
        "uppaal_q":    os.path.join(_CASES, "hri-w_ref1.q"),
    },
    "hri_v5": {
        "label":       "HRI V5",
        "json":        os.path.join(_CASES, "hri_v5.json"),
        "uppaal_xml":  os.path.join(_CASES, "hri-w_ref-V5.xml"),
        "uppaal_q":    os.path.join(_CASES, "hri-w_ref5.q"),
    },
    "energy": {
        "label":       "ENERGY",
        "json":        os.path.join(_CASES, "energy.json"),
        "csv_files":   ENERGY_CSV_FILES,
    },
    "green": {
        "label":       "GREEN",
        "payload":     GREEN_PAYLOAD,
        "csv_files":   GREEN_CSV_FILES,
    },
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_case(key, cfg, trigger=True):
    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")

    # Create case study
    if "payload" in cfg:
        cs_id = _post_case_study_inline(
            cfg["payload"],
            uppaal_model=cfg.get("uppaal_xml"),
            uppaal_query=cfg.get("uppaal_q"),
        )
    else:
        cs_id = _post_case_study(
            cfg["json"],
            uppaal_model=cfg.get("uppaal_xml"),
            uppaal_query=cfg.get("uppaal_q"),
        )

    if cs_id is None:
        print(f"  FAILED to create {cfg['label']}.")
        return None

    # Upload CSV files if any
    for csv_path in cfg.get("csv_files", []):
        if not os.path.exists(csv_path):
            print(f"  WARNING: CSV not found – {csv_path}  (skipping)")
            continue
        _upload_csv(cs_id, csv_path)

    # Trigger algorithm
    task_id = None
    if trigger:
        task_id = _trigger(cs_id)
    else:
        print("  (--no-trigger: skipping algorithm run)")

    return {"id": cs_id, "task_id": task_id}


def main():
    parser = argparse.ArgumentParser(description="Post and run all LSHA case studies")
    parser.add_argument("--only", default="",
                        help="Comma-separated keys to run (default: all). "
                             f"Valid: {', '.join(CASE_STUDIES)}")
    parser.add_argument("--no-trigger", action="store_true",
                        help="Post case studies but do NOT trigger the algorithm.")
    args = parser.parse_args()

    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        unknown = [k for k in keys if k not in CASE_STUDIES]
        if unknown:
            print(f"ERROR: unknown case study keys: {unknown}")
            sys.exit(1)
    else:
        keys = list(CASE_STUDIES)

    print(f"Server  : {BASE_URL}")
    print(f"Running : {keys}")
    print(f"Trigger : {'no' if args.no_trigger else 'yes'}")

    results = {}
    for key in keys:
        info = run_case(key, CASE_STUDIES[key], trigger=not args.no_trigger)
        if info:
            results[key] = info

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for key, info in results.items():
        print(f"  {CASE_STUDIES[key]['label']:15s}  id={info['id']}  task={info.get('task_id', 'N/A')}")

    if not args.no_trigger:
        print()
        print("Monitor the Celery worker log for progress.")
        print(f"Check results at: {BASE_URL}/api/case-study/<id>/")

    return results


if __name__ == "__main__":
    main()
