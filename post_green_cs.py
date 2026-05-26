"""
post_green_cs.py
================
Creates the GREEN (GR3N decanter) case study via the Django REST API,
uploads the three relevant CSV files, and triggers the LSHA algorithm.

Run from the project root:
    python post_green_cs.py

Requires: requests  (pip install requests)
Server must be running at http://localhost:8000
"""

import os
import sys
import json
import time
import requests

BASE_URL = "http://localhost:8000"
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads", "GREEN")

# Three files needed:
#   PumpSpeed  → contributes SpeedSP column after vertical pivot
#   Temperature→ contributes Value   column after vertical pivot  (2TT-001A sensor)
#   REAL Decanter → contributes TCuscinettiAlimentazione column   (column_remap needed)
CSV_FILES = [
    "20250514_PumpSpeedData_JanToMay_REAL.csv",
    "20250514_TemperatureData_JanToMay_REAL.csv",
    "20250514_DecanterData_JanToMay_REAL.csv",
]

# ---------------------------------------------------------------------------
# JSON payload
# ---------------------------------------------------------------------------
# Event design (follows gr3n/sul_definition.py, extended for actual data range)
#
#  Pump speed (SpeedSP) unique operational values: 0,5,10,20,25,30,40,50,60,70,75,80,85,90,95,100
#  sp events: 10 buckets in 10-rpm steps covering 0–100
#  When sp==0 (pump stopped): three temperature-qualified stop states
#  Catch-all "s": unconditional fallback
#
#  Ambient temperature (2TT-001A 'Value'): range 0–83 °C, mean 48 °C
#  Thresholds at 30 and 55 °C split the stopped-pump state into
#  cold / warm / hot idle modes so the learner can distinguish bearing
#  temperature behaviour during different ambient conditions.
#
#  Main variable: TCuscinettiAlimentazione (bearing input temperature, Ta)
#  Model type:    MEAN  (constant absorption, matching gr3n modello_assorbimento)

USER_JSON = {
    "variables": [
        {
            "name":   "sp",
            "source": "SpeedSP",
            "role":   "driver",
            "type":   "REAL"
        },
        {
            "name":   "tmp",
            "source": "Value",
            "role":   "driver",
            "type":   "REAL"
        },
        {
            "name":   "Ta",
            "source": "TCuscinettiAlimentazione",
            "role":   "estimated",
            "type":   "REAL"
        }
    ],
    "constants": {
        "SP_STEP":   10,
        "TMP_LOW":   30,
        "TMP_HIGH":  55,
        "tolerances": {
            "sp":  1,
            "tmp": 5
        }
    },
    "models": [
        {"id": 0, "name": "Running", "type": "MEAN"}
    ],
    "events": [
        # ── pump running – three broad speed bands ────────────────────────────
        # Consolidated from 10 fine-grained buckets to three well-supported
        # bands.  Counts from Phase-2 histogram (sp_0..sp_9 over 453 pts):
        #   sp_low  covers sp_0 (119) + sp_1 (109) = 228 observations
        #   sp_mid  covers sp_2 (52)  + sp_3 (10)  = 62  observations
        #   sp_high covers sp_4..sp_9 (18+10+10+9+8+4) = 59 observations
        # All three bands have ≥ 59 change-points — well above n_min.
        {"symbol": "sp_low",  "model_id": 0, "guard": "sp > 0 and sp <= 20"},
        {"symbol": "sp_mid",  "model_id": 0, "guard": "sp > 20 and sp <= 40"},
        {"symbol": "sp_high", "model_id": 0, "guard": "sp > 40"},
        # ── pump stopped, qualified by ambient temperature ────────────────────
        # Threshold 40 °C splits the 73-sample sp_stop_hot (tmp > 55) and the
        # 31-sample cold/warm combined (tmp <= 40) groups into two bands with
        # enough observations each.
        {"symbol": "sp_stop_cold", "model_id": 0, "guard": "sp == 0 and tmp <= 40"},
        {"symbol": "sp_stop_hot",  "model_id": 0, "guard": "sp == 0 and tmp > 40"},
        # ── unconditional catch-all ───────────────────────────────────────────
        {"symbol": "s", "model_id": 0, "guard": ""},
    ],
    "trace_generation": {
        "strategy": "SIM",
        "csv": {
            # SCADA vertical (long) format:
            #   DataObjectName | DataObjectField | Value | TimeStamp
            "wide_or_long":       "long",
            "timestamp_column":   "TimeStamp",
            "field_column":       "DataObjectField",
            "value_column":       "Value",
            # The REAL decanter file uses lowercase column names ('time', 'value').
            # column_remap normalises them to match the other files BEFORE concat.
            "column_remap": {
                "time":  "TimeStamp",
                "value": "Value"
            },
            # SpeedSP is a discrete setpoint – round to nearest 5 rpm to
            # remove any floating-point noise before event guard evaluation.
            "round_columns": {
                "SpeedSP": 5
            },
            "interpolate_method": "linear",
            # All 3 files cover the same time window (Jan–May 2025) but each
            # holds different signals.  bundle_all_csv=true makes the L* engine
            # merge them into one trace instead of processing each separately.
            "bundle_all_csv": True,
            # Split the single 271-event merged trace into sub-traces of
            # ≤10 events each (~27 sub-traces).  Each word prefix then has
            # multiple matching segments, satisfying n_min=2 and bounding
            # counterexample length to ~10 instead of ~271.
            "max_trace_events": 10
        }
    }
}

CASE_STUDY_PAYLOAD = {
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
    "user_json":          USER_JSON,
}


def post_case_study():
    url = f"{BASE_URL}/api/case-study/"
    resp = requests.post(url, json=CASE_STUDY_PAYLOAD, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cs_id = data["id"]
    print(f"[1/3] Case study created: id={cs_id}  name={data['name']}")
    return cs_id


def upload_csv_files(cs_id):
    url = f"{BASE_URL}/api/csv-files/"
    uploaded = []
    for fname in CSV_FILES:
        fpath = os.path.join(UPLOADS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: file not found – {fpath}  (skipping)")
            continue
        with open(fpath, "rb") as fh:
            resp = requests.post(
                url,
                data={"case_study": cs_id},
                files={"file": (fname, fh, "text/csv")},
                timeout=60,
            )
        resp.raise_for_status()
        fid = resp.json()["id"]
        print(f"  uploaded {fname}  → csv_file id={fid}")
        uploaded.append(fid)
    print(f"[2/3] Uploaded {len(uploaded)} CSV file(s) for case study {cs_id}")
    return uploaded


def trigger_algorithm(cs_id):
    url = f"{BASE_URL}/run/"
    resp = requests.post(url, json={"case_study_id": cs_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("task_id", "?")
    print(f"[3/3] Algorithm triggered: task_id={task_id}")
    print()
    print("Monitor Celery worker output for progress.")
    print(f"Retrieve result: GET {BASE_URL}/api/case-study/{cs_id}/")
    return task_id


def main():
    print("=" * 55)
    print("  GREEN (GR3N) Case Study – POST & Run")
    print("=" * 55)
    print(f"Server : {BASE_URL}")
    print(f"Files  : {UPLOADS_DIR}")
    print()

    cs_id = post_case_study()
    upload_csv_files(cs_id)
    task_id = trigger_algorithm(cs_id)

    print()
    print(f"Case study id : {cs_id}")
    print(f"Celery task   : {task_id}")


if __name__ == "__main__":
    main()
