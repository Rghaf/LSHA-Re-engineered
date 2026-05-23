#!/usr/bin/env python3
"""POST a CaseStudy JSON + optional file attachments, return the new id."""
import json
import sys
import requests

BASE = "http://127.0.0.1:8000"


def post_case_study(json_path, files=None):
    with open(json_path) as f:
        cfg = json.load(f)
    data = {}
    for k, v in cfg.items():
        if isinstance(v, (dict, list)):
            data[k] = json.dumps(v)
        elif isinstance(v, bool):
            data[k] = "true" if v else "false"
        else:
            data[k] = str(v)
    files_payload = {}
    if files:
        for field, path in files.items():
            files_payload[field] = open(path, "rb")
    r = requests.post(f"{BASE}/api/case-study/", data=data, files=files_payload, timeout=60)
    if r.status_code not in (200, 201):
        print(f"ERROR {r.status_code}: {r.text}"); sys.exit(1)
    body = r.json(); print(f"id={body['id']} name='{body['name']}' strategy={body['resample_strategy']}")
    return body["id"]


def post_csv(case_study_id, csv_path):
    r = requests.post(f"{BASE}/api/csv-files/", data={"case_study": case_study_id},
                      files={"file": open(csv_path, "rb")}, timeout=60)
    if r.status_code not in (200, 201):
        print(f"CSV ERROR {r.status_code}: {r.text}"); sys.exit(1)
    print(f"  csv uploaded: id={r.json()['id']}  file={r.json()['file'].split('/')[-1]}")


def trigger(case_study_id):
    r = requests.post(f"{BASE}/run/", json={"case_study_id": case_study_id}, timeout=30)
    print(f"  trigger: {r.json()}")
    return r.json().get("task_id")


if __name__ == "__main__":
    target = sys.argv[1]
    csv_map = {
        "green":      "/home/rghaf/Projects/lsha_web/case_files/decanter_one_day.csv",
        "green-month":"/home/rghaf/Projects/lsha_web/case_files/decanter_jan2025.csv",
        "green-full": "/home/rghaf/Projects/lsha_web/media/uploads/csv/files/20250514_DecanterData_JanToMay_BOOL.csv",
    }
    if target == "energy":
        cs_id = post_case_study("case_files/energy.json")
        for p in [
            "/home/rghaf/Projects/lsha_web/media/uploads/csv/files/W7_2019-10-14_part0_N0NQaMK.csv",
            "/home/rghaf/Projects/lsha_web/media/uploads/csv/files/W7_2019-10-14_part2_G4yZUmz.csv",
            "/home/rghaf/Projects/lsha_web/media/uploads/csv/files/W7_2019-10-14_part3_0_fRS5swj.csv",
        ]:
            post_csv(cs_id, p)
        trigger(cs_id)
    elif target.startswith("green"):
        cs_id = post_case_study("case_files/green.json")
        post_csv(cs_id, csv_map[target])
        trigger(cs_id)
    else:
        print(f"unknown target: {target}"); sys.exit(1)
    print(f"NEW_ID={cs_id}")
