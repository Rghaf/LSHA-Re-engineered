# lsha_web

A **Django + Celery** web backend that wraps the **L\*SHA** (Learning Stochastic Hybrid Automata) algorithm. It exposes a REST API for submitting learning jobs (case studies), runs the learning pipeline asynchronously via Celery, and returns learned automata models, plots, and PDF reports.

---

## Architecture

| Component | Purpose |
|---|---|
| `lsha_web/` | Django project (settings, URLs, WSGI, Celery app) |
| `rest_api/` | REST API endpoints (job submission, results) |
| `core_algorithm/` | Django app + the L\*SHA engine (`tasks.py` = Celery tasks) |
| `core_algorithm/lsha/` | Vendored L\*SHA learning package |
| `core_algorithm/autotwin_automata_learning/` | Vendored automata-learning package |
| `core_algorithm/skg_connector/` | Vendored Neo4j / knowledge-graph connector |
| `core_algorithm/sha2dt_semantic_mapper/` | Vendored SHA → digital-twin mapper |

> The four packages under `core_algorithm/` are **vendored in-tree** (not pip-installed). They are placed on `sys.path` at runtime by `core_algorithm/tasks.py`, so they travel with the git repository — no separate install step is required.

---

## Requirements

- **Python 3.10** (required — the vendored sub-packages declare `python = "^3.10"`)
- **Redis** — Celery broker and result backend
- **Graphviz** (system package + dev headers) — needed to build `pygraphviz`
- **UPPAAL / `verifyta`** — for UPPAAL-based case studies (trace generation). *Optional* if you only run CSV-based case studies.
- **Neo4j** — *optional*, only if using the knowledge-graph (SKG) connector

---

## Setup

### 1. Get the code

```bash
git clone <your-repo-url> lsha_web
cd lsha_web
```

If migrating from another machine, also copy the runtime data that isn't reproducible from code:

```
db.sqlite3      # the database (or run migrations for a fresh one — see below)
media/          # uploaded media
uploads/        # uploaded case-study files
case_files/     # case-study inputs
results/        # generated outputs
```

### 2. Install system packages

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv redis-server graphviz graphviz-dev
sudo systemctl enable --now redis-server
```

### 3. Create the Python environment and install dependencies

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>Prefer conda? (mirrors the original development env)</summary>

```bash
conda create -n lsha-web python=3.10
conda activate lsha-web
pip install -r requirements.txt
```
</details>

### 4. UPPAAL (optional — needed for UPPAAL case studies)

Download UPPAAL and place `verifyta` at:

```
/opt/uppaal/lib/app/bin/verifyta
```

The path is set in `core_algorithm/tasks.py` (`UPPAAL_BIN`), with a fallback to `/usr/bin/verifyta`. If your UPPAAL lives elsewhere, edit that line. CSV-based case studies use the `SIM`/`CSV` strategy and **do not** require UPPAAL.

### 5. Configuration (optional environment variables)

The following default to local development values and can be overridden via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `NEO4J_URI` | `127.0.0.1:7687` | Neo4j connection |
| `NEO4J_USERNAME` | `neo4j` | Neo4j user |
| `NEO4J_PASSWORD` | `123456789` | Neo4j password |
| `NEO4J_SCHEMA` | `croma` | Neo4j schema |

Redis is expected at `redis://localhost:6379/0` (set in `lsha_web/settings.py`).

---

## Database

The project ships with a SQLite database (`db.sqlite3`).

- **To reuse existing data:** copy `db.sqlite3` from the old machine — no migration needed.
- **For a fresh database:**

```bash
python manage.py migrate
python manage.py createsuperuser   # optional, for the Django admin
```

---

## Running the project

You need **two processes** running simultaneously (Redis must be up first).

**Terminal 1 — Django development server:**

```bash
source .venv/bin/activate
python manage.py runserver
```

The API is served at `http://127.0.0.1:8000/`.

**Terminal 2 — Celery worker:**

```bash
source .venv/bin/activate
celery -A lsha_web worker -l info
```

CORS is pre-configured for front-ends at `http://localhost:3000` and `http://localhost:5173`.

---

## Submitting case studies

Helper scripts are included to post case studies to the running API:

```bash
python post_all_case_studies.py   # posts the bundled UPPAAL + CSV case studies
python post_green_cs.py           # posts the GREEN (energy) CSV case study
```

---

## Notes

- `DEBUG = True` and a development `SECRET_KEY` are set in `lsha_web/settings.py`. **Change both before any production deployment.**
- `core_algorithm/lsha/requirements.txt` and `autotwin_automata_learning/environment.yml` are legacy/stale. The top-level **`requirements.txt` is the source of truth**.
- Some case-study strategies: the UI submits `SIM` for CSV case studies; the backend normalises `SIM` → `CSV`.
