# LSHA-Web — A Universal, Configuration-Driven L\*-SHA Learner

**Thesis-style technical documentation**

---

## Abstract

This document accompanies the `lsha_web` codebase and describes the work
of transforming a research-grade, case-study-specific implementation of
the L\*-SHA active-learning algorithm into a **universal, configuration-
driven engine** that can learn Stochastic Hybrid Automata (SHA) from any
of four independent industrial case studies — a domestic thermostat
(THERMO), a Human-Robot Interaction model (HRI), a CNC spindle-energy
log (ENERGY), and an olive-oil decanter centrifuge (GREEN) — without
requiring a line of Python code to be rewritten between case studies.

The contribution has three layers:

1. **A declarative JSON schema** that lets a domain expert express the
   physical model, the discrete events, the change-point semantics, and
   the data-source layout for any new SHA-shaped problem.
2. **A *dynamic SUL* (`dynamic_sul.py`)** that interprets that schema at
   runtime in place of the previous hand-written `sul_functions.py`
   files (one per case study).
3. **A unified pipeline** (Django + Celery + L\*-SHA core) that runs the
   same `Learner` / `Teacher` / `ObsTable` triple for both UPPAAL-driven
   simulation traces and static CSV bundles uploaded through the web UI.

The remainder of the document is organised as a thesis: motivation,
background, case-study physics, schema specification, an in-depth
description of every component of the dynamic engine, two end-to-end
worked examples, and a discussion of remaining limitations.

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Theoretical Background](#2-theoretical-background)
- [3. The Four Case Studies](#3-the-four-case-studies)
  - [3.1 THERMO — Thermostat](#31-thermo--thermostat)
  - [3.2 HRI — Human-Robot Interaction](#32-hri--human-robot-interaction)
  - [3.3 ENERGY — CNC Spindle](#33-energy--cnc-spindle)
  - [3.4 GREEN — Decanter Centrifuge](#34-green--decanter-centrifuge)
- [4. From Hard-Coded to Universal](#4-from-hard-coded-to-universal)
- [5. The User-Facing JSON Schema](#5-the-user-facing-json-schema)
- [6. The Dynamic SUL — `dynamic_sul.py`](#6-the-dynamic-sul--dynamic_sulpy)
- [7. The Trace Generator — `dynamic_tracegenerator.py`](#7-the-trace-generator--dynamic_tracegeneratorpy)
- [8. The Teacher — `teacher.py`](#8-the-teacher--teacherpy)
- [9. The Observation Table — `dynamic_obstable.py`](#9-the-observation-table--dynamic_obstablepy)
- [10. End-to-End Pipeline — `tasks.py`](#10-end-to-end-pipeline--taskspy)
- [11. Worked Example A — THERMO V1](#11-worked-example-a--thermo-v1)
- [12. Worked Example B — ENERGY V1](#12-worked-example-b--energy-v1)
- [13. Worked Example C — GREEN V1](#13-worked-example-c--green-v1)
- [14. Limitations and Future Work](#14-limitations-and-future-work)
- [Appendix A — Full JSON Schemas](#appendix-a--full-json-schemas)
- [Appendix B — File-by-File Reference](#appendix-b--file-by-file-reference)
- [References](#references)

---

## 1. Introduction

### 1.1 Motivation

Stochastic Hybrid Automata (SHAs) are the *lingua franca* of formal
modelling for cyber-physical systems: they marry a discrete-event
controller (the *modes*) with continuous-time, stochastic flows that
describe the physical state inside each mode. SHAs underpin the
analyses of room thermostats (Henzinger 1996), industrial process
controllers (Quintáns 2014), human-machine interaction (Vicentini et
al. 2020), and energy-efficiency monitors (Pradhan et al. 2023).

Building an SHA model by hand is hard: the physicist must enumerate
every mode, every event, the per-mode flow equations, and the
probability distribution of every parameter. Active automata learning
algorithms — most famously Angluin's **L\*** (Angluin 1987) and its
hybrid extension **L\*-SHA** (Pradhan, Bartocci, Vinarskii 2024) —
*derive* the SHA from observed traces, asking the user (or a simulator)
only a small number of equivalence and refinement questions.

The L\*-SHA reference implementation, however, was originally written
as a **research artefact**, with one bespoke Python module per case
study. Adding a fifth case study meant copying ~400 lines of
`sul_functions.py` from an existing one, renaming variables, rewriting
the trace parser to match the new sensor format, and re-deriving the
flow-fitting code from scratch. The contribution of `lsha_web` is to
**turn that bespoke pattern into a single, JSON-configurable engine**
that any practitioner can drive from a web UI without touching Python.

### 1.2 What "universal" means here

We aim for a learner that, given:

* a JSON description of the system (variables, models, events,
  trace-generation strategy, optional symbolic constants), and
* either an UPPAAL `.xml` model file *or* a bundle of CSV trace files,

will produce the learned SHA — with the same `Learner` / `Teacher` /
`ObsTable` core code path executed regardless of case study.

We deliberately do **not** aim for:

* automatic discovery of variables or events (the user still declares
  these in the JSON);
* automatic guess of the flow type (`EXP_DECAY` vs `LINEAR_GROWTH`
  etc.) — the user picks from a fixed library;
* learning systems whose state cannot be summarised by a single scalar
  observable (the SUL still has one `main` variable per case study).

These remain promising directions; see Section 14.

### 1.3 Roadmap

Sections 2 and 3 give the theoretical and physical background needed to
make sense of the engine. Section 4 spells out the design tension that
motivated the dynamic SUL. Sections 5 to 10 form the technical core:
the JSON grammar followed by a component-by-component description of the
runtime. Sections 11, 12, and 13 walk through three end-to-end examples.

---

## 2. Theoretical Background

### 2.1 Stochastic Hybrid Automata

A **Stochastic Hybrid Automaton** is a tuple

$$
\mathcal{H} \;=\; \bigl( L,\; \ell_0,\; X,\; F,\; E,\; \Sigma,\; G,\; R,\; D \bigr)
$$

where

| Symbol | Meaning |
|---|---|
| $L$ | finite set of *locations* (modes) |
| $\ell_0 \in L$ | initial location |
| $X = \{x_1,\dots,x_n\}$ | continuous state variables |
| $F : L \rightarrow \mathcal{F}$ | flow condition assigned to each location |
| $E \subseteq L \times L$ | discrete transitions (edges) |
| $\Sigma$ | alphabet of event symbols |
| $G : E \rightarrow 2^{X \cup \Sigma}$ | guard predicates over $X$ and $\Sigma$ |
| $R : E \rightarrow (X \rightarrow X)$ | reset maps |
| $D : L \times \mathcal{F} \rightarrow \mathcal{D}$ | per-location, per-flow probability distributions |

Each *flow condition* $f \in \mathcal{F}$ is a parametric ODE over $X$;
in this work the supported flow families are **constant**, **linear**,
**exponential decay**, and **exponential growth** — see Section 6.7.
The *parameter* of a flow condition (its rate, slope, or asymptote) is
sampled from a probability distribution $D(\ell, f)$ that the learner
must also identify.

A *trajectory* of $\mathcal{H}$ is an alternation of continuous flows
and discrete jumps. Recording a trajectory yields a *trace* — a finite
sequence of timestamped event symbols paired with sampled signal values.
The objective of L\*-SHA is to **reconstruct $\mathcal{H}$ from a finite
collection of traces**.

### 2.2 Active Automata Learning (L\*)

Angluin's L\* algorithm (Angluin 1987) learns the *minimal DFA*
recognising a regular language using two oracle queries:

* **Membership query** $\textsc{Mem}(w)$ — does the target language
  contain word $w$?
* **Equivalence query** $\textsc{Eq}(\mathcal{A})$ — is hypothesis
  automaton $\mathcal{A}$ equivalent to the target? If not, return a
  counterexample word.

The algorithm maintains an *observation table* $T = (S, E, \mathit{obs})$,
indexed by row words $S \cup S\!\cdot\!\Sigma$ and column words $E$,
whose cell $(s, e)$ stores $\mathit{obs}(s\!\cdot\!e)$. The table is
*closed* when every lower row matches some upper row, and *consistent*
when equivalent rows behave the same on every $a \in \Sigma$. When both
properties hold, the table induces a unique minimal DFA.

### 2.3 L\*-SHA — Extension to SHA

Pradhan et al. (2024) extend L\* with two extra oracle queries that
account for the continuous-stochastic nature of SHA:

* **Model-Identification query** $\textsc{MI}(w)$ — given prefix $w$,
  which flow condition $f \in \mathcal{F}$ best fits the observed
  signal segments that follow $w$ across all known traces? Implemented
  via Dynamic Time Warping (DDTW) against ideal flow curves.
* **Hypothesis-Testing query** $\textsc{HT}(w, f)$ — given prefix $w$
  and chosen flow $f$, which probability distribution
  $\delta \in D(\cdot, f)$ best matches the parameter samples extracted
  from the segments? Two backends are supported: **deterministic**
  exact-match (D) and **stochastic** Kolmogorov–Smirnov two-sample test (S).

The cell $(s, e)$ of the L\*-SHA table is therefore not a simple bit
but a pair $(f, \delta)$. Closedness and consistency lift to this
richer alphabet via *strict* (cell-for-cell) or *weak* (wildcard for
unobserved) equality, configurable per case study.

A third query — **refinement** $\textsc{Ref}(T)$ — is invoked whenever
the table has under-observed cells: the Teacher asks the trace generator
for additional traces and feeds them through the SUL parser, populating
empty cells until either no ambiguity remains or no fresh traces can be
produced.

---

## 3. The Four Case Studies

The four case studies span two trace-generation regimes (synthetic via
UPPAAL, or real via CSV) and four very different physical systems.
Section 3.x first explains the **physics**, then the **raw data
shape**, then how each maps to the SHA being learned.

### 3.1 THERMO — Thermostat

#### 3.1.1 Physics

A single-room thermostat controls the heater so that the room
temperature $T_r(t)$ stays inside a comfort band $[T_\min, T_\max]$.
The continuous dynamics follow a Newton-style first-order ODE:

$$
\dot T_r \;=\; \begin{cases} -\dfrac{T_r}{R} & \text{when heater OFF} \\[4pt]
  K - \dfrac{T_r}{R} & \text{when heater ON} \end{cases}
$$

with $R$ the *thermal resistance* (s) and $K$ the *thermal forcing*
(°C/s). Closed-form solutions are exponential:

$$
T_r(t) = T_0\,e^{-t/R} \quad \text{(OFF)} \qquad
T_r(t) = R\,K - (R\,K - T_0)\,e^{-t/R} \quad \text{(ON)}.
$$

Both $R$ and $K$ are *random variables* whose distributions depend on
window state — e.g. with both windows open, $R$ is shorter and $K$ is
smaller (cold air leaks in faster). The discrete state is
$(\textsf{ON/OFF}, \textsf{open}_0, \textsf{open}_1)$, giving up to
**8 modes**; the SHA we learn captures both the discrete transitions
between them and the parametric distributions of $(R, K)$ in each.

#### 3.1.2 Raw data

Generated synthetically by **UPPAAL** with the `verifyta -t0` flag.
Each invocation writes a plain-text file in this shape:

```
T_r:
[0]: (0.00, 15.20) (1.00, 15.05) (2.00, 14.90) … (200.00, 18.45)
[1]: (0.00, 15.20) (1.00, 15.10) …                           ← second sim run, ignored
t.ON:
[0]: (0.00, 0) (4.25, 1) (39.15, 0) (91.06, 1) …
r.open:
[0]: (0.00, 0) (36.00, 1) …
```

Three "section blocks", each with a section header followed by one or
more `[i]:` lines of `(time, value)` tuples. Subsequent runs `[1]:`,
`[2]:` are deadlock-driven duplicates and are intentionally skipped (see
[`_parse_uppaal`](core_algorithm/dynamic_sul.py)).

#### 3.1.3 Mapping to Python

| UPPAAL section | Friendly JSON name | `signals` dict key | Role |
|---|---|---|---|
| `T_r` | `T` | `'main'` | estimated |
| `t.ON` | `t.ON` | `'t.ON'` | driver |
| `r.open` | `r.open` | `'r.open'` | driver |

The friendly-vs-physical name divergence (`T` ≠ `T_r`) is bridged by
the JSON `aliases` map (Section 5.7) or per-variable `source` field.

#### 3.1.4 Discrete events

Three event symbols suffice for V1:

| Symbol | Trigger | Guard | Flow id |
|---|---|---|---|
| `h_0` | heater turns on | `t.ON == 1` | 1 (EXP_GROWTH) |
| `c_0` | heater turns off, windows closed | `t.ON == 0 and r.open == 0` | 0 (EXP_DECAY) |
| `c_1` | heater turns off, a window is open | `t.ON == 0 and r.open > 0` | 0 (EXP_DECAY) |

Higher case-study versions (V3, V8, V10 in the legacy `config.ini`)
introduce additional events for the *both windows open* combination
and a linear flow alternative.

---

### 3.2 HRI — Human-Robot Interaction

#### 3.2.1 Physics

Human muscular **fatigue** $F(t) \in [0, 1]$ is well modelled by an
exponential dynamic that switches between two regimes (Konz 1998;
Ma et al. 2009):

$$
\dot F = \lambda\,(1 - F) \quad \text{when person is BUSY (working)}
$$

$$
\dot F = -\mu\,F \quad \text{when person is IDLE (resting)}
$$

with $\lambda$ the *fatigue rate* and $\mu$ the *recovery rate*, both
profile-dependent: a `young_healthy` worker has small $\lambda$ and
rapid $\mu$, whereas an `elderly_sick` worker has the reverse. The
**closed-form** solutions are again exponentials:

$$
F(t) = 1 - (1 - F_0)\,e^{-\lambda t} \quad \text{(busy)} \qquad
F(t) = F_0\,e^{-\mu t} \quad \text{(idle)}.
$$

A worker who reaches $F \geq 1$ "passes out" — modelled as an absorbing
state in the SHA. Position-dependent location effects (sitting vs
standing vs running, in office vs in waiting room) further modulate
$\lambda$ and $\mu$, giving a richer mode space.

#### 3.2.2 Raw data

Same UPPAAL text format as THERMO; the section headers are the
real-valued variables exposed by the UPPAAL `simulate{}` query. From
the V1 model file (see [uploads/uppaal/models/](uploads/uppaal/models/)):

```
humanFatigue[currH - 1]:
[0]: (0.0, 0.0) (31.0, 0.0045) (32.0, 0.0090) …
humanPositionX[currH - 1]:
[0]: …
amy.busy || amy.p_2:
[0]: (0.0, 0) (31.0, 1) (60.0, 0) (91.0, 1) …
```

Note the awkward UPPAAL identifiers: bracketed array accesses
(`humanFatigue[currH - 1]`) and Boolean-OR composites
(`amy.busy || amy.p_2`). These **cannot be renamed without modifying
the UPPAAL model**, so the dynamic SUL ships an *alias* mechanism that
lets the user keep friendly JSON names while the parser reads the awkward
physical headers.

#### 3.2.3 Mapping to Python

Recommended JSON `aliases` map for V1:

```json
"aliases": {
  "F":          "humanFatigue[currH - 1]",
  "h.busy":     "amy.busy || amy.p_2",
  "h.location": "humanPositionX[currH - 1]"
}
```

The parser strips whitespace from the section headers when matching, so
the user can write `"h.busy": "amy.busy||amy.p_2"` (no spaces) and it
will still match `amy.busy || amy.p_2`.

#### 3.2.4 Discrete events

| Symbol | Guard | Flow |
|---|---|---|
| `h_start` | `h.busy == 1` | EXP_GROWTH (busy) |
| `h_stop_idle` | `h.busy == 0 and h.location != 1` | EXP_DECAY (idle) |
| `h_stop_sit` | `h.busy == 0 and h.location == 1` | EXP_DECAY (sitting) |
| `h_pass_out` | `F >= 1.0` | EXP_DECAY (absorbing) |

The `h.location` integer encodes `0=standing/walking`, `1=sitting`,
`2=running`, etc. — one of the abstractions the user provides via the
UPPAAL model and the `aliases`/`source` field.

---

### 3.3 ENERGY — CNC Spindle

#### 3.3.1 Physics

A CNC milling machine's **spindle motor** consumes electrical energy at
a rate that depends almost entirely on its angular velocity (RPM). For
the W7 dataset (provided by an industrial partner) the relationship is
approximately piecewise-constant: the spindle has a small number of
*operating regimes* (idle, low-speed roughing, high-speed finishing,
…) and within each regime the instantaneous power $P(t)$ is well
approximated by a constant $P_i + \varepsilon$ with $\varepsilon$ a
small Gaussian noise.

The *cumulative energy* counter (in joules) is therefore a piecewise-
linear function of time, whose slope encodes the regime:

$$
E(t) = E(t_0) + \int_{t_0}^{t} P(\tau)\,d\tau \;\approx\; E(t_0) + P_i \cdot (t - t_0).
$$

The SHA we learn has one mode per RPM bin and constant flows.

#### 3.3.2 Raw data

A bundle of **wide-format CSV** files with the columns

```
,TimestampUTC,HEADSTOCK__SPINDLE_DRIVE___1___ENERGY,HEADSTOCK__SPINDLE_MOTOR___1___RPM,RT__PALLET_LOCKING___1___PRESSURE
49,2019-10-14 06:03:55,,2285.39,
50,2019-10-14 06:04:00,2575.98,2285.84,
51,2019-10-14 06:04:05,,2285.89,
…
```

Two characteristics matter:

1. **Sparse sensor cadence** — the energy counter and pressure sensor
   report once every ~5 s, while RPM reports every 1 s. Empty cells
   are filled by linear interpolation (Step 5 of `_parse_csv`).
2. **High-frequency RPM jitter** — even at "constant" speed RPM
   fluctuates by ±5 around the bin centre. The user can either round
   it via `round_columns: {"…RPM": 100}` or rely on the change-point
   detector's CHECK 1 (label change) to ignore intra-bin noise.

Multiple files are concatenated into one pandas `DataFrame` and sorted
chronologically before further processing.

#### 3.3.3 Mapping to Python

| CSV column | JSON name | `signals` key | Role |
|---|---|---|---|
| `HEADSTOCK__SPINDLE_DRIVE___1___ENERGY` | (same) | `'main'` | estimated |
| `HEADSTOCK__SPINDLE_MOTOR___1___RPM` | (same) | (driver name) | driver |
| `RT__PALLET_LOCKING___1___PRESSURE` | (same) | (driver name) | driver |

No aliasing needed because the JSON references the physical column
names verbatim.

#### 3.3.4 Discrete events

| Symbol | Guard | Mode |
|---|---|---|
| `l` | `PRESSURE >= p_min` | pallet locked (loading) |
| `u` | `PRESSURE < p_min` | pallet unlocked (unloading) |
| `i_0` | `RPM < 100` | idle |
| `m_1` | `100 <= RPM < 1100` | low speed |
| `m_2` | `1100 <= RPM < 2100` | medium speed |
| `m_3` | `2100 <= RPM <= 3100` | high speed |

The thresholds (`p_min = 50`, `s_min = 100`, `s_max = 3100`) are
parameters of the `constants` block — symbolic so the user can tune
them without rewriting guards.

---

### 3.4 GREEN — Decanter Centrifuge

#### 3.4.1 Physics

An olive-oil **decanter centrifuge** separates the oil emulsion from
the wet solids. While running, its electric motor draws an absorbed
power *Assorbimento* (Italian for "absorption") that varies linearly
with the **torque** *Coppia* on the bowl. With the decanter OFF the
absorbed power is zero (the constant flow); with the decanter ON it
follows a linear flow whose slope depends on the torque load:

$$
A(t) = \begin{cases} 0 & \text{decanter OFF} \\
  A_0 + \alpha(\textit{Coppia})\,(t - t_0) & \text{decanter ON} \end{cases}
$$

where $\alpha$ is a piecewise-constant function of torque. The SHA
captures the start/stop discrete transitions and the four torque-binned
linear flows.

#### 3.4.2 Raw data

A **long-format SCADA dump** spread across several CSV files, one per
data type (boolean signals, real-valued signals). The structure is

```
"DataObjectName","DataObjectField","Value","TimeStamp"
"DecanterMB","TermicaCoclea",False,"2025-01-08 10:18:38.451"
"DecanterMB","MarciaDecanter",True,"2025-01-10 09:13:29.388"
"DecanterMB","MarciaDecanter",False,"2025-01-10 12:15:00.987"
…
```

Three transformations are required before the SUL can use this:

1. **Filter** rows by `DataObjectName == "DecanterMB"` (or whatever
   ``filter_object`` the user specified) so we don't accidentally
   pivot in unrelated equipment.
2. **Pivot** from long to wide so each `DataObjectField` becomes its
   own column (`MarciaDecanter`, `Coppia`, `Assorbimento`, …) with
   `Value` as the cell.
3. **Coerce booleans** — the literal Python `True`/`False` and the
   strings `"True"`/`"False"` are mapped to numeric `1`/`0` so
   `pivot_table(aggfunc="mean")` does not choke and so guards can
   compare them with `==`.

The boolean and real CSVs are simply concatenated; the long-format
pivot puts everything back together by timestamp.

#### 3.4.3 Mapping to Python

| Pivoted column | JSON name | `signals` key | Role |
|---|---|---|---|
| `Assorbimento` | `Assorbimento` | `'main'` | estimated |
| `MarciaDecanter` | `MarciaDecanter` | (driver name) | driver |
| `Coppia` | `Coppia` | (driver name) | driver |

#### 3.4.4 Discrete events

| Symbol | Guard | Notes |
|---|---|---|
| `decanter_ON` | `MarciaDecanter == 1 and prev.MarciaDecanter == 0` | rising-edge detection |
| `decanter_OFF` | `MarciaDecanter == 0 and prev.MarciaDecanter == 1` | falling-edge detection |
| `torque_0_20` | `MarciaDecanter == 1 and 0 <= Coppia < 20` | torque bin |
| `torque_20_40` | `MarciaDecanter == 1 and 20 <= Coppia < 40` | torque bin |
| `torque_40_60` | `MarciaDecanter == 1 and 40 <= Coppia < 60` | torque bin |
| `torque_60_80` | `MarciaDecanter == 1 and 60 <= Coppia < 80` | torque bin |

Note the use of `prev.MarciaDecanter` for edge detection — a syntactic
convenience the dynamic SUL accepts and rewrites to the internal
`prev_MarciaDecanter` lookup before guard evaluation.

---

## 4. From Hard-Coded to Universal

### 4.1 The legacy code path

The L\*-SHA reference implementation organises its case-study-specific
code under

```
core_algorithm/lsha/sha_learning/case_studies/
├── thermostat/   sul_functions.py + sul_definition.py
├── hri/          sul_functions.py + sul_definition.py
├── energy/       sul_functions.py + sul_definition.py
├── energy_made/  sul_functions.py + sul_definition.py
├── energy_sim/   sul_functions.py + sul_definition.py
├── gr3n/         sul_functions.py + sul_definition.py
└── auto_twin/    sul_functions.py + sul_definition.py
```

Each `sul_functions.py` exports four functions that the SUL adapter
calls polymorphically:

```python
def parse_data(file_path) -> List[SampledSignal]
def is_chg_pt(curr, prev)  -> bool
def label_event(events, signals, t) -> Event
def get_<x>_param(segment, flow) -> float
```

The function bodies hard-code the column names of the trace files, the
threshold constants, the guard logic, and the flow-fitting maths.
Adding a new case study meant copying ~400 lines into a new directory
and editing every constant by hand. Editing an existing case study
required rebuilding the Python package because the dispatch was
controlled by the `CASE_STUDY` field of `config.ini`.

### 4.2 Pain points

The legacy organisation suffered from three concrete problems:

1. **Code duplication.** Five of the seven case studies had near-
   identical CSV-loading skeletons that diverged only in column names
   and tolerance constants. A bug fix in one had to be backported by
   hand to the others.
2. **Tight coupling between data and analysis.** Changing the
   `_parse_uppaal` regex (because UPPAAL switched its pretty-printer)
   required editing every case study, even the CSV-only ones, because
   the function lived inside each module.
3. **No web-driven configurability.** A new domain expert wanting to
   try a fresh case study had no way to do so without (a) writing
   Python, (b) re-deploying the Django app, and (c) restarting Celery.

### 4.3 Design goals for the universal SUL

The replacement is governed by four principles:

> **G1 — Schema-first.** Everything that distinguishes one case study
> from another must be expressible in a JSON file the user can edit in
> the web UI.
>
> **G2 — Behaviour-preserving.** Existing case studies must continue
> to learn the same SHAs they did under the legacy code path; the
> dynamic engine is a refactoring, not a rewrite of the algorithm.
>
> **G3 — Best-effort fallback.** Missing JSON fields must trigger
> sensible defaults rather than crashes; missing CSV columns should
> become zero-filled signals plus a warning rather than an exception.
>
> **G4 — Sandbox safety.** Guard expressions are evaluated by Python
> `eval`; we must restrict the environment to a tiny, safe sub-language
> so a user-supplied JSON cannot read the disk or import modules.

Section 5 spells out the schema; Section 6 walks through the engine.

---

## 5. The User-Facing JSON Schema

### 5.1 Top-level structure

```json
{
  "case_study":  "<free-text label>",
  "version":     "<free-text version, e.g. V1>",
  "variables":   [ /* §5.2 */ ],
  "models":      [ /* §5.3 */ ],
  "events":      [ /* §5.4 */ ],
  "constants":   { /* §5.5 */ },
  "aliases":     { /* §5.6 */ },
  "trace_generation": { /* §5.7 */ }
}
```

`case_study` and `version` are informational — they appear in
generated filenames and reports but are otherwise inert. The four
substantive sections are detailed below.

### 5.2 `variables[]`

Each entry declares one signal:

```json
{
  "name":   "<friendly name used in guards>",
  "source": "<physical column name in the trace>",   // optional
  "type":   "REAL" | "INT" | "BOOL",                  // informational
  "role":   "estimated" | "driver" | "context"
}
```

Field semantics:

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | The label that guards refer to (`F`, `RPM`, `MarciaDecanter`). |
| `source` | no | The physical column inside the UPPAAL section header or CSV column. Defaults to `name`. |
| `type` | no | Documentation; the parser coerces to numeric anyway. |
| `role` | yes | Determines downstream handling: |

* `estimated` — the single observable variable the SHA is *learning*.
  Mapped internally to the special key `'main'`.
* `driver` — a discrete or continuous signal whose value drives mode
  transitions. Multiple allowed.
* anything else (e.g. `context`) — kept as auxiliary signal, available
  to guards but not used for change-point detection.

Exactly **one** `estimated` variable is supported per case study.

### 5.3 `models[]`

Each entry declares one *flow condition* (a parametric mode-internal
ODE family):

```json
{
  "id":   <integer>,
  "name": "<human-readable label>",
  "type": "MEAN" | "CONSTANT" | "LINEAR" |
          "EXP_DECAY" | "EXP_GROWTH" |
          "LINEAR_DECAY" | "LINEAR_GROWTH",
  "params": { /* optional, model-specific */ }
}
```

The supported `type` values map to the formulas in Section 6.7.
`MEAN` and `CONSTANT` are aliases (mean of segment, zero rate);
`LINEAR` reports a signed slope; the four directional types
(`*_DECAY`, `*_GROWTH`) are kept distinct so the higher-level
hypothesis-testing query can refuse a sign mismatch.

### 5.4 `events[]`

Each entry declares one discrete event symbol:

```json
{
  "symbol":        "<unique alphabet symbol>",
  "trigger_value": <int, optional>,
  "model_id":      <id from models[]>,
  "guard":         "<Python boolean expression>"
}
```

* `symbol` — the alphabet letter that appears in the L\* word.
* `trigger_value` — for UPPAAL-driven case studies, the integer the
  trace generator stuffs into `force_act[]` to *replay* this event in
  the next simulation. Ignored in CSV mode.
* `model_id` — the flow that is active **after** this event fires.
* `guard` — a boolean Python expression evaluated by `safe_eval`
  (Section 6.3). May reference any signal name, any `prev.X`, any
  symbolic constant from §5.5, and the constants `True`/`False`.

Guards are checked in **definition order** at every sample; the first
matching guard wins. The order therefore encodes priority and lets the
user write catch-all branches by leaving the last guard empty.

### 5.5 `constants{}`

A flat map of *symbolic constants* injected into every guard's
evaluation context. Useful for keeping magic numbers out of the
guard expressions:

```json
"constants": { "p_min": 50.0, "s_min": 100, "s_max": 3100 }
```

A guard like `"PRESSURE >= p_min"` is then far more meaningful than
the equivalent `"PRESSURE >= 50.0"`, and tuning `p_min` requires no
change to the guards themselves.

The legacy `trace_generation.physics_constants` block is **also**
merged in; either location works.

### 5.6 `aliases{}` and per-variable `source`

Two equivalent ways to translate friendly names to physical column
names. The dictionary form is convenient for HRI:

```json
"aliases": {
  "F":          "humanFatigue[currH - 1]",
  "h.busy":     "amy.busy || amy.p_2",
  "h.location": "humanPositionX[currH - 1]"
}
```

The per-variable form keeps the mapping next to the role declaration:

```json
"variables": [
  { "name": "F", "source": "humanFatigue[currH - 1]",
    "type": "REAL", "role": "estimated" }
]
```

If both are present, both are merged; conflicts resolve in favour of
the per-variable `source` (more specific wins).

### 5.7 `trace_generation{}`

This block is the dispatch point between UPPAAL and CSV strategies:

```json
"trace_generation": {
  "strategy": "UPPAAL" | "CSV" | "SIM",   // SIM is a legacy alias of CSV
  "max_length": 15,                       // padding for force_act[]

  // ── UPPAAL-only fields ─────────────────────────────────
  "xml_force_variable":  "force_act",     // single-driver
  "xml_force_variables": ["force_act"],   // multi-driver
  "xml_action_variable": "force_exe",

  // ── CSV-only fields (flat or nested under "csv") ─────
  "csv": {
    "files":              [ "..." ],      // informational
    "boolean_files":      [ "..." ],      // GREEN: extra files to merge
    "real_files":         [ "..." ],      //   "
    "timestamp_column":   "TimeStamp",
    "wide_or_long":       "wide" | "long",
    "object_name_column": "DataObjectName",
    "field_column":       "DataObjectField",
    "value_column":       "Value",
    "filter_object":      "DecanterMB",
    "max_length":         50
  },

  // ── Legacy flat CSV fields (still accepted) ──────────
  "csv_format":         "horizontal" | "vertical",
  "key_column":         "DataObjectField",
  "value_column":       "Value",
  "time_column":        "TimeStamp",
  "time_format":        "datetime" | "numeric",
  "replace_values":     { "OFF": 0, "ON": 1 },
  "interpolate_method": "linear",
  "round_columns":      { "RPM": 100 },
  "physics_constants":  { /* same shape as §5.5 */ }
}
```

The new `csv: {...}` nested block exists for UI cleanliness — the user
can ship a single object instead of ten flat keys. Internally the
nested values are flattened on top of the legacy keys with `nested
wins`, and aliases are normalised: `wide_or_long="wide"` becomes
`csv_format="horizontal"`, `timestamp_column` becomes `time_column`,
`field_column` becomes `key_column`.

---

## 6. The Dynamic SUL — `dynamic_sul.py`

The dynamic SUL replaces the per-case-study `sul_functions.py` files
with a single 800-line module that interprets the JSON at runtime. It
exposes four entry points to the rest of the system:

```python
parse_data_dynamic(file_paths, args)        -> dict[str, np.ndarray]
is_chg_pt_dynamic(signals, index, args)     -> bool
label_event_dynamic(signals, index, args)   -> str           # event symbol
get_physics_param_dynamic(signals, s, e, args) -> dict[str,float]
```

All four take an `args` dict — the same dict assembled by `tasks.py` —
and consult its `'trace_generation'`, `'events'`, `'models'`, and
`'variables'` keys.

### 6.1 The signals dictionary — internal data type

Every entry point returns or consumes a uniform structure:

```python
{
  'time'           : np.ndarray,    # absolute time (s) from trace start
  'main'           : np.ndarray,    # the estimated variable
  '<friendly_name>': np.ndarray,    # one entry per driver / context
  ...
}
```

All arrays share the same length and the same indexing — `signals[k][i]`
is always "the value of signal `k` at the i-th sample". The
`'main'` key is reserved; everything else is keyed by the friendly
name from `variables[]` (or the legacy `driver_signal` field).

### 6.2 Variable resolution — `_resolve_variable_roles` and `_build_target_vars`

These two helpers convert the `variables[]` block into the legacy
fields used by the rest of the engine:

```python
_resolve_variable_roles(args)
    # Reads args['variables'], args['aliases']
    # Writes  args['main_var'], args['driver'],
    #         args['context_variables'], args['alias_map']
```

The function is **idempotent** and **non-destructive**: an
explicitly-set legacy field always wins over a derived one, so a
domain expert who fills in `cs_instance.driver_signal` from the UI
can still override what the JSON would have suggested.

```python
_build_target_vars(args) -> dict[str, str]
    # Returns { physical_column_name : internal_signal_key }
```

This is the table consulted by both parsers (CSV and UPPAAL) when they
need to know which column to read for which output key. The keys are
*physical* names (so the parser can find them in the file); the values
are *friendly* keys (so guards can refer to them by their UI name).

### 6.3 Guard sandbox — `safe_eval`

User-supplied guard expressions are evaluated by Python `eval` inside
a tightly controlled environment:

```python
safe_eval("MarciaDecanter == 1 and prev.MarciaDecanter == 0", context)
```

The pipeline is:

1. **`prev.X` rewrite.** A precompiled regex `\bprev\.` substitutes
   `prev.X` → `prev_X` so the user's natural attribute-style syntax
   maps to the internal `prev_<key>` lookup written by
   `label_event_dynamic`.
2. **Placeholder substitution.** Each context key is sorted
   *longest-first* (so `RPM` cannot clobber `prev_RPM`), then replaced
   with a positional placeholder `__VAR_<i>__`. The actual value goes
   into the eval namespace under that placeholder. This step is what
   makes dotted column names like `t.ON` work — they are matched as
   plain strings without Python's attribute lookup ever firing.
3. **Restricted `eval`.** The namespace contains only the placeholder
   bindings and the `math` module; `__builtins__` is set to `None`.
   Any exception (e.g. a typo in the guard) is silently absorbed and
   the guard is treated as `False`, so a bad guard never crashes the
   learning run.

### 6.4 CSV parsing — `_parse_csv`

Eight steps, applied in this fixed order:

| # | Step | Why |
|---|---|---|
| 0 | Splice in `boolean_files` / `real_files` | GREEN-style multi-file groups |
| 1 | `read_csv` and `concat` | Multi-file ENERGY / GREEN |
| 1b | Filter rows by `filter_object` | Long-format GREEN dump |
| 1c | Coerce `True`/`False` literals to `1`/`0` | Pivot needs numeric Value |
| 2 | Pivot long → wide if `csv_format == 'vertical'` | GREEN |
| 3 | Apply user `replace_values` map | Custom token mapping |
| 4 | Parse and zero-reference the time axis | Both `datetime` and `numeric` |
| 5 | Linear-interpolate NaN gaps + bfill/ffill | Sparse sensor cadence |
| 6 | Round selected columns to bins via `round_columns` | RPM discretisation |
| 7 | Build `signals` dict, zero-fill missing columns + warn | Robustness (G3) |

The whole function is wrapped in `try / except` so any catastrophic
parse error returns an empty dict (with a stack trace logged) rather
than aborting the Celery task.

### 6.5 UPPAAL parsing — `_parse_uppaal`

Three passes:

1. **Pass 1.** Walk the file line by line, recognising headers
   (lines ending in `:`) and `[0]:` data lines. Tuples
   `(t, v)` are stripped of parentheses and split on commas. Lines
   beginning with `[1]:`, `[2]:` etc. are intentionally ignored —
   they correspond to *additional* simulation runs caused by
   non-deterministic deadlocks in the UPPAAL model and would otherwise
   double-count events.
2. **Pass 2.** Build the union time axis from every variable's data
   points and sort it; this is the master grid.
3. **Pass 3.** For each requested variable, advance a pointer through
   its `(t, v)` list and **forward-fill** the value at every grid
   timestamp. Forward-fill is exactly UPPAAL's own semantics — a
   variable holds its last value until a transition writes a new one.

Whitespace inside section headers (`amy.busy || amy.p_2` vs
`amy.busy||amy.p_2`) is normalised away before matching, so the user's
JSON can use either form.

### 6.6 Change-point detection — `is_chg_pt_dynamic`

A change point is the start of a new "segment" — a maximal
constant-mode interval inside which the SHA's flow condition is
unchanged. The detector applies two checks in order:

1. **CHECK 1 — label change.** Call `label_event_dynamic` for the
   current and previous index; if the resulting symbols differ, a
   discrete event has fired and the index is a change point. This
   catches every transition in every case study, regardless of which
   driver caused it.
2. **CHECK 2 — driver step.** For each declared driver, compare the
   current and previous numeric values; if the absolute change exceeds
   the per-driver tolerance (`physics_constants.tolerances[<driver>]`,
   falling back to legacy `SPEED_RANGE`), fire a change point.

When **no tolerance is configured** for a numeric driver, CHECK 2 is
deliberately *skipped* (the previous behaviour of "tolerance defaults
to 0, so every micro-jitter fires" produced too many false change
points for ENERGY/GREEN). For non-numeric drivers (booleans, strings)
strict equality is always used.

### 6.7 Event labelling — `label_event_dynamic`

Builds the guard-evaluation context from:

* every signal's current value (key = friendly name, including `'main'`);
* every signal's previous value, prefixed `prev_`;
* `trace_generation.physics_constants`;
* top-level `constants`.

Then walks `events[]` in definition order, evaluates each `guard` via
`safe_eval`, and returns the first matching `symbol`. If none match,
the symbol of the first event is returned as a safe fallback (this is
the "default branch" semantics).

### 6.8 Physics parameter extraction — `get_physics_param_dynamic`

For a signal segment $[s_i, s_{i+1}, \dots, s_e]$ extracted from
`signals['main']`, returns a dict `{'mean': μ, 'rate': r}` where:

* **`mean`** is always the arithmetic mean — used by the deterministic
  HT query for CONSTANT/MEAN flows and as a fallback elsewhere.
* **`rate`** depends on the flow type of the event that fired at
  `s`:

| Flow type | Formula |
|---|---|
| `MEAN`, `CONSTANT` | $r = 0$ |
| `LINEAR`, `LINEAR_GROWTH` | $r = \dfrac{1}{n}\sum_i (s_i - s_{i-1})$ |
| `LINEAR_DECAY` | $r = \dfrac{1}{n}\sum_i (s_{i-1} - s_i)$ |
| `EXP_DECAY` | $r = -\dfrac{1}{n}\sum_i \ln(s_i / s_{i-1})$ (positive entries only) |
| `EXP_GROWTH` | $r = \dfrac{1}{n}\sum_i (s_i - s_{i-1})$ (used as $K$ proxy) |

The flow type is looked up by:

1. Calling `label_event_dynamic` at the segment start to get the
   active event symbol.
2. Looking up the symbol's `model_id` in the `events[]` list.
3. Looking up the model's `type` in the `models[]` list.

---

## 7. The Trace Generator — `dynamic_tracegenerator.py`

The trace generator exposes a single method to the Teacher:

```python
custom_tg.get_traces(n: int) -> list[str]   # paths
```

and dispatches internally to one of two strategies.

### 7.1 Strategy normalisation

Three different spellings have been used for "CSV mode" over the
project's lifetime: `"SIM"` (the original config.ini), `"CSV"` (the
new web UI), and `"sim"`/`"static"` (informal). The generator's
`__init__` normalises them all via `_normalise_strategy`:

```python
_CSV_ALIASES    = {'CSV', 'SIM', 'CSV_FILES', 'STATIC'}
_UPPAAL_ALIASES = {'UPPAAL', 'UPP', 'VERIFYTA'}
```

`tasks.py` and `teacher.py` apply the same normalisation, so every
component agrees on the canonical spelling regardless of where the
strategy string came from.

### 7.2 UPPAAL strategy

`get_traces_uppaal(n)` performs three steps per call:

1. **Patch the `.xml` model file in place** via `fix_model`: rewrite
   the line `int force_act[MAX_E] = {…};` with the trigger sequence
   built from the current word's events; force `bool force_exe = true`
   so the model honours the forced events instead of running its
   native non-deterministic logic; rewrite `const int TAU = …;` to be
   at least 200 and proportional to the event count.
2. **Spawn `verifyta`** as a subprocess with `-t0`, redirecting stdout
   to a fresh trace file in `output_dir`. The filename embeds a
   random 32-bit integer to avoid collisions across parallel runs.
3. **Validate** the resulting file is non-empty and append its path
   to the return list.

`build_event_strings` handles both single-driver case studies (THERMO
V1 uses `force_act`; HRI V1 uses `force_open`) and multi-driver case
studies (`xml_force_variables` lists multiple arrays, each populated
from the per-event `trigger_values` dict).

### 7.3 CSV strategy — yield-once short-circuit

`get_traces_csv(n)` is more interesting. Because L\* refinement keeps
asking the trace generator for more data while it's still uncertain,
a static CSV bundle has to **lie convincingly** about being exhausted
the moment it's been consumed once:

```python
def get_traces_csv(self, n=1):
    if not self.csv_files:
        return []                     # nothing to give
    if not self.csv_yielded:
        self.csv_yielded = True
        return self.csv_files          # full bundle, ONCE
    return []                          # all subsequent calls
```

The Teacher treats an empty list as "no further refinement possible"
and gracefully exits the loop, which is exactly what we want: the SHA
learned from one full pass over the CSV bundle is the best we can
achieve without acquiring more data from the field.

### 7.4 Multi-file groups

GREEN ships several CSVs split by data type (boolean signals in one
file, real-valued signals in another). The generator's `__init__`
augments `self.csv_files` with everything it finds under
`trace_gen_config['boolean_files']`, `['real_files']`, and
`['extra_files']` (and the same keys inside the nested `csv: {…}`
block), de-duplicating against paths that already exist on disk.

---

## 8. The Teacher — `teacher.py`

`CustomTeacher` is the L\*-SHA oracle implementation. It receives
hyperparameters as a plain dict (so the Django UI can override the
legacy `config.ini` values) and performs five oracle queries.

### 8.1 `mi_query(word)` — Model Identification

For prefix `word`, collect every signal segment in the trace bank
that follows it; for each segment compute the DDTW distance against
every candidate `FlowCondition`'s ideal curve and pick the closest.
A flow that wins ≥75 % of the segments is reported as "the model"; a
narrower win returns `None` so the table cell stays empty and forces
more refinement.

When the UI's `mi_query` toggle is OFF, the Teacher answers with the
SUL's `default_m` flow — useful for ENERGY/GREEN where the user only
cares about discrete event ordering.

### 8.2 `ht_query(word, flow)` — Hypothesis Testing

Given a prefix and the flow chosen by `mi_query`, identify the
probability distribution of the flow's parameter:

* **Deterministic (D)** — exact-match on the scalar parameter; used
  for noiseless UPPAAL-derived metrics in THERMO and HRI.
* **Stochastic (S)** — two-sample Kolmogorov–Smirnov test against
  every existing distribution; used when sensors are noisy
  (ENERGY, GREEN). The user-supplied `p_value` and `noise` knobs
  control the KS rejection threshold.

If neither variant finds an existing distribution that fits, a fresh
`ProbDistribution` is allocated and registered with the SUL.

### 8.3 `eqr_query(row1, row2, strict)` — Row Equality

* **Strict** (`s`) — element-wise equality including unobserved cells.
  The published L\*-SHA semantics; right for dense traces.
* **Weak** (`w`) — wildcard for unobserved cells; a row's `(None,
  None)` cells match anything. Right for sparse real-world traces
  where requiring full population would stall the L\* loop forever.

### 8.4 `ref_query(table)` — Refinement

Walks every row of the observation table and flags ambiguous words
(those whose row is consistent with multiple existing rows, or simply
under-observed below `n_min` samples). For each ambiguous word the
Teacher requests fresh traces:

* **In UPPAAL mode** each call to `self.TG.get_traces(n)` runs `n`
  fresh `verifyta` simulations.
* **In CSV mode** the trace generator yields the file list once and
  empties out, so the inner refinement loop self-terminates.

### 8.5 `get_counterexample(table)` — Counterexample Search

The third L\* oracle question. Iterates every prefix of every
observed trace; for any prefix not already in $S \cup S\!\cdot\!\Sigma$,
materialises its hypothetical row and checks two failure modes
against the current table:

* **Non-closedness** — the new row is not equivalent to any existing
  upper row.
* **Non-consistency** — there exists an event $a$ and a row $s_w$
  in $S$ such that row($s_w$) ≡ row(prefix) but row($s_w \cdot a$) ≢
  row(prefix$\cdot a$).

Returning the prefix forces the Learner to either add it to $S$ (fix
closedness) or extend the suffix set $E$ (fix consistency). In CSV
mode an extra heuristic checks for unconsumed event symbols and
seeds one more L\* iteration before exit.

---

## 9. The Observation Table — `dynamic_obstable.py`

`ObsTable` is a thin wrapper around four lists:

```
__S       : List[Trace]   # upper-row words (the agreed prefixes)
__low_S   : List[Trace]   # lower-row words (one-step extensions)
__E       : List[Trace]   # column words (distinguishing suffixes)
__upp_obs : List[Row]     # one Row per S, len(state) == len(E)
__low_obs : List[Row]     # one Row per low_S, same shape
```

Each `Row` carries a `state: List[State]` of `(model_id, distr_id)`
pairs. `is_populated()` returns `True` when at least one cell is
non-`(None, None)` — used by closedness checks to skip placeholder
rows. Equality is strict cell-for-cell; the Teacher's `eqr_query` is
where the strict-vs-weak distinction is implemented.

The `to_sha(teacher)` method is the table's most interesting
operation: it walks the upper rows in order, deduplicates them via
`eqr_query`, and emits one SHA `Location` per unique row. Edges are
then added by walking every `(s, e)` pair whose cell is observed and
finding a destination location for the resulting row. A handful of
defensive code paths handle (a) the 1-D vs 2-D variable layout
(ENERGY has one observable, HRI has three) and (b) the empty-table
edge case where an early run of L\* hasn't seen a single row yet.

---

## 10. End-to-End Pipeline — `tasks.py`

The Celery task `run_lsha_learning_task(case_study_id)` is the
orchestrator that ties everything together:

| Phase | Action |
|---|---|
| 0 | Load `CaseStudy` model from the DB |
| 1 | Normalise `RESAMPLE_STRATEGY` to `'CSV'` or `'UPPAAL'` |
| 2 | Parse the user's JSON; pre-resolve `variables[]` so `MAIN_VARIABLE` and `DRIVER_SIGNAL` are populated even when the DB-level fields were left blank |
| 3 | Construct the `CustomTraceGenerator` |
| 4 | (UPPAAL only) Run a one-shot `verifyta` to confirm the model patches and produces a non-empty trace; (CSV only) verify the user has uploaded at least one CSV |
| 5 | Sanity-check the SUL pipeline: parse one trace, find the first few change points, label the first event, fit one segment |
| 6 | Build the SUL adapter functions (`parse_adapter`, `is_chg_pt_adapter`, `label_event_adapter`, `get_physics_param_adapter`) — these wrap the dynamic SUL functions and translate between the dict-of-arrays internal format and the `SampledSignal` / `SignalPoint` types the L\* core expects |
| 7 | Construct the `SystemUnderLearning` |
| 8 | Construct the `CustomTeacher` |
| 9 | Construct the initial `ObsTable` (one row per event in $\Sigma$) |
| 10 | Run `Learner.run_lsha(filter_empty=True)` |
| 11 | Render the resulting SHA to PDF + DOT source via `graphviz` |
| 12 | Save PDF and DOT files back into the `CaseStudy` model fields and mark status `COMPLETED` |
| 13 | Generate the experimental report (event tables, distributions, runtime) via `lsha_report.save_data` |

The adapter functions in step 6 are worth a closer look:

* **`parse_adapter`** receives a path, calls `parse_data_dynamic`, and
  re-emits the resulting dict as a list of `CustomSignal` objects with
  the right labels. It also caches the dict on `args` under the key
  `'__current_trace_cache__'` so the other three adapters can look up
  values by index without re-parsing the file.
* **`is_chg_pt_adapter`** receives `(curr, prev)` value tuples from
  the L\* core, recovers the index by looking up `curr.t.to_secs()`
  in the cached time array, then delegates to `is_chg_pt_dynamic`.
* **`label_event_adapter`** does the analogous lookup for change-point
  timestamps.
* **`get_physics_param_adapter`** maps a list of `SignalPoint`s to
  start/end indices and delegates to `get_physics_param_dynamic`.
  It also rounds the returned float to four decimal places to dodge
  an L\* infinite-loop pathology where two segments produce metrics
  that differ only in floating-point noise.

---

## 11. Worked Example A — THERMO V1

This section traces a complete run of the engine on the THERMO V1
configuration so the abstractions of Sections 5–10 become concrete.

### 11.1 User input

The user uploads `thermostat-v1.xml` (an UPPAAL model), opens the
case-study form in the web UI, and fills in:

```json
{
  "case_study": "THERMO",
  "version": "V1",
  "variables": [
    { "name": "T", "source": "T_r", "type": "REAL", "role": "estimated" },
    { "name": "t.ON",   "type": "BOOL", "role": "driver" },
    { "name": "r.open", "type": "INT",  "role": "driver" }
  ],
  "models": [
    { "id": 0, "name": "Cooling Mode (Decay)",  "type": "EXP_DECAY"  },
    { "id": 1, "name": "Heating Mode (Growth)", "type": "EXP_GROWTH" }
  ],
  "events": [
    { "symbol": "h_0", "trigger_value": 0, "model_id": 1,
      "guard": "t.ON == 1" },
    { "symbol": "c_0", "trigger_value": 1, "model_id": 0,
      "guard": "t.ON == 0 and r.open == 0" },
    { "symbol": "c_1", "trigger_value": 2, "model_id": 0,
      "guard": "t.ON == 0 and r.open > 0" }
  ],
  "trace_generation": {
    "strategy": "UPPAAL",
    "xml_force_variable": "force_act",
    "max_length": 20
  }
}
```

Note the per-variable `source: "T_r"` for `T` — the friendly UI name
`T` maps to the physical UPPAAL section header `T_r`. The two driver
variables (`t.ON`, `r.open`) need no aliasing because their UPPAAL
section headers are identical.

### 11.2 Phase 1 — strategy normalisation

`tasks.py` reads `cs_instance.resample_strategy = "UPPAAL"` and
canonicalises to `RESAMPLE_STRATEGY = "UPPAAL"`, `IS_CSV_MODE = False`.

### 11.3 Phase 2 — variable resolution

`_resolve_variable_roles(sul_args)` populates:

```python
sul_args['main_var']   = 'T'
sul_args['driver']     = ['t.ON', 'r.open']
sul_args['alias_map']  = {'T': 'T_r'}
```

`_build_target_vars(sul_args)` returns:

```python
{ 'T_r': 'main', 't.ON': 't.ON', 'r.open': 'r.open' }
```

Note the asymmetry: `T` is mapped to `'main'` *via its physical name*
`T_r`. The drivers, having no alias, map identity-on-identity.

### 11.4 Phase 3 — trace generation

`CustomTraceGenerator(resample_strategy="UPPAAL", …)` patches
`thermostat-v1.xml` so `force_act[]` contains the trigger values
of the current word's events. For an empty initial word it writes:

```
int force_act[MAX_E] = {-1, -1, -1, …};
```

Then `verifyta -t0 thermostat-v1.xml query.q` is invoked, producing
a fresh trace file under `/results/upp_results/`.

### 11.5 Phase 5 — SUL sanity check

`parse_data_dynamic([trace_path], args=sul_args)` loads the trace and
returns:

```python
{
  'time'  : array([ 0.0,  1.0,  2.0, …, 200.0]),
  'main'  : array([15.20, 15.05, 14.90, …, 18.45]),
  't.ON'  : array([ 0,  0,  0,  0,  1,  1, …,  1]),
  'r.open': array([ 0,  0,  …,  1,  1,  …]),
}
```

`is_chg_pt_dynamic(signals, 4, args)` runs CHECK 1: at index 4 the
guard `t.ON == 1` becomes true (label `h_0`), at index 3 the guard
`t.ON == 0 and r.open == 0` was true (label `c_0`). Labels differ →
returns `True`. The first change point is index 4, time 4 s.

`label_event_dynamic(signals, 4, args)` returns `'h_0'`.

`get_physics_param_dynamic(signals, 4, 39, args)` extracts
`signals['main'][4:39]`, computes `mean ≈ 17.5`, looks up the active
flow (model_id 1, type EXP_GROWTH), and computes `rate ≈ 0.011` (mean
increment per step). The pair `{'mean': 17.5, 'rate': 0.011}` is
returned and then rounded to 4 dp by the adapter.

### 11.6 Phase 7-10 — L\* loop

The L\*-SHA learner starts with $S = \{\varepsilon\}$,
$\Sigma = \{h_0, c_0, c_1\}$, $E = \{\varepsilon\}$. Each iteration:

1. **Refinement** — Teacher asks for ~10 fresh traces, processes them,
   updates the per-cell `(model_id, distr_id)` States.
2. **Closedness check** — every lower row is compared against every
   upper row using `eqr_query` with `strict=True` (THERMO uses dense
   UPPAAL traces, so strict equality is correct).
3. **Consistency check** — for every pair of equivalent rows and every
   alphabet symbol, the extension rows are compared.
4. **Counterexample search** — `get_counterexample` walks every
   observed trace prefix.

When all three checks pass, `to_sha(teacher)` builds the SHA: one
location per unique row, one edge per observed (row, event)
transition, plus an `__init__` location with edges into every row that
is reachable in one step from $\varepsilon$.

### 11.7 Phase 11–13 — output

The graphviz renderer writes
`results/final_results/<name>_UPPAAL.gv` and `.pdf`; the report writer
emits a Markdown table of `(symbol, flow, distribution)` rows plus the
runtime. Both files are saved back into the `CaseStudy` model's
`final_result_pdf` / `final_result_txt` fields, and the status is
flipped to `COMPLETED`.

---

## 12. Worked Example B — ENERGY V1

ENERGY exercises the **CSV path** of the engine and the
*tolerance-free* change-point heuristic.

### 12.1 User input

The user uploads 11 CSV files of the W7 dataset to the
`CsvFile` table and submits:

```json
{
  "case_study": "ENERGY",
  "version": "V1",
  "variables": [
    { "name": "HEADSTOCK__SPINDLE_DRIVE___1___ENERGY", "type": "REAL", "role": "estimated" },
    { "name": "HEADSTOCK__SPINDLE_MOTOR___1___RPM",    "type": "REAL", "role": "driver" },
    { "name": "RT__PALLET_LOCKING___1___PRESSURE",     "type": "REAL", "role": "driver" }
  ],
  "models": [
    { "id": 0, "name": "Idle / Stopped", "type": "CONSTANT", "params": {"value": 0.0} },
    { "id": 1, "name": "Bin 1 Power",    "type": "CONSTANT" },
    { "id": 2, "name": "Bin 2 Power",    "type": "CONSTANT" },
    { "id": 3, "name": "Bin 3 Power",    "type": "CONSTANT" }
  ],
  "events": [
    { "symbol": "l",   "trigger_value": 0, "model_id": 0,
      "guard": "RT__PALLET_LOCKING___1___PRESSURE >= p_min" },
    { "symbol": "u",   "trigger_value": 1, "model_id": 0,
      "guard": "RT__PALLET_LOCKING___1___PRESSURE < p_min" },
    { "symbol": "i_0", "trigger_value": 2, "model_id": 0,
      "guard": "HEADSTOCK__SPINDLE_MOTOR___1___RPM < 100" },
    { "symbol": "m_1", "trigger_value": 3, "model_id": 1,
      "guard": "100 <= HEADSTOCK__SPINDLE_MOTOR___1___RPM < 1100" },
    { "symbol": "m_2", "trigger_value": 4, "model_id": 2,
      "guard": "1100 <= HEADSTOCK__SPINDLE_MOTOR___1___RPM < 2100" },
    { "symbol": "m_3", "trigger_value": 5, "model_id": 3,
      "guard": "2100 <= HEADSTOCK__SPINDLE_MOTOR___1___RPM <= 3100" }
  ],
  "constants": { "p_min": 50.0, "s_min": 100, "s_max": 3100 },
  "trace_generation": {
    "strategy": "CSV",
    "csv": { "timestamp_column": "TimestampUTC", "wide_or_long": "wide", "max_length": 100 }
  }
}
```

### 12.2 Phase 1 — strategy

`RESAMPLE_STRATEGY = "CSV"`, `IS_CSV_MODE = True`. The CSV branch of
Phase 1 in tasks.py queries `CsvFile.objects.filter(...)`, finds 11
rows, and stuffs the absolute paths into `trace_gen_config['csv_files']`.

### 12.3 Phase 5 — CSV parsing

`_parse_csv` is invoked with all 11 files:

* Step 0 — no `boolean_files` / `real_files` to splice.
* Step 1 — concat all 11 files; result is a single ~50 000-row
  DataFrame.
* Step 1b — no `filter_object` — skip.
* Step 1c — bulk boolean replace; ENERGY has no booleans so this is a
  no-op.
* Step 2 — `wide_or_long="wide"` → `csv_format="horizontal"` → no pivot.
* Step 3 — no `replace_values` — skip.
* Step 4 — `time_column="TimestampUTC"`, `time_format="datetime"`
  (the default). Sort by time; convert to seconds.
* Step 5 — linear interpolation fills the NaN cells where the energy
  counter was silent.
* Step 6 — no `round_columns` — RPM stays at its native ~5-RPM
  resolution. (CHECK 1 will catch bin transitions; intra-bin jitter is
  ignored thanks to the no-tolerance fall-through.)
* Step 7 — build the signals dict with three keys plus `'time'`.

### 12.4 Phase 7-10 — L\* loop

The single CSV bundle is processed exactly once (CSV yield-once
contract). The Teacher's first `ref_query` triggers another call to
`get_traces`, which returns `[]`. Refinement therefore exits
immediately and the Learner proceeds to closedness/consistency checks
on the data already gathered.

ENERGY uses the **MEAN/CONSTANT** flow type, so `get_physics_param`
returns `rate=0` and `mean=<segment mean power>`. The deterministic HT
query bins the means and assigns one distribution per RPM bin, giving
the four flows (`Idle`, `Bin 1`, `Bin 2`, `Bin 3`) declared in the
JSON. The discrete events (`l`, `u`, `i_0`, `m_1..m_3`) populate the
columns of the observation table; weak equality is appropriate because
the trace bundle covers some prefixes far more than others.

### 12.5 Output

The resulting SHA has six modes (one per distinct row in the table)
connected by the six event symbols, plus an `__init__` location whose
out-edges encode the typical power-on sequence
`u → l → i_0 → m_1 → m_2 → m_3 → m_2 → … → u`.

---

## 13. Worked Example C — GREEN V1

GREEN exercises the **long-format CSV pivot** path.

### 13.1 User input

The user uploads at least one boolean CSV (sample
`20250514_DecanterData_JanToMay_BOOL.csv`) plus three real-valued
CSVs (pump speed, temperature, …) and submits:

```json
{
  "case_study": "GREEN",
  "version": "V1",
  "variables": [
    { "name": "Assorbimento",   "type": "REAL", "role": "estimated" },
    { "name": "MarciaDecanter", "type": "BOOL", "role": "driver" },
    { "name": "Coppia",         "type": "REAL", "role": "driver" }
  ],
  "models": [
    { "id": 0, "name": "Constant Assorbimento", "type": "CONSTANT" },
    { "id": 1, "name": "Linear Assorbimento",   "type": "LINEAR"   }
  ],
  "events": [
    { "symbol": "decanter_ON",  "trigger_value": 0, "model_id": 1,
      "guard": "MarciaDecanter == 1 and prev.MarciaDecanter == 0" },
    { "symbol": "decanter_OFF", "trigger_value": 1, "model_id": 0,
      "guard": "MarciaDecanter == 0 and prev.MarciaDecanter == 1" },
    { "symbol": "torque_0_20",  "trigger_value": 2, "model_id": 0,
      "guard": "MarciaDecanter == 1 and 0 <= Coppia < 20" },
    { "symbol": "torque_20_40", "trigger_value": 3, "model_id": 0,
      "guard": "MarciaDecanter == 1 and 20 <= Coppia < 40" },
    { "symbol": "torque_40_60", "trigger_value": 4, "model_id": 0,
      "guard": "MarciaDecanter == 1 and 40 <= Coppia < 60" },
    { "symbol": "torque_60_80", "trigger_value": 5, "model_id": 0,
      "guard": "MarciaDecanter == 1 and 60 <= Coppia < 80" }
  ],
  "trace_generation": {
    "strategy": "CSV",
    "csv": {
      "boolean_files": [ "20250514_DecanterData_JanToMay_BOOL.csv" ],
      "real_files":    [ "20250514_DecanterData_JanToMay_REAL.csv" ],
      "timestamp_column": "TimeStamp",
      "wide_or_long": "long",
      "object_name_column": "DataObjectName",
      "field_column": "DataObjectField",
      "value_column": "Value",
      "filter_object": "DecanterMB",
      "max_length": 50
    }
  }
}
```

### 13.2 Long-format pivot

`_parse_csv` walks through:

* Step 0 — splice both `boolean_files` and `real_files` into the
  CSV list (the trace generator already merged them once; this is the
  belt-and-braces).
* Step 1 — concat all the long-format rows into one DataFrame.
* Step 1b — apply `DataObjectName == "DecanterMB"` filter; rows with
  other object names (PumpController, TemperatureSensor, …) are
  dropped. If the filter removes *every* row, a loud warning is
  printed and the filter is undone — the user has likely mistyped the
  object name.
* Step 1c — coerce `True`/`False` literals (and their string variants)
  to `1`/`0`. Without this, the next step would crash because
  `pivot_table(aggfunc="mean")` cannot mean booleans.
* Step 2 — pivot: `index=TimeStamp`, `columns=DataObjectField`,
  `values=Value`, `aggfunc=mean`. The result is a wide DataFrame with
  one column per field (`MarciaDecanter`, `CicloAttivo`, …).
* Steps 3–7 — proceed as for the wide-format ENERGY case.

### 13.3 Edge-detection guards

The two `decanter_ON` / `decanter_OFF` guards rely on
`prev.MarciaDecanter`. The dynamic SUL rewrites this to the internal
`prev_MarciaDecanter` lookup before evaluation, so the guard becomes:

```python
MarciaDecanter == 1 and prev_MarciaDecanter == 0
```

which evaluates to `True` exactly on the rising edge — i.e. the first
sample at which the decanter switches on. Without this, the SHA
would have one `decanter_ON` event per sample of every "on" segment.

### 13.4 LINEAR flow

The `decanter_ON` event activates the `LINEAR` flow (model id 1). For
each on-segment the Teacher's HT query computes the segment's signed
slope (mean per-step increment) and either matches it against an
existing distribution or allocates a new one. Over enough segments
the distribution naturally clusters by torque bin.

---

## 14. Limitations and Future Work

The current dynamic engine still has rough edges:

1. **Missing column → silent zeros.** When the user's `aliases` map
   doesn't match the actual UPPAAL section header (or the CSV column
   doesn't exist), the engine fills the signal with zeros and prints a
   warning. The L\* loop then learns a degenerate SHA whose only
   discrete event is the catch-all default. A stricter mode that
   *fails fast* on missing columns would be valuable for production
   use.
2. **Single estimated variable.** The internal `'main'` key admits
   exactly one `estimated` signal per case study. Multi-output SHAs
   (e.g. a thermostat that learns both temperature and humidity flows
   in parallel) would require generalising the SUL to a list of
   primary observables. The L\*-SHA core already supports it; only
   the dynamic SUL adapter would need work.
3. **Flow type catalogue is fixed.** `MEAN`, `CONSTANT`, `LINEAR`,
   `EXP_DECAY`, `EXP_GROWTH`, `LINEAR_DECAY`, `LINEAR_GROWTH` cover
   the four case studies in this thesis but exclude e.g. logistic or
   sinusoidal flows. Adding new types is a 20-line patch in
   `_compute_rate` plus a one-line entry in `tasks.py`'s
   `make_ideal_flow` factory; turning the catalogue into a plugin
   registry would be tidier.
4. **No auto-discovery of variables.** The user still has to declare
   every signal in `variables[]`. A future version could scan the
   first row of an uploaded CSV and propose roles automatically.
5. **Driver list mutation in `parse_adapter`.** The legacy adapter in
   `tasks.py` rewrites `args['driver']` from a list to its first
   element on the first call (so a single string is forwarded
   downstream). This is benign because change-point CHECK 1 (label
   change) catches transitions for every driver, but it is fragile and
   should be replaced with a proper "primary driver" abstraction.
6. **CSV refinement is single-shot.** The yield-once short-circuit
   trades L\* completeness for termination; in principle one could
   subdivide a long CSV into windows and serve them incrementally,
   exposing the Learner to fresh evidence over many refinements.
7. **No formal proof of correctness.** We rely on the empirical
   observation that the four case studies learn the same SHAs as the
   legacy code, but we have not proven the dynamic engine is
   semantically equivalent to the per-case-study modules. A bisimulation-
   style argument would strengthen the result.
8. **Sandbox boundary.** `safe_eval` removes `__builtins__` but still
   permits Python operator overloading and arbitrary attribute lookup
   on numbers. A truly hostile JSON could not exfiltrate data but
   could in principle induce a denial-of-service via a pathological
   guard expression (e.g. `(2 ** 2) ** 2 ** 30`). For a public-facing
   deployment a dedicated tiny-expression parser would be safer.

---

## Appendix A — Full JSON Schemas

Below are the four complete, copy-pasteable JSON files used in the
case studies that ship with the engine.

### A.1 THERMO V1 (UPPAAL)

```json
{
  "case_study": "THERMO",
  "version": "V1",
  "variables": [
    { "name": "T",      "source": "T_r", "type": "REAL", "role": "estimated" },
    { "name": "t.ON",   "type": "BOOL", "role": "driver" },
    { "name": "r.open", "type": "INT",  "role": "driver" }
  ],
  "models": [
    { "id": 0, "name": "Cooling Mode (Decay)",  "type": "EXP_DECAY"  },
    { "id": 1, "name": "Heating Mode (Growth)", "type": "EXP_GROWTH" }
  ],
  "events": [
    { "symbol": "h_0", "trigger_value": 0, "model_id": 1,
      "guard": "t.ON == 1" },
    { "symbol": "c_0", "trigger_value": 1, "model_id": 0,
      "guard": "t.ON == 0 and r.open == 0" },
    { "symbol": "c_1", "trigger_value": 2, "model_id": 0,
      "guard": "t.ON == 0 and r.open > 0" }
  ],
  "trace_generation": {
    "strategy": "UPPAAL",
    "xml_force_variable": "force_act",
    "max_length": 20
  }
}
```

### A.2 HRI V1 (UPPAAL)

```json
{
  "case_study": "HRI",
  "version": "V1",
  "variables": [
    { "name": "F",          "type": "REAL", "role": "estimated" },
    { "name": "h.busy",     "type": "BOOL", "role": "driver" },
    { "name": "h.location", "type": "INT",  "role": "driver" }
  ],
  "aliases": {
    "F":          "humanFatigue[currH - 1]",
    "h.busy":     "amy.busy || amy.p_2",
    "h.location": "humanPositionX[currH - 1]"
  },
  "models": [
    { "id": 0, "name": "Idle Recovery (Decay)", "type": "EXP_DECAY"  },
    { "id": 1, "name": "Busy Fatigue (Growth)", "type": "EXP_GROWTH" }
  ],
  "events": [
    { "symbol": "h_start",     "trigger_value": 0, "model_id": 1,
      "guard": "h.busy == 1" },
    { "symbol": "h_stop_idle", "trigger_value": 1, "model_id": 0,
      "guard": "h.busy == 0 and h.location != 1" },
    { "symbol": "h_stop_sit",  "trigger_value": 2, "model_id": 0,
      "guard": "h.busy == 0 and h.location == 1" },
    { "symbol": "h_pass_out",  "trigger_value": 3, "model_id": 0,
      "guard": "F >= 1.0" }
  ],
  "trace_generation": {
    "strategy": "UPPAAL",
    "xml_force_variable": "force_open",
    "max_length": 30
  }
}
```

### A.3 ENERGY V1 (CSV wide)

```json
{
  "case_study": "ENERGY",
  "version": "V1",
  "variables": [
    { "name": "HEADSTOCK__SPINDLE_DRIVE___1___ENERGY", "type": "REAL", "role": "estimated" },
    { "name": "HEADSTOCK__SPINDLE_MOTOR___1___RPM",    "type": "REAL", "role": "driver" },
    { "name": "RT__PALLET_LOCKING___1___PRESSURE",     "type": "REAL", "role": "driver" }
  ],
  "models": [
    { "id": 0, "name": "Idle / Stopped", "type": "CONSTANT", "params": {"value": 0.0} },
    { "id": 1, "name": "Bin 1 Power",    "type": "CONSTANT" },
    { "id": 2, "name": "Bin 2 Power",    "type": "CONSTANT" },
    { "id": 3, "name": "Bin 3 Power",    "type": "CONSTANT" }
  ],
  "events": [
    { "symbol": "l",   "trigger_value": 0, "model_id": 0,
      "guard": "RT__PALLET_LOCKING___1___PRESSURE >= p_min" },
    { "symbol": "u",   "trigger_value": 1, "model_id": 0,
      "guard": "RT__PALLET_LOCKING___1___PRESSURE < p_min" },
    { "symbol": "i_0", "trigger_value": 2, "model_id": 0,
      "guard": "HEADSTOCK__SPINDLE_MOTOR___1___RPM < 100" },
    { "symbol": "m_1", "trigger_value": 3, "model_id": 1,
      "guard": "100 <= HEADSTOCK__SPINDLE_MOTOR___1___RPM < 1100" },
    { "symbol": "m_2", "trigger_value": 4, "model_id": 2,
      "guard": "1100 <= HEADSTOCK__SPINDLE_MOTOR___1___RPM < 2100" },
    { "symbol": "m_3", "trigger_value": 5, "model_id": 3,
      "guard": "2100 <= HEADSTOCK__SPINDLE_MOTOR___1___RPM <= 3100" }
  ],
  "constants": { "p_min": 50.0, "s_min": 100, "s_max": 3100 },
  "trace_generation": {
    "strategy": "CSV",
    "csv": {
      "timestamp_column": "TimestampUTC",
      "wide_or_long": "wide",
      "max_length": 100
    }
  }
}
```

### A.4 GREEN V1 (CSV long)

```json
{
  "case_study": "GREEN",
  "version": "V1",
  "variables": [
    { "name": "Assorbimento",   "type": "REAL", "role": "estimated" },
    { "name": "MarciaDecanter", "type": "BOOL", "role": "driver" },
    { "name": "Coppia",         "type": "REAL", "role": "driver" }
  ],
  "models": [
    { "id": 0, "name": "Constant Assorbimento", "type": "CONSTANT" },
    { "id": 1, "name": "Linear Assorbimento",   "type": "LINEAR"   }
  ],
  "events": [
    { "symbol": "decanter_ON",  "trigger_value": 0, "model_id": 1,
      "guard": "MarciaDecanter == 1 and prev.MarciaDecanter == 0" },
    { "symbol": "decanter_OFF", "trigger_value": 1, "model_id": 0,
      "guard": "MarciaDecanter == 0 and prev.MarciaDecanter == 1" },
    { "symbol": "torque_0_20",  "trigger_value": 2, "model_id": 0,
      "guard": "MarciaDecanter == 1 and 0 <= Coppia < 20" },
    { "symbol": "torque_20_40", "trigger_value": 3, "model_id": 0,
      "guard": "MarciaDecanter == 1 and 20 <= Coppia < 40" },
    { "symbol": "torque_40_60", "trigger_value": 4, "model_id": 0,
      "guard": "MarciaDecanter == 1 and 40 <= Coppia < 60" },
    { "symbol": "torque_60_80", "trigger_value": 5, "model_id": 0,
      "guard": "MarciaDecanter == 1 and 60 <= Coppia < 80" }
  ],
  "trace_generation": {
    "strategy": "CSV",
    "csv": {
      "boolean_files": [ "20250514_DecanterData_JanToMay_BOOL.csv" ],
      "real_files":    [ "20250514_DecanterData_JanToMay_REAL.csv",
                         "20250514_PumpSpeedData_JanToMay_REAL.csv",
                         "20250514_TemperatureData_JanToMay_REAL.csv" ],
      "timestamp_column": "TimeStamp",
      "wide_or_long": "long",
      "object_name_column": "DataObjectName",
      "field_column": "DataObjectField",
      "value_column": "Value",
      "filter_object": "DecanterMB",
      "max_length": 50
    }
  }
}
```

---

## Appendix B — File-by-File Reference

A short tour of the eight files that make up the dynamic engine.

| File | Lines | Purpose |
|---|---:|---|
| [`core_algorithm/dynamic_sul.py`](core_algorithm/dynamic_sul.py) | ~830 | Universal SUL: parse_data_dynamic, is_chg_pt_dynamic, label_event_dynamic, get_physics_param_dynamic |
| [`core_algorithm/dynamic_tracegenerator.py`](core_algorithm/dynamic_tracegenerator.py) | ~270 | UPPAAL model patcher + CSV yield-once short-circuit |
| [`core_algorithm/teacher.py`](core_algorithm/teacher.py) | ~620 | L\*-SHA Teacher: mi_query, ht_query (D and S), eqr_query, ref_query, get_counterexample |
| [`core_algorithm/dynamic_obstable.py`](core_algorithm/dynamic_obstable.py) | ~310 | Observation table data structure + to_sha conversion |
| [`core_algorithm/tasks.py`](core_algorithm/tasks.py) | ~480 | Celery orchestration: DB → JSON → SUL adapters → Learner → graphviz |
| [`core_algorithm/lsha/sha_learning/`](core_algorithm/lsha/sha_learning/) | (vendored) | The published L\*-SHA core (Pradhan et al. 2024) — Learner, FlowCondition, ProbDistribution, etc. |
| [`rest_api/models.py`](rest_api/models.py) | ~150 | Django CaseStudy + CsvFile models that store user input |
| [`rest_api/views.py`](rest_api/views.py) | ~250 | REST API endpoints invoked by the React frontend |

### B.1 Where each JSON field is consumed

| JSON field | Consumed by |
|---|---|
| `case_study`, `version` | `tasks.py` (filenames, reports) |
| `variables[].name`, `.role` | `dynamic_sul._resolve_variable_roles`, `_build_target_vars` |
| `variables[].source` | `dynamic_sul._resolve_variable_roles` (alias_map) |
| `aliases` (top-level) | `dynamic_sul._resolve_variable_roles` |
| `models[]` | `dynamic_sul._get_model_type_for_segment`, `tasks.make_ideal_flow` |
| `events[].symbol`, `.guard` | `dynamic_sul.label_event_dynamic` |
| `events[].trigger_value` | `dynamic_tracegenerator.build_event_strings` (UPPAAL only) |
| `events[].model_id` | `dynamic_sul._get_model_type_for_segment` |
| `constants` | `dynamic_sul.label_event_dynamic` (guard context) |
| `trace_generation.strategy` | `tasks.py`, `dynamic_tracegenerator.__init__`, `dynamic_sul.parse_data_dynamic` |
| `trace_generation.xml_force_variable[s]` | `dynamic_tracegenerator.build_event_strings` |
| `trace_generation.xml_action_variable` | `dynamic_tracegenerator.fix_model` |
| `trace_generation.max_length` | `dynamic_tracegenerator.__init__` (force_act padding) |
| `trace_generation.csv.timestamp_column` | `dynamic_sul._parse_csv` (alias of `time_column`) |
| `trace_generation.csv.wide_or_long` | `dynamic_sul._parse_csv` (alias of `csv_format`) |
| `trace_generation.csv.field_column` | `dynamic_sul._parse_csv` (alias of `key_column`) |
| `trace_generation.csv.object_name_column` | `dynamic_sul._parse_csv` (long-format filter target) |
| `trace_generation.csv.filter_object` | `dynamic_sul._parse_csv` (long-format row filter) |
| `trace_generation.csv.boolean_files`, `.real_files` | `dynamic_sul._parse_csv` (Step 0 splice) and `dynamic_tracegenerator.__init__` |
| `trace_generation.csv_format`, `time_column`, etc. | `dynamic_sul._parse_csv` (legacy flat keys) |
| `trace_generation.replace_values` | `dynamic_sul._parse_csv` (Step 3) |
| `trace_generation.interpolate_method` | `dynamic_sul._parse_csv` (Step 5) |
| `trace_generation.round_columns` | `dynamic_sul._parse_csv` (Step 6) |
| `trace_generation.physics_constants` | `dynamic_sul.label_event_dynamic`, `is_chg_pt_dynamic` |

### B.2 The four "dynamic_*" entry points at a glance

```text
parse_data_dynamic(file_paths, args)
    └── _build_target_vars(args)
    └── _parse_csv(...)        ─── CSV / SIM strategies
    └── _parse_uppaal(...)     ─── UPPAAL strategy
    return signals: dict[str, np.ndarray]

is_chg_pt_dynamic(signals, index, args)
    ├── CHECK 1: label_event_dynamic(curr) != label_event_dynamic(prev)
    └── CHECK 2: any driver step > tolerance
    return bool

label_event_dynamic(signals, index, args)
    └── safe_eval(guard, context)   for each event in order
    return symbol of first matching event

get_physics_param_dynamic(signals, start, end, args)
    └── _get_model_type_for_segment(signals, start, args)
    └── _compute_rate(segment, type)
    return {'mean': float, 'rate': float}
```

---

## References

* **Angluin (1987).** *Learning regular sets from queries and
  counterexamples.* Information and Computation 75(2), 87–106.
* **Henzinger (1996).** *The theory of hybrid automata.*
  Proceedings, 11th Annual IEEE Symposium on Logic in Computer
  Science.
* **Konz (1998).** *Work design — industrial ergonomics.*
  Publishing Horizons.
* **Ma, Chablat, Bennis & Zhang (2009).** *A new simple dynamic
  muscle fatigue model and its validation.* International Journal of
  Industrial Ergonomics 39(1), 211–220.
* **Quintáns (2014).** *Petri nets in industrial process control:
  modelling and analysis.* Springer.
* **Vicentini, Pedrocchi, Beschi, Giussani, Iannacci, Magnoni,
  Pellegrinelli, Roveda, Villagrossi, Askarpour, Maurtua, Tellaeche,
  Becchi, Stellin & Fogliazza (2020).** *PIROS: Cooperative,
  safe and reconfigurable robotic companion for CNC pallets load /
  unload stations.* Bringing Innovative Robotic Technologies from
  Research Labs to Industrial End-users 23(4), 57–96.
* **Pradhan, Bartocci, Vinarskii (2024).** *Active Learning of
  Stochastic Hybrid Automata.* (L\*-SHA reference paper that the
  vendored `core_algorithm/lsha/` implements.)
* **Pradhan, Vinarskii, Bartocci (2023).** *Hybrid Automata
  Learning for Energy-Efficiency Monitoring of Industrial Plants.*

---

*End of document.*
