"""
generate_report.py
Run with:  /home/rghaf/miniforge3/envs/lsha-web/bin/python generate_report.py
Produces:  lsha_dynamic_explanation.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

OUTPUT = "lsha_dynamic_explanation.pdf"

# ── Colour palette ───────────────────────────────────────────────────────────
C_NAVY   = colors.HexColor("#1a2a4a")
C_BLUE   = colors.HexColor("#2563eb")
C_LBLUE  = colors.HexColor("#dbeafe")
C_GREEN  = colors.HexColor("#166534")
C_LGREEN = colors.HexColor("#dcfce7")
C_AMBER  = colors.HexColor("#92400e")
C_LAMBER = colors.HexColor("#fef3c7")
C_RED    = colors.HexColor("#991b1b")
C_LRED   = colors.HexColor("#fee2e2")
C_GRAY   = colors.HexColor("#6b7280")
C_LGRAY  = colors.HexColor("#f3f4f6")
C_WHITE  = colors.white
C_BLACK  = colors.black

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)

styles = getSampleStyleSheet()
W = A4[0] - 4*cm   # usable text width

# ── Custom styles ─────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

sTitle = S("sTitle", fontSize=26, textColor=C_NAVY, spaceAfter=6,
           fontName="Helvetica-Bold", alignment=TA_CENTER)
sSubtitle = S("sSubtitle", fontSize=13, textColor=C_GRAY, spaceAfter=20,
              fontName="Helvetica", alignment=TA_CENTER)
sH1 = S("sH1", fontSize=16, textColor=C_WHITE, spaceBefore=18, spaceAfter=4,
        fontName="Helvetica-Bold", backColor=C_NAVY,
        leftIndent=-6, rightIndent=-6, borderPad=6)
sH2 = S("sH2", fontSize=13, textColor=C_NAVY, spaceBefore=14, spaceAfter=4,
        fontName="Helvetica-Bold", borderPad=3,
        borderWidth=0, leftIndent=0)
sH3 = S("sH3", fontSize=11, textColor=C_BLUE, spaceBefore=10, spaceAfter=3,
        fontName="Helvetica-Bold")
sBody = S("sBody", fontSize=10, textColor=C_BLACK, spaceAfter=6,
          fontName="Helvetica", leading=15, alignment=TA_JUSTIFY)
sCode = S("sCode", fontSize=8.5, textColor=C_BLACK, spaceAfter=4,
          fontName="Courier", leading=13, backColor=C_LGRAY,
          leftIndent=8, rightIndent=8, borderPad=5)
sCodeGreen = S("sCodeGreen", fontSize=8.5, textColor=C_BLACK, spaceAfter=4,
               fontName="Courier", leading=13, backColor=C_LGREEN,
               leftIndent=8, rightIndent=8, borderPad=5)
sCodeRed = S("sCodeRed", fontSize=8.5, textColor=C_BLACK, spaceAfter=4,
             fontName="Courier", leading=13, backColor=C_LRED,
             leftIndent=8, rightIndent=8, borderPad=5)
sNote = S("sNote", fontSize=9.5, textColor=C_AMBER, spaceAfter=6,
          fontName="Helvetica-Oblique", backColor=C_LAMBER,
          leftIndent=8, rightIndent=8, borderPad=5, leading=14)
sInfo = S("sInfo", fontSize=9.5, textColor=C_GREEN, spaceAfter=6,
          fontName="Helvetica", backColor=C_LGREEN,
          leftIndent=8, rightIndent=8, borderPad=5, leading=14)
sBullet = S("sBullet", fontSize=10, textColor=C_BLACK, spaceAfter=3,
            fontName="Helvetica", leading=14, leftIndent=16,
            bulletIndent=6)
sCaption = S("sCaption", fontSize=8.5, textColor=C_GRAY, spaceAfter=6,
             fontName="Helvetica-Oblique", alignment=TA_CENTER)

def hr(): return HRFlowable(width="100%", thickness=1, color=C_LBLUE, spaceAfter=6)
def sp(h=6): return Spacer(1, h)

def h1(txt): return Paragraph(f"  {txt}", sH1)
def h2(txt): return Paragraph(txt, sH2)
def h3(txt): return Paragraph(txt, sH3)
def p(txt):  return Paragraph(txt, sBody)
def code(txt, style=sCode):
    safe = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(safe, style)
def note(txt): return Paragraph(f"⚠  {txt}", sNote)
def info(txt): return Paragraph(f"✔  {txt}", sInfo)
def bullet(txt): return Paragraph(f"• {txt}", sBullet)

def compare_table(old_lines, new_lines, caption=""):
    """Two-column old/new comparison table."""
    old_cell = "\n".join(old_lines)
    new_cell = "\n".join(new_lines)
    data = [
        [Paragraph("<b>OLD  (hard-coded)</b>", S("th", fontSize=9, fontName="Helvetica-Bold",
                   textColor=C_WHITE, backColor=C_RED)),
         Paragraph("<b>NEW  (dynamic)</b>", S("th2", fontSize=9, fontName="Helvetica-Bold",
                   textColor=C_WHITE, backColor=C_GREEN))],
        [Paragraph(f'<font face="Courier" size="8">{old_cell.replace(chr(38),"&amp;").replace("<","&lt;")}</font>',
                   S("tc", fontSize=8, fontName="Courier", leading=12)),
         Paragraph(f'<font face="Courier" size="8">{new_cell.replace(chr(38),"&amp;").replace("<","&lt;")}</font>',
                   S("tc2", fontSize=8, fontName="Courier", leading=12))]
    ]
    col = (W - 0.4*cm) / 2
    tbl = Table(data, colWidths=[col, col], spaceBefore=4, spaceAfter=4)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,0), C_RED),
        ("BACKGROUND", (1,0), (1,0), C_GREEN),
        ("BACKGROUND", (0,1), (0,1), colors.HexColor("#fff5f5")),
        ("BACKGROUND", (1,1), (1,1), colors.HexColor("#f0fff4")),
        ("GRID",        (0,0), (-1,-1), 0.5, C_GRAY),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    items = [tbl]
    if caption:
        items.append(Paragraph(caption, sCaption))
    return items

def flow_table(rows, col_widths=None, header=True):
    """Styled data table."""
    col_widths = col_widths or [W/len(rows[0])]*len(rows[0])
    tbl = Table(rows, colWidths=col_widths, spaceBefore=4, spaceAfter=6)
    style = [
        ("GRID",         (0,0), (-1,-1), 0.5, C_GRAY),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
        ("FONTNAME",     (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
    ]
    if header:
        style += [
            ("BACKGROUND", (0,0), (-1,0), C_NAVY),
            ("TEXTCOLOR",  (0,0), (-1,0), C_WHITE),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ]
    tbl.setStyle(TableStyle(style))
    return tbl

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
story = []

# ── Cover ─────────────────────────────────────────────────────────────────────
story += [
    sp(60),
    Paragraph("LSHA Web Platform", sSubtitle),
    Paragraph("Making the Learning Engine Dynamic", sTitle),
    sp(8),
    hr(),
    sp(8),
    Paragraph(
        "A deep-dive technical document explaining how three core components — "
        "the SUL Functions, the Observation Table, and the Teacher — were "
        "transformed from hard-coded, case-study-specific scripts into a single "
        "fully-generic engine driven entirely by user input from the Django web UI.",
        sBody
    ),
    sp(12),
    flow_table([
        ["Component", "Old File", "New File"],
        ["SUL Functions", "case_studies/&lt;cs&gt;/sul_functions.py", "core_algorithm/dynamic_sul.py"],
        ["Observation Table", "lsha/.../obstable.py", "core_algorithm/dynamic_obstable.py"],
        ["Teacher", "lsha/.../teacher.py", "core_algorithm/teacher.py (CustomTeacher)"],
    ], col_widths=[4*cm, 7*cm, 7*cm]),
    PageBreak(),
]

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SUL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
story += [
    h1("PART 1 — Dynamic SUL Functions"),
    sp(4),
    p(
        "The <b>System Under Learning (SUL)</b> is the software layer that sits between the raw "
        "trace data and the LSHA learning algorithm. It answers three questions: "
        "<i>Where are the change points in a trace?</i>  "
        "<i>What event label applies at each point?</i>  "
        "<i>What physics parameter characterises each segment?</i>"
    ),
    p(
        "In the original project these three questions were answered by a separate "
        "<b>sul_functions.py</b> file inside each case-study folder "
        "(thermostat, HRI, energy_sim, gr3n). Each file contained hard-coded variable "
        "names, threshold constants, and physics formulas specific to that experiment. "
        "Adding a new case study required writing a brand-new file from scratch."
    ),
    sp(4),
]

story += [
    h2("1.1  What the old sul_functions.py files looked like"),
    p(
        "Here is a concrete example. The <b>energy_sim</b> case study hard-codes three "
        "constants from config.ini and uses them directly in every function:"
    ),
    code(
        "# energy_sim/sul_functions.py  (OLD)\n"
        "SPEED_RANGE = int(config['ENERGY CS']['SPEED_RANGE'])   # = 200\n"
        "MIN_SPEED   = int(config['ENERGY CS']['MIN_SPEED'])     # = 100\n"
        "MAX_SPEED   = int(config['ENERGY CS']['MAX_SPEED'])     # = 10000\n"
        "\n"
        "def is_chg_pt(curr, prev):\n"
        "    # 'curr' is a 2-tuple: (speed_value, pressure_value)\n"
        "    return abs(curr[0] - prev[0]) > SPEED_RANGE or curr[1] != prev[1]\n"
        "\n"
        "def label_event(events, signals, t):\n"
        "    speed_sig    = signals[1]   # hard-coded index!\n"
        "    pressure_sig = signals[2]   # hard-coded index!\n"
        "    # ... builds intervals from MIN_SPEED / SPEED_RANGE / MAX_SPEED ...\n"
        "    for i, interval in enumerate(SPEED_INTERVALS):\n"
        "        if interval[0] <= curr_speed < interval[1]:\n"
        "            identified_event = events[i]   # picks from hard-coded list\n"
        "\n"
        "def parse_data(path):\n"
        "    # reads CSV columns by FIXED position: row[2]=time, row[3]=speed, row[4]=power\n"
        "    ts       = parse_ts(row[2])\n"
        "    speed_v  = round(float(row[3]) / 100) * 100\n"
        "    power_v  = float(row[4])\n"
    ),
    p(
        "The <b>thermostat</b> sul_functions.py is equally rigid — it hard-codes signal "
        "indices (signals[0]=heater, signals[1]=temperature, signals[2]=window), reads "
        "ON_R=100.0 as a literal constant, and branches on CS_VERSION 1-10 with explicit "
        "if/else chains."
    ),
    p(
        "If a user wanted to try a machine with different speed thresholds, or rename a "
        "column, or add a third signal, they had to edit Python source files. That is exactly "
        "what the dynamic approach eliminates."
    ),
    sp(6),
]

story += [
    h2("1.2  What the user provides instead"),
    p(
        "Instead of editing code, the user fills in a web form. The key fields are:"
    ),
    flow_table([
        ["Django field", "Example value", "What it controls"],
        ["resample_strategy", '"UPPAAL" or "SIM"', "Which data source to use"],
        ["main_variable", '"T_r"', "Column/signal that is the main observable"],
        ["driver_signal", '"t.ON"', "Column that causes discrete state changes"],
        ["context_variables", '["r.open"]', "Extra columns available to guards"],
        ["user_json", "{models, events, trace_generation}", "Full physics + event definition"],
    ], col_widths=[4.5*cm, 4.5*cm, 8*cm]),
    p("The <b>user_json</b> field is a JSON text area. For the thermostat V1 it looks like:"),
    code(
        '{\n'
        '  "models": [\n'
        '    { "id": 0, "name": "Cooling", "type": "EXP_DECAY"  },\n'
        '    { "id": 1, "name": "Heating", "type": "EXP_GROWTH" }\n'
        '  ],\n'
        '  "events": [\n'
        '    { "symbol": "h_0", "guard": "t.ON == 1.0", "model_id": 1, "trigger_value": 0 },\n'
        '    { "symbol": "c_0", "guard": "t.ON == 0.0", "model_id": 0, "trigger_value": 0 }\n'
        '  ],\n'
        '  "trace_generation": {\n'
        '    "xml_force_variable":  "force_open",\n'
        '    "xml_action_variable": "force_exe",\n'
        '    "max_length": 15\n'
        '  }\n'
        '}'
    ),
    sp(4),
]

story += [
    h2("1.3  How dynamic_sul.py is structured"),
    p(
        "The new file has one public entry point for each of the three SUL questions. "
        "Everything else is a private helper:"
    ),
    flow_table([
        ["Function", "Question it answers"],
        ["parse_data_dynamic(file_paths, args)", "Load trace file(s) → signals dictionary"],
        ["is_chg_pt_dynamic(signals, index, args)", "Is index a change point?"],
        ["label_event_dynamic(signals, index, args)", "Which event symbol is active at index?"],
        ["get_physics_param_dynamic(signals, s, e, args)", "What rate/mean characterises segment [s..e]?"],
    ], col_widths=[8*cm, 9*cm]),
    p("The <b>args</b> dictionary passed to every function is assembled in tasks.py from the Django model:"),
    code(
        "sul_args = {\n"
        "    'resample_strategy': cs_instance.resample_strategy,\n"
        "    'main_var':          cs_instance.main_variable,\n"
        "    'driver':            cs_instance.driver_signal,\n"
        "    'context_variables': cs_instance.context_variables,\n"
        "    'events':            data_dict['events'],     # from user_json\n"
        "    'models':            data_dict['models'],     # from user_json\n"
        "    'trace_generation':  data_dict['trace_generation'],\n"
        "}"
    ),
    sp(6),
]

story += [
    h2("1.4  Change 1 — parse_data_dynamic: generic trace parser"),
    h3("What it replaced"),
    p(
        "Every old sul_functions.py had its own <b>parse_data(path)</b> that hard-coded "
        "the file format. The energy version read CSV columns by numeric index. "
        "The thermostat version parsed a UPPAAL text format by searching for known "
        "variable name headers. You could not reuse either for any other case study."
    ),
    h3("How the new version works — routing by strategy"),
    p(
        "parse_data_dynamic checks <b>args['resample_strategy']</b> (or "
        "trace_generation.strategy) and routes to one of two private parsers:"
    ),
    code(
        "strategy = trace_config.get('strategy') or args.get('resample_strategy', 'UPPAAL')\n"
        "\n"
        "if strategy == 'SIM':\n"
        "    return _parse_csv(file_paths, target_vars, trace_config)\n"
        "else:\n"
        "    return _parse_uppaal(file_paths[0], target_vars)"
    ),
    h3("_build_target_vars — the column-to-key map"),
    p(
        "Both parsers need to know which columns to extract and how to name them internally. "
        "This map is built once from the user's fields:"
    ),
    code(
        "# User set: main_variable='T_r', driver_signal='t.ON', context=['r.open']\n"
        "target_vars = {\n"
        "    'T_r'    : 'main',    # main observable always mapped to 'main'\n"
        "    't.ON'   : 't.ON',    # driver keeps its original name\n"
        "    'r.open' : 'r.open',  # context variable keeps its name\n"
        "}"
    ),
    p(
        "The output signals dictionary always has this shape, no matter which case study "
        "or how many variables there are:"
    ),
    code(
        "signals = {\n"
        "    'time'   : np.array([0.0, 1.0, 2.0, ...]),   # seconds from start\n"
        "    'main'   : np.array([15.2, 15.04, 14.89, ...]),  # T_r values\n"
        "    't.ON'   : np.array([0.0,  0.0,   0.0, ...]),    # heater state\n"
        "    'r.open' : np.array([0.0,  0.0,   0.0, ...]),    # window state\n"
        "}"
    ),
    info(
        "The key insight: downstream code (is_chg_pt, label_event, get_physics_param) "
        "never needs to know column positions. It only uses dictionary keys — "
        "keys that the user defined."
    ),
    sp(4),
]

story += [
    h3("_parse_csv — reading the energy / SIM case studies"),
    p(
        "The CSV parser applies a fixed six-step preprocessing pipeline driven entirely "
        "by the trace_generation sub-object in user_json:"
    ),
    flow_table([
        ["Step", "What happens", "Config key used"],
        ["1. Load", "Read all uploaded CSV files and concatenate", "—"],
        ["2. Pivot", "If vertical format, pivot rows to columns", "csv_format, key_column, value_column"],
        ["3. Replace", "Convert string tokens to numbers", "replace_values: {\"OFF\": 0, \"ON\": 1}"],
        ["4. Time", "Parse timestamp → elapsed seconds", "time_column, time_format"],
        ["5. Interpolate", "Fill NaN gaps (sensors report at different rates)", "interpolate_method"],
        ["6. Round", "Discretise continuous signals to buckets", "round_columns: {\"RPM\": 200}"],
    ], col_widths=[1.2*cm, 8*cm, 7.8*cm]),
    p(
        "<b>Example for the energy CSV files (W7/W9 data):</b> "
        "The column HEADSTOCK__SPINDLE_MOTOR___1___RPM contains continuous values like "
        "2285.4, 2285.8, 2286.1. After step 6 with bucket size 200, all three become 2200. "
        "Without rounding, every tiny floating-point wobble would look like a new event. "
        "With rounding, the signal is cleanly discrete."
    ),
    h3("_parse_uppaal — reading the thermostat / HRI trace files"),
    p(
        "UPPAAL verifyta produces a plain text file where each variable has a header line "
        "followed by space-separated (time, value) pairs. The parser:"
    ),
    bullet("Reads variable name headers (lines ending with ':')"),
    bullet("Normalises whitespace in names so 'amy.busy || amy.p_2' matches 'amy.busy||amy.p_2'"),
    bullet("Ignores [1]:, [2]: lines (duplicate simulation runs caused by UPPAAL deadlocks)"),
    bullet("Forward-fills values onto a unified time axis (UPPAAL hold-last-value semantics)"),
    sp(6),
]

story += [
    h2("1.5  Change 2 — is_chg_pt_dynamic: generic change-point detection"),
    h3("What the old version did"),
    p(
        "Each case study had its own is_chg_pt with hardcoded logic. Energy:"
    ),
    code("def is_chg_pt(curr, prev):\n"
         "    return abs(curr[0] - prev[0]) > SPEED_RANGE or curr[1] != prev[1]\n"
         "    # curr[0] is always speed, curr[1] is always pressure — fixed positions!"),
    p("Thermostat:"),
    code("def is_chg_pt(curr, prev):\n"
         "    return curr != prev   # checks the entire 3-tuple for any change"),
    h3("What the new version does — two independent checks"),
    p(
        "The new function applies two checks in order. Either one returning True "
        "declares a change point:"
    ),
    code(
        "# CHECK 1 — Did the event label change?\n"
        "curr_label = label_event_dynamic(signals, index,     args)\n"
        "prev_label = label_event_dynamic(signals, index - 1, args)\n"
        "if curr_label != prev_label:\n"
        "    return True          # e.g. heater just turned on: c_0 -> h_0\n"
        "\n"
        "# CHECK 2 — Did any driver signal step beyond its tolerance?\n"
        "for driver_name in drivers:           # e.g. ['t.ON'] or ['RPM']\n"
        "    curr_val = signals[driver_name][index]\n"
        "    prev_val = signals[driver_name][index - 1]\n"
        "    tol      = tolerances.get(driver_name, 0)  # from physics_constants\n"
        "    if abs(curr_val - prev_val) > tol:\n"
        "        return True"
    ),
    p(
        "<b>Where the tolerance comes from:</b> the user puts it inside "
        "trace_generation.physics_constants.tolerances in user_json:"
    ),
    code(
        '"physics_constants": {\n'
        '    "MIN_SPEED": 100,\n'
        '    "tolerances": {\n'
        '        "HEADSTOCK__SPINDLE_MOTOR___1___RPM": 200\n'
        '    }\n'
        '}'
    ),
    p(
        "For thermostat, t.ON only takes values 0 and 1, so any change is significant — "
        "tolerance is 0 (the default). For energy RPM, a tolerance of 200 means small "
        "floating-point wobble around 2285 RPM does not trigger a change point; only "
        "a jump bigger than 200 does."
    ),
    sp(4),
]

story += [
    h2("1.6  Change 3 — label_event_dynamic: guard-based event identification"),
    h3("What the old version did"),
    p(
        "The old label_event functions contained chains of if/elif statements that "
        "checked hard-coded signal indices and constants. HRI had nested conditions "
        "across CS_VERSION and SAMPLE_STRATEGY. Thermostat branched on CS_VERSION 1-10. "
        "None of this logic was reusable."
    ),
    h3("What the new version does — evaluate guard strings"),
    p(
        "The user defines guards in the events array of user_json. The function evaluates "
        "them in order and returns the symbol of the first one that matches:"
    ),
    code(
        "# Build the evaluation context\n"
        "context = {}\n"
        "for key, arr in signals.items():\n"
        "    context[key]           = arr[index]      # current value\n"
        "    context[f'prev_{key}'] = arr[index - 1]  # previous value\n"
        "context.update(physics_constants)             # e.g. MIN_SPEED=100\n"
        "\n"
        "# Evaluate each event guard in definition order\n"
        "for event_def in args['events']:\n"
        "    if safe_eval(event_def['guard'], context):\n"
        "        return event_def['symbol']"
    ),
    h3("safe_eval — how guards are evaluated securely"),
    p(
        "Guards are Python-style boolean expressions supplied by the user as strings. "
        "Calling eval() on user input is dangerous, so safe_eval sandboxes it:"
    ),
    bullet("No Python built-ins are available (no __import__, no open, no exec)"),
    bullet("Only the math module and the explicitly provided context variables are accessible"),
    bullet(
        "Variable names are replaced by positional placeholders (__VAR_0__, __VAR_1__, …) "
        "before eval() is called, sorted by descending length to prevent shorter names "
        "from accidentally matching inside longer ones"
    ),
    p(
        "<b>Example — thermostat guard 't.ON == 1.0':</b> "
        "The context has the key 't.ON' (a string with a period). "
        "safe_eval replaces the literal substring 't.ON' with '__VAR_0__' in the expression, "
        "giving '__VAR_0__ == 1.0'. Python evaluates this cleanly as a simple comparison. "
        "t.ON would have been illegal as a Python identifier, but the replacement "
        "happens at the string level before eval ever sees it."
    ),
    p(
        "<b>Example — HRI guard 'amy.busy || amy.p_2 == 1.0':</b> "
        "The signal key is 'amy.busy || amy.p_2' (the full UPPAAL expression, 20 characters). "
        "Because keys are sorted longest-first, the whole substring 'amy.busy || amy.p_2' "
        "is replaced with '__VAR_0__' before eval. The result '__VAR_0__ == 1.0' is valid Python. "
        "Without the replacement, Python would misparse '||' as a syntax error; "
        "with it, the replacement absorbs both the name and the operator."
    ),
    p(
        "<b>Example — energy guard with symbolic constants:</b>"
    ),
    code(
        'guard: "MIN_SPEED <= HEADSTOCK__SPINDLE_MOTOR___1___RPM and\n'
        '        HEADSTOCK__SPINDLE_MOTOR___1___RPM < 1000"\n'
        "\n"
        "context['MIN_SPEED']                           = 100   # from physics_constants\n"
        "context['HEADSTOCK__SPINDLE_MOTOR___1___RPM']  = 600   # current RPM\n"
        "\n"
        "# After safe_eval replacement:\n"
        "# '__VAR_1__ <= __VAR_0__ and __VAR_0__ < 1000'\n"
        "# Evaluates to: 100 <= 600 and 600 < 1000  ->  True  ->  returns 'e_low'"
    ),
    sp(4),
]

story += [
    h2("1.7  Change 4 — get_physics_param_dynamic: model-aware parameter extraction"),
    h3("What the old versions did"),
    p(
        "Each case study had a dedicated parameter function with a completely different "
        "formula hard-coded for its specific physics:"
    ),
    bullet("thermostat: get_thermo_param — exponential heating/cooling with R=100 and K constants"),
    bullet("energy: get_power_param — arithmetic mean of power values"),
    bullet("HRI: get_ftg_param — log-ratio formula for fatigue growth and recovery"),
    p(
        "None of these formulas were shared. Adding a new case study meant inventing "
        "and coding a new formula."
    ),
    h3("What the new version does — model-type dispatch"),
    p(
        "The new function looks up the model type from the user's JSON and dispatches "
        "to the appropriate formula. The model type for the active segment is found by:"
    ),
    bullet("Calling label_event_dynamic to get the event symbol at start_i"),
    bullet("Finding that symbol in args['events'] to get its model_id"),
    bullet("Finding that model_id in args['models'] to get its type string"),
    code(
        "# User JSON says:\n"
        "# events: [{symbol:'c_0', model_id:0}, {symbol:'h_0', model_id:1}]\n"
        "# models: [{id:0, type:'EXP_DECAY'}, {id:1, type:'EXP_GROWTH'}]\n"
        "\n"
        "# At a segment starting during cooling (c_0 is active):\n"
        "symbol   = 'c_0'      # from label_event_dynamic\n"
        "model_id = 0          # from events lookup\n"
        "type     = 'EXP_DECAY'  # from models lookup"
    ),
    p("Then _compute_rate applies the right formula for that type:"),
    flow_table([
        ["Model type", "Formula", "Captures", "Example"],
        ["MEAN", "arithmetic mean", "Average value in segment", "Power: 2575 W"],
        ["EXP_DECAY", "mean(−log(v[i]/v[i−1]))", "Decay constant R", "Cooling: R ≈ 0.01"],
        ["EXP_GROWTH", "mean(v[i] − v[i−1])", "Growth increment K", "Heating: K ≈ 0.55 °C/step"],
        ["LINEAR_DECAY", "mean(v[i−1] − v[i])", "Slope α", "Custom: 0.3 per second"],
        ["LINEAR_GROWTH", "mean(v[i] − v[i−1])", "Slope α", "Custom: 0.3 per second"],
    ], col_widths=[3.5*cm, 5*cm, 4.5*cm, 4*cm]),
    p(
        "The function always returns both mean and rate so that the adapter in tasks.py "
        "can use whichever is appropriate for the configured case study."
    ),
    PageBreak(),
]

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — OBSERVATION TABLE
# ══════════════════════════════════════════════════════════════════════════════
story += [
    h1("PART 2 — Dynamic Observation Table"),
    sp(4),
    p(
        "The <b>Observation Table (ObsTable)</b> is the central data structure of the "
        "L* learning algorithm. It is a two-part matrix: the upper half (S set) contains "
        "the candidate prefixes for automaton locations; the lower half (low_S) contains "
        "extensions being tested. Each cell holds the physics state (flow model + "
        "probability distribution) observed when the system was run with that word."
    ),
    p(
        "The table is also responsible for two structural operations: "
        "<b>to_sha</b> converts the table into a Stochastic Hybrid Automaton (the "
        "learned result), and <b>get_loc_from_word</b> maps any word to its automaton "
        "location by comparing rows."
    ),
    sp(6),
]

story += [
    h2("2.1  The single hardcoded parameter: EQ_CONDITION"),
    p(
        "The original obstable.py had exactly one parameter that controlled behaviour, "
        "and it was baked in at import time:"
    ),
]
story += compare_table(
    [
        "import configparser",
        "config.read('config.ini')",
        "EQ_CONDITION = config",
        "  ['LSHA PARAMETERS']",
        "  ['EQ_CONDITION'].lower()",
        "",
        "# Module-level constant —",
        "# fixed for the entire process",
    ],
    [
        "# No configparser, no .ini file",
        "# No module-level constant",
        "",
        "# Read live inside each method:",
        "eq_condition = teacher.config",
        "  .get('eq_condition', 's')",
        "  .lower()",
    ],
    "Removing the config.ini dependency from obstable.py"
)
story += [
    p(
        "<b>EQ_CONDITION</b> controls whether two rows in the table are considered "
        "equivalent (same automaton location) using strict or weak equality:"
    ),
    flow_table([
        ["Value", "Meaning", "Effect on the learned automaton"],
        ['"s"  (strict)', "Rows must match exactly in every filled cell", "Fewer merged locations; may produce more states but is safer"],
        ['"w"  (weak)',   "Rows match if no filled cell explicitly disagrees", "More aggressive merging; automaton may be non-deterministic but converges faster"],
    ], col_widths=[2.5*cm, 7*cm, 7.5*cm]),
    sp(4),
]

story += [
    h2("2.2  Where eq_condition now comes from — the data flow"),
    p(
        "The value travels from the user's browser to every row comparison inside the "
        "observation table through the following chain:"
    ),
    code(
        "# Step 1: User selects 's' or 'w' in the React web UI\n"
        "#         Django saves it to:  CaseStudy.eq_condition  (CharField)\n"
        "\n"
        "# Step 2: tasks.py builds the teacher config dict:\n"
        "teacher_config = {\n"
        "    'eq_condition': getattr(cs_instance, 'eq_condition', 's'),\n"
        "    ...  # noise, p_value, ht_query, n_min, etc.\n"
        "}\n"
        "\n"
        "# Step 3: CustomTeacher stores it:\n"
        "class CustomTeacher:\n"
        "    def __init__(self, sul, trace_generator, config_data):\n"
        "        self.config = config_data         # the whole dict\n"
        "        self.eq_condition = self.config.get('eq_condition', 's').lower()\n"
        "\n"
        "# Step 4: dynamic_obstable.py reads it at call time:\n"
        "def get_loc_from_word(self, word, locations, seq_to_loc, teacher):\n"
        "    eq_condition = teacher.config.get('eq_condition', 's').lower()\n"
        "    ...\n"
        "    if eq_condition == 's':\n"
        "        teacher.eqr_query(curr_row, row, strict=True)\n"
        "    else:\n"
        "        teacher.eqr_query(curr_row, row, strict=False)"
    ),
    note(
        "In the old version, two users running different case studies with different "
        "eq_condition settings in the same Python process would both get the same value "
        "because EQ_CONDITION was a module-level singleton set at import time. "
        "Now each call reads from the teacher object, which was created fresh "
        "for that specific case study run."
    ),
    sp(6),
]

story += [
    h2("2.3  Where eq_condition is used inside the table — three locations"),
    h3("Location A — get_loc_from_word (edge routing)"),
    p(
        "When building the SHA edges, every word must be mapped to an automaton location. "
        "The function walks the upper observations and asks: 'Is the row for this word "
        "equivalent to any row that already has a named location?' The equivalence check "
        "uses eq_condition."
    ),
    code(
        "eq_condition = teacher.config.get('eq_condition', 's').lower()\n"
        "for i, row in enumerate(self.get_upper_observations()):\n"
        "    if eq_condition == 's':\n"
        "        match = teacher.eqr_query(curr_row, row, strict=True)\n"
        "    else:\n"
        "        match = teacher.eqr_query(curr_row, row, strict=False)"
    ),
    h3("Location B — to_sha (location creation)"),
    p(
        "When converting the table to an automaton, the method groups rows into unique "
        "locations. Two rows that are 'equivalent' become the same location. "
        "Again eq_condition drives the grouping:"
    ),
    code(
        "eq_condition = teacher.config.get('eq_condition', 's').lower()\n"
        "for i, row in enumerate(upp_obs):\n"
        "    for seq in unique_sequences:\n"
        "        row_2 = upp_obs[self.get_S().index(seq)]\n"
        "        if eq_condition == 's':\n"
        "            already_present = teacher.eqr_query(row, row_2, strict=True)\n"
        "        else:\n"
        "            already_present = teacher.eqr_query(row, row_2, strict=False)"
    ),
    sp(6),
]

story += [
    h2("2.4  The hard-coded flow label — the most impactful fix"),
    p(
        "This is the change that actually broke every non-HRI case study in the old version. "
        "The flow label is the text written on each automaton location — it describes the "
        "physics that govern the system while in that state (e.g. 'cooling R=0.01')."
    ),
]
story += compare_table(
    [
        "# in to_sha:",
        "new_flow = (",
        "  row.state[0].vars[0][0].label",
        "  + ', ' +",
        "  row.state[0].vars[0][1].label",
        ")",
        "# Assumes EXACTLY 2 physics",
        "# variables: vars[0][0] and",
        "# vars[0][1].",
        "# Crashes for Energy (1 var),",
        "# Thermostat (1 var), and any",
        "# custom case study.",
    ],
    [
        "try:",
        "  if hasattr(row.state[0], 'vars')",
        "    and isinstance(..., (list,tuple)):",
        "    flat_vars = []",
        "    for v in row.state[0].vars:",
        "      if isinstance(v, (list,tuple)):",
        "        flat_vars.extend(v)",
        "      else:",
        "        flat_vars.append(v)",
        "    new_flow = ', '.join([",
        "      getattr(v,'label',str(v))",
        "      for v in flat_vars])",
        "except Exception:",
        "  new_flow = str(row.state[0])",
    ],
    "Flow label construction: hard-coded 2-variable access vs. generic flatten"
)
story += [
    p(
        "<b>What vars[0] contains depends entirely on the case study.</b> "
        "For HRI it is [(lambda_var, mu_var)] — a list containing a tuple of two variables. "
        "For Energy it is [mean_var] — a list with one variable. "
        "For a custom 3-variable case study it would be [(v1, v2, v3)]."
    ),
    p(
        "The new code flattens whatever structure exists and joins all variable labels "
        "with commas. Example outputs:"
    ),
    flow_table([
        ["Case study", "vars structure", "Old result", "New result"],
        ["HRI",        "[(lambda, mu)]",     "lambda, mu  ✔", "lambda, mu  ✔"],
        ["Energy",     "[mean_power]",        "IndexError  ✗", "mean_power  ✔"],
        ["Thermostat", "[R]",                 "IndexError  ✗", "R  ✔"],
        ["Custom 3-var","[(v1, v2, v3)]",    "IndexError  ✗", "v1, v2, v3  ✔"],
    ], col_widths=[2.8*cm, 3.5*cm, 4.5*cm, 4.5*cm]),
    sp(6),
]

story += [
    h2("2.5  The safe max() fix — preventing crashes on empty tables"),
    p(
        "At the very start of the L* algorithm, before any observations have been collected, "
        "the observation table is empty. If the __str__ method is called at this point "
        "(e.g. for logging), the old code crashed:"
    ),
]
story += compare_table(
    [
        "max_tabs = max(",
        "    [len(str(word))",
        "     for i, word in enumerate(...)",
        "     if i in populated_rows]",
        ")",
        "# If populated_rows is empty,",
        "# max([]) raises ValueError:",
        "# 'max() arg is an empty sequence'"
    ],
    [
        "valid_lens = [",
        "    len(str(word))",
        "    for i, word in enumerate(...)",
        "    if i in populated_rows",
        "]",
        "max_tabs = max(valid_lens) if valid_lens else 8",
        "# Returns default width 8",
        "# when the table is empty.",
    ],
    "Safe max() with default fallback in __str__"
)
story += [PageBreak()]

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — TEACHER
# ══════════════════════════════════════════════════════════════════════════════
story += [
    h1("PART 3 — Dynamic Teacher (CustomTeacher)"),
    sp(4),
    p(
        "The <b>Teacher</b> is the orchestrator of the L* learning loop. It is responsible "
        "for four types of queries:"
    ),
    flow_table([
        ["Query", "Abbreviation", "What it does"],
        ["Model Identification", "mi_query", "Fits a physics model (EXP_DECAY etc.) to a segment"],
        ["Hypothesis Testing",   "ht_query", "Assigns a probability distribution to a flow condition"],
        ["Row Equality",         "eqr_query", "Checks if two table rows represent the same state"],
        ["Counterexample",       "get_counterexample", "Finds a word that proves the table is not closed or consistent"],
    ], col_widths=[4.5*cm, 3.5*cm, 9*cm]),
    p(
        "In the original project, the Teacher was a single class in "
        "lsha/sha_learning/learning_setup/teacher.py that read ALL its configuration "
        "from config.ini at module import time. This made it impossible to run different "
        "case studies with different settings without manually editing the .ini file."
    ),
    sp(6),
]

story += [
    h2("3.1  How the old Teacher was configured"),
    p(
        "The old file had a block of module-level constants that were read once and "
        "never changed during the lifetime of the Python process:"
    ),
    code(
        "# lsha/sha_learning/learning_setup/teacher.py  (OLD)\n"
        "\n"
        "config = configparser.ConfigParser()\n"
        "config.read('...sha_learning/resources/config/config.ini')\n"
        "\n"
        "CS           = config['SUL CONFIGURATION']['CASE_STUDY']     # e.g. 'THERMO'\n"
        "NOISE        = float(config['LSHA PARAMETERS']['DELTA'])      # e.g. 1.0\n"
        "P_VALUE      = 0.0                                            # hard-coded!\n"
        "MI_QUERY     = config['LSHA PARAMETERS']['MI_QUERY'] == 'True'\n"
        "PLOT_DDTW    = config['LSHA PARAMETERS']['PLOT_DDTW'] == 'True'\n"
        "HT_QUERY     = config['LSHA PARAMETERS']['HT_QUERY'] == 'True'\n"
        "HT_QUERY_TYPE= config['LSHA PARAMETERS']['HT_QUERY_TYPE']    # 'D' or 'S'\n"
        "EQ_CONDITION = config['LSHA PARAMETERS']['EQ_CONDITION'].lower()\n"
        "\n"
        "class Teacher:\n"
        "    def __init__(self, sul, pov=None, start_dt=None, ...):\n"
        "        self.sul = sul\n"
        "        self.TG  = TraceGenerator(pov=pov, ...)\n"
        "        self.hist = {}"
    ),
    p(
        "Every method then used these module-level variables directly. For example "
        "mi_query hard-coded a special case for the thermostat:"
    ),
    code(
        "def mi_query(self, word):\n"
        "    if not MI_QUERY or word == '':\n"
        "        return self.flows[0][self.sul.default_m]\n"
        "    ...\n"
        "    if CS == 'THERMO' and word[-1].symbol == 'h_0':\n"
        "        return self.flows[0][2]    # hard-coded index for thermostat model!"
    ),
    note(
        "P_VALUE was even hard-coded as 0.0 in the module, not read from config. "
        "This meant the statistical test threshold could never be changed "
        "without editing the source code."
    ),
    sp(6),
]

story += [
    h2("3.2  How the new CustomTeacher is configured"),
    p(
        "The new CustomTeacher receives a <b>config_data dictionary</b> at construction time. "
        "This dictionary is built in tasks.py directly from the Django database model, "
        "so every field reflects exactly what the user set in the web UI:"
    ),
    code(
        "# tasks.py\n"
        "teacher_config = {\n"
        "    'noise':        getattr(cs_instance, 'noise',        0.0),\n"
        "    'p_value':      getattr(cs_instance, 'p_value',      0.05),\n"
        "    'mi_query':     getattr(cs_instance, 'mi_query',     False),\n"
        "    'plot_ddtw':    getattr(cs_instance, 'plot_ddtw',    False),\n"
        "    'ht_query':     getattr(cs_instance, 'ht_query',     False),\n"
        "    'ht_query_type':getattr(cs_instance, 'ht_query_type','D'),\n"
        "    'eq_condition': getattr(cs_instance, 'eq_condition', 's'),\n"
        "    'n_min':        getattr(cs_instance, 'n_min',        10),\n"
        "    'is_aggregation':getattr(cs_instance,'is_aggregation',False),\n"
        "}\n"
        "\n"
        "teacher = CustomTeacher(\n"
        "    sul=sul,\n"
        "    config_data=teacher_config,\n"
        "    trace_generator=custom_tg\n"
        ")"
    ),
    p(
        "Inside CustomTeacher.__init__, each value is stored as an instance attribute "
        "and logged so the user can verify their settings took effect:"
    ),
    code(
        "class CustomTeacher:\n"
        "    def __init__(self, sul, trace_generator=None, config_data=None):\n"
        "        self.config          = config_data if config_data else {}\n"
        "        self.noise           = float(self.config.get('noise',        0.0))\n"
        "        self.p_value         = float(self.config.get('p_value',      0.05))\n"
        "        self.mi_query_flag   = self.config.get('mi_query',    False)\n"
        "        self.plot_ddtw       = self.config.get('plot_ddtw',   False)\n"
        "        self.ht_query_flag   = self.config.get('ht_query',    False)\n"
        "        self.ht_query_type   = self.config.get('ht_query_type','D')\n"
        "        self.eq_condition    = str(self.config.get('eq_condition','s')).lower()\n"
        "        self.n_min           = int(self.config.get('n_min',       10))\n"
        "        self.is_aggregation  = self.config.get('is_aggregation', False)\n"
        "        self.TG              = trace_generator  # not built internally!\n"
        "        self.hist            = {}"
    ),
    sp(4),
]

story += [
    h2("3.3  Change-by-change breakdown"),
    h3("Change A — All module-level constants removed"),
    p(
        "Every constant from the old teacher.py top block (CS, NOISE, P_VALUE, MI_QUERY, "
        "PLOT_DDTW, HT_QUERY, HT_QUERY_TYPE, EQ_CONDITION) is now an instance attribute "
        "set from config_data. The configparser import is also gone."
    ),
    flow_table([
        ["Old constant", "Old source", "New attribute", "Django field"],
        ["CS",           "config.ini", "— (removed)",   "Case study name is just a label now"],
        ["NOISE",        "DELTA in .ini", "self.noise",  "CaseStudy.noise  (FloatField)"],
        ["P_VALUE",      "hard-coded 0.0", "self.p_value","CaseStudy.p_value (FloatField)"],
        ["MI_QUERY",     "config.ini", "self.mi_query_flag","CaseStudy.mi_query (BooleanField)"],
        ["PLOT_DDTW",    "config.ini", "self.plot_ddtw", "CaseStudy.plot_ddtw (BooleanField)"],
        ["HT_QUERY",     "config.ini", "self.ht_query_flag","CaseStudy.ht_query (BooleanField)"],
        ["HT_QUERY_TYPE","config.ini", "self.ht_query_type","CaseStudy.ht_query_type (CharField)"],
        ["EQ_CONDITION", "config.ini", "self.eq_condition","CaseStudy.eq_condition (CharField)"],
    ], col_widths=[3*cm, 3.2*cm, 4.2*cm, 6.6*cm]),
    sp(6),
]

story += [
    h3("Change B — mi_query: removed the CS='THERMO' special case"),
]
story += compare_table(
    [
        "def mi_query(self, word):",
        "    if not MI_QUERY or word == '':",
        "        return self.flows[0]",
        "                   [self.sul.default_m]",
        "    ...",
        "    # HARD-CODED THERMOSTAT BRANCH:",
        "    if CS == 'THERMO' and \\",
        "       word[-1].symbol == 'h_0':",
        "        return self.flows[0][2]",
        "    ...",
    ],
    [
        "def mi_query(self, word):",
        "    if not self.mi_query_flag \\",
        "       or word == '':",
        "        return self.flows[0]",
        "                   [self.sul.default_m]",
        "    ...",
        "    # No CS-specific branch.",
        "    # The model at index 2 comes",
        "    # from the user's models list,",
        "    # not a hard-coded number.",
    ],
    "mi_query: removing the CS='THERMO' hard-coded model index"
)
story += [
    p(
        "The old code forced the thermostat to always use model index 2 for event h_0. "
        "This was correct for the specific thermostat version used in the research paper "
        "but would silently produce wrong results for any other case study or any "
        "thermostat variant where h_0 mapped to a different model. "
        "The new code simply trusts the model index declared in the user's JSON."
    ),
    sp(4),
]

story += [
    h3("Change C — ht_query: routing by self.ht_query_type instead of HT_QUERY_TYPE"),
]
story += compare_table(
    [
        "def ht_query(self, word, flow, save=True):",
        "    if not HT_QUERY or word == '':",
        "        return self.distributions",
        "                   [self.sul.default_d]",
        "    if HT_QUERY_TYPE == 'D':",
        "        return self.ht_d_query(...)",
        "    else:",
        "        return self.ht_s_query(...)",
    ],
    [
        "def ht_query(self, word, flow, save=True):",
        "    if not self.ht_query_flag \\",
        "       or word == '':",
        "        return self.distributions",
        "                   [self.sul.default_d]",
        "    if self.ht_query_type == 'D':",
        "        return self.ht_d_query(...)",
        "    else:",
        "        return self.ht_s_query(...)",
    ],
    "ht_query: switching from module constants to instance attributes"
)
story += [
    p(
        "<b>What 'D' vs 'S' means:</b> "
        "HT_QUERY_TYPE='D' (Deterministic) uses exact parameter matching — two segments "
        "are considered the same flow condition only if their physics parameters are "
        "numerically identical. "
        "HT_QUERY_TYPE='S' (Stochastic) uses the Kolmogorov-Smirnov test — two segments "
        "are the same if their parameter distributions cannot be statistically distinguished. "
        "The user chooses which is appropriate for their system via the web UI's "
        "'HT Query Type' dropdown."
    ),
    sp(4),
]

story += [
    h3("Change D — ht_s_query: NOISE and P_VALUE replaced by instance attributes"),
    p(
        "The stochastic hypothesis test contains two hardcoded decisions from the old code: "
        "the noise level added to the comparison samples, and the statistical significance "
        "threshold. Also, the thermostat case study was handled specially (no noise for THERMO)."
    ),
]
story += compare_table(
    [
        "# OLD noise handling:",
        "if CS == 'THERMO':",
        "    v1 = metrics",
        "    noise1 = [0] * len(v1)",
        "else:",
        "    v1 = [avg] * 50",
        "    noise1 = np.random.normal(",
        "        0.0, NOISE, size=len(v1))",
        "",
        "# OLD significance threshold:",
        "if pvalue >= P_VALUE:  # was 0.0",
        "    best_fit = fits[0]",
    ],
    [
        "# NEW noise handling (is_aggregation",
        "# field replaces CS=='THERMO'):",
        "if not self.is_aggregation:",
        "    v1 = metrics",
        "    noise1 = [0] * len(v1)",
        "else:",
        "    v1 = [avg] * 50",
        "    noise1 = np.random.normal(",
        "        0.0, self.noise, size=...)",
        "",
        "# NEW significance threshold:",
        "if pvalue >= self.p_value:",
        "    best_fit = fits[0]",
    ],
    "ht_s_query: replacing CS-specific branch and hard-coded P_VALUE=0.0 with UI values"
)
story += [
    p(
        "<b>What is_aggregation means:</b> "
        "When is_aggregation=False (the default), each observed metric is used directly "
        "as a data point in the KS test — no noise, no expansion. This is appropriate "
        "when the data is deterministic or the user wants exact matching. "
        "When is_aggregation=True, metrics are expanded to 50 samples with Gaussian noise "
        "of standard deviation self.noise. This simulates stochastic variation and is "
        "suitable for noisy physical systems like HRI fatigue measurement. "
        "Previously this choice was implicit in whether CS was 'THERMO'."
    ),
    p(
        "<b>What p_value controls:</b> "
        "The KS test compares two distributions and returns a p-value. If the p-value is "
        "above self.p_value, the two distributions are considered indistinguishable and "
        "the same probability distribution is reused. A higher p_value threshold is more "
        "permissive (merges more distributions). The old code hard-coded this as 0.0, "
        "meaning distributions were NEVER merged — every segment got a new distribution. "
        "The new default of 0.05 is a statistically conventional choice."
    ),
    sp(6),
]

story += [
    h3("Change E — ref_query: n_min from self.n_min; CSV vs UPPAAL path split"),
]
story += compare_table(
    [
        "def ref_query(self, table):",
        "    n_resample = int(",
        "        config['LSHA PARAMETERS']",
        "               ['N_min'])",
        "    ...",
        "    for word in uq:",
        "        self.TG.set_word(word + e)",
        "        path = self.TG.get_traces(",
        "                   n_resample)",
        "        if path is not None:",
        "            for sim in path:",
        "                self.sul",
        "                    .process_data(sim)",
    ],
    [
        "def ref_query(self, table):",
        "    n_resample = int(self.n_min)",
        "    ...",
        "    for word in uq:",
        "        self.TG.set_word(word + e)",
        "        path = self.TG.get_traces(",
        "                   n_resample)",
        "        if path and len(path) > 0:",
        "          if TG.resample_strategy=='CSV':",
        "            self.sul.process_data(path)",
        "          else:",
        "            for sim in path:",
        "              self.sul.process_data(sim)",
        "        # empty list -> do nothing",
    ],
    "ref_query: n_min from instance + CSV vs UPPAAL path handling"
)
story += [
    p(
        "<b>What n_min controls:</b> "
        "When the teacher refines its knowledge, it resamples ambiguous words this many "
        "times. A higher n_min gives more statistically reliable estimates but takes longer. "
        "The user sets this in the 'N Min' field of the web UI."
    ),
    p(
        "<b>Why the CSV path is different:</b> "
        "For UPPAAL, get_traces() returns a list of file paths — one per simulation run. "
        "Each file must be processed individually because each is an independent trace. "
        "For CSV, get_traces() returns the list of ALL uploaded CSV files at once. "
        "These must be passed together so the parser can concatenate them and produce "
        "one unified time series. Passing them one at a time would give partial data "
        "with gaps at every file boundary."
    ),
    p(
        "<b>Why an empty list is silently ignored:</b> "
        "The CustomTraceGenerator uses a csv_yielded flag. After yielding the CSV files "
        "once, every subsequent call to get_traces() returns an empty list. This prevents "
        "the ref_query loop from running forever. The check 'if path and len(path) > 0' "
        "cleanly handles this without any error or log spam."
    ),
    sp(6),
]

story += [
    h3("Change F — not_closed and not_consistent: EQ_CONDITION removed"),
]
story += compare_table(
    [
        "def not_closed(self, table, new_row):",
        "    if EQ_CONDITION == 's':",
        "        eq_rows = [r for r in ...",
        "          self.eqr_query(...,",
        "                        strict=True)]",
        "    else:",
        "        eq_rows = [r for r in ...",
        "          self.eqr_query(...,",
        "                        strict=False)]",
        "",
        "    # Also in not_consistent:",
        "    if EQ_CONDITION == 's' and ...",
        "    elif EQ_CONDITION == 'w' and ...",
    ],
    [
        "def not_closed(self, table, new_row):",
        "    is_strict = (self.eq_condition == 's')",
        "    eq_rows = [r for r in ...",
        "      self.eqr_query(new_row, r,",
        "                     strict=is_strict)]",
        "",
        "    # Also in not_consistent:",
        "    is_strict = (self.eq_condition == 's')",
        "    equal = self.eqr_query(...,",
        "                          strict=is_strict)",
        "    # Single branch — no if/elif",
    ],
    "not_closed / not_consistent: collapsing the double if/elif into a boolean flag"
)
story += [
    p(
        "Beyond removing the global constant, the new code also collapses the "
        "if EQ_CONDITION=='s' / elif EQ_CONDITION=='w' double branches into a single "
        "boolean is_strict = (self.eq_condition == 's'). This reduces code duplication "
        "and makes it immediately obvious what the flag controls."
    ),
    sp(6),
]

story += [
    h3("Change G — get_counterexample: removed the CS-specific ENERGY branch"),
]
story += compare_table(
    [
        "# End of get_counterexample:",
        "if CS in ['ENERGY','AUTO_TWIN']",
        "   and len(not_counter) > 0:",
        "    new_events = set([",
        "        e.symbol for x in not_counter",
        "        for e in x.events",
        "    ]) - set([",
        "        e.symbol for t in S",
        "        for e in t.events",
        "    ])",
        "    if len(new_events) > 0:",
        "        return not_counter[-1]",
        "    else:",
        "        return None",
    ],
    [
        "# get_counterexample remains",
        "# structurally the same but",
        "# the CS=='ENERGY' special case",
        "# is gone.",
        "",
        "# The standard L* algorithm",
        "# handles all case studies.",
        "# No hard-coded exceptions",
        "# for specific system names.",
        "",
        "return None",
    ],
    "get_counterexample: removing the ENERGY/AUTO_TWIN special-case branch"
)
story += [
    p(
        "The old code had a branch that returned a counterexample only for the ENERGY "
        "and AUTO_TWIN case studies when new event symbols were found in traces that "
        "were not yet in the observation table. This was a workaround for a specific "
        "behaviour of those case studies, not a general algorithm property. "
        "Removing it makes the teacher apply the same L* logic to every case study."
    ),
    PageBreak(),
]

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
story += [
    h1("SUMMARY — Complete picture of what changed and why"),
    sp(6),
    p(
        "The table below maps every removed constant and hard-coded value to the user "
        "field that replaced it, and shows which module each change appears in."
    ),
    flow_table([
        ["What was hard-coded", "Where", "Now comes from", "Django field"],
        ["SPEED_RANGE, MIN_SPEED, MAX_SPEED", "energy sul_functions", "physics_constants in user_json", "user_json → trace_generation"],
        ["Signal indices signals[0], [1], [2]", "all sul_functions", "_build_target_vars from field names", "main_variable, driver_signal, context_variables"],
        ["Column positions row[2], row[3]", "energy parse_data", "column names in target_vars", "main_variable + driver_signal"],
        ["CS_VERSION branches 1-10", "thermostat label_event", "guard expressions in events[]", "user_json → events[].guard"],
        ["CS='THERMO' model index 2", "old teacher mi_query", "Removed entirely", "models[].id in user_json"],
        ["CS='ENERGY' counterexample branch", "old teacher get_counterexample", "Removed entirely", "Standard L* for all"],
        ["NOISE / DELTA = 1.0", "old teacher ht_s_query", "self.noise", "CaseStudy.noise"],
        ["P_VALUE = 0.0 (hard-coded)", "old teacher ht_s_query", "self.p_value", "CaseStudy.p_value"],
        ["MI_QUERY from config.ini", "old teacher mi_query", "self.mi_query_flag", "CaseStudy.mi_query"],
        ["HT_QUERY from config.ini", "old teacher ht_query", "self.ht_query_flag", "CaseStudy.ht_query"],
        ["HT_QUERY_TYPE from config.ini", "old teacher ht_query", "self.ht_query_type", "CaseStudy.ht_query_type"],
        ["EQ_CONDITION from config.ini", "old teacher + old obstable", "self.eq_condition / teacher.config", "CaseStudy.eq_condition"],
        ["N_min from config.ini", "old teacher ref_query", "self.n_min", "CaseStudy.n_min"],
        ["Flow label vars[0][0]+vars[0][1]", "old obstable to_sha", "Generic flatten loop", "models[].type (indirectly)"],
        ["max() crash on empty table", "old obstable __str__", "Safe max with default=8", "— (robustness fix)"],
        ["CSV vs UPPAAL file passing", "old teacher ref_query", "resample_strategy branch", "CaseStudy.resample_strategy"],
    ], col_widths=[5.5*cm, 3.5*cm, 4.5*cm, 3.5*cm]),
    sp(8),
    info(
        "The result is a system where adding a completely new case study requires "
        "zero Python code changes. The user fills in the web form, uploads their "
        "trace files or UPPAAL model, and the engine adapts entirely to what they provided."
    ),
]

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF written to: {OUTPUT}")
