import re
import numpy as np
from scipy.stats import linregress

def build_robust_pattern(var_name):
    """
    Makes the regex search completely immune to whitespace differences and 
    translates pythonic 'or'/'and' into UPPAAL '||'/'&&'.
    """
    # 1. Translate pythonic logic to UPPAAL logic
    v = re.sub(r'\bor\b', '||', str(var_name))
    v = re.sub(r'\band\b', '&&', v)
    
    # 2. Strip all spaces
    v = v.replace(' ', '')
    
    # 3. Escape each character and insert \s* (optional spaces) between them
    flexible_var = r"\s*".join([re.escape(c) for c in v])
    
    # 4. Construct final regex allowing newlines/spaces around the colon and [0]:
    pattern = flexible_var + r"\s*:\s*\[0\]:\s*((?:\([0-9\.]+\s*,\s*[0-9\.]+\)\s*)+)"
    return pattern

def parse_data_dynamic(file_path, args=None):
    print(f"[SUL] Parsing UPPAAL trace: {file_path}")
    if not args:
        return {}

    # 1. Identify Target Variables
    target_vars = {
        args['driver']: 'driver', 
        args.get('main_var') or args.get('main_variable'): 'main'
    }
    
    context_vars = args.get('context_variables')
    if context_vars is None:
        context_vars = []
        
    for cv in context_vars:
        target_vars[cv] = cv

    # 2. Read File Content
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"[SUL Read Error] {e}")
        return {}

    # 3. Regex Parsing Strategy
    raw_signals = {}
    all_time_points = set()

    for var_name, internal_key in target_vars.items():
        pattern = build_robust_pattern(var_name)
        match = re.search(pattern, content)
        
        if match:
            pairs_str = match.group(1)
            # Find all (time, value) tuples, allowing for spaces inside the parentheses
            points = re.findall(r"\(([0-9\.]+)\s*,\s*([0-9\.]+)\)", pairs_str)
            data_points = [[float(t), float(v)] for t, v in points]
            raw_signals[internal_key] = data_points
            
            for t, v in data_points:
                all_time_points.add(t)
        else:
            print(f"[SUL Warning] Variable '{var_name}' not found in trace.")
            raw_signals[internal_key] = []

    # 4. Alignment / Resampling
    if not all_time_points:
        print("[SUL Error] No time points found. Trace parsing failed.")
        return {}

    sorted_times = sorted(list(all_time_points))
    final_signals = {'time': np.array(sorted_times)}
    
    for internal_key, points in raw_signals.items():
        if not points:
            final_signals[internal_key] = np.zeros(len(sorted_times))
            continue
            
        val_map = {t: v for t, v in points}
        aligned_values = []
        current_val = points[0][1] 
        
        for t in sorted_times:
            if t in val_map:
                current_val = val_map[t]
            aligned_values.append(current_val)
            
        final_signals[internal_key] = np.array(aligned_values)

    return final_signals


def is_chg_pt_dynamic(signals, index, args=None):
    if index == 0: return True
    return signals['driver'][index] != signals['driver'][index - 1]


def label_event_dynamic(signals, index, args=None):
    if not args or 'events' not in args: return "unknown"

    driver_val = signals['driver'][index]
    
    # 1. Build Context (Sanitized for Python 'eval')
    current_context = {}
    for key in signals:
        if key not in ['time', 'driver', 'main']:
            # Strip spaces, then replace all non-alphanumeric chars (like [] or .) with underscores
            clean_key = re.sub(r'[^a-zA-Z0-9]', '_', key.replace(' ', ''))
            current_context[clean_key] = signals[key][index]

    # 2. Match JSON Events
    for event_def in args['events']:
        # Check Trigger
        try:
            if float(event_def.get('trigger_value', -999)) != float(driver_val):
                continue
        except ValueError:
            continue

        # Check Guard
        guard_expr = event_def.get('guard')
        if not guard_expr:
            return event_def['symbol']
            
        try:
            if eval(guard_expr, {}, current_context):
                return event_def['symbol']
        except Exception:
            continue

    return "unknown"


def get_physics_param_dynamic(signals, start_idx, end_idx, args=None):
    if not args: return {}

    evt_symbol = label_event_dynamic(signals, start_idx, args)
    
    model_type = "UNKNOWN"
    for e in args['events']:
        if e['symbol'] == evt_symbol:
            m_id = e.get('model_id')
            for m in args['models']:
                if m['id'] == m_id:
                    model_type = m['type']
            break

    if 'main' not in signals or len(signals['main']) <= end_idx:
        return {'rate': 0.0, 'variance': 0.0}

    y = signals['main'][start_idx:end_idx]
    x = signals['time'][start_idx:end_idx]
    
    if len(y) < 2: return {'rate': 0.0, 'variance': 0.0}

    x = x - x[0] # Normalize time
    params = {}
    
    if 'LINEAR' in model_type:
        slope, _, _, _, _ = linregress(x, y)
        residuals = y - (slope * x + y[0])
        params['rate'] = abs(slope)
        params['variance'] = np.var(residuals) if len(residuals) > 0 else 0.0

    elif 'EXP' in model_type or 'GROWTH' in model_type:
        valid = y > 0
        if np.sum(valid) > 2:
            y_log = np.log(y[valid])
            x_log = x[valid]
            slope, _, _, _, _ = linregress(x_log, y_log)
            params['mean'] = abs(slope) 
            pred = np.exp(y_log[0] + slope * x_log) 
            params['variance'] = np.var(y[valid] - pred)
        else:
            params['mean'] = 0.1
            params['variance'] = 0.1
    else:
        params['rate'] = 0.0
        params['variance'] = 0.0

    return params