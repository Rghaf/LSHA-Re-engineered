import math

def bounded_growth_model(params, x):

    # 1. Extract parameters from JSON with safe defaults
    power = float(params.get('power', 1.0))
    resistance = float(params.get('resistance', 10.0))
    print(f"params, x: {params}, {x}")
    # 2. Prevent division by zero error
    if resistance == 0:
        return power

    # 3. Calculate the derivative (Rate of change)
    rate = power - (x / resistance)
    print(f"Calculated rate: {rate}")
    return rate


def exponential_decay_model(params, x):
    mean = float(params.get('mean', 0.0)) # The target value
    k = float(params.get('k', 0.1))       # The speed of decay
    print(f"params, x: {params}, {x}")
    
    # Newton's Law of Cooling logic
    rate = -k * (x - mean)
    print(f"Calculated rate: {rate}")
    
    return rate


def linear_model(params, t, x):
    rate = float(params.get('rate', 1.0))
    try:
        result = rate * x
        print(f"DEBUG: linear_model | params={params}, x={x}, result={result}")
        return result
    except Exception:
        return 0.0

def exponential_model(params, t, x):
    k = float(params.get('k', 0.1))
    try:
        result = k * x
        print(f"DEBUG: exponential_model | params={params}, x={x}, result={result}")
        return result
    except Exception:
        return 0.0

def constant_model(params, t, x):
    result = float(params.get('value', 0.0))
    print(f"DEBUG: constant_model | params={params}, x={x}, result={result}")
    return result


def constant_model(params, x):
    return 0.0



def get_physics_model(model_type):
    # Normalize the input string (uppercase, strip whitespace) to prevent typos
    # e.g., "Bounded Growth " -> "BOUNDED_GROWTH"
    mt = str(model_type or '').upper().strip().replace(" ", "_")
    
    # 1. Thermostat Heating Logic
    if mt in ['BOUNDED_GROWTH', 'HEATING']:
        return bounded_growth_model
    
    # 2. Thermostat Cooling / HRI Trust Logic
    elif mt in ['EXPONENTIAL_DECAY', 'EXP_DECAY', 'EXPONENTIAL', 'COOLING']:
        return exponential_decay_model
    
    # 3. HRI Fatigue Logic / Simple Counters
    elif mt in ['LINEAR', 'LIN', 'GROWTH']:
        print("LINEAR")
        
        return linear_model
        
    # 4. Fallback / Idle
    elif mt in ['CONSTANT', 'CONST', 'IDLE', 'OFF']:
        return constant_model

    # Default Safety: If we don't recognize the name, return Linear (safe fallback)
    return linear_model