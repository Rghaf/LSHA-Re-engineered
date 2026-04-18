import os
import random
import subprocess

class CustomTraceGenerator:
    def __init__(self, cs_name, resample_strategy, output_dir, trace_gen_config, uppaal_bin_path=None, uppaal_model_path=None, uppaal_query_path=None, csv_file=None):
        self.cs_name = cs_name.replace(" ", "_")
        self.resample_strategy = resample_strategy

        self.uppaal_bin_path = uppaal_bin_path
        self.uppaal_model_path = uppaal_model_path 
        self.uppaal_query_path = uppaal_query_path
        
        # This will hold the absolute path to our CSV file
        self.csv_file = csv_file
        
        self.output_dir = output_dir
        
        self.config = trace_gen_config if trace_gen_config else {}
        self.max_e = int(self.config.get('max_length', 15))
        
        self.xml_force_var = self.config.get('xml_force_variable', 'force_open')
        self.xml_action_var = self.config.get('xml_action_variable', 'force_exe')

        self.word = None       
        self.events = []   
        
        # Flag to prevent infinite loops when processing static CSV files
        self.csv_yielded = False 
        
    def set_word(self, w):
        self.word = w
        self.events = w.events

    def build_event_strings(self):
        """
        Dynamically builds the UPPAAL array strings. 
        Returns a dictionary mapping the XML variable name to its string format.
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

    def fix_model(self):
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

    def get_traces(self, n: int = 1):
        # --- THE ROUTER ---
        if self.resample_strategy == 'CSV':
            print(f"[TraceGen] CSV Strategy Detected.")
            return self.get_traces_csv(n)            
        elif self.resample_strategy == 'UPPAAL':
            return self.get_traces_uppaal(n)
            
        return []

    def get_traces_uppaal(self, n: int):
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

    def get_traces_csv(self, n: int = 1):
        # 1. Validate that the CSV file path was provided
        if not self.csv_file or self.csv_file == '':
            print("[TraceGen] The CSV file was not found or the path is empty!!")
            return []
        
        # 2. Prevent L* infinite loops! 
        # A CSV is static data. We only hand the file path to the SUL engine ONCE per learning run.
        if not self.csv_yielded:
            self.csv_yielded = True
            sims = [self.csv_file]
            print(f"[TraceGen] Yielding CSV file to SUL Engine: {sims}")
            return sims
        else:
            # Silently return empty arrays for all subsequent requests to stop the Teacher from looping
            return []