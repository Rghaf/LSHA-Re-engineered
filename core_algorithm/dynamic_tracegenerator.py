import os
import random
import subprocess

class CustomTraceGenerator:
    def __init__(self, cs_name, resample_strategy, uppaal_bin_path, uppaal_model_path, uppaal_query_path, output_dir, trace_gen_config):
        self.cs_name = cs_name.replace(" ", "_")
        self.resample_strategy = resample_strategy
        self.uppaal_bin_path = uppaal_bin_path
        self.uppaal_model_path = uppaal_model_path
        self.uppaal_query_path = uppaal_query_path
        self.output_dir = output_dir
        
        self.config = trace_gen_config if trace_gen_config else {}
        self.max_e = int(self.config.get('max_length', 15))
        
        self.xml_force_var = self.config.get('xml_force_variable', 'force_open')
        self.xml_action_var = self.config.get('xml_action_variable', 'force_exe')

        self.word = None       
        self.events = []       
        self.evt_int = []  
        
    def set_word(self, w):
        self.word = w
        self.events = w.events
        self.evt_int = []

    def evts_to_ints(self):
        self.evt_int = []
        for e in self.events:
            if hasattr(e, 'trigger_value') and e.trigger_value is not None:
                self.evt_int.append(int(e.trigger_value))
            else:
                self.evt_int.append(-1)

    def get_evt_str(self):
        self.evts_to_ints()
        res = '{'
        i = 0
        for evt in self.evt_int:
            res += str(evt) + ', '
            i += 1
        while i < self.max_e - 1:
            res += '-1, '
            i += 1
        res += '-1};\n'
        return res

    def fix_model(self):
        print(f"[TraceGen] Fixing model at: {self.uppaal_model_path}")
        if not os.path.exists(self.uppaal_model_path):
            print(f"Error: Model file not found at {self.uppaal_model_path}")
            return

        with open(self.uppaal_model_path, 'r') as f:
            lines = f.readlines()

        values_str = self.get_evt_str()
        print(f"[TraceGen] Injecting Values into '{self.xml_force_var}'")
        
        tau_val = max(len(self.evt_int) * 50, 200)

        target_force_key = f"int {self.xml_force_var}[MAX_E] ="
        target_action_key = f"bool {self.xml_action_var} ="
        target_tau_key = "const int TAU ="

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(target_force_key):
                lines[i] = f"int {self.xml_force_var}[MAX_E] = {values_str}"
            elif stripped.startswith(target_action_key):
                lines[i] = f"bool {self.xml_action_var} = true;\n"
            elif stripped.startswith(target_tau_key):
                lines[i] = f"const int TAU = {tau_val};\n"

        with open(self.uppaal_model_path, 'w') as f:
            f.writelines(lines)
        print("[TraceGen] Model patched successfully.")

    def get_traces(self, n: int = 1):
        if self.resample_strategy == 'UPPAAL':
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