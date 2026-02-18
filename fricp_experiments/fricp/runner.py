import subprocess
import os
import time
import numpy as np

class FRICPRunner:
    def __init__(self, exe_path):
        self.exe_path = os.path.abspath(exe_path)

    def run(self, target_ply, source_ply, output_dir, method=3, skip_if_exists=False):
        """
        Runs FRICP.exe and returns the time taken and the resulting transformation.
        If skip_if_exists is True and output files are present, returns existing results.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        trans_file = os.path.join(output_dir, f"m{method}trans.txt")
        time_file = os.path.join(output_dir, f"m{method}time.txt")
        reg_pc_file = os.path.join(output_dir, f"m{method}reg_pc.ply")

        if skip_if_exists and os.path.exists(trans_file) and os.path.exists(reg_pc_file):
            exec_time = 0.0
            if os.path.exists(time_file):
                try:
                    with open(time_file, 'r') as f:
                        exec_time = float(f.read().strip())
                except: pass
            
            transformation = np.loadtxt(trans_file)
            return {
                "time": exec_time,
                "transformation": transformation,
                "stdout": "Skipped (already exists)",
                "stderr": ""
            }

        cmd = [
            self.exe_path,
            os.path.abspath(target_ply),
            os.path.abspath(source_ply),
            os.path.abspath(output_dir) + os.sep,
            str(method)
        ]
        
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        end_time = time.time()
        
        exec_time = end_time - start_time
        
        # Save time for later skipping
        with open(time_file, 'w') as f:
            f.write(f"{exec_time}\n")
        
        # Parse transformation
        if os.path.exists(trans_file):
            transformation = np.loadtxt(trans_file)
        else:
            transformation = None
            
        # Parse stats from stdout
        iters = np.nan
        final_energy = np.nan
        for line in stdout.split('\n'):
            if "STATS:" in line:
                try:
                    parts = line.strip().split()
                    # Expected: STATS: iters=15 energy=0.00451
                    iters = int(parts[1].split('=')[1])
                    final_energy = float(parts[2].split('=')[1])
                except:
                    pass

        return {
            "time": exec_time,
            "transformation": transformation,
            "iters": iters,
            "final_energy": final_energy,
            "stdout": stdout,
            "stderr": stderr
        }
