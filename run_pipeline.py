import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_phase(script_name):
    logging.info(f"========== Starting {script_name} ==========")
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        logging.info(f"========== Finished {script_name} ==========\n")
    except subprocess.CalledProcessError as e:
        logging.error(f"[ERROR] {script_name} failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    phases = [
        "phase1_data_profiling.py",
        "phase2_sequence_construction.py",
        "phase3_feature_engineering.py",
        "phase4_model_architecture.py",
        "phase5_evaluation.py",
        "phase6_preference_alignment.py",
        "phase6_rlhf.py",
        "phase7_reporting_layer.py",
        "phase7_reporting.py",
        "robustness_audit.py"
    ]
    
    for phase in phases:
        run_phase(phase)
        
    logging.info("ALL PHASES COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
