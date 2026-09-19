import os
import argparse
import subprocess

def eval_beaf(results_dir, beaf_data_dir):
    results_file = os.path.join(results_dir, "raw_outputs.json")
    metric_script = os.path.join(beaf_data_dir, "beaf_metric.py")
    
    # We call the official beaf_metric.py from the dataset folder
    print(f"Calling official BEAF metric script: {metric_script}")
    
    # Normally, beaf_metric.py takes the predictions file as an argument.
    # E.g., python beaf_metric.py --pred_file raw_outputs.json
    # The exact arguments depend on their script, but we'll try the standard approach.
    cmd = ["python", metric_script, "--pred_file", results_file]
    
    try:
        # We redirect output to metrics.txt
        out_file = os.path.join(results_dir, "metrics.txt")
        with open(out_file, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"Metrics saved to {out_file}. Please check this file for TU, IAF, SF, UF scores.")
    except Exception as e:
        print(f"Error running beaf_metric.py: {e}")
        print("You might need to run beaf_metric.py manually on raw_outputs.json depending on its required arguments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--beaf_data_dir", type=str, required=True)
    args = parser.parse_args()
    
    eval_beaf(args.results_dir, args.beaf_data_dir)
