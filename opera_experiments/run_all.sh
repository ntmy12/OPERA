#!/bin/bash

echo "======================================"
echo " Starting OPERA Benchmark Pipeline..."
echo "======================================"

# Activate environment (uncomment and adjust if running directly)
# source ~/.bashrc
# conda activate opera_env

MODELS=("llava" "qwen2vl")
USE_OPERA_FLAGS=("" "--use_opera")

for model in "${MODELS[@]}"; do
    for flag in "${USE_OPERA_FLAGS[@]}"; do
        
        mode="baseline"
        if [ "$flag" == "--use_opera" ]; then
            mode="opera"
        fi
        
        echo "------------------------------------------------"
        echo " Running Model: $model | Mode: $mode"
        echo "------------------------------------------------"
        
        # 1. POPE
        echo "[1/3] Running POPE (Random split)..."
        cd benchmarks/pope
        python run_pope.py --model $model $flag --split random
        cd ../..
        
        # 2. CHAIR
        echo "[2/3] Running CHAIR..."
        cd benchmarks/chair
        python run_chair_captions.py --model $model $flag
        cd ../..
        
        # 3. BEAF
        echo "[3/3] Running BEAF..."
        cd benchmarks/beaf
        python run_beaf.py --model $model $flag
        cd ../..
        
    done
done

echo "======================================"
echo " All Benchmarks Finished!"
echo " Results are saved in the results/ directory."
echo " Use summarize_results.py to aggregate metrics."
echo "======================================"
