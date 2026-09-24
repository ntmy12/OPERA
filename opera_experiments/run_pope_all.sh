#!/bin/bash
# ==============================================================================
# Script to run POPE benchmark (all 3 splits) on LLaVA or Qwen2-VL with OPERA
# ==============================================================================

set -e

MODEL=${1:-"llava"}           # Options: "llava" or "qwen2vl"
USE_OPERA=${2:-"--use_opera"}  # Pass "--use_opera" or leave empty for baseline
DTYPE=${3:-"bf16"}
MAX_SAMPLES=${4:-""}          # Optional: e.g. 10 for timing tests

EXTRA_ARGS=""
if [ -n "$MAX_SAMPLES" ]; then
    EXTRA_ARGS="--max_samples ${MAX_SAMPLES}"
fi

echo "=========================================================="
echo " Starting POPE Benchmark with OPERA"
echo " Model:          ${MODEL}"
echo " Mode:           ${USE_OPERA:-baseline}"
echo " Precision:      ${DTYPE}"
echo " Splits:         all (random, popular, adversarial)"
echo " Max New Tokens: 6 (enforced)"
echo " Decoding:       Greedy (do_sample=False, temp=0.0)"
if [ -n "$MAX_SAMPLES" ]; then
    echo " Max Samples:    ${MAX_SAMPLES} per split (timing benchmark)"
fi
echo "=========================================================="

python benchmarks/pope/run_pope.py \
    --model "${MODEL}" \
    ${USE_OPERA} \
    --split all \
    --max_new_tokens 6 \
    --dtype "${DTYPE}" \
    --device auto \
    --seed 42 \
    ${EXTRA_ARGS}

echo "=========================================================="
echo " POPE Benchmark Run Completed!"
echo "=========================================================="
