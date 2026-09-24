#!/bin/bash
# ==============================================================================
# Script to run CHAIR benchmark on LLaVA or Qwen2-VL with OPERA
# ==============================================================================

set -e

MODEL=${1:-"llava"}           # Options: "llava" or "qwen2vl"
USE_OPERA=${2:-"--use_opera"}  # Pass "--use_opera" or leave empty for baseline
DTYPE=${3:-"bf16"}
NUM_SAMPLES=${4:-"10"}         # Number of images to evaluate (default 10 for timing)

EXTRA_ARGS=""
if [ -n "$NUM_SAMPLES" ]; then
    EXTRA_ARGS="--num_samples ${NUM_SAMPLES}"
fi

echo "=========================================================="
echo " Starting CHAIR Benchmark with OPERA"
echo " Model:          ${MODEL}"
echo " Mode:           ${USE_OPERA:-baseline}"
echo " Precision:      ${DTYPE}"
echo " Num Samples:    ${NUM_SAMPLES} (average runtime profiling)"
echo " Max New Tokens: 128 (image captioning)"
echo " Decoding:       Greedy (do_sample=False, temp=0.0)"
echo "=========================================================="

python benchmarks/chair/run_chair.py \
    --model "${MODEL}" \
    ${USE_OPERA} \
    --max_new_tokens 128 \
    --dtype "${DTYPE}" \
    --device auto \
    --seed 2027 \
    ${EXTRA_ARGS}

echo "=========================================================="
echo " CHAIR Benchmark Run Completed!"
echo "=========================================================="
