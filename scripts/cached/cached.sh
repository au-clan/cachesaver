#!/bin/bash

benchmark="hle"
method="cot"
split="test"
repeats=5

provider="openai"
model="gpt-4.1-nano"

# Decoding parameters
source scripts/configs/$benchmark.env

# Override MAX_COMPLETION_TOKENS if method is "io" or "cot"
if [[ "$method" == "io" || "$method" == "cot" ]]; then
  MAX_COMPLETION_TOKENS=10000
fi

python scripts/repeats/repeats.py \
    --benchmark "$benchmark" \
    --method "$method" \
    --model "$model" \
    --batch_size 1 \
    --timeout 2.0 \
    --temperature "$TEMPERATURE" \
    --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
    --top_p "$TOP_P" \
    --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
    --split "$split" \
    --correctness 1 \
    --allow_batch_overflow 1 \
    --ns_ratio 0.0 \
    --provider "$provider" \
    ${STOP:+--stop "$STOP"} \
    --value_cache \
    --repeats "$repeats"
