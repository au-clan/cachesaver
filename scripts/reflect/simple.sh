#!/bin/bash

benchmark="game24"
method="foa"
split="single"

provider="openai"
model="gpt-4.1-nano"

python scripts/simple/simple.py \
    --benchmark "$benchmark" \
    --method "$method" \
    --model "$model" \
    --batch_size 1 \
    --timeout 2.0 \
    --temperature 0.7 \
    --max_completion_tokens 200 \
    --top_p 1.0 \
    --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
    --split "$split" \
    --correctness 1 \
    --allow_batch_overflow 1 \
    --ns_ratio 0.0 \
    --provider "$provider" \
    --value_cache
