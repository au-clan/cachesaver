#!/bin/bash
python "scripts/perms/game24.py" \
    --provider "openai" \
    --model "gpt-4.1-nano" \
    --batch_size 300 \
    --timeout 2.0 \
    --temperature 0.7 \
    --max_completion_tokens 200 \
    --top_p 1.0 \
    --dataset_path "datasets/dataset_game24.csv.gz" \
    --split "test" \
    --conf_path "scripts/perms/game24.yaml" \
    --correctness 0 \
    --value_cache