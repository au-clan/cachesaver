#!/bin/bash


# Define benchmarks
methods=("foa" "tot_bfs" "got")

# Define methods
benchmark="humaneval"

# Define models
provider="openai"
base_url="http://139.19.179.53:19999/v1"
# llama-4-meverick
# model="/DS/dsg-ml2/nobackup/cxu/weights/unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"

# llama-4-scout
model="/DS/dsg-ml/nobackup/cxu/weights/Llama-4-Scout-17B-16E-Instruct/"

# Define number of retrials
split="test"

# Delete caches if they exist

for method in "${methods[@]}"; do
    FLAG_NAME="frameworks_humaneval_ncs_${method}"
    echo $base_url > $FLAG_NAME.time.log
    echo "START $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log

    python "scripts/frameworks/${benchmark}/${benchmark}.py" \
        --provider "$provider" \
        --model "$model" \
        --batch_size 1 \
        --timeout 2.0 \
        --temperature 0.7 \
        --max_completion_tokens 200 \
        --top_p 1.0 \
        --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
        --split "$split" \
        --method "$method" \
        --conf_path "scripts/old/${benchmark}.yaml" \
        --correctness 1 \
        --value_cache

    echo "END__ $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log

done