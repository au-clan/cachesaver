#!/bin/bash


# Define benchmarks
methods=("foa" "tot_bfs" "got")

# Define methods
benchmark="game24"

# Define models
provider="openai"
model="gpt-4.1-nano"

# Define number of retrials
split="test"

# Delete caches if they exist

for method in "${methods[@]}"; do
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
done