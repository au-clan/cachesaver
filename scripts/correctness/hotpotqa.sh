#!/bin/bash

# Define benchmarks
benchmarks=("hotpotqa")

# Define methods
methods=("tot_bfs" "got" "react" "rap")

# Define models
provider="openai"
model="gpt-4.1-nano"

# Define number of retrials
retrials=5
split="test"

# Delete caches if they exist
for method in "${methods[@]}"; do
    FOLDER="caches/correctness/hotpotqa/${method}"
    if [ -d "$FOLDER" ]; then
        rm -rf "$FOLDER"
    fi
done

for benchmark in "${benchmarks[@]}"; do
    for method in "${methods[@]}"; do
        for ((i=1; i<=retrials; i++)); do
            echo "Running $benchmark with $method (trial $i/$retrials)"
            python "scripts/correctness/${benchmark}.py" \
                --provider "$provider" \
                --model "$model" \
                --batch_size 1 \
                --timeout 2.0 \
                --temperature 0.7 \
                --max_completion_tokens 300 \
                --top_p 1.0 \
                --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
                --split "$split" \
                --method "$method" \
                --conf_path "scripts/correctness/${benchmark}.yaml" \
                --correctness 1 \
                --value_cache

            python "scripts/correctness/${benchmark}.py" \
                --provider "$provider" \
                --model "$model" \
                --batch_size 300 \
                --timeout 2.0 \
                --temperature 0.7 \
                --max_completion_tokens 300 \
                --top_p 1.0 \
                --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
                --split "$split" \
                --method "$method" \
                --conf_path "scripts/correctness/${benchmark}.yaml" \
                --correctness 0 \
                --value_cache
        done
    done
done