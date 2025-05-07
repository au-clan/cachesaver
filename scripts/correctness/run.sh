#!/bin/bash

# Define benchmarks
benchmarks=("game24") # =("game24" "hotpotqa" "humaneval")

# Define methods
methods=("foa") # =("foa" "tot_bfs" "tot_dfs" "got" "rap" "react" "reflexion" "rafa" "rest_mcts")

# Define models
provider="openai"
model="gpt-4.1-nano"

# Define number of retrials
retrials=1

for benchmark in "${benchmarks[@]}"; do
    for method in "${methods[@]}"; do
        for ((i=1; i<=retrials; i++)); do
            echo "Running $benchmark with $method (trial $i/$retrials)"
            python "scripts/correctness/${benchmark}.py" \
                --provider "$provider" \
                --model "$model" \
                --batch_size 1 \
                --timeout 0.05 \
                --temperature 0.7 \
                --max_completion_tokens 200 \
                --top_p 1.0 \
                --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
                --split "single" \
                --method "$method" \
                --conf_path "scripts/correctness/${benchmark}.yaml" \
                --value_cache

            python "scripts/correctness/${benchmark}.py" \
                --provider "$provider" \
                --model "$model" \
                --batch_size 200 \
                --timeout 0.05 \
                --temperature 0.7 \
                --max_completion_tokens 200 \
                --top_p 1.0 \
                --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
                --split "single" \
                --method "$method" \
                --conf_path "scripts/correctness/${benchmark}.yaml" \
                --value_cache
        done
    done
done