#!/bin/bash
namespace_fraction=(0.0 0.2 0.4 0.6 0.8 1.0)

# Define benchmarks
benchmark="game24"

# Define methods
methods=("got") # =("foa" "tot_bfs" "tot_dfs" "got" "rap" "react" "reflexion" "rafa" "rest_mcts")

# Define models
provider="openai"
model="gpt-4.1-nano"

# Define number of retrials
retrials=1
split="test"

# Delete caches if they exist
for method in "${methods[@]}"; do
    FOLDER="caches/namespace/game24/${method}"
    if [ -d "$FOLDER" ]; then
        rm -rf "$FOLDER"
    fi
done

for frac in "${namespace_fraction[@]}"; do
    for method in "${methods[@]}"; do
        for ((i=1; i<=retrials; i++)); do
            echo "Running $benchmark with $method and fraction $frac (trial $i/$retrials)"
            python "scripts/namespace/${benchmark}.py" \
                --provider "$provider" \
                --model "$model" \
                --batch_size 300 \
                --timeout 2.0 \
                --temperature 0.7 \
                --max_completion_tokens 200 \
                --top_p 1.0 \
                --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
                --split "$split" \
                --method "$method" \
                --conf_path "scripts/namespace/${benchmark}.yaml" \
                --correctness 0\
                --namespace_fraction $frac \
                --value_cache
        done
    done    
done
