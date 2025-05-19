#!/bin/bash
namespace_fraction=(0.4)

# Define benchmarks
benchmark="game24"

# Define methods
methods=("foa" "tot_bfs" "got") # =("foa" "tot_bfs" "tot_dfs" "got" "rap" "react" "reflexion" "rafa" "rest_mcts")

# Define models
provider="openai"
base_url="http://139.19.179.48:19999/v1"
# llama-4-meverick
# model="/DS/dsg-ml2/nobackup/cxu/weights/unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"

# llama-4-scout
model="/DS/dsg-ml/nobackup/cxu/weights/Llama-4-Scout-17B-16E-Instruct/"

# Define number of retrials
retrials=10
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

            FLAG_NAME="namespace_game24_${benchmark}_${frac}_${method}_${i}_s1"
            echo $base_url > $FLAG_NAME.time.log
            echo "START $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log

            python "scripts/namespace/${benchmark}.py" \
                --provider "$provider" \
                --model "$model" \
                --base_url "$base_url" \
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

            echo "END__ $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log

        done
    done    
done
