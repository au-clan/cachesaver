#!/bin/bash


# Define benchmarks
benchmark="humaneval"

# Define methods
method="foa" # =("foa" "tot_bfs" "tot_dfs" "got" "rap" "react" "reflexion" "rafa" "rest_mcts")

# Define models
provider="openai"
model="gpt-4.1-nano"

# Define number of retrials
split="test"

# Delete caches if they exist

echo "Resampling"
python "scripts/ablations/${method}/${benchmark}.py" \
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
    --conf_path "scripts/old/${benchmark}.yaml" \
    --correctness 0 \
    --value_cache \
    --resampling "max_unique" \
    --backtrack 0.5 \
    --selection 1 

echo "Backtracking"
python "scripts/ablations/${method}/${benchmark}.py" \
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
    --conf_path "scripts/old/${benchmark}.yaml" \
    --correctness 0 \
    --value_cache \
    --resampling "linear_filtered" \
    --backtrack 0 \
    --selection 1 

echo "Resampling"
python "scripts/ablations/${method}/${benchmark}.py" \
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
    --conf_path "scripts/old/${benchmark}.yaml" \
    --correctness 0 \
    --value_cache \
    --resampling "linear_filtered" \
    --backtrack 0.5 \
    --selection 0 