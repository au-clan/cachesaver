#!/bin/bash


# Define benchmarks
benchmark="game24"

# Define methods
method="tot_bfs" # =("foa" "tot_bfs" "tot_dfs" "got" "rap" "react" "reflexion" "rafa" "rest_mcts")

# Define models
provider="openai"
model="gpt-4.1-nano"

# Define number of retrials
split="test"

# Delete caches if they exist

for num_selections in 5 3 1; do
    for num_steps in 4; do # Game of 24
        for num_evaluations in 3 2 1; do
            echo "Running with num_selections: $num_selections, num_steps: $num_steps, num_evaluations: $num_evaluations"
            python "scripts/hyper_search/${method}/${benchmark}.py" \
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
                --conf_path "scripts/correctness/${benchmark}.yaml" \
                --correctness 1 \
                --value_cache \
                --num_selections "$num_selections" \
                --num_steps "$num_steps" \
                --num_evaluations "$num_evaluations" 
        done
    done
done