#!/bin/bash


# Define benchmarks
benchmark="scibench"

# Define methods
method="got" # =("foa" "tot_bfs" "tot_dfs" "got" "rap" "react" "reflexion" "rafa" "rest_mcts")

# Define models
provider="openai"
base_url="http://139.19.179.52:19999/v1"
# llama-4-meverick
# model="/DS/dsg-ml2/nobackup/cxu/weights/unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"

# llama-4-scout
model="/DS/dsg-ml/nobackup/cxu/weights/Llama-4-Scout-17B-16E-Instruct/"

# Define number of retrials
split="test"

# Delete caches if they exist

FLAG_NAME='hyper_search_got_scibench'
echo $base_url > $FLAG_NAME.time.log
echo "START $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log

for num_selections in 5 3 1; do
    for num_steps in 8 6 4; do 
        for num_evaluations in 3 1 2; do
            echo "Running with num_selections: $num_selections, num_steps: $num_steps, num_evaluations: $num_evaluations"
            python "scripts/hyper_search/${method}/${benchmark}.py" \
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
                --conf_path "scripts/old/${benchmark}.yaml" \
                --correctness 0 \
                --value_cache \
                --task "100" \
                --num_selections "$num_selections" \
                --num_steps "$num_steps" \
                --num_evaluations "$num_evaluations" 
        done
    done
done

echo "END__ $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log