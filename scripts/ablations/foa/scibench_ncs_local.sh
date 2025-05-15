#!/bin/bash


# Define benchmarks
benchmark="scibench"

# Define methods
method="foa" # =("foa" "tot_bfs" "tot_dfs" "got" "rap" "react" "reflexion" "rafa" "rest_mcts")

# Define models
provider="openai"
base_url="http://139.19.179.51:19999/v1"
# llama-4-meverick
# model="/DS/dsg-ml2/nobackup/cxu/weights/unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"

# llama-4-scout
model="/DS/dsg-ml/nobackup/cxu/weights/Llama-4-Scout-17B-16E-Instruct/"

# Define number of retrials
split="test"

# Delete caches if they exist
FLAG_NAME='ablations_foa_scibench_ncs_Resampling'
echo $base_url > $FLAG_NAME.time.log
echo "START $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log

echo "Resampling"
python "scripts/ablations/${method}/${benchmark}.py" \
    --provider "$provider" \
    --model "$model" \
    --base_url "$base_url" \
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
    --value_cache \
    --task "100" \
    --resampling "max_unique" \
    --backtrack 0.5 \
    --selection 1 

echo "END__ $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log


FLAG_NAME='ablations_foa_scibench_ncs_Backtracking'
echo $base_url > $FLAG_NAME.time.log
echo "START $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log


echo "Backtracking"
python "scripts/ablations/${method}/${benchmark}.py" \
    --provider "$provider" \
    --model "$model" \
    --base_url "$base_url" \
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
    --value_cache \
    --task "100" \
    --resampling "linear_filtered" \
    --backtrack 0 \
    --selection 1 

echo "END__ $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log


FLAG_NAME='ablations_foa_scibench_ncs_Selection'
echo $base_url > $FLAG_NAME.time.log
echo "START $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log


echo "Selection"
python "scripts/ablations/${method}/${benchmark}.py" \
    --provider "$provider" \
    --model "$model" \
    --base_url "$base_url" \
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
    --value_cache \
    --task "100" \
    --resampling "linear_filtered" \
    --backtrack 0.5 \
    --selection 0 

echo "END__ $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log
