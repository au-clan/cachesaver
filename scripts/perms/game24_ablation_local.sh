#!/bin/bash
base_url="http://139.19.179.51:19999/v1"
# llama-4-meverick
# model="/DS/dsg-ml2/nobackup/cxu/weights/unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"

# llama-4-scout
model="/DS/dsg-ml/nobackup/cxu/weights/Llama-4-Scout-17B-16E-Instruct/"


FLAG_NAME="perms_game24_ablation"
echo $base_url > $FLAG_NAME.time.log
echo "START $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log

python "scripts/perms/game24_ablation.py" \
    --provider "openai" \
    --model "${model}" \
    --base_url "${base_url}" \
    --batch_size 300 \
    --timeout 2.0 \
    --temperature 0.7 \
    --max_completion_tokens 200 \
    --top_p 1.0 \
    --dataset_path "datasets/dataset_game24.csv.gz" \
    --split "test" \
    --conf_path "scripts/perms/game24.yaml" \
    --correctness 0 \
    --value_cache

echo "END__ $(date +'%Y%m%d_%H%M%S')" >> $FLAG_NAME.time.log
