# Shared NS
python scripts/hotpotqa.py\
    --provider together\
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free\
    --batch_size 800\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 300\
    --top_p 1.0\
    --dataset_path datasets/dataset_hotpotqa.csv.gz\
    --split train\
    --share_ns \
    --method tot\
    --conf_path scripts/game24.yaml\
    --value_cache

python scripts/hotpotqa.py\
    --provider together\
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free\
    --batch_size 800\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 300\
    --top_p 1.0\
    --dataset_path datasets/dataset_hotpotqa.csv.gz\
    --split train\
    --share_ns \
    --method foa\
    --conf_path scripts/hotpotqa.yaml\
    --value_cache

# No Shared NS
python scripts/hotpotqa.py\
    --provider together\
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free\
    --batch_size 800\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 300\
    --top_p 1.0\
    --dataset_path datasets/dataset_hotpotqa.csv.gz\
    --split train\
    --method tot\
    --conf_path scripts/game24.yaml\
    --value_cache

python scripts/hotpotqa.py\
    --provider together\
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free\
    --batch_size 800\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 300\
    --top_p 1.0\
    --dataset_path datasets/dataset_hotpotqa.csv.gz\
    --split train\
    --method foa\
    --conf_path scripts/hotpotqa.yaml\
    --value_cache

# Shared NS + Batch Size = 1
python scripts/hotpotqa.py\
    --provider together\
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free\
    --batch_size 1\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 300\
    --top_p 1.0\
    --dataset_path datasets/dataset_hotpotqa.csv.gz\
    --split train\
    --share_ns \
    --method tot\
    --conf_path scripts/game24.yaml\
    --value_cache

python scripts/hotpotqa.py\
    --provider together\
    --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free\
    --batch_size 1\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 300\
    --top_p 1.0\
    --dataset_path datasets/dataset_hotpotqa.csv.gz\
    --split train\
    --share_ns \
    --method foa\
    --conf_path scripts/hotpotqa.yaml\
    --value_cache