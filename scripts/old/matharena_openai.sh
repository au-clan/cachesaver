# Shared NS
python scripts/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 400\
    --timeout 0.2\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --share_ns \
    --method tot\
    --conf_path scripts/matharena.yaml\
    --value_cache

python scripts/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 400\
    --timeout 0.2\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --share_ns \
    --method foa\
    --conf_path scripts/matharena.yaml\
    --value_cache

# No Shared NS
python scripts/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 400\
    --timeout 0.2\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --method tot\
    --conf_path scripts/matharena.yaml\
    --value_cache

python scripts/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 400\
    --timeout 0.2\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --method foa\
    --conf_path scripts/matharena.yaml\
    --value_cache

# Shared NS + Batch Size = 1
python scripts/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 1\
    --timeout 0.2\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --share_ns \
    --method tot\
    --conf_path scripts/matharena.yaml\
    --value_cache

python scripts/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 1\
    --timeout 0.2\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --share_ns \
    --method foa\
    --conf_path scripts/matharena.yaml\
    --value_cache