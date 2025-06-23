# Shared NS
python scripts/old/matharena.py\
    --provider openai\
    --model gpt-4.1\
    --batch_size 100\
    --timeout 10.0\
    --temperature 0.5\
    --max_completion_tokens 750\
    --top_p 0.9\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split mini\
    --share_ns \
    --method got\
    --conf_path scripts/old/matharena.yaml\
    --value_cache

python scripts/old/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 400\
    --timeout 2.0\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --share_ns \
    --method foa\
    --conf_path scripts/old/matharena.yaml\
    --value_cache

# No Shared NS
python scripts/old/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 400\
    --timeout 2.0\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --method tot\
    --conf_path scripts/old/matharena.yaml\
    --value_cache

python scripts/old/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 400\
    --timeout 2.0\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --method foa\
    --conf_path scripts/old/matharena.yaml\
    --value_cache

# Shared NS + Batch Size = 1
python scripts/old/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 1\
    --timeout 2.0\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --share_ns \
    --method tot\
    --conf_path scripts/old/matharena.yaml\
    --value_cache

python scripts/old/matharena.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 1\
    --timeout 2.0\
    --temperature 0.8\
    --max_completion_tokens 500\
    --top_p 1.0\
    --dataset_path datasets/dataset_mathArena_aime2023.jsonl.gz\
    --split train\
    --share_ns \
    --method foa\
    --conf_path scripts/old/matharena.yaml\
    --value_cache