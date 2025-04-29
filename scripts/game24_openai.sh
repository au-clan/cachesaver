# Shared NS
python scripts/game24.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 800\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 150\
    --top_p 1.0\
    --dataset_path datasets/dataset_game24.csv.gz\
    --split train\
    --share_ns \
    --method tot\
    --conf_path scripts/game24.yaml\
    --value_cache

python scripts/game24.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 800\
    --timeout 0.1\
    --temperature 2.0\
    --max_completion_tokens 150\
    --top_p 1.0\
    --dataset_path datasets/dataset_game24.csv.gz\
    --split train\
    --share_ns \
    --method foa\
    --conf_path scripts/game24.yaml\
    --value_cache

python scripts/game24.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 800\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 100\
    --top_p 1.0\
    --dataset_path datasets/dataset_game24.csv.gz\
    --split train\
    --share_ns \
    --method got\
    --conf_path scripts/game24.yaml\
    --value_cache

# No Shared NS
python scripts/game24.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 800\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 150\
    --top_p 1.0\
    --dataset_path datasets/dataset_game24.csv.gz\
    --split train\
    --method tot\
    --conf_path scripts/game24.yaml\
    --value_cache

python scripts/game24.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 800\
    --timeout 0.1\
    --temperature 2.0\
    --max_completion_tokens 150\
    --top_p 1.0\
    --dataset_path datasets/dataset_game24.csv.gz\
    --split train\
    --method foa\
    --conf_path scripts/game24.yaml\
    --value_cache

# Shared NS + Batch Size = 1
python scripts/game24.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 1\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 150\
    --top_p 1.0\
    --dataset_path datasets/dataset_game24.csv.gz\
    --split train\
    --share_ns \
    --method tot\
    --conf_path scripts/game24.yaml\
    --value_cache

python scripts/game24.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 1\
    --timeout 0.1\
    --temperature 2.0\
    --max_completion_tokens 150\
    --top_p 1.0\
    --dataset_path datasets/dataset_game24.csv.gz\
    --split train\
    --share_ns \
    --method foa\
    --conf_path scripts/game24.yaml\
    --value_cache