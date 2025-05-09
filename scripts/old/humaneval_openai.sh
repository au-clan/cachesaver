# Shared NS
python scripts/humaneval.py\
    --provider openai\
    --model gpt-4o-mini\
    --batch_size 800\
    --timeout 0.1\
    --temperature 0.7\
    --max_completion_tokens 4000\ 
    --top_p 1.0\
    --dataset_path datasets/humaneval-py-sorted.csv.gz\
    --split mini\
    --share_ns \
    --method got\
    --conf_path scripts/humaneval.yaml\
    --value_cache