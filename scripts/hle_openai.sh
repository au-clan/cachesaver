# Shared NS with ToT Algorithm
python scripts/hle.py \
    --provider openai \
    --model gpt-4o-mini \
    --batch_size 600 \
    --timeout 0.2 \
    --temperature 0.8 \
    --max_completion_tokens 400 \
    --top_p 1.0 \
    --dataset_path datasets/dataset_hle_sample_without_images.jsonl.gz \
    --split train \
    --share_ns \
    --method tot \
    --conf_path scripts/hle.yaml \
    --value_cache

# Shared NS with FoA Algorithm
python scripts/hle.py \
    --provider openai \
    --model gpt-4o-mini \
    --batch_size 600 \
    --timeout 0.2 \
    --temperature 0.8 \
    --max_completion_tokens 400 \
    --top_p 1.0 \
    --dataset_path datasets/dataset_hle_sample_without_images.jsonl.gz \
    --split train \
    --share_ns \
    --method foa \
    --conf_path scripts/hle.yaml \
    --value_cache

# No Shared NS with ToT Algorithm
python scripts/hle.py  \
    --provider openai \
    --model gpt-4o-mini \
    --batch_size 600 \
    --timeout 0.2 \
    --temperature 0.8 \
    --max_completion_tokens 400 \
    --top_p 1.0 \
    --dataset_path datasets/dataset_hle_sample_without_images.jsonl.gz \
    --split train \
    --method tot \
    --conf_path scripts/hle.yaml \
    --value_cache

# No Shared NS with FoA Algorithm
python scripts/hle.py \
    --provider openai \
    --model gpt-4o-mini \
    --batch_size 600 \
    --timeout 0.2 \
    --temperature 0.8 \
    --max_completion_tokens 400 \
    --top_p 1.0 \
    --dataset_path datasets/dataset_hle_sample_without_images.jsonl.gz \
    --split train \
    --method foa \
    --conf_path scripts/hle.yaml \
    --value_cache

# Shared NS + Batch Size = 1 with ToT Algorithm
python scripts/hle.py \
    --provider openai \
    --model gpt-4o-mini \
    --batch_size 1 \
    --timeout 0.2 \
    --temperature 0.8 \
    --max_completion_tokens 400 \
    --top_p 1.0 \
    --dataset_path datasets/dataset_hle_sample_without_images.jsonl.gz \
    --split train \
    --share_ns  \
    --method tot \
    --conf_path scripts/hle.yaml \
    --value_cache

# Shared NS + Batch Size = 1 with FoA Algorithm
python scripts/hle.py \
    --provider openai \
    --model gpt-4o-mini \
    --batch_size 1 \
    --timeout 0.2 \
    --temperature 0.8 \
    --max_completion_tokens 400 \
    --top_p 1.0 \
    --dataset_path datasets/dataset_hle_sample_without_images.jsonl.gz \
    --split train \
    --share_ns  \
    --method foa \
    --conf_path scripts/hle.yaml \
    --value_cache