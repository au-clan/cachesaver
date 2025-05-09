#!/bin/bash

# Common parameters
DATASET="datasets/dataset_game24.csv.gz"
CONF_PATH="scripts/game24.yaml"
MAX_TOKENS=150
TOP_P=1.0
TIMEOUT=0.1

# Function to run a single test
run_test() {
    local method=$1
    local temperature=$2
    local batch_size=$3
    local share_ns=$4
    local provider=$5
    local model=$6
    local base_url=$7

    local ns_flag=""
    if [ "$share_ns" = "true" ]; then
        ns_flag="--share_ns"
    fi

    local base_url_flag=""
    if [ -n "$base_url" ]; then
        base_url_flag="--base_url $base_url"
    fi

    echo "Running $method (temperature=$temperature, batch_size=$batch_size, share_ns=$share_ns)..."
    python scripts/game24.py \
        --provider "$provider" \
        --model "$model" \
        $base_url_flag \
        --batch_size "$batch_size" \
        --timeout "$TIMEOUT" \
        --temperature "$temperature" \
        --max_completion_tokens "$MAX_TOKENS" \
        --top_p "$TOP_P" \
        --dataset_path "$DATASET" \
        --split train \
        $ns_flag \
        --method "$method" \
        --conf_path "$CONF_PATH" \
        --value_cache
}

# Function to run all tests for a given batch size
run_batch_tests() {
    local batch_size=$1
    local share_ns=$2
    local provider=$3
    local model=$4
    local base_url=$5

    echo "Running tests with batch_size=$batch_size, share_ns=$share_ns"
    echo "----------------------------------------"

    # TOT method with temperature 0.7
    run_test "tot" 0.7 "$batch_size" "$share_ns" "$provider" "$model" "$base_url"

    # FOA method with temperature 2.0
    run_test "foa" 2.0 "$batch_size" "$share_ns" "$provider" "$model" "$base_url"
}

# Function to run all test configurations for a single model
run_all_tests_for_model() {
    local provider=$1
    local model=$2
    local base_url=$3

    echo "Starting Game24 tests with $model"
    echo "========================================"

    # Run tests with shared namespace
    echo "Running tests with shared namespace..."
    run_batch_tests 800 "true" "$provider" "$model" "$base_url"
    run_batch_tests 1 "true" "$provider" "$model" "$base_url"

    # Run tests without shared namespace
    echo "Running tests without shared namespace..."
    run_batch_tests 800 "false" "$provider" "$model" "$base_url"
}

# Function to run all test configurations for multiple models
run_all_tests() {
    local provider=$1
    local base_url=$2
    shift 2  # Remove provider and base_url from arguments
    local models=("$@")  # Remaining arguments are models

    for model in "${models[@]}"; do
        run_all_tests_for_model "$provider" "$model" "$base_url"
    done
} 