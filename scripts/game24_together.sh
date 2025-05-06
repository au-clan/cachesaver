#!/bin/bash

# Source the base script
source "$(dirname "$0")/game24_base.sh"

# Together specific parameters
PROVIDER="together"
BASE_URL=""  # Empty for default Together endpoint

# Models to test
MODELS=(
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
)

# Run all tests for all models
run_all_tests "$PROVIDER" "$BASE_URL" "${MODELS[@]}" 