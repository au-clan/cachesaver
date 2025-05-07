#!/bin/bash

# Source the base script
source "$(dirname "$0")/game24_base.sh"

# vLLM specific parameters
PROVIDER="openai"
BASE_URL="http://localhost:8000/v1"

# Models to test
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "unsloth/Llama-4-Scout-17B-16E-Instruct"
    "unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"
)

# Run all tests for all models
run_all_tests "$PROVIDER" "$BASE_URL" "${MODELS[@]}" 