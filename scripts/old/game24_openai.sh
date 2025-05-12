#!/bin/bash

# Source the base script
source "$(dirname "$0")/game24_base.sh"

# OpenAI specific parameters
PROVIDER="openai"
BASE_URL=""  # Empty for default OpenAI endpoint

# Models to test
MODELS=(
    "gpt-4o-mini"
    "gpt-4o"
    "gpt-3.5-turbo"
)

# Run all tests for all models
run_all_tests "$PROVIDER" "$BASE_URL" "${MODELS[@]}" 