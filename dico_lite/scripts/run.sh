#!/usr/bin/env bash
set -e

RUN_TRAIN=${1:-"false"}

# Navigate to the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "\033[32mRunning DiCo-lite Comparative Experiment Suite\033[0m"

echo -e "\n\033[36m[1/2] Running Uniform Baseline...\033[0m"
python -m src.main --config configs/config.json --method uniform --run_train "$RUN_TRAIN"

echo -e "\n\033[36m[2/2] Running Module-DiCo-lite Baseline...\033[0m"
python -m src.main --config configs/config.json --method dico --run_train "$RUN_TRAIN"

echo -e "\n\033[32mComparative Suite Complete!\033[0m"
