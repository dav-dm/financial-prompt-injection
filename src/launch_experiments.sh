#!/bin/bash

set -euo pipefail

TASKS="financial_sentiment twitter_news"
MODELS="llama3.1:8b gpt-oss:20b"

RESULTS_DIR="../results"

for MODEL in $MODELS
do
    for TASK in $TASKS
    do
        RUN_ID=$(date +"%Y-%m-%d_%H-%M-%S")
        RUN_DIR="$RESULTS_DIR/$RUN_ID"

        mkdir -p "$RUN_DIR"

        echo "Running experiment for model: $MODEL and task: $TASK"

        python main.py \
            --model "$MODEL" \
            --task "$TASK" \
            --clean \
            --log_dir "$RESULTS_DIR" \
            --run-id "$RUN_ID" \
            > "$RUN_DIR/experiment_${MODEL}_${TASK}.log" 2>&1

    done
done

echo "All experiments completed successfully."