#!/bin/bash

# Uncomment if you want the script to exit immediately if any command fails
# set -euo pipefail

RESULTS_DIR="../results"

# Task specification, passed via the -t flag.
# Accepts a single task or a comma-separated list (e.g. -t financial_sentiment,twitter_news).
TASKS="financial_sentiment,twitter_news"

# Model specification, passed via the -m flag.
# Accepts a single model or a comma-separated list (e.g. -m llama3.1:8b,gpt-oss:20b).
MODELS="llama3.1:8b,gpt-oss:20b"

# Seed specification, passed via the -s flag.
# Accepts a single seed (e.g. -s 42) or a range "start-end" (e.g. -s 0-9).
# Defaults to seed 0 if not specified.
SEEDS="0"

# Attack specification, passed via the -a flag.
# Accepts a single attack (e.g. -a attack1) or a comma-separated list
# (e.g. -a attack1,attack2).
ATTACKS=""

# Defence specification, passed via the -d flag.
# Accepts a single defence (e.g. -d defence1) or a comma-separated list
# (e.g. -d defence1,defence2).
DEFENCES=""

# Execution mode is derived from -a / -d, mirroring main.py's --mode:
#   - neither -a nor -d  -> clean mode         (--mode c)
#   - -a only            -> attack-only mode   (--mode a)
#   - -a and -d together -> attack+defence mode (--mode ad)
#   - -d only            -> invalid (defence requires an attack), falls back to clean mode
#
# Directory layout: "$RESULTS_DIR/$RUN_ID/<attack>_<defence>|<attack>|clean/<seed>/"

usage() {
    echo "Usage: $0 [-t <task[,task...]>] [-m <model[,model...]>] -s <seed | start-end> [-a <attack[,attack...]>] [-d <defence[,defence...]>]" >&2
    echo "  -t   Single task or comma-separated list of tasks. Default: financial_sentiment,twitter_news" >&2
    echo "  -m   Single model or comma-separated list of models. Default: llama3.1:8b,gpt-oss:20b" >&2
    echo "  -s   Single seed (e.g. 42) or seed range (e.g. 0-9). Default: 0" >&2
    echo "  -a   Single attack or comma-separated list of attacks (e.g. attack1,attack2)." >&2
    echo "  -d   Single defence or comma-separated list of defences (e.g. defence1,defence2)." >&2
    echo "  Neither -a nor -d          -> clean mode" >&2
    echo "  -a only                    -> attack-only mode" >&2
    echo "  -a and -d together         -> attack+defence mode" >&2
    exit 1
}

while getopts ":t:m:s:a:d:h" opt; do
    case "$opt" in
        t) TASKS="$OPTARG" ;;
        m) MODELS="$OPTARG" ;;
        s) SEEDS="$OPTARG" ;;
        a) ATTACKS="$OPTARG" ;;
        d) DEFENCES="$OPTARG" ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument" >&2; usage ;;
    esac
done

# Expand the SEEDS specification into a list of individual seed values
expand_seeds() {
    local spec="$1"
    if [[ "$spec" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    elif [[ "$spec" =~ ^[0-9]+$ ]]; then
        echo "$spec"
    else
        echo "Error: invalid seed format: '$spec'. Use a single number (e.g. 42) or a range (e.g. 0-9)." >&2
        exit 1
    fi
}

# Expand a comma-separated specification (tasks/models/attacks/defences) into a list of values.
expand_list() {
    local spec="$1"
    IFS=',' read -ra items <<< "$spec"
    for item in "${items[@]}"; do
        echo "$item" | xargs
    done
}

SEED_LIST=$(expand_seeds "$SEEDS")
mapfile -t TASK_LIST < <(expand_list "$TASKS")
mapfile -t MODEL_LIST < <(expand_list "$MODELS")

# Determine the execution mode from -a / -d, matching main.py's --mode choices.
if [[ -n "$DEFENCES" && -z "$ATTACKS" ]]; then
    echo "Warning: -d requires -a to also be specified. Falling back to clean mode; the provided -d value will be ignored." >&2
    MODE="c"
    ATTACK_LIST=("")
    DEFENCE_LIST=("")
elif [[ -n "$ATTACKS" && -n "$DEFENCES" ]]; then
    MODE="ad"
    mapfile -t ATTACK_LIST < <(expand_list "$ATTACKS")
    mapfile -t DEFENCE_LIST < <(expand_list "$DEFENCES")
elif [[ -n "$ATTACKS" ]]; then
    MODE="a"
    mapfile -t ATTACK_LIST < <(expand_list "$ATTACKS")
    DEFENCE_LIST=("")
else
    MODE="c"
    ATTACK_LIST=("")
    DEFENCE_LIST=("")
fi

for MODEL in "${MODEL_LIST[@]}"
do
    for TASK in "${TASK_LIST[@]}"
    do
        RUN_ID=$(date +"%Y-%m-%d_%H-%M-%S")
        RUN_ID="${RUN_ID}_${MODEL}_${TASK}"
        RUN_DIR="$RESULTS_DIR/$RUN_ID"

        for ATTACK in "${ATTACK_LIST[@]}"
        do
            for DEFENCE in "${DEFENCE_LIST[@]}"
            do
                EXTRA_ARGS=(--mode "$MODE")
                case "$MODE" in
                    c)
                        COMBO_DIR="clean"
                        ;;
                    a)
                        EXTRA_ARGS+=(--attack "$ATTACK")
                        COMBO_DIR="$ATTACK"
                        ;;
                    ad)
                        EXTRA_ARGS+=(--attack "$ATTACK" --defence "$DEFENCE")
                        COMBO_DIR="${ATTACK}_${DEFENCE}"
                        ;;
                esac

                COMBO_DIR_PATH="$RUN_DIR/$COMBO_DIR"

                for SEED in $SEED_LIST
                do
                    SEED_DIR="$COMBO_DIR_PATH/$SEED"
                    mkdir -p "$SEED_DIR"

                    echo "Running experiment for model: $MODEL, task: $TASK, mode: $MODE, attack: ${ATTACK:-none}, defence: ${DEFENCE:-none}, seed: $SEED"

                    python main.py \
                        --model "$MODEL" \
                        --task "$TASK" \
                        --seed "$SEED" \
                        "${EXTRA_ARGS[@]}" \
                        --log_dir "$COMBO_DIR_PATH" \
                        --run-id "$SEED" \
                        > "$SEED_DIR/experiment_${MODEL}_${TASK}_seed${SEED}.log" 2>&1

                done
            done
        done
    done
done

echo "All experiments completed successfully."