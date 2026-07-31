#!/usr/bin/env bash
# Runs every translate_*.py script's `translate` subcommand, once per model (gemma,
# qwen are loaded one at a time -- see translate_common.py for why two vLLM engines
# don't share a process), covering both languages (es, fr) in each run.
# tensor_parallel_size is set here (not baked into translate_common.py) so it's easy
# to tune per-machine without touching the scripts; other per-model engine kwargs
# (max_model_len, qwen's reasoning_parser, etc.) are defaults in MODELS there.
#
# Usage:
#   ./run_all.sh
#
# Safe to interrupt and re-run: every script's `translate` subcommand is checkpointed
# and skips already-translated items. If one script/model combination fails, this
# continues on to the rest and reports a summary of failures at the end.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

SCRIPTS=(
    translate_mantis.py
    translate_wow.py
    translate_statcan.py
    translate_clarqa.py
    translate_qrecc.py
    translate_coqa_abg.py
    translate_coral.py
    translate_ikat.py
    translate_clinc_oos.py
    translate_daily_dialog.py
    translate_faithdial.py
    translate_xrisawoz.py
    translate_multiwoz.py
    translate_air_dialogue.py
    translate_canard.py
    translate_topiocqa.py
)

MODELS=(gemma qwen)
LANGS=(es fr)
ENGINE_KWARGS=(--engine-kwarg tensor_parallel_size=2)

failures=()
total=0

for script in "${SCRIPTS[@]}"; do
    name="${script%.py}"
    name="${name#translate_}"
    for model in "${MODELS[@]}"; do
        total=$((total + 1))
        log_file="$LOG_DIR/${name}_${model}.log"
        echo "=== [$total] $name / $model -- logging to $log_file ==="
        if python3 "$SCRIPT_DIR/$script" translate --model "$model" --langs "${LANGS[@]}" "${ENGINE_KWARGS[@]}" \
            2>&1 | tee "$log_file"; then
            echo "=== [$total] $name / $model: done ==="
        else
            echo "=== [$total] $name / $model: FAILED (see $log_file) ==="
            failures+=("$name/$model")
        fi
    done
done

echo
echo "==================== summary ===================="
echo "$total runs attempted, ${#failures[@]} failed"
if [ "${#failures[@]}" -gt 0 ]; then
    printf ' - %s\n' "${failures[@]}"
    exit 1
fi
