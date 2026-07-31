#!/usr/bin/env bash
# Runs every translate_*.py script's `assemble` subcommand. No vLLM/GPU needed --
# this just reads whatever translation checkpoints already exist and builds/saves the
# final datasets (translations/*_final/), skipping any script/language combo whose
# checkpoints aren't complete yet (each script reports exactly what's missing).
#
# Usage:
#   ./assemble_all.sh
#
# Safe to re-run at any time (e.g. after a `translate` run finishes for one more
# model/split). If one script fails outright, this continues on to the rest and
# reports a summary of failures at the end.

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

LANGS=(es fr)

failures=()
total=0

for script in "${SCRIPTS[@]}"; do
    name="${script%.py}"
    name="${name#translate_}"
    total=$((total + 1))
    log_file="$LOG_DIR/${name}_assemble.log"
    echo "=== [$total] $name assemble -- logging to $log_file ==="
    if python3 "$SCRIPT_DIR/$script" assemble --langs "${LANGS[@]}" \
        2>&1 | tee "$log_file"; then
        echo "=== [$total] $name assemble: done ==="
    else
        echo "=== [$total] $name assemble: FAILED (see $log_file) ==="
        failures+=("$name")
    fi
done

echo
echo "==================== summary ===================="
echo "$total assembles attempted, ${#failures[@]} failed"
if [ "${#failures[@]}" -gt 0 ]; then
    printf ' - %s\n' "${failures[@]}"
    exit 1
fi
