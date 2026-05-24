#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
run() {
    local script="$SCRIPT_DIR/$1"
    echo "=== $1 ==="
    uv run --no-sync "$script" "${@:2}"
    echo ""
}

cd "$ROOT"

run generate_dialogmteb_overview_table.py
run generate_dialogmteb_descriptive_stats.py
run generate_dialogmteb_model_table.py --all
run generate_dialogmteb_results_table.py
run generate_pertask_results_table.py
run generate_language_results_table.py
run generate_dataset_provenance_table.py
run generate_references_bib.py
run export_subset_results.py
run analyze_dialogmteb_task_correlations.py
run plot_dialog_vs_general.py
run plot_size_vs_performance.py
run plot_task_difficulty.py
run calculate_eval_time.py
run calculate_mteb_rank_correlation.py
run check_missing_results.py

echo "Done."
