"""Export DialogMTEB task results broken down by split and hf_subset to CSV.

Output: rows = (task, task_type, split, hf_subset, languages), columns = models.
Values are main_score (0–1 scale).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.dialogmteb._common import OUT_DIR, cache, load_unique_tasks, RUN_MODELS


def main():
    tasks = load_unique_tasks()
    task_type_map = {t.metadata.name: t.metadata.type for t in tasks}
    print(f"Tasks: {len(tasks)}")

    results = cache.load_results(tasks=tasks, models=RUN_MODELS)
    print(f"Models with results: {len(results.model_results)}")

    rows: list[dict] = []
    for model_result in results.model_results:
        model_name = model_result.model_name
        for task_result in model_result.task_results:
            task_name = task_result.task_name
            task_type = task_type_map.get(task_name, "")

            for split, subset_list in task_result.scores.items():
                for entry in subset_list:
                    hf_subset = entry.get("hf_subset", "default")
                    languages = entry.get("languages", [])
                    rows.append(
                        {
                            "task": task_name,
                            "task_type": task_type,
                            "split": split,
                            "hf_subset": hf_subset,
                            "languages": "; ".join(languages)
                            if isinstance(languages, list)
                            else str(languages),
                            "model": model_name,
                            "main_score": entry.get("main_score"),
                        }
                    )

    long_df = pd.DataFrame(rows)

    wide_df = long_df.pivot_table(
        index=["task", "task_type", "split", "hf_subset", "languages"],
        columns="model",
        values="main_score",
        aggfunc="first",
    )
    wide_df.columns.name = None
    wide_df = wide_df.sort_index()

    out = OUT_DIR / "dialogmteb_subset_results.csv"
    wide_df.to_csv(out)
    print(f"\nRows (index entries): {len(wide_df)}")
    print(f"Model columns: {len(wide_df.columns)}")
    print(f"CSV written to {out}")

    print(f"\nUnique tasks  : {long_df['task'].nunique()}")
    print(f"Unique splits : {sorted(long_df['split'].unique())}")
    print(f"Unique subsets: {long_df['hf_subset'].nunique()}")


if __name__ == "__main__":
    main()
