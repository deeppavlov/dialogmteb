from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def load_multi2woz_data(language_dir):
    """Load Multi2WOZ data for a specific language."""
    lang_code = language_dir.name  # 'ar', 'de', 'ru', 'zh'
    all_slots = set()
    datasets = {}

    # Get file suffix based on language
    suffix = "cn" if lang_code == "zh" else lang_code
    log_key = f"log-{suffix}"

    # Process each file in the language directory
    for file_path in language_dir.glob(f"*_full_{suffix}.json"):
        split_name = "test" if "test" in file_path.name else "validation"

        with file_path.open(encoding="utf-8") as f:
            data = json.load(f)

        # First pass: collect all possible slot names
        for dialogue_id, dialogue_data in data.items():
            for turn in dialogue_data.get(log_key, []):
                if "metadata" in turn:
                    for domain in ["taxi", "police", "restaurant", "bus",
                                   "hospital", "hotel", "attraction", "train"]:
                        if domain in turn["metadata"] and "semi" in turn["metadata"][domain]:
                            for slot_name in turn["metadata"][domain]["semi"].keys():
                                all_slots.add(f"{domain}-{slot_name}")
        all_slots = set(sorted(all_slots))

        # Second pass: transform the dialogue data
        transformed_data = []
        for dialogue_id, dialogue_data in data.items():
            history = []

            for turn_idx, turn in enumerate(dialogue_data.get(log_key, [])):
                # Process only user turns (even indices)
                if turn_idx % 2 == 0:
                    # Initialize entry with all slots as "none"
                    entry = {slot: "none" for slot in all_slots}
                    entry["dialogue_id"] = dialogue_id
                    entry["text"] = turn["text"]
                    entry["history"] = history.copy()

                    # User turns don't have metadata (metadata comes from system turns)
                    # We will process metadata in the next iteration

                    transformed_data.append(entry)
                    history.append({"role": "user", "content": turn["text"]})
                else:
                    # System turns: add to history and process metadata
                    if "text" in turn:
                        history.append({"role": "assistant", "content": turn["text"]})

                    # Process metadata from system turn
                    if "metadata" in turn:
                        # Get the entry we just added (for the previous user turn)
                        entry = transformed_data[-1]

                        for domain in ["taxi", "police", "restaurant", "bus",
                                       "hospital", "hotel", "attraction", "train"]:
                            if domain in turn["metadata"] and "semi" in turn["metadata"][domain]:
                                for slot_name, slot_value in turn["metadata"][domain]["semi"].items():
                                    full_slot_name = f"{domain}-{slot_name}"
                                    if slot_value and slot_value != "not mentioned":
                                        entry[full_slot_name] = slot_value

        # Create dataset for this split
        datasets[split_name] = Dataset.from_list(transformed_data)
        print(f"Created {split_name} dataset with {len(transformed_data)} examples for {lang_code}")

    return DatasetDict(datasets), sorted(all_slots)


if __name__ == "__main__":
    base_path = Path("/home/samoed/Desktop/mteb/dialogmteb/datasets/Multi2WOZ")
    all_configs = {}

    # Process each language directory
    for lang_path in base_path.glob("*"):
        lang_code = lang_path.name
        print(f"\nProcessing language: {lang_code}")

        # Load data for this language
        ds, slots = load_multi2woz_data(lang_path)
        print(f"Found {len(slots)} slots for language '{lang_code}'")

        # Save as configuration in dictionary
        all_configs[lang_code] = ds

    # Push all configurations to the hub as a single repository
    for lang_code, ds in all_configs.items():
        ds.push_to_hub(
            "DeepPavlov/Multi2WOZ",
            config_name=lang_code
        )
        print(f"Pushed {lang_code} configuration to hub")