from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def process_bitod_data(file_path, all_columns=None):
    """Process BiToD JSON data to datasets.Dataset format with appropriate columns."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Initialize columns or use provided ones
    all_action_columns = set()
    all_state_columns = set()

    # First pass: collect all possible action and state columns
    for dialogue_id, dialogue_data in data.items():
        for event in dialogue_data.get("Events", []):
            if event["Agent"] not in ["User", "Wizard"]:
                continue

            # Process actions
            if isinstance(event.get("Actions", []), list):
                for action in event.get("Actions", []):
                    if isinstance(action, dict) and "act" in action:
                        act_name = action["act"]
                        slot_name = action.get("slot", "")
                        if slot_name:
                            all_action_columns.add(f"action-{act_name}-{slot_name}")
                        else:
                            all_action_columns.add(f"action-{act_name}")

            # Process state
            if "state" in event:
                for domain, slots in event["state"].items():
                    for slot_name in slots.keys():
                        all_state_columns.add(f"state-{domain}-{slot_name}")

    # Use provided columns if available
    if all_columns is not None:
        all_action_columns = all_columns["action"]
        all_state_columns = all_columns["state"]

    # Second pass: transform the dialogue data
    transformed_data = []

    for dialogue_id, dialogue_data in data.items():
        history = []

        for event in dialogue_data.get("Events", []):
            if event["Agent"] not in ["User", "Wizard"]:
                continue

            if event["Agent"] == "User":
                # Initialize entry with default values
                entry = {col: "none" for col in all_action_columns | all_state_columns}
                entry["dialogue_id"] = dialogue_id
                entry["text"] = event.get("Text", "")
                entry["history"] = history.copy()

                # Process actions
                if isinstance(event.get("Actions", []), list):
                    for action in event.get("Actions", []):
                        if isinstance(action, dict) and "act" in action:
                            act_name = action["act"]
                            slot_name = action.get("slot", "")

                            if slot_name:
                                column_name = f"action-{act_name}-{slot_name}"
                                # Safely handle empty lists
                                if isinstance(action.get("value", []), list):
                                    values = action.get("value", [])
                                    value = values[0] if values else ""
                                else:
                                    value = action.get("value", "")
                                entry[column_name] = str(value)
                            # else:
                            #     column_name = f"action-{act_name}"
                            #     # Safely handle empty lists
                            #     if isinstance(action.get("value", []), list):
                            #         values = action.get("value", [])
                            #         value = values[0] if values else ""
                            #     else:
                            #         value = action.get("value", "")
                            #     entry[column_name] = str(value)

                # Process state
                if "state" in event:
                    for domain, slots in event["state"].items():
                        for slot_name, slot_info in slots.items():
                            column_name = f"state-{domain}-{slot_name}"

                            relation = slot_info.get("relation", "")
                            values = slot_info.get("value", [])
                            value = values[0] if values else ""

                            if relation == "at_least":
                                entry[column_name] = f"<{value}"
                            elif relation == "equal_to":
                                entry[column_name] = str(value)
                            else:
                                entry[column_name] = f"{relation} {value}".strip()

                transformed_data.append(entry)
                history.append({"role": "user", "content": event.get("Text", "")})

            elif event["Agent"] == "Wizard" and "Text" in event:
                # Add wizard's response to history
                history.append({"role": "assistant", "content": event["Text"]})

    return Dataset.from_list(transformed_data), {"action": all_action_columns, "state": all_state_columns}


def load_bitod_data():
    """Load and process all BiToD data files."""
    base_path = Path("/home/samoed/Desktop/mteb/dialogmteb/datasets/dialogues/dialogues/bitod/data")
    all_datasets = {}

    for lang in ["en", "zh"]:
        print(f"\nProcessing {lang} data...")
        datasets = {}

        # First pass: collect all columns across splits
        all_action_columns = set()
        all_state_columns = set()

        for file_path in base_path.glob(f"{lang}_*.json"):
            if "fewshot" in file_path.stem:
                continue

            split_name = file_path.stem.split('_')[-1]
            if split_name not in ["test", "train", "valid"]:
                continue

            print(f"First pass collecting columns from {file_path.name}...")
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            for dialogue_id, dialogue_data in data.items():
                for event in dialogue_data.get("Events", []):
                    if event["Agent"] not in ["User", "Wizard"]:
                        continue

                    if isinstance(event.get("Actions", []), list):
                        for action in event.get("Actions", []):
                            if isinstance(action, dict) and "act" in action:
                                act_name = action["act"]
                                slot_name = action.get("slot", "")
                                if slot_name:
                                    all_action_columns.add(f"action-{act_name}-{slot_name}")
                                else:
                                    all_action_columns.add(f"action-{act_name}")

                    if "state" in event:
                        for domain, slots in event["state"].items():
                            for slot_name in slots.keys():
                                all_state_columns.add(f"state-{domain}-{slot_name}")

        all_columns = {"action": all_action_columns, "state": all_state_columns}

        # Second pass: create datasets with consistent columns
        for file_path in base_path.glob(f"{lang}_*.json"):
            if "fewshot" in file_path.stem:
                continue

            split_name = file_path.stem.split('_')[-1]
            if split_name not in ["test", "train", "valid"]:
                continue

            print(f"Processing {file_path.name}...")
            dataset, _ = process_bitod_data(file_path, all_columns)
            datasets[split_name] = dataset

            print(f"Created {split_name} dataset with {len(dataset)} examples")

        all_datasets[lang] = DatasetDict(datasets)

    return all_datasets


if __name__ == "__main__":
    # Process BiToD data
    all_datasets = load_bitod_data()

    # Push to hub
    for lang, ds in all_datasets.items():
        if ds:
            ds.push_to_hub(
                "DeepPavlov/BiToD",
                config_name=lang
            )
            print(f"Pushed {lang} configuration to hub")