from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def transform_dialogue_data(input_json, all_slots):
    """Transform MultiWOZ-style dialogue data to a datasets.Dataset with simplified structure."""
    transformed_data = []

    for dialogue_entry in input_json:
        dialogue_id = dialogue_entry["dialogue_idx"]

        # Build conversation history from previous turns
        history = []
        for turn_idx, turn in enumerate(dialogue_entry["dialogue"]):
            # Create a new entry for each turn with all slots initialized to None
            entry = {slot: "none" for slot in all_slots}

            # Add dialogue ID
            entry["dialogue_id"] = dialogue_id

            if len(turn["system_transcript"]) > 0:
                history.append(
                    {"role": "assistant", "content": turn["system_transcript"]}
                )

            # Set current user query and history
            entry["history"] = history.copy()
            entry["text"] = turn["transcript"]

            # Extract all slot values from belief_state
            if "belief_state" in turn:
                for state in turn["belief_state"]:
                    if "slots" in state:
                        for slot in state["slots"]:
                            if len(slot) == 2:
                                slot_name, slot_value = slot
                                entry[slot_name] = slot_value

            history.append({"role": "user", "content": turn["transcript"]})

            transformed_data.append(entry)

    # Create a datasets.Dataset from the transformed data
    return Dataset.from_list(transformed_data)


if __name__ == "__main__":
    # Load the input JSON file
    # First pass: collect all possible slot categories
    for lang_path in Path("/home/samoed/Desktop/mteb/dialogmteb/datasets/Multi2WOZ/").glob("*"):
        all_slots = set()
        for split_path in lang_path.glob("*.json"):
            with split_path.open() as f:
                input_data = json.load(f)

            for dialogue_entry in input_data:
                for turn in dialogue_entry["dialogue"]:
                    if "belief_state" in turn:
                        for state in turn["belief_state"]:
                            if "slots" in state:
                                for slot in state["slots"]:
                                    if len(slot) == 2:
                                        all_slots.add(slot[0])
                                    else:
                                        raise ValueError(
                                            f"Unexpected slot format: {slot}. Expected a tuple of length 2."
                                        )
        all_slots = sorted(all_slots)

        ds = DatasetDict()
        for split_path in ["train", "dev", "test"]:
            with open(
                f"/home/samoed/Desktop/ToD-BERT/dialog_datasets/dialog_datasets/MultiWOZ-2.1/{split_path}_dials.json",
            ) as f:
                input_data = json.load(f)

            # Transform the data
            dataset = transform_dialogue_data(input_data, all_slots)

            # Print some info about the dataset
            print(f"Dataset created with {len(dataset)} examples")
            print(f"Features: {dataset.features}")
            print(f"First example: {dataset[0]}")

            ds[split_path] = dataset
            # Example of how to access the dataset
            print("\nExample slot values from first entry:")
            example = dataset[0]
            for key in example:
                if (
                    key not in ["history", "user query", "dialogue_id"]
                    and example[key] is not None
                ):
                    print(f"  {key}: {example[key]}")

        ds.push_to_hub("DeepPavlov/MultiWOZ-2.1")
