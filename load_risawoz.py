from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict


def extract_all_slots(data):
    """Extract all possible slots from the RiSAWOZ data for consistency."""
    all_slots = set()

    for dialogue in data:
        for turn in dialogue.get("dialogue", []):
            belief_state = turn.get("belief_state", [])
            for state in belief_state:
                for slot_pair in state.get("slots", []):
                    if len(slot_pair) == 2:
                        slot_name = slot_pair[0]
                        all_slots.add(slot_name)

    return sorted(list(all_slots))


def process_risawoz_data(file_path, all_slots):
    """Process RiSAWOZ JSON data to datasets.Dataset format."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Transform the dialogue data
    transformed_data = []

    for dialogue in data:
        dialogue_id = dialogue.get("dialogue_idx", "")
        domains = dialogue.get("domains", [])

        # Initialize dialogue history
        history = []

        # Process each turn in the dialogue
        for turn in dialogue.get("dialogue", []):
            turn_id = turn.get("turn_idx", 0)
            turn_domain = turn.get("domain", "")

            # Get user and system utterances
            user_text = turn.get("transcript", "")
            system_text = turn.get("system_transcript", "")

            # Create entry for this turn
            entry = {
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "domains": domains,
                "turn_domain": turn_domain,
                "text": user_text,
                "history": history.copy(),
                "user_acts": turn.get("user_acts", [])
            }

            # Add system acts if available
            if turn.get("system_acts"):
                entry["system_acts"] = turn.get("system_acts", [])

            # Initialize all slots with "not mentioned"
            for slot in all_slots:
                entry[slot] = "not mentioned"

            # Fill in the values for this turn from belief state
            belief_state = turn.get("belief_state", [])
            for state in belief_state:
                for slot_pair in state.get("slots", []):
                    if len(slot_pair) == 2:
                        slot_name = slot_pair[0]
                        slot_value = slot_pair[1]
                        entry[slot_name] = slot_value

            transformed_data.append(entry)

            # Update history with the current turn
            history.append({"role": "user", "content": user_text})

            # Add system response to history if available
            if system_text:
                history.append({"role": "assistant", "content": system_text})

    return Dataset.from_list(transformed_data)


def load_risawoz_data():
    """Load and process all RiSAWOZ data files."""
    base_path = Path("/home/samoed/Desktop/mteb/dialogmteb/datasets/RiSAWOZ/RiSAWOZ-data/task2-data-DST")

    # Define file mapping (filename -> split_name)
    file_mapping = {
        "dst_all_train10000.json": "train",
        "dst_all_dev600.json": "validation",
        "dst_all_test600_new.json": "test"
    }

    print("\nProcessing RiSAWOZ data...")

    # First pass: collect all possible slot names across all splits
    all_slots = set()

    for filename, split_name in file_mapping.items():
        file_path = base_path / filename
        if file_path.exists():
            print(f"First pass: collecting slots from {filename}...")
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            found_slots = extract_all_slots(data)
            all_slots.update(found_slots)

    all_slots = sorted(list(all_slots))
    print(f"Found {len(all_slots)} slots: {', '.join(all_slots[:5])}...")

    # Second pass: create datasets with consistent columns
    datasets = {}
    for filename, split_name in file_mapping.items():
        file_path = base_path / filename
        if file_path.exists():
            print(f"Second pass: processing {filename}...")
            dataset = process_risawoz_data(file_path, all_slots)
            datasets[split_name] = dataset
            print(f"Created {split_name} dataset with {len(dataset)} examples")

    return DatasetDict(datasets)


if __name__ == "__main__":
    # Process RiSAWOZ data
    dataset_dict = load_risawoz_data()

    # Push to hub
    dataset_dict.push_to_hub(
        "DeepPavlov/RISAWOZ",
        # config_name="zh"  # Chinese language configuration
    )
    print("Pushed RISAWOZ dataset to hub")