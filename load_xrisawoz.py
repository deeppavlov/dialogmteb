from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

from datasets import Dataset, DatasetDict


def extract_all_slots(data):
    """Extract all possible slots from the data for consistency."""
    inform_slots = set()
    request_slots = set()

    for dialogue in data:
        for turn in dialogue.get("dialogue", []):
            belief_state = turn.get("belief_state", {})

            # Extract inform slots
            inform_values = belief_state.get("inform slot-values", {})
            for slot in inform_values:
                inform_slots.add(f"inform-{slot}")

            # Extract request slots
            request_values = belief_state.get("turn request", [])
            for slot in request_values:
                request_slots.add(f"request-{slot}")

    return inform_slots, request_slots


def process_xrisawoz_data(file_path, all_inform_slots, all_request_slots):
    """Process CrossRISAWOZ JSON data to datasets.Dataset format."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Transform the dialogue data
    transformed_data = []

    for dialogue in data:
        dialogue_id = dialogue.get("dialogue_id", "")
        domains = dialogue.get("domains", [])

        # Initialize dialogue history
        history = []

        # Process each turn in the dialogue
        for turn in dialogue.get("dialogue", []):
            turn_id = turn.get("turn_id", 0)
            turn_domain = turn.get("turn_domain", [])

            # Get user utterance
            user_text = turn.get("user_utterance", [""])[0] if isinstance(turn.get("user_utterance", []), list) else ""

            # Extract belief state information
            belief_state = turn.get("belief_state", {})
            inform_slot_values = belief_state.get("inform slot-values", {})
            turn_request = belief_state.get("turn request", [])

            # Create entry for this turn
            entry = {
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "domains": domains,
                "turn_domain": turn_domain,
                "text": user_text,
                "history": history.copy(),
                "user_actions": turn.get("user_actions", [])
            }

            # Add any system actions if available
            if turn.get("system_actions"):
                entry["system_actions"] = turn.get("system_actions", [])

            # Initialize all slots with None/False
            for slot in all_inform_slots:
                entry[slot] = "none"
            for slot in all_request_slots:
                entry[slot] = False

            # Fill in the values for this turn
            for domain_slot, value in inform_slot_values.items():
                entry[f"inform-{domain_slot}"] = value

            # Mark requested slots as True
            for request_slot in turn_request:
                slot_key = f"request-{request_slot}"
                if slot_key in all_request_slots:
                    entry[slot_key] = True

            transformed_data.append(entry)

            # Update history with the current turn
            history.append({"role": "user", "content": user_text})

            # Add system response to history if available
            system_text = turn.get("system_utterance", [""])[0] if isinstance(turn.get("system_utterance", []),
                                                                              list) else ""
            if system_text:
                history.append({"role": "assistant", "content": system_text})

    return Dataset.from_list(transformed_data)


def load_xrisawoz_data():
    """Load and process all CrossRISAWOZ data files."""
    base_path = Path("/home/samoed/Desktop/mteb/dialogmteb/datasets/dialogues/dialogues/risawoz/data/original")
    all_datasets = {}

    # Find all language configurations
    langs = set()
    for file_path in base_path.glob("*.json"):
        if "_" in file_path.stem:
            lang_code = file_path.stem.split('_')[0]
            langs.add(lang_code)

    # Process each language configuration
    for lang in langs:
        print(f"\nProcessing {lang} data...")

        # First pass: collect all possible slot names across all splits
        all_inform_slots = set()
        all_request_slots = set()

        for split_type in ["test", "valid", "fewshot"]:
            file_path = base_path / f"{lang}_{split_type}.json"
            if file_path.exists():
                print(f"First pass: collecting slots from {file_path.name}...")
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                inform, request = extract_all_slots(data)
                all_inform_slots.update(inform)
                all_request_slots.update(request)

        print(f"Found {len(all_inform_slots)} inform slots and {len(all_request_slots)} request slots")

        # Second pass: create datasets with consistent columns
        datasets = {}
        for split_type in ["test", "valid", "fewshot"]:
            file_path = base_path / f"{lang}_{split_type}.json"
            if file_path.exists():
                print(f"Second pass: processing {file_path.name}...")
                dataset = process_xrisawoz_data(file_path, all_inform_slots, all_request_slots)

                # Map fewshot to train for consistency
                split_name = "train" if split_type == "fewshot" else split_type
                datasets[split_name] = dataset
                print(f"Created {split_name} dataset with {len(dataset)} examples")

        if datasets:
            all_datasets[lang] = DatasetDict(datasets)

    return all_datasets


if __name__ == "__main__":
    # Process CrossRISAWOZ data
    all_datasets = load_xrisawoz_data()

    # Push to hub
    for lang, ds in all_datasets.items():
        if ds:
            ds.push_to_hub(
                "DeepPavlov/XRISAWOZ",
                config_name=lang
            )
            print(f"Pushed {lang} configuration to hub")