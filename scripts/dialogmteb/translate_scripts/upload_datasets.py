#!/usr/bin/env python3
"""Upload every assembled translation to the Hub, one repo per dataset, one config per
(language, model, part) combination -- e.g. `translations/wow_final/es-gemma-corpus`
becomes config `es-gemma-corpus` in repo `dialogmteb/wow-translated`.

Discovers every `translations/*_final/` directory next to this script and pushes each
of its immediate subdirectories (each one a `save_to_disk`'d Dataset/DatasetDict) as a
config of `{org}/{task_name}-translated`, where `task_name` is the `*_final` directory
name with the `_final` suffix stripped.

This is a real, hard-to-reverse, publicly-visible action (creates/updates repos under
the `dialogmteb` org), so by default this only PRINTS the upload plan. Pass --push to
actually upload.

Usage:
    python upload_datasets.py                       # dry run: show what would be pushed
    python upload_datasets.py --push                 # actually upload everything found
    python upload_datasets.py --push --only wow statcan   # only these tasks
    python upload_datasets.py --push --org my-org    # different Hub org/user
    python upload_datasets.py --push --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSLATIONS_DIR = SCRIPT_DIR / "translations"


def discover_uploads(only: list[str] | None) -> list[tuple[str, str, Path]]:
    """Returns a list of (task_name, config_name, local_path)."""
    uploads = []
    for final_dir in sorted(TRANSLATIONS_DIR.glob("*_final")):
        if not final_dir.is_dir():
            continue
        task_name = final_dir.name.removesuffix("_final")
        if only and task_name not in only:
            continue
        for config_dir in sorted(final_dir.iterdir()):
            if config_dir.is_dir():
                uploads.append((task_name, config_dir.name, config_dir))
    return uploads


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true", help="actually upload (default is a dry run that only prints the plan)")
    parser.add_argument("--org", default="dialogmteb", help="Hub org/user prefix (default: dialogmteb)")
    parser.add_argument("--only", nargs="+", default=None, help="only upload these task names, e.g. --only wow statcan")
    parser.add_argument("--private", action="store_true", help="create/push repos as private")
    args = parser.parse_args()

    uploads = discover_uploads(args.only)
    if not uploads:
        print(f"nothing to upload -- no translations/*_final directories found under {TRANSLATIONS_DIR}")
        return

    print(f"{'Would push' if not args.push else 'Pushing'} {len(uploads)} configs:")
    for task_name, config_name, local_path in uploads:
        repo_id = f"{args.org}/{task_name}-translated"
        print(f"  {local_path.relative_to(SCRIPT_DIR)} -> {repo_id} (config={config_name})")

    if not args.push:
        print("\nDry run only -- pass --push to actually upload.")
        return

    from datasets import load_from_disk
    from huggingface_hub import whoami

    try:
        user = whoami()
    except Exception as e:  # noqa: BLE001
        print(f"\nNot logged in to the Hugging Face Hub ({e}). Run `huggingface-cli login` or set HF_TOKEN first.")
        sys.exit(1)
    print(f"\nLogged in as {user.get('name', user)}. Uploading...\n")

    failures = []
    for task_name, config_name, local_path in uploads:
        repo_id = f"{args.org}/{task_name}-translated"
        try:
            ds = load_from_disk(str(local_path))
            print(f"Pushing {local_path.relative_to(SCRIPT_DIR)} -> {repo_id} (config={config_name})...")
            ds.push_to_hub(repo_id, config_name=config_name, private=args.private)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            failures.append((repo_id, config_name, str(e)))

    print(f"\n{len(uploads) - len(failures)}/{len(uploads)} configs pushed successfully.")
    if failures:
        print("Failures:")
        for repo_id, config_name, err in failures:
            print(f"  {repo_id} (config={config_name}): {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
