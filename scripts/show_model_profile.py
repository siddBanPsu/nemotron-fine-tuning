#!/usr/bin/env python3
"""Print a pinned model profile for shell setup and troubleshooting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from nemotron_ft_lab.model_profiles import MODEL_PROFILES, get_model_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--tsv", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\n".join(sorted(MODEL_PROFILES)))
        return
    profile = get_model_profile(args.profile)
    if args.tsv:
        print(
            "\t".join(
                (profile.model_id, profile.revision, profile.megatron_checkpoint_name)
            )
        )
        return
    print(json.dumps(profile.as_dict(), indent=2))


if __name__ == "__main__":
    main()
