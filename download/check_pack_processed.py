#!/usr/bin/env python
"""
Helper script to check if a pack has already been processed.
Usage: python check_pack_processed.py <checkpoints_file> <pack_name>
Returns: 1 if processed, 0 if not processed
"""

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("0")
        return 0
    
    checkpoints_file = sys.argv[1]
    pack_name = sys.argv[2]
    
    checkpoint_path = Path(checkpoints_file)
    if not checkpoint_path.exists():
        print("0")
        return 0
    
    try:
        with checkpoint_path.open("r", encoding="utf-8") as f:
            processed_packs = set(json.load(f))
        
        if pack_name in processed_packs:
            print("1")
            return 1
        else:
            print("0")
            return 0
    except Exception:
        print("0")
        return 0


if __name__ == "__main__":
    exit(main())
