#!/usr/bin/env python3
"""Print agent replies that could not be read as JSON, to see what went wrong.

    python show_failures.py results/claude_tiny          # first 5 failures
    python show_failures.py results/claude_tiny 20       # first 20
"""
import glob
import json
import os
import sys

from masec.agents import parse_reply

folder = sys.argv[1] if len(sys.argv) > 1 else "results/mock"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
shown = total = 0
for path in sorted(glob.glob(os.path.join(folder, "traces", "*.json"))):
    with open(path) as f:
        for ev in json.load(f):
            if ev.get("event") != "agent" or parse_reply(ev["raw_output"])[2]:
                continue
            total += 1
            if shown < limit:
                shown += 1
                print(f"=== {os.path.basename(path)} | {ev['role']} ===")
                print(ev["raw_output"][:1500] or "(empty reply)")
                print()
print(f"{total} unreadable replies in {folder}" + (f" (showing {shown})" if total else ""))
