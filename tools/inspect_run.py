#!/usr/bin/env python3
"""Quick CLI to inspect summary/session metadata from a run directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: inspect_run.py <run_dir>")
    run_dir = Path(sys.argv[1])
    for name in ["session_info.json", "summary.json"]:
        path = run_dir / name
        if path.exists():
            print(f"=== {name} ===")
            print(json.dumps(json.loads(path.read_text(encoding='utf-8')), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
