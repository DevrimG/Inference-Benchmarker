#!/usr/bin/env python3
"""Thin wrapper: run a fixed benchmark from a config file.

    python tools/run_fixed_benchmark.py --config examples/lab.example.toml

All logic lives in llm_load_tester.lab_cli; any of its flags may be passed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_load_tester.lab_cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--mode" not in args:
        args = ["--mode", "fixed", *args]
    sys.exit(main(args))
