"""Run the AMD optimization benchmark. See src/amdlora/bench.py for the options.

    python scripts/bench_amd.py --list
    python scripts/bench_amd.py --variants plain,eager,math,efficient,flash,hipblaslt,expandable
"""
from amdlora.bench import main

raise SystemExit(main())