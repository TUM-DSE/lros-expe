#!/usr/bin/env python3
"""Regenerate every mock figure.

Usage: mock_all.py [OUT_DIR]

OUT_DIR defaults to $PLOT_OUT_DIR, else the paper's plots/mock folder. The
argument is passed on to each script unchanged, so running one of them alone
writes to the same place.

The mock scripts carry synthetic numbers so the paper can be laid out before
the measurements land; each one is replaced by a data-driven script as the
corresponding experiment starts producing results.
"""
import os
import runpy
import sys

SCRIPTS = [
    "mock_e2e.py",        # end-to-end performance
    "mock_accel.py",      # accelerators / VIAI
    "mock_memory.py",     # memory management
    "mock_sched.py",      # scheduler
    "mock_unikernel.py",  # unikernel microbenchmarks
]

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, here)

from common import parse_out_dir  # noqa: E402  (needs `here` on sys.path)

out_dir = parse_out_dir(description=__doc__)
print(f"Writing figures to {out_dir}")

for script in SCRIPTS:
    print(f"== {script}")
    # each script parses the same argv, so it picks up out_dir on its own
    runpy.run_path(os.path.join(here, script), run_name="__main__")
