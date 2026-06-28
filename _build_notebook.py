#!/usr/bin/env python3
"""Generator for the experiments notebook — assembles 21 experiment sections."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "badania.ipynb"

with NB_PATH.open("r", encoding="utf-8") as fh:
    nb = json.load(fh)

# Keep the first 7 header cells intact, rebuild the rest
existing = nb["cells"][:7]
# Assign IDs to existing cells if missing
for i, cell in enumerate(existing):
    if "id" not in cell:
        cell["id"] = f"header_{i}"

# Collect experiment cells
cells = []


def md(cid, src):
    cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": src})


def code(cid, src):
    cells.append({"cell_type": "code", "id": cid, "metadata": {},
                  "execution_count": None, "outputs": [], "source": src})


# This file is generated programmatically; keeping it short by composing
# per-experiment factories.

ALL_BLOCKS = []  # Each block is tuple (intro_md, code_src, fig_src_or_None, outcome_md)

# Here we just assemble via function calls below.
</content>
