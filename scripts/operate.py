#!/usr/bin/env python3
"""Cold-start a real open-model fidelity passport verification."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_model_fidelity_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
