"""Run the HuSkinDB benchmark inference pipeline from the project root."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skin_permeation.huskin import main


if __name__ == "__main__":
    raise SystemExit(main())
