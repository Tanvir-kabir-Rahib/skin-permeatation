from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_table(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=False)
    elif destination.suffix.lower() in {".xlsx", ".xls"}:
        frame.to_excel(destination, index=False)
    else:
        raise ValueError(f"Unsupported table format for {destination}.")
