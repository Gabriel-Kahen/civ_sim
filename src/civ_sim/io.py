from __future__ import annotations

import json
from pathlib import Path

from civ_sim.sim import Simulation


def save_simulation(simulation: Simulation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = simulation.snapshot()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def load_simulation(path: str | Path) -> Simulation:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Simulation.from_snapshot(payload)
