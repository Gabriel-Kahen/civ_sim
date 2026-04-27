from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

from civ_sim.sim import Simulation


def _is_pickle_path(path: Path) -> bool:
    return path.suffix in {".pkl", ".pickle"}


def _is_compressed_pickle_path(path: Path) -> bool:
    return path.suffixes[-2:] in [[".pkl", ".gz"], [".pickle", ".gz"]]


def save_simulation(simulation: Simulation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_compressed_pickle_path(path):
        simulation.controller._param_cache.clear()
        with gzip.open(path, "wb", compresslevel=1) as handle:
            pickle.dump(simulation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    if _is_pickle_path(path):
        simulation.controller._param_cache.clear()
        with path.open("wb") as handle:
            pickle.dump(simulation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    payload = simulation.snapshot()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def load_simulation(path: str | Path) -> Simulation:
    path = Path(path)
    if _is_compressed_pickle_path(path):
        with gzip.open(path, "rb") as handle:
            simulation = pickle.load(handle)
        if not isinstance(simulation, Simulation):
            raise TypeError(f"pickle did not contain a Simulation: {path}")
        simulation.controller._param_cache.clear()
        return simulation

    if _is_pickle_path(path):
        with path.open("rb") as handle:
            simulation = pickle.load(handle)
        if not isinstance(simulation, Simulation):
            raise TypeError(f"pickle did not contain a Simulation: {path}")
        simulation.controller._param_cache.clear()
        return simulation

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Simulation.from_snapshot(payload)
