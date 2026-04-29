from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import fields
from pathlib import Path

from civ_sim.config import SimConfig
from civ_sim.sim import Simulation


def _is_pickle_path(path: Path) -> bool:
    return path.suffix in {".pkl", ".pickle"}


def _is_compressed_pickle_path(path: Path) -> bool:
    return path.suffixes[-2:] in [[".pkl", ".gz"], [".pickle", ".gz"]]


def save_simulation(simulation: Simulation, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    simulation._observation_buffer = None
    simulation._observation_tensor = None
    if _is_compressed_pickle_path(path):
        simulation.controller._param_cache.clear()
        simulation.controller._numpy_param_cache.clear()
        simulation.controller._trait_cache.clear()
        with gzip.open(path, "wb", compresslevel=1) as handle:
            pickle.dump(simulation, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    if _is_pickle_path(path):
        simulation.controller._param_cache.clear()
        simulation.controller._numpy_param_cache.clear()
        simulation.controller._trait_cache.clear()
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
        return _prepare_loaded_simulation(simulation)

    if _is_pickle_path(path):
        with path.open("rb") as handle:
            simulation = pickle.load(handle)
        if not isinstance(simulation, Simulation):
            raise TypeError(f"pickle did not contain a Simulation: {path}")
        return _prepare_loaded_simulation(simulation)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _prepare_loaded_simulation(Simulation.from_snapshot(payload))


def _prepare_loaded_simulation(simulation: Simulation) -> Simulation:
    _upgrade_config(simulation.config)
    simulation.world.config = simulation.config
    simulation.world.generator.config = simulation.config
    if not hasattr(simulation.world.generator, "_patch_anchor_cache"):
        simulation.world.generator._patch_anchor_cache = {}
    if not hasattr(simulation.world.generator, "_region_profile_cache"):
        simulation.world.generator._region_profile_cache = {}
    if not hasattr(simulation.world, "_regen_masks"):
        simulation.world._regen_masks = {}
    if not hasattr(simulation.world, "_chunk_lookup_cache"):
        simulation.world._chunk_lookup_cache = {}
    simulation._observation_buffer = None
    simulation._observation_tensor = None
    if (
        not hasattr(simulation.world, "structure_positions")
        or not hasattr(simulation.world, "structures_by_id")
        or not hasattr(simulation.world, "structures_by_position")
        or not hasattr(simulation.world, "structure_ids_by_kind")
        or not hasattr(simulation.world, "inventory_structure_ids")
    ):
        simulation.world.reindex_structure_positions()
    else:
        simulation.world.reindex_structure_positions()
    simulation.controller.config = simulation.config
    if not hasattr(simulation.controller, "_trait_cache"):
        simulation.controller._trait_cache = {}
    if not hasattr(simulation.controller, "_numpy_param_cache"):
        simulation.controller._numpy_param_cache = {}
    simulation.controller._param_cache.clear()
    simulation.controller._numpy_param_cache.clear()
    simulation.controller._trait_cache.clear()
    return simulation


def _upgrade_config(config: SimConfig) -> None:
    defaults = SimConfig()
    for field in fields(SimConfig):
        if not hasattr(config, field.name):
            setattr(config, field.name, getattr(defaults, field.name))
