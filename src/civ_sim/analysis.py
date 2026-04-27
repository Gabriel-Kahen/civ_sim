from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from civ_sim.constants import HARVESTABLE_TERRAIN_TO_RESOURCE, StructureType, TerrainType
from civ_sim.sim import Simulation


def lineage_color(lineage_id: int) -> tuple[int, int, int]:
    value = (lineage_id * 2654435761) & 0xFFFFFFFF
    return (
        40 + ((value >> 16) & 0x7F),
        50 + ((value >> 8) & 0x7F),
        60 + (value & 0x7F),
    )


def export_metrics(simulation: Simulation, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "metrics.csv"
    if not simulation.stats_history:
        return
    fieldnames = sorted({key for row in simulation.stats_history for key in row.keys() if not isinstance(row[key], dict)})
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in simulation.stats_history:
            flat = {key: value for key, value in row.items() if key in fieldnames}
            writer.writerow(flat)

    latest = simulation.stats_history[-1]
    with (output_dir / "latest_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(latest, handle, indent=2)

    with (output_dir / "lineages.json").open("w", encoding="utf-8") as handle:
        json.dump(lineage_summary(simulation), handle, indent=2)
    with (output_dir / "hubs.json").open("w", encoding="utf-8") as handle:
        json.dump(hub_summary(simulation), handle, indent=2)
    with (output_dir / "districts.json").open("w", encoding="utf-8") as handle:
        json.dump(district_summary(simulation), handle, indent=2)


def road_map(simulation: Simulation) -> np.ndarray:
    min_x, min_y, max_x, max_y = simulation.world.active_world_bounds()
    width = max_x - min_x
    height = max_y - min_y
    grid = np.zeros((height, width), dtype=np.uint8)
    for (world_x, world_y), structure in simulation.world.iter_active_structures():
        if structure.kind == StructureType.PATH:
            grid[world_y - min_y, world_x - min_x] = 255
    return grid


def frontier_map(simulation: Simulation) -> np.ndarray:
    min_x, min_y, max_x, max_y = simulation.world.active_world_bounds()
    width = max_x - min_x
    height = max_y - min_y
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            influence = simulation.world.influence_tiles.get((x, y), 0.0)
            terrain = simulation.world.terrain_at(x, y)
            active = simulation.world.chunk_coords(x, y) in simulation.world.active_chunks
            green = int(min(255, influence * 80))
            red = 180 if terrain == TerrainType.HAZARD else 40
            blue = 110 if active else 30
            grid[y - min_y, x - min_x] = (red, green, blue)
    return grid


def lineage_map(simulation: Simulation) -> np.ndarray:
    min_x, min_y, max_x, max_y = simulation.world.active_world_bounds()
    width = max_x - min_x
    height = max_y - min_y
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    for (chunk_x, chunk_y), chunk in simulation.world.active_chunks.items():
        world_x = chunk_x * simulation.config.chunk_size
        world_y = chunk_y * simulation.config.chunk_size
        for local_y in range(simulation.config.chunk_size):
            for local_x in range(simulation.config.chunk_size):
                lineage_id = int(chunk.lineage_map[local_y, local_x])
                if lineage_id == 0:
                    continue
                grid[world_y + local_y - min_y, world_x + local_x - min_x] = lineage_color(lineage_id)
    return grid


def district_map(simulation: Simulation) -> np.ndarray:
    min_x, min_y, max_x, max_y = simulation.world.active_world_bounds()
    width = max_x - min_x
    height = max_y - min_y
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    block = 4
    colors = {
        "food": (105, 178, 87),
        "wood": (66, 120, 76),
        "stone": (132, 129, 140),
        "mixed": (178, 160, 92),
    }
    for start_y in range(min_y, max_y, block):
        for start_x in range(min_x, max_x, block):
            totals = {"food": 0.0, "wood": 0.0, "stone": 0.0}
            for y in range(start_y, min(start_y + block, max_y)):
                for x in range(start_x, min(start_x + block, max_x)):
                    terrain = simulation.world.terrain_at(x, y)
                    amount = simulation.world.resource_amount_at(x, y)
                    if terrain == TerrainType.FERTILE:
                        totals["food"] += amount
                    elif terrain == TerrainType.FOREST:
                        totals["wood"] += amount
                    elif terrain == TerrainType.STONE:
                        totals["stone"] += amount
            dominant = max(totals, key=totals.get)
            total = sum(totals.values())
            label = dominant if total > 0.0 and totals[dominant] / total > 0.5 else "mixed"
            color = colors[label]
            grid[start_y - min_y : min(start_y + block, max_y) - min_y, start_x - min_x : min(start_x + block, max_x) - min_x] = color
    return grid


def lineage_summary(simulation: Simulation) -> list[dict]:
    lineages: dict[int, dict] = {}
    for agent in simulation.agents.values():
        entry = lineages.setdefault(
            agent.lineage_id,
            {
                "lineage_id": agent.lineage_id,
                "population": 0,
                "avg_energy": 0.0,
                "avg_generation": 0.0,
            },
        )
        entry["population"] += 1
        entry["avg_energy"] += agent.energy
        entry["avg_generation"] += agent.generation
    for entry in lineages.values():
        if entry["population"] > 0:
            entry["avg_energy"] /= entry["population"]
            entry["avg_generation"] /= entry["population"]
    return sorted(lineages.values(), key=lambda item: item["population"], reverse=True)


def hub_summary(simulation: Simulation) -> list[dict]:
    hubs = []
    for (world_x, world_y), structure in simulation.world.iter_active_structures():
        if structure.kind not in {StructureType.STORAGE, StructureType.WORKSHOP, StructureType.BEACON, StructureType.HOME}:
            continue
        traffic = 0.0
        path_neighbors = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                chunk = simulation.world.get_chunk_for_tile(world_x + dx, world_y + dy, activate=False)
                local_x, local_y = simulation.world.local_coords(world_x + dx, world_y + dy)
                traffic += float(chunk.traffic[local_y, local_x])
                neighbor = simulation.world.get_structure(world_x + dx, world_y + dy)
                if neighbor is not None and neighbor.kind == StructureType.PATH:
                    path_neighbors += 1
        hubs.append(
            {
                "x": world_x,
                "y": world_y,
                "kind": structure.kind.value,
                "lineage_id": structure.lineage_id,
                "traffic": traffic,
                "path_neighbors": path_neighbors,
                "inventory_total": 0.0 if structure.inventory is None else structure.inventory.total(),
            }
        )
    return sorted(hubs, key=lambda item: item["traffic"], reverse=True)[:32]


def district_summary(simulation: Simulation) -> list[dict]:
    min_x, min_y, max_x, max_y = simulation.world.active_world_bounds()
    districts = []
    block = 6
    for start_y in range(min_y, max_y, block):
        for start_x in range(min_x, max_x, block):
            resources = {"food": 0.0, "wood": 0.0, "stone": 0.0}
            structures = {}
            traffic = 0.0
            for y in range(start_y, min(start_y + block, max_y)):
                for x in range(start_x, min(start_x + block, max_x)):
                    terrain = simulation.world.terrain_at(x, y)
                    if terrain in HARVESTABLE_TERRAIN_TO_RESOURCE:
                        resource = HARVESTABLE_TERRAIN_TO_RESOURCE[terrain]
                        resources[resource.value] += simulation.world.resource_amount_at(x, y)
                    structure = simulation.world.get_structure(x, y)
                    if structure is not None:
                        structures[structure.kind.value] = structures.get(structure.kind.value, 0) + 1
                    chunk = simulation.world.get_chunk_for_tile(x, y, activate=False)
                    local_x, local_y = simulation.world.local_coords(x, y)
                    traffic += float(chunk.traffic[local_y, local_x])
            total_resource = sum(resources.values())
            dominant_resource = max(resources, key=resources.get) if total_resource > 0 else "mixed"
            districts.append(
                {
                    "x": start_x,
                    "y": start_y,
                    "dominant_resource": dominant_resource,
                    "resource_totals": resources,
                    "structures": structures,
                    "traffic": traffic,
                }
            )
    return sorted(districts, key=lambda item: item["traffic"], reverse=True)
