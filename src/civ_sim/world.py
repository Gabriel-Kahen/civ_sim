from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from civ_sim.config import SimConfig
from civ_sim.constants import (
    BLOCKING_STRUCTURES,
    RESOURCE_INDEX,
    INDEX_TO_TERRAIN,
    INVENTORY_STRUCTURES,
    PASSABLE_TERRAIN,
    StructureType,
    TerrainType,
)
from civ_sim.models import Chunk, Inventory, Structure
from civ_sim.worldgen import WorldGenerator


class World:
    def __init__(self, config: SimConfig):
        self.config = config
        self.generator = WorldGenerator(config)
        self.active_chunks: dict[tuple[int, int], Chunk] = {}
        self.dormant_chunks: dict[tuple[int, int], Chunk] = {}
        self.seed_chunks: dict[tuple[int, int], Chunk] = {}
        self.influence_tiles: dict[tuple[int, int], float] = {}
        self.next_structure_id = 1

    def initialize(self) -> None:
        radius = self.config.initial_active_radius_chunks
        for chunk_y in range(-radius, radius + 1):
            for chunk_x in range(-radius, radius + 1):
                self.ensure_chunk_active(chunk_x, chunk_y)

    def chunk_coords(self, x: int, y: int) -> tuple[int, int]:
        size = self.config.chunk_size
        return x // size, y // size

    def local_coords(self, x: int, y: int) -> tuple[int, int]:
        size = self.config.chunk_size
        return x % size, y % size

    def ensure_chunk_active(self, chunk_x: int, chunk_y: int) -> Chunk:
        key = (chunk_x, chunk_y)
        if key in self.active_chunks:
            return self.active_chunks[key]
        if key in self.dormant_chunks:
            chunk = self.dormant_chunks.pop(key)
            chunk.active = True
            self.active_chunks[key] = chunk
            return chunk
        if key in self.seed_chunks:
            chunk = self.seed_chunks.pop(key)
            chunk.active = True
            self.active_chunks[key] = chunk
            return chunk
        chunk = self.generator.generate_chunk(chunk_x, chunk_y)
        chunk.active = True
        self.active_chunks[key] = chunk
        return chunk

    def get_chunk_if_loaded(self, chunk_x: int, chunk_y: int) -> Chunk | None:
        return self.active_chunks.get((chunk_x, chunk_y)) or self.dormant_chunks.get((chunk_x, chunk_y))

    def get_chunk_for_tile(self, x: int, y: int, activate: bool = False) -> Chunk:
        chunk_x, chunk_y = self.chunk_coords(x, y)
        key = (chunk_x, chunk_y)
        if activate:
            return self.ensure_chunk_active(chunk_x, chunk_y)
        chunk = self.active_chunks.get(key)
        if chunk is not None:
            return chunk
        dormant = self.dormant_chunks.get(key)
        if dormant is not None:
            return dormant
        seed_chunk = self.seed_chunks.get(key)
        if seed_chunk is None:
            seed_chunk = self.generator.generate_chunk(chunk_x, chunk_y)
            seed_chunk.active = False
            self.seed_chunks[key] = seed_chunk
        return seed_chunk

    def terrain_at(self, x: int, y: int) -> TerrainType:
        chunk = self.get_chunk_for_tile(x, y, activate=False)
        local_x, local_y = self.local_coords(x, y)
        return INDEX_TO_TERRAIN[int(chunk.terrain[local_y, local_x])]

    def resource_amount_at(self, x: int, y: int) -> float:
        chunk = self.get_chunk_for_tile(x, y, activate=False)
        local_x, local_y = self.local_coords(x, y)
        return float(chunk.resource_amount[local_y, local_x])

    def resource_quality_at(self, x: int, y: int) -> float:
        chunk = self.get_chunk_for_tile(x, y, activate=False)
        local_x, local_y = self.local_coords(x, y)
        return float(chunk.resource_quality[local_y, local_x])

    def hazard_at(self, x: int, y: int) -> float:
        chunk = self.get_chunk_for_tile(x, y, activate=False)
        local_x, local_y = self.local_coords(x, y)
        return float(chunk.hazard[local_y, local_x])

    def modify_resource(self, x: int, y: int, delta: float) -> float:
        chunk = self.get_chunk_for_tile(x, y, activate=True)
        local_x, local_y = self.local_coords(x, y)
        current = float(chunk.resource_amount[local_y, local_x])
        updated = max(0.0, current + delta)
        chunk.resource_amount[local_y, local_x] = updated
        return updated - current

    def get_structure(self, x: int, y: int) -> Structure | None:
        chunk = self.get_chunk_for_tile(x, y, activate=False)
        local_x, local_y = self.local_coords(x, y)
        return chunk.structures.get((local_x, local_y))

    def add_structure(self, kind: StructureType, x: int, y: int, lineage_id: int, tick: int) -> Structure:
        chunk = self.get_chunk_for_tile(x, y, activate=True)
        local_x, local_y = self.local_coords(x, y)
        inventory = Inventory() if kind in INVENTORY_STRUCTURES else None
        structure = Structure(
            structure_id=self.next_structure_id,
            kind=kind,
            x=local_x,
            y=local_y,
            lineage_id=lineage_id,
            health=self.config.structure_max_health(kind),
            max_health=self.config.structure_max_health(kind),
            created_tick=tick,
            inventory=inventory,
        )
        self.next_structure_id += 1
        chunk.structures[(local_x, local_y)] = structure
        return structure

    def remove_structure(self, x: int, y: int) -> Structure | None:
        chunk = self.get_chunk_for_tile(x, y, activate=False)
        local_x, local_y = self.local_coords(x, y)
        return chunk.structures.pop((local_x, local_y), None)

    def structure_inventory(self, x: int, y: int) -> Inventory | None:
        structure = self.get_structure(x, y)
        return None if structure is None else structure.inventory

    def is_passable(self, x: int, y: int) -> bool:
        terrain = self.terrain_at(x, y)
        if terrain not in PASSABLE_TERRAIN:
            return False
        structure = self.get_structure(x, y)
        if structure is None:
            return True
        return structure.kind not in BLOCKING_STRUCTURES

    def move_cost_multiplier(self, x: int, y: int) -> float:
        structure = self.get_structure(x, y)
        if structure is not None and structure.kind == StructureType.PATH:
            return self.config.path_move_cost
        terrain = self.terrain_at(x, y)
        if terrain == TerrainType.FOREST:
            return self.config.forest_move_cost
        if terrain == TerrainType.STONE:
            return self.config.stone_move_cost
        if terrain == TerrainType.HAZARD:
            return self.config.hazard_move_cost
        return 1.0

    def mark_traffic(self, x: int, y: int, lineage_id: int) -> None:
        chunk = self.get_chunk_for_tile(x, y, activate=True)
        local_x, local_y = self.local_coords(x, y)
        chunk.traffic[local_y, local_x] += 1.0
        chunk.lineage_map[local_y, local_x] = lineage_id

    def ground_resource_vector(self, x: int, y: int, activate: bool = False) -> np.ndarray:
        chunk = self.get_chunk_for_tile(x, y, activate=activate)
        local_x, local_y = self.local_coords(x, y)
        return chunk.ground_resources[local_y, local_x]

    def add_ground_resource(self, x: int, y: int, resource, amount: float) -> float:
        chunk = self.get_chunk_for_tile(x, y, activate=True)
        local_x, local_y = self.local_coords(x, y)
        index = RESOURCE_INDEX[resource]
        chunk.ground_resources[local_y, local_x, index] += amount
        return float(chunk.ground_resources[local_y, local_x, index])

    def take_ground_resource(self, x: int, y: int, resource, amount: float) -> float:
        chunk = self.get_chunk_for_tile(x, y, activate=True)
        local_x, local_y = self.local_coords(x, y)
        index = RESOURCE_INDEX[resource]
        current = float(chunk.ground_resources[local_y, local_x, index])
        taken = min(current, amount)
        chunk.ground_resources[local_y, local_x, index] -= taken
        return taken

    def decay_buffers(self) -> None:
        for chunk in self.active_chunks.values():
            chunk.traffic *= self.config.traffic_decay

    def iter_active_structures(self) -> Iterable[tuple[tuple[int, int], Structure]]:
        for (chunk_x, chunk_y), chunk in self.active_chunks.items():
            for structure in chunk.structures.values():
                yield (
                    (
                        chunk_x * self.config.chunk_size + structure.x,
                        chunk_y * self.config.chunk_size + structure.y,
                    ),
                    structure,
                )

    def iter_loaded_structures(self) -> Iterable[tuple[tuple[int, int], Structure]]:
        for chunks in (self.active_chunks, self.dormant_chunks):
            for (chunk_x, chunk_y), chunk in chunks.items():
                for structure in chunk.structures.values():
                    yield (
                        (
                            chunk_x * self.config.chunk_size + structure.x,
                            chunk_y * self.config.chunk_size + structure.y,
                        ),
                        structure,
                    )

    def recompute_influence(self, agent_positions: set[tuple[int, int]]) -> dict[str, int]:
        self.influence_tiles = defaultdict(float)
        activated_chunks: set[tuple[int, int]] = set()
        for chunk in self.active_chunks.values():
            chunk.frontier_value.fill(0.0)

        radius = self.config.influence_radius
        for (world_x, world_y), structure in self.iter_active_structures():
            strength = self.config.structure_influence_strength(structure.kind)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    tx = world_x + dx
                    ty = world_y + dy
                    distance = (dx * dx + dy * dy) ** 0.5
                    if distance > radius:
                        continue
                    influence = max(0.0, strength * (1.0 - distance / (radius + 0.5)))
                    if influence <= 0.0:
                        continue
                    self.influence_tiles[(tx, ty)] += influence

        can_expand = self._expansion_density(agent_positions) >= self.config.expansion_density_threshold
        structure_count = sum(1 for _ in self.iter_active_structures())
        for (x, y), value in self.influence_tiles.items():
            chunk_x, chunk_y = self.chunk_coords(x, y)
            chunk = self.active_chunks.get((chunk_x, chunk_y))
            if chunk is not None:
                local_x, local_y = self.local_coords(x, y)
                chunk.frontier_value[local_y, local_x] = value
            elif (
                can_expand
                and structure_count >= self.config.expansion_min_structures
                and value >= self.config.influence_activation_threshold
                and self._has_active_neighbor(chunk_x, chunk_y)
            ):
                activated_chunks.add((chunk_x, chunk_y))

        for chunk_x, chunk_y in sorted(activated_chunks):
            self.ensure_chunk_active(chunk_x, chunk_y)

        removed_wild = 0
        sleeping_built = 0
        for key, chunk in list(self.active_chunks.items()):
            has_agents = any(self.chunk_coords(x, y) == key for x, y in agent_positions)
            if has_agents:
                continue
            frontier_max = float(chunk.frontier_value.max())
            if frontier_max >= self.config.dormancy_threshold:
                continue
            if chunk.structures:
                chunk.active = False
                self.dormant_chunks[key] = chunk
                del self.active_chunks[key]
                sleeping_built += 1
            else:
                del self.active_chunks[key]
                removed_wild += 1
        return {
            "activated": len(activated_chunks),
            "removed_wild": removed_wild,
            "sleeping_built": sleeping_built,
        }

    def _has_active_neighbor(self, chunk_x: int, chunk_y: int) -> bool:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (chunk_x + dx, chunk_y + dy) in self.active_chunks:
                return True
        return False

    def _expansion_density(self, agent_positions: set[tuple[int, int]]) -> float:
        structure_score = 0.0
        for _, structure in self.iter_active_structures():
            if structure.kind == StructureType.HOME:
                structure_score += 0.35
            elif structure.kind == StructureType.PATH:
                structure_score += 1.0
            else:
                structure_score += 1.4
        agent_score = len(agent_positions) * 0.08
        traffic_score = 0.0
        for chunk in self.active_chunks.values():
            traffic_score += float(chunk.traffic.sum()) * 0.002
        return (structure_score + agent_score + traffic_score) / max(1, len(self.active_chunks))

    def update_resources(self) -> None:
        for chunk in self.active_chunks.values():
            terrain = chunk.terrain
            fertile_mask = terrain == 3
            forest_mask = terrain == 1
            stone_mask = terrain == 2
            regen_mask = fertile_mask | forest_mask | stone_mask
            chunk.resource_amount[regen_mask] += self.config.resource_regen_rate * chunk.resource_quality[regen_mask]
            drift = (
                np.sin((chunk.chunk_x + 17) * 0.17 + chunk.resource_quality * 0.3)
                + np.cos((chunk.chunk_y - 11) * 0.19 + chunk.resource_amount * 0.07)
            )
            chunk.resource_quality[regen_mask] = np.clip(
                chunk.resource_quality[regen_mask] + drift[regen_mask] * self.config.quality_drift_strength,
                0.25,
                1.75,
            )

    def active_world_bounds(self) -> tuple[int, int, int, int]:
        if not self.active_chunks:
            return (0, 0, 0, 0)
        size = self.config.chunk_size
        xs = [chunk_x for chunk_x, _ in self.active_chunks.keys()]
        ys = [chunk_y for _, chunk_y in self.active_chunks.keys()]
        min_x = min(xs) * size
        min_y = min(ys) * size
        max_x = (max(xs) + 1) * size
        max_y = (max(ys) + 1) * size
        return min_x, min_y, max_x, max_y

    def snapshot(self) -> dict:
        return {
            "next_structure_id": self.next_structure_id,
            "active_chunks": [chunk.to_dict() for chunk in self.active_chunks.values()],
            "dormant_chunks": [chunk.to_dict() for chunk in self.dormant_chunks.values()],
        }

    @classmethod
    def from_snapshot(cls, config: SimConfig, payload: dict) -> "World":
        world = cls(config)
        world.next_structure_id = int(payload["next_structure_id"])
        for chunk_data in payload["active_chunks"]:
            chunk = Chunk.from_dict(chunk_data)
            world.active_chunks[(chunk.chunk_x, chunk.chunk_y)] = chunk
        for chunk_data in payload["dormant_chunks"]:
            chunk = Chunk.from_dict(chunk_data)
            world.dormant_chunks[(chunk.chunk_x, chunk.chunk_y)] = chunk
        return world
