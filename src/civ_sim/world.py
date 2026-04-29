from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterable

import numpy as np

from civ_sim.config import SimConfig
from civ_sim.constants import (
    BLOCKING_STRUCTURES,
    RESOURCE_INDEX,
    INDEX_TO_TERRAIN,
    INVENTORY_STRUCTURES,
    TERRAIN_INDEX,
    StructureType,
    TerrainType,
)
from civ_sim.models import Chunk, Inventory, Structure
from civ_sim.worldgen import WorldGenerator

PASSABLE_TERRAIN_INDICES = {
    TERRAIN_INDEX[TerrainType.GRASS],
    TERRAIN_INDEX[TerrainType.FOREST],
    TERRAIN_INDEX[TerrainType.STONE],
    TERRAIN_INDEX[TerrainType.FERTILE],
    TERRAIN_INDEX[TerrainType.HAZARD],
}
FOREST_INDEX = TERRAIN_INDEX[TerrainType.FOREST]
STONE_INDEX = TERRAIN_INDEX[TerrainType.STONE]
HAZARD_INDEX = TERRAIN_INDEX[TerrainType.HAZARD]


@lru_cache(maxsize=16)
def _influence_kernel(radius: int) -> tuple[tuple[int, int, float], ...]:
    return tuple(
        (dx, dy, max(0.0, 1.0 - ((dx * dx + dy * dy) ** 0.5) / (radius + 0.5)))
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if (dx * dx + dy * dy) ** 0.5 <= radius
    )


class World:
    def __init__(self, config: SimConfig):
        self.config = config
        self.generator = WorldGenerator(config)
        self.active_chunks: dict[tuple[int, int], Chunk] = {}
        self.dormant_chunks: dict[tuple[int, int], Chunk] = {}
        self.seed_chunks: dict[tuple[int, int], Chunk] = {}
        self.influence_tiles: dict[tuple[int, int], float] = {}
        self.next_structure_id = 1
        self.structure_positions: dict[int, tuple[int, int]] = {}
        self.structures_by_id: dict[int, Structure] = {}
        self.structures_by_position: dict[tuple[int, int], Structure] = {}
        self.structure_ids_by_kind: dict[StructureType, set[int]] = defaultdict(set)
        self.inventory_structure_ids: set[int] = set()
        self._regen_masks: dict[tuple[int, int], np.ndarray] = {}
        self._chunk_lookup_cache: dict[tuple[int, int], Chunk] = {}

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
            chunk = self.active_chunks[key]
            self._chunk_lookup_cache[key] = chunk
            return chunk
        if key in self.dormant_chunks:
            chunk = self.dormant_chunks.pop(key)
            chunk.active = True
            self.active_chunks[key] = chunk
            self._chunk_lookup_cache[key] = chunk
            return chunk
        if key in self.seed_chunks:
            chunk = self.seed_chunks.pop(key)
            chunk.active = True
            self.active_chunks[key] = chunk
            self._chunk_lookup_cache[key] = chunk
            return chunk
        chunk = self.generator.generate_chunk(chunk_x, chunk_y)
        chunk.active = True
        self.active_chunks[key] = chunk
        self._chunk_lookup_cache[key] = chunk
        return chunk

    def get_chunk_if_loaded(self, chunk_x: int, chunk_y: int) -> Chunk | None:
        return self.active_chunks.get((chunk_x, chunk_y)) or self.dormant_chunks.get((chunk_x, chunk_y))

    def _chunk_for_key(self, chunk_x: int, chunk_y: int, activate: bool = False) -> Chunk:
        key = (chunk_x, chunk_y)
        if activate:
            return self.ensure_chunk_active(chunk_x, chunk_y)
        cached = self._chunk_lookup_cache.get(key)
        if cached is not None:
            return cached
        chunk = self.active_chunks.get(key)
        if chunk is not None:
            self._chunk_lookup_cache[key] = chunk
            return chunk
        dormant = self.dormant_chunks.get(key)
        if dormant is not None:
            self._chunk_lookup_cache[key] = dormant
            return dormant
        seed_chunk = self.seed_chunks.get(key)
        if seed_chunk is None:
            seed_chunk = self.generator.generate_chunk(chunk_x, chunk_y)
            seed_chunk.active = False
            self.seed_chunks[key] = seed_chunk
        self._chunk_lookup_cache[key] = seed_chunk
        return seed_chunk

    def get_chunk_for_tile(self, x: int, y: int, activate: bool = False) -> Chunk:
        chunk_x, chunk_y = self.chunk_coords(x, y)
        return self._chunk_for_key(chunk_x, chunk_y, activate=activate)

    def is_active_tile(self, x: int, y: int) -> bool:
        return self.chunk_coords(x, y) in self.active_chunks

    def is_frontier_tile(self, x: int, y: int) -> bool:
        chunk_x, chunk_y = self.chunk_coords(x, y)
        return (chunk_x, chunk_y) not in self.active_chunks and self._has_active_neighbor(chunk_x, chunk_y)

    def can_enter_tile(self, x: int, y: int) -> bool:
        return self.is_passable(x, y)

    def activate_tile(self, x: int, y: int) -> None:
        chunk_x, chunk_y = self.chunk_coords(x, y)
        self.ensure_chunk_active(chunk_x, chunk_y)

    def chunk_and_local(self, x: int, y: int, activate: bool = False) -> tuple[Chunk, int, int]:
        size = self.config.chunk_size
        chunk_x, local_x = divmod(x, size)
        chunk_y, local_y = divmod(y, size)
        return self._chunk_for_key(chunk_x, chunk_y, activate=activate), local_x, local_y

    def terrain_at(self, x: int, y: int) -> TerrainType:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=False)
        return INDEX_TO_TERRAIN[int(chunk.terrain[local_y, local_x])]

    def resource_amount_at(self, x: int, y: int) -> float:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=False)
        return float(chunk.resource_amount[local_y, local_x])

    def resource_quality_at(self, x: int, y: int) -> float:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=False)
        return float(chunk.resource_quality[local_y, local_x])

    def hazard_at(self, x: int, y: int) -> float:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=False)
        return float(chunk.hazard[local_y, local_x])

    def modify_resource(self, x: int, y: int, delta: float) -> float:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=True)
        current = float(chunk.resource_amount[local_y, local_x])
        updated = max(0.0, current + delta)
        chunk.resource_amount[local_y, local_x] = updated
        return updated - current

    def get_structure(self, x: int, y: int) -> Structure | None:
        return self.structures_by_position.get((x, y))

    def add_structure(self, kind: StructureType, x: int, y: int, lineage_id: int, tick: int) -> Structure:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=True)
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
        self.structure_positions[structure.structure_id] = (x, y)
        self.structures_by_id[structure.structure_id] = structure
        self.structures_by_position[(x, y)] = structure
        self.structure_ids_by_kind[structure.kind].add(structure.structure_id)
        if structure.inventory is not None:
            self.inventory_structure_ids.add(structure.structure_id)
        return structure

    def remove_structure(self, x: int, y: int) -> Structure | None:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=False)
        structure = chunk.structures.pop((local_x, local_y), None)
        if structure is not None:
            self.structure_positions.pop(structure.structure_id, None)
            self.structures_by_id.pop(structure.structure_id, None)
            self.structures_by_position.pop((x, y), None)
            self.inventory_structure_ids.discard(structure.structure_id)
            ids = self.structure_ids_by_kind.get(structure.kind)
            if ids is not None:
                ids.discard(structure.structure_id)
        return structure

    def structure_by_id(self, structure_id: int) -> Structure | None:
        structure = self.structures_by_id.get(structure_id)
        if structure is not None:
            return structure
        self.reindex_structure_positions()
        return self.structures_by_id.get(structure_id)

    def structure_position(self, structure: Structure) -> tuple[int, int] | None:
        position = self.structure_positions.get(structure.structure_id)
        if position is not None:
            return position
        return self.reindex_structure_positions().get(structure.structure_id)

    def reindex_structure_positions(self) -> dict[int, tuple[int, int]]:
        positions: dict[int, tuple[int, int]] = {}
        structures: dict[int, Structure] = {}
        by_position: dict[tuple[int, int], Structure] = {}
        ids_by_kind: dict[StructureType, set[int]] = defaultdict(set)
        inventory_ids: set[int] = set()
        for (chunk_x, chunk_y), chunk in {**self.active_chunks, **self.dormant_chunks}.items():
            base_x = chunk_x * self.config.chunk_size
            base_y = chunk_y * self.config.chunk_size
            for structure in chunk.structures.values():
                positions[structure.structure_id] = (base_x + structure.x, base_y + structure.y)
                structures[structure.structure_id] = structure
                by_position[(base_x + structure.x, base_y + structure.y)] = structure
                ids_by_kind[structure.kind].add(structure.structure_id)
                if structure.inventory is not None:
                    inventory_ids.add(structure.structure_id)
        self.structure_positions = positions
        self.structures_by_id = structures
        self.structures_by_position = by_position
        self.structure_ids_by_kind = ids_by_kind
        self.inventory_structure_ids = inventory_ids
        return positions

    def structure_inventory(self, x: int, y: int) -> Inventory | None:
        structure = self.get_structure(x, y)
        return None if structure is None else structure.inventory

    def is_passable(self, x: int, y: int) -> bool:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=False)
        terrain_index = int(chunk.terrain[local_y, local_x])
        if terrain_index not in PASSABLE_TERRAIN_INDICES:
            return False
        structure = self.get_structure(x, y)
        if structure is None:
            return True
        return structure.kind not in BLOCKING_STRUCTURES

    def move_cost_multiplier(self, x: int, y: int) -> float:
        structure = self.get_structure(x, y)
        if structure is not None and structure.kind == StructureType.PATH:
            return self.config.path_move_cost
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=False)
        terrain_index = int(chunk.terrain[local_y, local_x])
        if terrain_index == FOREST_INDEX:
            return self.config.forest_move_cost
        if terrain_index == STONE_INDEX:
            return self.config.stone_move_cost
        if terrain_index == HAZARD_INDEX:
            return self.config.hazard_move_cost
        return 1.0

    def mark_traffic(self, x: int, y: int, lineage_id: int) -> None:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=True)
        chunk.traffic[local_y, local_x] += 1.0
        chunk.lineage_map[local_y, local_x] = lineage_id

    def ground_resource_vector(self, x: int, y: int, activate: bool = False) -> np.ndarray:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=activate)
        return chunk.ground_resources[local_y, local_x]

    def add_ground_resource(self, x: int, y: int, resource, amount: float) -> float:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=True)
        index = RESOURCE_INDEX[resource]
        chunk.ground_resources[local_y, local_x, index] += amount
        return float(chunk.ground_resources[local_y, local_x, index])

    def take_ground_resource(self, x: int, y: int, resource, amount: float) -> float:
        chunk, local_x, local_y = self.chunk_and_local(x, y, activate=True)
        index = RESOURCE_INDEX[resource]
        current = float(chunk.ground_resources[local_y, local_x, index])
        taken = min(current, amount)
        chunk.ground_resources[local_y, local_x, index] -= taken
        return taken

    def decay_buffers(self, ticks: int = 1) -> None:
        decay = self.config.traffic_decay ** max(1, ticks)
        for chunk in self.active_chunks.values():
            chunk.traffic *= decay

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

    def iter_loaded_structures_in_bounds(
        self,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
    ) -> Iterable[tuple[tuple[int, int], Structure]]:
        size = self.config.chunk_size
        min_chunk_x = min_x // size
        max_chunk_x = (max_x - 1) // size
        min_chunk_y = min_y // size
        max_chunk_y = (max_y - 1) // size
        for chunks in (self.active_chunks, self.dormant_chunks):
            for chunk_y in range(min_chunk_y, max_chunk_y + 1):
                for chunk_x in range(min_chunk_x, max_chunk_x + 1):
                    chunk = chunks.get((chunk_x, chunk_y))
                    if chunk is None:
                        continue
                    base_x = chunk_x * size
                    base_y = chunk_y * size
                    for structure in chunk.structures.values():
                        world_x = base_x + structure.x
                        world_y = base_y + structure.y
                        if min_x <= world_x < max_x and min_y <= world_y < max_y:
                            yield (world_x, world_y), structure

    def recompute_influence(self, agent_positions: set[tuple[int, int]]) -> dict[str, int]:
        self.influence_tiles = defaultdict(float)
        activated_chunks: set[tuple[int, int]] = set()
        agent_chunks = {self.chunk_coords(x, y) for x, y in agent_positions}
        for chunk in self.active_chunks.values():
            chunk.frontier_value.fill(0.0)

        radius = self.config.influence_radius
        kernel = _influence_kernel(radius)
        active_structures = list(self.iter_active_structures())
        for (world_x, world_y), structure in active_structures:
            strength = self.config.structure_influence_strength(structure.kind)
            for dx, dy, factor in kernel:
                influence = strength * factor
                if influence > 0.0:
                    self.influence_tiles[(world_x + dx, world_y + dy)] += influence

        can_expand = self._expansion_density(agent_positions, active_structures) >= self.config.expansion_density_threshold
        structure_count = len(active_structures)
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
            if key in agent_chunks:
                continue
            frontier_max = float(chunk.frontier_value.max())
            if frontier_max >= self.config.dormancy_threshold:
                continue
            if chunk.structures:
                chunk.active = False
                self.dormant_chunks[key] = chunk
                del self.active_chunks[key]
                self._chunk_lookup_cache[key] = chunk
                sleeping_built += 1
            else:
                del self.active_chunks[key]
                self._chunk_lookup_cache.pop(key, None)
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

    def _expansion_density(
        self,
        agent_positions: set[tuple[int, int]],
        active_structures: Iterable[tuple[tuple[int, int], Structure]] | None = None,
    ) -> float:
        structure_score = 0.0
        if active_structures is None:
            active_structures = self.iter_active_structures()
        for _, structure in active_structures:
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

    def update_resources(self, ticks: int = 1) -> None:
        ticks = max(1, ticks)
        for chunk in self.active_chunks.values():
            terrain = chunk.terrain
            regen_key = (chunk.chunk_x, chunk.chunk_y)
            regen_mask = self._regen_masks.get(regen_key)
            if regen_mask is None:
                regen_mask = (terrain == 3) | (terrain == 1) | (terrain == 2)
                self._regen_masks[regen_key] = regen_mask
            quality = chunk.resource_quality[regen_mask]
            chunk.resource_amount[regen_mask] += self.config.resource_regen_rate * ticks * quality
            amount = chunk.resource_amount[regen_mask]
            drift = (
                np.sin((chunk.chunk_x + 17) * 0.17 + quality * 0.3)
                + np.cos((chunk.chunk_y - 11) * 0.19 + amount * 0.07)
            )
            chunk.resource_quality[regen_mask] = np.clip(
                quality + drift * self.config.quality_drift_strength * ticks,
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
        world.reindex_structure_positions()
        world._chunk_lookup_cache = {
            **world.active_chunks,
            **world.dormant_chunks,
        }
        return world
