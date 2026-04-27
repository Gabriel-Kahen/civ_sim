from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from civ_sim.config import SimConfig
from civ_sim.constants import INDEX_TO_TERRAIN, TERRAIN_INDEX, TerrainType
from civ_sim.models import Chunk
from civ_sim.noise import fractal_noise, hash_float, smoothstep, warped_noise


@dataclass(slots=True)
class PatchAnchor:
    x: float
    y: float
    terrain: TerrainType
    radius: float
    weight: float


@dataclass(slots=True)
class RegionProfile:
    moisture_bias: float
    roughness_bias: float
    fertility_bias: float
    danger_bias: float
    grass_weight: float
    forest_weight: float
    stone_weight: float
    fertile_weight: float
    water_weight: float
    hazard_weight: float


class WorldGenerator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.region_cell_size = config.chunk_size * 10
        self.patch_cell_size = config.chunk_size * 5
        self.max_patch_radius = 42.0

    def generate_chunk(self, chunk_x: int, chunk_y: int) -> Chunk:
        size = self.config.chunk_size
        terrain = np.zeros((size, size), dtype=np.int16)
        resource_amount = np.zeros((size, size), dtype=np.float32)
        resource_quality = np.zeros((size, size), dtype=np.float32)
        hazard = np.zeros((size, size), dtype=np.float32)
        anchors = self._collect_anchors(chunk_x, chunk_y)
        for local_y in range(size):
            for local_x in range(size):
                world_x = chunk_x * size + local_x
                world_y = chunk_y * size + local_y
                terrain_type, amount, quality, hazard_value = self._score_tile(world_x, world_y, anchors)
                terrain[local_y, local_x] = TERRAIN_INDEX[terrain_type]
                resource_amount[local_y, local_x] = amount
                resource_quality[local_y, local_x] = quality
                hazard[local_y, local_x] = hazard_value
        terrain = self._smooth_terrain(terrain)
        self._seed_starter_basin(chunk_x, chunk_y, terrain, resource_amount, resource_quality, hazard)
        return Chunk(
            chunk_x=chunk_x,
            chunk_y=chunk_y,
            terrain=terrain,
            resource_amount=resource_amount,
            resource_quality=resource_quality,
            hazard=hazard,
        )

    def _collect_anchors(self, chunk_x: int, chunk_y: int) -> list[PatchAnchor]:
        anchors: list[PatchAnchor] = []
        size = self.config.chunk_size
        min_world_x = chunk_x * size - int(self.max_patch_radius * 1.5)
        max_world_x = (chunk_x + 1) * size + int(self.max_patch_radius * 1.5)
        min_world_y = chunk_y * size - int(self.max_patch_radius * 1.5)
        max_world_y = (chunk_y + 1) * size + int(self.max_patch_radius * 1.5)
        min_cell_x = min_world_x // self.patch_cell_size
        max_cell_x = max_world_x // self.patch_cell_size
        min_cell_y = min_world_y // self.patch_cell_size
        max_cell_y = max_world_y // self.patch_cell_size
        for patch_cell_y in range(min_cell_y, max_cell_y + 1):
            for patch_cell_x in range(min_cell_x, max_cell_x + 1):
                anchors.extend(self._anchors_for_patch_cell(patch_cell_x, patch_cell_y))
        if abs(chunk_x) <= 3 and abs(chunk_y) <= 3:
            anchors.extend(self._starter_basin_anchors())
        return anchors

    def _anchors_for_patch_cell(self, patch_cell_x: int, patch_cell_y: int) -> list[PatchAnchor]:
        seed = self.config.seed
        base_x = patch_cell_x * self.patch_cell_size
        base_y = patch_cell_y * self.patch_cell_size
        center_x = base_x + self.patch_cell_size * 0.5
        center_y = base_y + self.patch_cell_size * 0.5
        profile = self._region_profile(center_x, center_y)
        moisture = self._macro_value(seed + 11, center_x, center_y, profile.moisture_bias)
        roughness = self._macro_value(seed + 23, center_x, center_y, profile.roughness_bias)
        danger = self._macro_value(seed + 31, center_x, center_y, profile.danger_bias)
        fertility = self._macro_value(seed + 47, center_x, center_y, profile.fertility_bias)

        anchor_count = 1 + int(hash_float(seed + 67, patch_cell_x, patch_cell_y) * 2.4)
        anchors: list[PatchAnchor] = []
        for index in range(anchor_count):
            local_x = hash_float(seed + 71 + index, patch_cell_x, patch_cell_y, index) * self.patch_cell_size
            local_y = hash_float(seed + 79 + index, patch_cell_x, patch_cell_y, index) * self.patch_cell_size
            selector = hash_float(seed + 89 + index, patch_cell_x, patch_cell_y, index)
            terrain = self._select_anchor_terrain(
                selector=selector,
                profile=profile,
                moisture=moisture,
                roughness=roughness,
                fertility=fertility,
                danger=danger,
            )
            patch_scale = 0.85 + hash_float(seed + 97 + index, patch_cell_x, patch_cell_y, index) * 0.55
            anchors.append(
                PatchAnchor(
                    x=base_x + local_x,
                    y=base_y + local_y,
                    terrain=terrain,
                    radius=9.0 + selector * 12.0 * patch_scale,
                    weight=0.9 + selector * 1.35,
                )
            )
        return anchors

    def _select_anchor_terrain(
        self,
        selector: float,
        profile: RegionProfile,
        moisture: float,
        roughness: float,
        fertility: float,
        danger: float,
    ) -> TerrainType:
        weights = {
            TerrainType.GRASS: max(0.05, profile.grass_weight + (1.0 - roughness) * 0.25 + (1.0 - danger) * 0.1),
            TerrainType.FOREST: max(0.05, profile.forest_weight + moisture * 0.55 - roughness * 0.08),
            TerrainType.STONE: max(0.05, profile.stone_weight + roughness * 0.65),
            TerrainType.FERTILE: max(0.05, profile.fertile_weight + fertility * 0.6 + moisture * 0.08),
            TerrainType.WATER: max(0.02, profile.water_weight + moisture * 0.35 - roughness * 0.05),
            TerrainType.HAZARD: max(0.01, profile.hazard_weight + danger * 0.5 + roughness * 0.06),
        }
        total = sum(weights.values())
        threshold = selector * total
        running = 0.0
        for terrain, weight in weights.items():
            running += weight
            if threshold <= running:
                return terrain
        return TerrainType.GRASS

    def _region_profile(self, world_x: float, world_y: float) -> RegionProfile:
        seed = self.config.seed
        region_x = int(np.floor(world_x / self.region_cell_size))
        region_y = int(np.floor(world_y / self.region_cell_size))
        moisture = fractal_noise(seed + 301, region_x, region_y, (4.0, 2.0))
        roughness = fractal_noise(seed + 313, region_x, region_y, (4.0, 2.0))
        fertility = fractal_noise(seed + 317, region_x, region_y, (4.0, 2.0))
        danger = fractal_noise(seed + 331, region_x, region_y, (4.0, 2.0))
        selector = hash_float(seed + 337, region_x, region_y)

        if danger > 0.72 and roughness > 0.45:
            return RegionProfile(0.0, 0.28, -0.08, 0.45, 0.22, 0.08, 0.24, 0.06, 0.02, 0.5)
        if moisture > 0.7 and fertility > 0.46:
            return RegionProfile(0.22, -0.08, 0.2, -0.15, 0.18, 0.3, 0.05, 0.3, 0.23, 0.02)
        if roughness > 0.68:
            return RegionProfile(-0.05, 0.35, -0.04, 0.08, 0.16, 0.1, 0.42, 0.08, 0.05, 0.16)
        if moisture > 0.6 and selector > 0.55:
            return RegionProfile(0.25, -0.04, 0.1, -0.05, 0.16, 0.34, 0.04, 0.17, 0.24, 0.02)
        if fertility > 0.56:
            return RegionProfile(0.08, -0.08, 0.26, -0.12, 0.26, 0.16, 0.05, 0.32, 0.05, 0.02)
        return RegionProfile(0.04, 0.02, 0.05, 0.0, 0.34, 0.18, 0.12, 0.14, 0.06, 0.04)

    def _macro_value(self, seed: int, world_x: float, world_y: float, bias: float = 0.0) -> float:
        return np.clip(fractal_noise(seed, world_x, world_y, (140.0, 72.0, 36.0)) + bias, 0.0, 1.0)

    def _score_tile(
        self,
        world_x: int,
        world_y: int,
        anchors: list[PatchAnchor],
    ) -> tuple[TerrainType, float, float, float]:
        seed = self.config.seed
        profile = self._region_profile(world_x, world_y)
        moisture = self._macro_value(seed + 101, world_x, world_y, profile.moisture_bias)
        roughness = self._macro_value(seed + 103, world_x, world_y, profile.roughness_bias)
        danger = self._macro_value(seed + 107, world_x, world_y, profile.danger_bias)
        fertility = self._macro_value(seed + 109, world_x, world_y, profile.fertility_bias)
        noise = warped_noise(seed + 127, world_x, world_y, 22.0, 48.0)
        river_band = warped_noise(seed + 211, world_x, world_y, 44.0, 120.0)
        river_strength = max(0.0, 1.0 - abs(river_band - 0.52) / 0.055)
        wetland_strength = max(0.0, 1.0 - abs(river_band - 0.52) / 0.11)

        scores = {
            TerrainType.GRASS: 0.42 + profile.grass_weight + (1.0 - roughness) * 0.18 - moisture * 0.05,
            TerrainType.FOREST: 0.08 + profile.forest_weight + moisture * 0.55 - roughness * 0.08,
            TerrainType.STONE: 0.06 + profile.stone_weight + roughness * 0.72 - moisture * 0.05,
            TerrainType.FERTILE: 0.08 + profile.fertile_weight + fertility * 0.7 + wetland_strength * 0.18,
            TerrainType.WATER: 0.02 + profile.water_weight + moisture * 0.32 + river_strength * 1.85 - roughness * 0.06,
            TerrainType.HAZARD: 0.01 + profile.hazard_weight + danger * 0.72 + roughness * 0.08,
        }
        for anchor in anchors:
            dx = world_x - anchor.x
            dy = world_y - anchor.y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance > anchor.radius * 1.6:
                continue
            influence = max(0.0, 1.0 - distance / max(anchor.radius * 1.2, 1e-6))
            influence = smoothstep(influence)
            scores[anchor.terrain] += influence * anchor.weight
            if anchor.terrain == TerrainType.WATER:
                scores[TerrainType.FERTILE] += influence * 0.15
            if anchor.terrain == TerrainType.HAZARD:
                scores[TerrainType.GRASS] -= influence * 0.1
        scores[TerrainType.GRASS] += (noise - 0.5) * 0.08
        scores[TerrainType.FOREST] += noise * 0.1
        scores[TerrainType.STONE] += (1.0 - noise) * 0.08
        scores[TerrainType.FERTILE] += wetland_strength * 0.08
        self._apply_starter_bias(world_x, world_y, scores)
        terrain = max(scores.items(), key=lambda item: item[1])[0]

        quality = 0.62 + warped_noise(seed + 149, world_x, world_y, 15.0, 40.0) * 0.68
        amount = 0.0
        hazard_value = 0.0
        if terrain == TerrainType.FERTILE:
            amount = 8.0 + fertility * 12.0 + wetland_strength * 4.0
        elif terrain == TerrainType.FOREST:
            amount = 10.0 + moisture * 11.0
        elif terrain == TerrainType.STONE:
            amount = 10.0 + roughness * 11.0
        elif terrain == TerrainType.HAZARD:
            hazard_value = 0.4 + danger * 0.7
            amount = 0.0
        elif terrain == TerrainType.WATER:
            amount = 0.0
        return terrain, amount, quality, hazard_value

    def _starter_basin_anchors(self) -> list[PatchAnchor]:
        return [
            PatchAnchor(0.0, 0.0, TerrainType.GRASS, 20.0, 2.2),
            PatchAnchor(10.0, -6.0, TerrainType.FERTILE, 12.0, 1.85),
            PatchAnchor(-14.0, 8.0, TerrainType.FOREST, 14.0, 1.75),
            PatchAnchor(6.0, 14.0, TerrainType.STONE, 11.0, 1.55),
            PatchAnchor(0.0, -18.0, TerrainType.WATER, 8.0, 1.2),
        ]

    def _apply_starter_bias(self, world_x: int, world_y: int, scores: dict[TerrainType, float]) -> None:
        distance = float(np.hypot(world_x, world_y))
        starter_radius = self.config.chunk_size * 3.2
        if distance >= starter_radius:
            return
        safety = smoothstep(max(0.0, 1.0 - distance / starter_radius))
        scores[TerrainType.HAZARD] -= 3.2 * safety
        scores[TerrainType.WATER] -= 0.7 * safety
        scores[TerrainType.GRASS] += 0.9 * safety
        scores[TerrainType.FERTILE] += 0.45 * safety
        scores[TerrainType.FOREST] += 0.28 * safety
        scores[TerrainType.STONE] += 0.18 * safety

    def _smooth_terrain(self, terrain: np.ndarray) -> np.ndarray:
        smoothed = terrain.copy()
        height, width = terrain.shape
        for _ in range(3):
            updated = smoothed.copy()
            for y in range(height):
                for x in range(width):
                    counts: dict[int, int] = {}
                    for ny in range(max(0, y - 1), min(height, y + 2)):
                        for nx in range(max(0, x - 1), min(width, x + 2)):
                            tile = int(smoothed[ny, nx])
                            counts[tile] = counts.get(tile, 0) + 1
                    current = int(smoothed[y, x])
                    winner, count = max(counts.items(), key=lambda item: item[1])
                    if count >= 5:
                        updated[y, x] = winner
                    else:
                        updated[y, x] = current
            smoothed = updated
        return smoothed

    def _seed_starter_basin(
        self,
        chunk_x: int,
        chunk_y: int,
        terrain: np.ndarray,
        resource_amount: np.ndarray,
        resource_quality: np.ndarray,
        hazard: np.ndarray,
    ) -> None:
        size = self.config.chunk_size
        for local_y in range(size):
            for local_x in range(size):
                world_x = chunk_x * size + local_x
                world_y = chunk_y * size + local_y
                distance = float(np.hypot(world_x, world_y))
                if distance > size * 3.0:
                    continue
                safety = smoothstep(max(0.0, 1.0 - distance / (size * 3.0)))
                hazard[local_y, local_x] *= 1.0 - 0.98 * safety
                resource_quality[local_y, local_x] = max(resource_quality[local_y, local_x], 0.72 + 0.2 * safety)
                terrain_type = INDEX_TO_TERRAIN[int(terrain[local_y, local_x])]
                if terrain_type == TerrainType.HAZARD and safety > 0.2:
                    terrain[local_y, local_x] = TERRAIN_INDEX[TerrainType.GRASS]
                    resource_amount[local_y, local_x] = max(resource_amount[local_y, local_x], 3.0 + safety * 4.0)
                elif terrain_type == TerrainType.FERTILE:
                    resource_amount[local_y, local_x] = max(resource_amount[local_y, local_x], 8.0 + safety * 6.0)
                elif terrain_type == TerrainType.FOREST:
                    resource_amount[local_y, local_x] = max(resource_amount[local_y, local_x], 8.5 + safety * 5.5)
                elif terrain_type == TerrainType.STONE:
                    resource_amount[local_y, local_x] = max(resource_amount[local_y, local_x], 7.0 + safety * 4.0)
