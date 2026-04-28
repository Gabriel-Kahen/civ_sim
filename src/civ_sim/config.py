from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from civ_sim.constants import ResourceType, StructureType


def _structure_values(
    path: float,
    storage: float,
    beacon: float,
    wall: float,
    gate: float,
    home: float,
    workshop: float,
) -> dict[str, float]:
    return {
        StructureType.PATH.value: path,
        StructureType.STORAGE.value: storage,
        StructureType.BEACON.value: beacon,
        StructureType.WALL.value: wall,
        StructureType.GATE.value: gate,
        StructureType.HOME.value: home,
        StructureType.WORKSHOP.value: workshop,
    }


def _parts_costs(
    path: float,
    storage: float,
    beacon: float,
    wall: float,
    gate: float,
    home: float,
    workshop: float,
) -> dict[str, dict[str, float]]:
    return {
        kind: {ResourceType.PARTS.value: amount}
        for kind, amount in _structure_values(path, storage, beacon, wall, gate, home, workshop).items()
    }


@dataclass(slots=True)
class SimConfig:
    seed: int = 7
    chunk_size: int = 16
    observation_radius: int = 5
    initial_active_radius_chunks: int = 2
    initial_home_count: int = 3
    initial_home_spacing: int = 18
    spawn_radius: int = 6
    initial_agents: int = 10
    max_agents: int = 100
    tick_limit: int = 0
    tile_size: int = 16
    fps: int = 30
    export_every: int = 50
    video_center_x: int = 0
    video_center_y: int = 9
    video_view_width_tiles: int = 128
    video_view_height_tiles: int = 128
    video_pixels_per_tile: int = 8
    influence_activation_threshold: float = 3.0
    influence_radius: int = 6
    dormancy_threshold: float = 0.65
    expansion_density_threshold: float = 8.0
    expansion_min_structures: int = 120
    resource_regen_rate: float = 0.006
    quality_drift_strength: float = 0.0008
    hazard_damage_scale: float = 0.06
    base_energy_decay: float = 0.024
    carried_energy_decay: float = 0.004
    founder_energy_decay_scale: float = 0.22
    founder_support_ticks: int = 1600
    home_support_radius: int = 14
    home_support_energy_threshold: float = 18.0
    home_support_food_amount: float = 0.12
    home_support_food_reserve: float = 95.0
    food_energy_gain: float = 9.0
    carried_food_eat_threshold: float = 12.0
    structure_food_eat_threshold: float = 12.5
    max_energy: float = 30.0
    founder_energy_min: float = 24.0
    founder_energy_random: float = 5.0
    founder_carried_parts: float = 7.0
    offspring_energy: float = 22.0
    agent_min_lifespan: int = 18000
    agent_max_lifespan: int = 45000
    harvest_amount: float = 1.0
    pickup_amount: float = 1.5
    deposit_amount: float = 3.0
    withdraw_amount: float = 1.5
    home_food_upkeep: float = 0.006
    home_starvation_damage: float = 0.004
    home_low_access_damage: float = 0.003
    reproduction_food_threshold: float = 20.0
    reproduction_parts_threshold: float = 2.5
    reproduction_food_cost: float = 3.0
    reproduction_parts_cost: float = 0.5
    home_birth_interval: int = 60
    opportunistic_deposit_amount: float = 1.25
    seed_home_food: float = 420.0
    seed_home_wood: float = 34.0
    seed_home_stone: float = 34.0
    seed_home_parts: float = 100.0
    seed_ground_cache: float = 24.0
    new_home_food: float = 45.0
    workshop_parts_output: float = 2.0
    workshop_wood_input: float = 1.0
    workshop_stone_input: float = 1.0
    structure_decay_multiplier: float = 0.55
    build_health_fraction: float = 0.75
    repair_health_fraction: float = 0.35
    guard_repair_bonus: float = 0.18
    anti_stuck_ticks: int = 12
    hidden_size: int = 32
    lineage_color_pool: int = 256
    path_traffic_threshold: float = 1.2
    storage_hub_threshold: float = 4.5
    workshop_resource_threshold: float = 0.0
    wall_hazard_threshold: float = 0.55
    beacon_frontier_min: float = 0.8
    home_min_distance: int = 8
    path_adjacent_settlement_threshold: int = 1
    storage_min_distance: int = 4
    storage_crossroads_threshold: int = 1
    storage_access_threshold: float = 1.0
    beacon_min_distance: int = 10
    beacon_traffic_multiplier: float = 1.2
    wall_min_influence: float = 0.7
    wall_edge_influence: float = 0.25
    wall_exposure_threshold: int = 2
    gate_wall_neighbors_threshold: int = 2
    gate_path_neighbors_threshold: int = 1
    home_min_influence: float = 0.75
    home_resource_radius: int = 4
    home_surplus_threshold: float = 55.0
    home_access_threshold: float = 1.0
    home_competition_radius: int = 6
    workshop_min_distance: int = 10
    workshop_resource_radius: int = 6
    workshop_access_threshold: float = 0.0
    traffic_decay: float = 0.985
    path_move_cost: float = 0.55
    forest_move_cost: float = 1.1
    stone_move_cost: float = 1.05
    hazard_move_cost: float = 1.2
    structure_health: dict[str, float] = field(
        default_factory=lambda: _structure_values(40.0, 120.0, 80.0, 130.0, 110.0, 180.0, 140.0)
    )
    structure_decay: dict[str, float] = field(
        default_factory=lambda: _structure_values(0.003, 0.006, 0.006, 0.007, 0.007, 0.005, 0.007)
    )
    structure_build_costs: dict[str, dict[str, float]] = field(
        default_factory=lambda: _parts_costs(0.6, 5.0, 4.0, 4.0, 5.0, 10.0, 8.0)
    )
    structure_repair_costs: dict[str, dict[str, float]] = field(
        default_factory=lambda: _parts_costs(0.15, 0.35, 0.3, 0.3, 0.35, 0.4, 0.4)
    )
    influence_strength: dict[str, float] = field(
        default_factory=lambda: _structure_values(0.45, 2.0, 2.8, 0.5, 1.0, 2.6, 2.2)
    )
    output_root: Path = field(default_factory=lambda: Path("exports"))
    autosave_path: Path = field(default_factory=lambda: Path("exports/autosave.pkl"))

    @property
    def observation_diameter(self) -> int:
        return self.observation_radius * 2 + 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SimConfig":
        payload = dict(payload)
        for path_field in ("output_root", "autosave_path"):
            if path_field in payload:
                payload[path_field] = Path(payload[path_field])
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        return cls(**payload)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"config YAML must contain a mapping: {path}")
        return cls.from_dict(payload)

    def structure_max_health(self, kind: StructureType) -> float:
        return float(self.structure_health[kind.value])

    def structure_decay_rate(self, kind: StructureType) -> float:
        return float(self.structure_decay[kind.value]) * self.structure_decay_multiplier

    def structure_build_cost(self, kind: StructureType, resource: ResourceType) -> float:
        return float(self.structure_build_costs[kind.value][resource.value])

    def structure_repair_cost(self, kind: StructureType, resource: ResourceType) -> float:
        return float(self.structure_repair_costs[kind.value][resource.value])

    def structure_influence_strength(self, kind: StructureType) -> float:
        return float(self.influence_strength[kind.value])


DEFAULT_CONFIG = SimConfig()
