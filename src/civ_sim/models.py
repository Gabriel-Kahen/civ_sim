from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from civ_sim.constants import ResourceType, StructureType


@dataclass(slots=True)
class Inventory:
    food: float = 0.0
    wood: float = 0.0
    stone: float = 0.0
    parts: float = 0.0

    def get(self, resource: ResourceType) -> float:
        return getattr(self, resource.value)

    def set(self, resource: ResourceType, value: float) -> None:
        setattr(self, resource.value, max(0.0, value))

    def add(self, resource: ResourceType, amount: float) -> float:
        current = self.get(resource)
        setattr(self, resource.value, current + amount)
        return amount

    def remove(self, resource: ResourceType, amount: float) -> float:
        current = self.get(resource)
        taken = min(current, amount)
        setattr(self, resource.value, current - taken)
        return taken

    def can_afford(self, costs: dict[ResourceType, float]) -> bool:
        return all(self.get(resource) >= amount for resource, amount in costs.items())

    def spend(self, costs: dict[ResourceType, float]) -> bool:
        if not self.can_afford(costs):
            return False
        for resource, amount in costs.items():
            self.remove(resource, amount)
        return True

    def total(self) -> float:
        return self.food + self.wood + self.stone + self.parts

    def to_dict(self) -> dict[str, float]:
        return {
            "food": self.food,
            "wood": self.wood,
            "stone": self.stone,
            "parts": self.parts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float] | None) -> "Inventory":
        payload = payload or {}
        return cls(
            food=float(payload.get("food", 0.0)),
            wood=float(payload.get("wood", 0.0)),
            stone=float(payload.get("stone", 0.0)),
            parts=float(payload.get("parts", 0.0)),
        )


@dataclass(slots=True)
class Structure:
    structure_id: int
    kind: StructureType
    x: int
    y: int
    lineage_id: int
    health: float
    max_health: float
    created_tick: int
    inventory: Inventory | None = None
    stored_throughput: float = 0.0
    processed_throughput: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "kind": self.kind.value,
            "x": self.x,
            "y": self.y,
            "lineage_id": self.lineage_id,
            "health": self.health,
            "max_health": self.max_health,
            "created_tick": self.created_tick,
            "inventory": None if self.inventory is None else self.inventory.to_dict(),
            "stored_throughput": self.stored_throughput,
            "processed_throughput": self.processed_throughput,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Structure":
        return cls(
            structure_id=int(payload["structure_id"]),
            kind=StructureType(payload["kind"]),
            x=int(payload["x"]),
            y=int(payload["y"]),
            lineage_id=int(payload["lineage_id"]),
            health=float(payload["health"]),
            max_health=float(payload["max_health"]),
            created_tick=int(payload["created_tick"]),
            inventory=Inventory.from_dict(payload.get("inventory")),
            stored_throughput=float(payload.get("stored_throughput", 0.0)),
            processed_throughput=float(payload.get("processed_throughput", 0.0)),
        )


@dataclass(slots=True)
class AgentGenome:
    hidden_size: int
    weights: dict[str, Any]
    traits: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "weights": {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in self.weights.items()
            },
            "traits": self.traits,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentGenome":
        return cls(
            hidden_size=int(payload["hidden_size"]),
            weights=dict(payload["weights"]),
            traits={key: float(value) for key, value in payload["traits"].items()},
        )


@dataclass(slots=True)
class Agent:
    agent_id: int
    lineage_id: int
    x: int
    y: int
    home_id: int | None
    age: int
    max_age: int
    energy: float
    carried_resource: str | None
    carried_amount: float
    hidden_state: list[float]
    genome: AgentGenome
    last_move_tick: int = 0
    birth_tick: int = 0
    generation: int = 0
    stuck_ticks: int = 0
    action_progress: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "lineage_id": self.lineage_id,
            "x": self.x,
            "y": self.y,
            "home_id": self.home_id,
            "age": self.age,
            "max_age": self.max_age,
            "energy": self.energy,
            "carried_resource": self.carried_resource,
            "carried_amount": self.carried_amount,
            "hidden_state": (
                self.hidden_state.tolist()
                if hasattr(self.hidden_state, "tolist")
                else self.hidden_state
            ),
            "genome": self.genome.to_dict(),
            "last_move_tick": self.last_move_tick,
            "birth_tick": self.birth_tick,
            "generation": self.generation,
            "stuck_ticks": self.stuck_ticks,
            "action_progress": self.action_progress,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Agent":
        return cls(
            agent_id=int(payload["agent_id"]),
            lineage_id=int(payload["lineage_id"]),
            x=int(payload["x"]),
            y=int(payload["y"]),
            home_id=payload.get("home_id"),
            age=int(payload["age"]),
            max_age=int(payload["max_age"]),
            energy=float(payload["energy"]),
            carried_resource=payload.get("carried_resource"),
            carried_amount=float(payload["carried_amount"]),
            hidden_state=[float(value) for value in payload["hidden_state"]],
            genome=AgentGenome.from_dict(payload["genome"]),
            last_move_tick=int(payload.get("last_move_tick", 0)),
            birth_tick=int(payload.get("birth_tick", 0)),
            generation=int(payload.get("generation", 0)),
            stuck_ticks=int(payload.get("stuck_ticks", 0)),
            action_progress=float(payload.get("action_progress", 0.0)),
        )


@dataclass(slots=True)
class Chunk:
    chunk_x: int
    chunk_y: int
    terrain: np.ndarray
    resource_amount: np.ndarray
    resource_quality: np.ndarray
    hazard: np.ndarray
    active: bool = True
    structures: dict[tuple[int, int], Structure] = field(default_factory=dict)
    traffic: np.ndarray | None = None
    lineage_map: np.ndarray | None = None
    frontier_value: np.ndarray | None = None
    ground_resources: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.traffic is None:
            self.traffic = np.zeros_like(self.resource_amount, dtype=np.float32)
        if self.lineage_map is None:
            self.lineage_map = np.zeros_like(self.resource_amount, dtype=np.int32)
        if self.frontier_value is None:
            self.frontier_value = np.zeros_like(self.resource_amount, dtype=np.float32)
        if self.ground_resources is None:
            self.ground_resources = np.zeros((*self.resource_amount.shape, 4), dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_x": self.chunk_x,
            "chunk_y": self.chunk_y,
            "terrain": self.terrain.tolist(),
            "resource_amount": self.resource_amount.tolist(),
            "resource_quality": self.resource_quality.tolist(),
            "hazard": self.hazard.tolist(),
            "active": self.active,
            "structures": [
                structure.to_dict()
                for structure in sorted(self.structures.values(), key=lambda item: item.structure_id)
            ],
            "traffic": self.traffic.tolist(),
            "lineage_map": self.lineage_map.tolist(),
            "frontier_value": self.frontier_value.tolist(),
            "ground_resources": self.ground_resources.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        chunk = cls(
            chunk_x=int(payload["chunk_x"]),
            chunk_y=int(payload["chunk_y"]),
            terrain=np.asarray(payload["terrain"], dtype=np.int16),
            resource_amount=np.asarray(payload["resource_amount"], dtype=np.float32),
            resource_quality=np.asarray(payload["resource_quality"], dtype=np.float32),
            hazard=np.asarray(payload["hazard"], dtype=np.float32),
            active=bool(payload["active"]),
            traffic=np.asarray(payload["traffic"], dtype=np.float32),
            lineage_map=np.asarray(payload["lineage_map"], dtype=np.int32),
            frontier_value=np.asarray(payload["frontier_value"], dtype=np.float32),
            ground_resources=np.asarray(payload["ground_resources"], dtype=np.float32),
        )
        chunk.structures = {}
        for item in payload["structures"]:
            structure = Structure.from_dict(item)
            chunk.structures[(structure.x, structure.y)] = structure
        return chunk


@dataclass(slots=True)
class SimulationStats:
    population: int = 0
    births: int = 0
    deaths: int = 0
    active_chunks: int = 0
    homes: int = 0
    road_length: int = 0
    storage_throughput: float = 0.0
    workshop_throughput: float = 0.0
    frontier_expansion_rate: float = 0.0
    extraction_food: float = 0.0
    extraction_wood: float = 0.0
    extraction_stone: float = 0.0
    delivered_food: float = 0.0
    delivered_wood: float = 0.0
    delivered_stone: float = 0.0
    delivered_parts: float = 0.0
    district_specialization: float = 0.0
    hub_centrality: float = 0.0
    lineage_sizes: dict[int, int] = field(default_factory=dict)
    structure_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "births": self.births,
            "deaths": self.deaths,
            "active_chunks": self.active_chunks,
            "homes": self.homes,
            "road_length": self.road_length,
            "storage_throughput": self.storage_throughput,
            "workshop_throughput": self.workshop_throughput,
            "frontier_expansion_rate": self.frontier_expansion_rate,
            "extraction_food": self.extraction_food,
            "extraction_wood": self.extraction_wood,
            "extraction_stone": self.extraction_stone,
            "delivered_food": self.delivered_food,
            "delivered_wood": self.delivered_wood,
            "delivered_stone": self.delivered_stone,
            "delivered_parts": self.delivered_parts,
            "district_specialization": self.district_specialization,
            "hub_centrality": self.hub_centrality,
            "lineage_sizes": self.lineage_sizes,
            "structure_counts": self.structure_counts,
        }
