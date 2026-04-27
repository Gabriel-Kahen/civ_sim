from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from civ_sim.config import SimConfig
from civ_sim.constants import ActionType, Direction
from civ_sim.models import AgentGenome


TRAIT_RANGES = {
    "vision_range": (3.0, 6.0),
    "carry_capacity": (2.0, 8.0),
    "move_speed": (0.55, 1.45),
    "harvest_food_bias": (-1.0, 1.0),
    "harvest_wood_bias": (-1.0, 1.0),
    "harvest_stone_bias": (-1.0, 1.0),
    "build_path_bias": (-1.0, 1.0),
    "build_storage_bias": (-1.0, 1.0),
    "build_beacon_bias": (-1.0, 1.0),
    "build_wall_bias": (-1.0, 1.0),
    "repair_bias": (-1.0, 1.0),
    "home_attachment": (-1.0, 1.0),
    "risk_tolerance": (-1.0, 1.0),
    "social_attraction": (-1.0, 1.0),
    "explore_bias": (-1.0, 1.0),
    "gate_preference": (-1.0, 1.0),
}

TRAIT_ORDER = tuple(TRAIT_RANGES.keys())

FOUNDER_ARCHETYPES = {
    "forager": {
        "traits": {
            "carry_capacity": 5.8,
            "move_speed": 1.15,
            "harvest_food_bias": 0.9,
            "harvest_wood_bias": 0.1,
            "harvest_stone_bias": -0.2,
            "social_attraction": 0.2,
            "explore_bias": 0.6,
            "risk_tolerance": 0.1,
        },
        "action_bias": {
            ActionType.MOVE: 0.8,
            ActionType.HARVEST: 1.0,
            ActionType.DEPOSIT: 0.7,
            ActionType.WITHDRAW: 0.2,
            ActionType.BUILD_PATH: 0.15,
            ActionType.IDLE: -0.4,
        },
    },
    "courier": {
        "traits": {
            "carry_capacity": 6.4,
            "move_speed": 1.2,
            "social_attraction": 0.7,
            "explore_bias": 0.2,
            "build_storage_bias": 0.35,
            "build_beacon_bias": 0.2,
        },
        "action_bias": {
            ActionType.MOVE: 0.9,
            ActionType.PICKUP: 0.7,
            ActionType.DEPOSIT: 1.0,
            ActionType.WITHDRAW: 0.8,
            ActionType.BUILD_STORAGE: 0.5,
            ActionType.BUILD_PATH: 0.35,
            ActionType.IDLE: -0.5,
        },
    },
    "builder": {
        "traits": {
            "carry_capacity": 4.4,
            "move_speed": 0.95,
            "build_path_bias": 0.9,
            "build_storage_bias": 0.55,
            "build_beacon_bias": 0.4,
            "home_attachment": 0.4,
            "repair_bias": 0.7,
            "social_attraction": 0.5,
            "explore_bias": 0.15,
        },
        "action_bias": {
            ActionType.MOVE: 0.5,
            ActionType.DEPOSIT: 0.7,
            ActionType.WITHDRAW: 0.75,
            ActionType.BUILD_PATH: 0.9,
            ActionType.BUILD_STORAGE: 0.6,
            ActionType.BUILD_BEACON: 0.45,
            ActionType.BUILD_HOME: 0.35,
            ActionType.BUILD_WORKSHOP: 0.45,
            ActionType.REPAIR: 0.8,
            ActionType.IDLE: -0.6,
        },
    },
    "mason": {
        "traits": {
            "carry_capacity": 4.8,
            "move_speed": 0.9,
            "harvest_stone_bias": 0.6,
            "build_wall_bias": 0.9,
            "gate_preference": 0.7,
            "repair_bias": 0.8,
            "risk_tolerance": 0.45,
        },
        "action_bias": {
            ActionType.MOVE: 0.45,
            ActionType.HARVEST: 0.35,
            ActionType.DEPOSIT: 0.55,
            ActionType.WITHDRAW: 0.6,
            ActionType.BUILD_WALL: 0.95,
            ActionType.BUILD_GATE: 0.7,
            ActionType.REPAIR: 0.85,
            ActionType.GUARD: 0.35,
            ActionType.IDLE: -0.5,
        },
    },
}


@dataclass(slots=True)
class ControllerOutput:
    action_logits: torch.Tensor
    direction_logits: torch.Tensor
    hidden_state: torch.Tensor


class ControllerRuntime:
    def __init__(self, config: SimConfig, observation_size: int):
        self.config = config
        self.observation_size = observation_size
        self.hidden_size = config.hidden_size
        self.action_size = len(ActionType)
        self.direction_size = len(Direction)

    def create_random_genome(self) -> AgentGenome:
        hidden_size = self.hidden_size
        input_size = self.observation_size
        weights = {
            "gru_weight_ih": (torch.randn(hidden_size * 3, input_size) * 0.12).tolist(),
            "gru_weight_hh": (torch.randn(hidden_size * 3, hidden_size) * 0.12).tolist(),
            "gru_bias_ih": (torch.randn(hidden_size * 3) * 0.02).tolist(),
            "gru_bias_hh": (torch.randn(hidden_size * 3) * 0.02).tolist(),
            "action_weight": (torch.randn(self.action_size, hidden_size) * 0.15).tolist(),
            "action_bias": torch.zeros(self.action_size).tolist(),
            "direction_weight": (torch.randn(self.direction_size, hidden_size) * 0.15).tolist(),
            "direction_bias": torch.zeros(self.direction_size).tolist(),
        }
        traits = {
            name: low + (high - low) * torch.rand(1).item()
            for name, (low, high) in TRAIT_RANGES.items()
        }
        return AgentGenome(hidden_size=hidden_size, weights=weights, traits=traits)

    def create_founder_genome(self, archetype: str) -> AgentGenome:
        genome = self.create_random_genome()
        spec = FOUNDER_ARCHETYPES[archetype]
        traits = dict(genome.traits)
        traits.update(spec.get("traits", {}))
        genome.traits = traits

        weights = {key: self._tensor(value) for key, value in genome.weights.items()}
        action_bias = weights["action_bias"]
        for action, value in spec.get("action_bias", {}).items():
            action_bias[int(action)] = value
        direction_bias = weights["direction_bias"]
        direction_bias[int(Direction.CENTER)] = -0.2
        weights["action_bias"] = action_bias
        weights["direction_bias"] = direction_bias
        genome.weights = {key: tensor.tolist() for key, tensor in weights.items()}
        return genome

    def mutate_genome(self, genome: AgentGenome, strength: float = 0.06) -> AgentGenome:
        weights: dict[str, list] = {}
        for key, value in genome.weights.items():
            tensor = self._tensor(value)
            noise = torch.randn_like(tensor) * strength
            if tensor.ndim == 1:
                noise *= 0.5
            weights[key] = (tensor + noise).tolist()

        traits: dict[str, float] = {}
        for name, old_value in genome.traits.items():
            low, high = TRAIT_RANGES[name]
            sigma = (high - low) * 0.08
            mutated = float(old_value + torch.randn(1).item() * sigma)
            traits[name] = min(high, max(low, mutated))
        return AgentGenome(hidden_size=genome.hidden_size, weights=weights, traits=traits)

    def forward(self, genome: AgentGenome, observation: torch.Tensor, hidden_state: torch.Tensor) -> ControllerOutput:
        params = {key: self._tensor(value) for key, value in genome.weights.items()}
        x = observation
        h = hidden_state
        gate_x = F.linear(x, params["gru_weight_ih"], params["gru_bias_ih"])
        gate_h = F.linear(h, params["gru_weight_hh"], params["gru_bias_hh"])
        i_r, i_z, i_n = gate_x.chunk(3)
        h_r, h_z, h_n = gate_h.chunk(3)
        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        next_hidden = (1.0 - update_gate) * new_gate + update_gate * h
        action_logits = F.linear(next_hidden, params["action_weight"], params["action_bias"])
        direction_logits = F.linear(next_hidden, params["direction_weight"], params["direction_bias"])
        return ControllerOutput(
            action_logits=action_logits,
            direction_logits=direction_logits,
            hidden_state=next_hidden,
        )

    def initial_hidden(self) -> list[float]:
        return [0.0 for _ in range(self.hidden_size)]

    def trait_vector(self, genome: AgentGenome) -> list[float]:
        vector = []
        for name in TRAIT_ORDER:
            low, high = TRAIT_RANGES[name]
            value = genome.traits[name]
            normalized = (value - low) / max(high - low, 1e-6)
            vector.append(normalized * 2.0 - 1.0)
        return vector

    def _tensor(self, values: list) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32)

    def carry_capacity(self, genome: AgentGenome) -> float:
        return genome.traits["carry_capacity"]

    def move_speed(self, genome: AgentGenome) -> float:
        return genome.traits["move_speed"]

    def vision_range(self, genome: AgentGenome) -> int:
        return max(3, min(self.config.observation_radius, int(math.floor(genome.traits["vision_range"]))))
